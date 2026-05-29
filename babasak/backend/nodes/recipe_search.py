"""
Neo4j 레시피 검색 노드.
db.py의 그래프 함수를 활용해 LangGraph state에 recipe_info를 채움.

지원 시나리오:
  1. 인기 레시피 폴백
  2. 조건 기반 추천 (난이도, 인분, 종류, 조리법)
  3. 재료 제외 검색 (알레르기 대응)
  4. 대체재 제안 (시나리오 4 — "두부 대신 쓸 거" 유사레시피 기반, 기존)
  5. 다중 재료 AND 검색
  6. 단일 재료 기반 검색
  7. 메뉴 키워드 검색 + 재료 상세 + 조리단계
     ↳ 없는 메뉴면 build_graph_relation_context로 그래프 조합 컨텍스트 생성
  8. 유사 레시피 추천 (재료 공유도 기반) — is_similar 또는 reference_menu
  9. 1:1 대체재 추천 (RAG 그래프) — substitute_for 들어왔을 때
     ↳ 같은 lv1 카테고리 + 해당 메뉴에 자주 등장하는 재료 후보 5개 반환
     ↳ cost_calculator가 이 후보를 가격사전과 매칭해 원가 낮은 거 선택 가능
"""
import re

from backend.db import (
    search_recipes,
    get_recipe_ingredients,
    get_recipe_detail,
    get_recipes_by_ingredient,
    get_recipes_by_multiple_ingredients,
    get_recipes_excluding_ingredient,
    recommend_recipes,
    get_popular_recipes,
    # 신규: 그래프 RAG 강화
    build_graph_relation_context,   # DB 없는 메뉴 → 그래프 조합 컨텍스트
    find_similar_recipes,           # 재료 공유도 기반 유사 레시피
    suggest_substitute_ingredient,  # 같은 카테고리 + 메뉴 빈도 기반 대체재
)
from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage

# 폴백용 키워드 리스트 (LLM 장애 시 보험)
_KNOWN_MENUS = [
    '김치찌개', '된장찌개', '불고기', '순두부찌개', '비빔밥',
    '제육볶음', '김치전', '떡볶이', '잡채', '오므라이스',
    '김밥', '부대찌개', '미역국', '소불고기덮밥', '치즈돈까스',
    '파전', '치킨카레', '삼계탕', '갈비탕', '냉면',
    '콩나물국', '마라탕', '부침개', '돈까스', '어묵국',
]

_KNOWN_INGREDIENTS = [
    '돼지고기', '소고기', '닭고기', '두부', '김치',
    '양파', '대파', '감자', '당근', '애호박',
    '버섯', '고추', '마늘', '계란', '우유',
    '밀가루', '설탕', '간장', '된장', '고춧가루',
    '참기름', '식용유', '소금', '후추', '새우',
]

_ALIASES = {
    '김찌': '김치찌개', '된찌': '된장찌개', '순찌': '순두부찌개',
    '부찌': '부대찌개', '김치찌게': '김치찌개', '된장찌게': '된장찌개',
    '돈까쓰': '돈까스', '떡볶히': '떡볶이',
}

# 양념/조미료/잡재료 — substitute 대상에서 제외 (대체할 가치 적음)
_SEASONING_TOKENS = {
    '소금', '후추', '설탕', '간장', '식용유', '참기름', '들기름', '올리브유',
    '마늘', '다진마늘', '생강', '고춧가루', '된장', '고추장', '다시다', '미원',
    '멸치액젓', '액젓', '맛술', '청주', '미림', '식초', '물', '깨', '통깨',
    '대파', '쪽파', '실파', '양파', '청양고추', '홍고추', '풋고추',
    '굴소스', '두반장', '연두', '간마늘', '월계수잎', '월계수',
}

# 비싼 단백질/해산물 — substitute 1순위 (원가 절감 효과 큼)
_PREMIUM_HINTS = (
    '고기', '살', '갈비', '삼겹', '목살', '안심', '등심', '닭', '돼지', '소', '오리',
    '새우', '오징어', '낙지', '문어', '조개', '꽃게', '전복', '관자',
    '연어', '참치', '고등어', '갈치', '명태', '코다리', '꽁치', '광어', '대구', '동태',
    '햄', '소시지', '베이컨', '스팸',
)
# 보조 주재료 — 단백질 없을 때 2순위
_SECONDARY_HINTS = ('두부', '유부', '버섯', '계란', '달걀', '어묵', '게맛살')


def _pick_main_ingredient(ings: list) -> str | None:
    """
    재료 리스트에서 substitute할 가치가 큰 '주재료' 1개를 추출.
    - 양념/조미료 스킵
    - 1순위: 비싼 단백질/해산물 (원가 절감 효과 큼)
    - 2순위: 두부/버섯/계란 등 보조 주재료
    - 그래도 없으면 양념 뺀 첫 재료
    """
    if not ings or not isinstance(ings, list):
        return None

    candidates = []
    for ing in ings:
        if isinstance(ing, dict):
            name = ing.get("name") or ing.get("ingredient") or ""
        else:
            name = str(ing)
        name = name.strip()
        if not name or name in _SEASONING_TOKENS or len(name) < 2:
            continue
        candidates.append(name)

    if not candidates:
        return None

    for name in candidates:
        if any(hint in name for hint in _PREMIUM_HINTS):
            return name
    for name in candidates:
        if any(hint in name for hint in _SECONDARY_HINTS):
            return name

    return candidates[0]

# ── LLM 키워드 추출 (lazy init) ──
_llm_keyword = None


def _get_llm_keyword():
    """키워드 추출용 경량 LLM (lazy init)"""
    global _llm_keyword
    if _llm_keyword is None:
        _llm_keyword = ChatDatabricks(endpoint="databricks-gpt-5-4-mini", temperature=0, max_tokens=20)
    return _llm_keyword


def _llm_extract_menu_keyword(query: str) -> str | None:
    """
    LLM으로 사용자 쿼리에서 음식/요리명만 추출.
    preprocessor가 entity 추출에 실패했을 때 fallback으로 사용.
    """
    try:
        messages = [
            SystemMessage(content=(
                "사용자 질문에서 요리 또는 음식 이름만 추출하세요. "
                "음식명만 출력하고 다른 말은 하지 마세요. "
                "음식명이 여러 개면 쉼표로 구분하세요. "
                "음식명이 없으면 NONE이라고만 출력하세요."
            )),
            HumanMessage(content=query),
        ]
        response = _get_llm_keyword().invoke(messages)
        result = response.content.strip()

        if not result or result.upper() == "NONE":
            return None

        # 쉼표로 구분된 경우 첫 번째만 사용
        keyword = result.split(",")[0].strip()
        # 빈 문자열이나 너무 긴 결과 필터링 (할루시네이션 방지)
        if not keyword or len(keyword) > 20:
            return None
        return keyword

    except Exception:
        return None


def _extract_keywords_from_query(query: str) -> dict:
    """원본 쿼리에서 메뉴/재료 키워드 직접 추출 (LLM 장애 시 보험용)"""
    found_menu = None
    found_ingredients = []

    for menu in _KNOWN_MENUS:
        if menu in query:
            found_menu = menu
            break

    if not found_menu:
        for alias, real in _ALIASES.items():
            if alias in query:
                found_menu = real
                break

    for ing in _KNOWN_INGREDIENTS:
        if ing in query:
            found_ingredients.append(ing)

    return {'menu': found_menu, 'ingredients': found_ingredients}


def recipe_search_node(state: dict) -> dict:
    """
    Neo4j에서 레시피/재료 정보를 조회하는 노드.
    entities의 의도 정보(exclude, conditions, is_popular, is_alternative)에 따라
    db.py의 적절한 함수를 선택하여 호출.
    """
    entities = state.get("entities", {})
    existing_errors = state.get("error_log", [])

    menu = entities.get("menu")
    ingredient = entities.get("ingredient")  # list or str or None
    exclude = entities.get("exclude")
    is_alternative = entities.get("is_alternative", False)
    conditions = entities.get("conditions")  # dict or None
    is_popular = entities.get("is_popular", False)
    # 신규(시나리오 8): "비슷한/유사한" 의도 + 기준 메뉴
    is_similar = entities.get("is_similar", False)
    reference_menu = entities.get("reference_menu") or menu
    # 신규(시나리오 9): "X 대신 Y" 1:1 대체재 추천 의도
    # preprocessor가 잡거나, cost_calculator가 원가 비싼 재료 발견 후 호출 가능
    substitute_for = entities.get("substitute_for")  # 대체할 재료 이름 (예: "돼지고기")

    # ingredient 정규화: str → list
    if isinstance(ingredient, str):
        ingredients = [ingredient]
    elif isinstance(ingredient, list):
        ingredients = ingredient
    else:
        ingredients = []

    # ★ fallback: entities 못 잡았으면 LLM → 고정 리스트 순서로 추출 시도
    if not menu and not ingredients and not is_popular and not conditions:
        query = state["messages"][-1].content if state.get("messages") else ""

        # 1차: LLM으로 음식명 추출
        llm_keyword = _llm_extract_menu_keyword(query)
        if llm_keyword:
            menu = llm_keyword
        else:
            # 2차: 고정 리스트 매칭 (LLM 장애 시 보험)
            fallback = _extract_keywords_from_query(query)
            menu = fallback.get("menu")
            ingredients = fallback.get("ingredients", [])

    # fallback 후에도 키워드 없으면 인기 레시피로 폴백
    if not menu and not ingredients and not conditions:
        is_popular = True

    recipe_data = []

    try:
        # === 시나리오 9: 대체재 1:1 추천 (RAG 그래프 기반) ===
        # "김치찌개에 돼지고기 대신 뭐 써?"
        # cost_calculator가 원가 비싼 재료 찾고 이 노드 재호출할 때도 사용 가능
        # → 같은 lv1 카테고리 + 해당 메뉴에 자주 나오는 재료 5개 반환
        if substitute_for and (menu or reference_menu):
            target_menu = menu or reference_menu
            substitutes = suggest_substitute_ingredient(
                target_menu, substitute_for, limit=5
            )
            return {"recipe_info": {
                "data": substitutes,
                "search_type": "substitute",
                "menu": target_menu,
                "missing_ingredient": substitute_for,
                "note": (
                    f"'{target_menu}'에서 '{substitute_for}' 대체 후보. "
                    "cost_calculator가 가격사전과 매칭해 원가 낮은 거 선택 가능."
                ),
            }}

        # === 시나리오 8: 유사 레시피 추천 (재료 공유도) ===
        # "김치찌개랑 비슷한 거", "김치찌개 대체메뉴" 등
        # 기준 메뉴 1개로 base recipe 잡고 → find_similar_recipes
        # steps도 같이 가져옴 (LLM 환각 방지, DB 원문 활용)
        if is_similar and reference_menu:
            base_recipes = search_recipes(reference_menu, limit=1)
            if base_recipes:
                base = base_recipes[0]
                similar = find_similar_recipes(base["id"], limit=3, min_shared=2)
                for recipe in similar:
                    ings = get_recipe_ingredients(recipe["id"])
                    detail = get_recipe_detail(recipe["id"])
                    recipe_data.append({
                        "menu": recipe["name"],
                        "id": recipe["id"],
                        "servings": recipe.get("servings"),
                        "difficulty": recipe.get("difficulty"),
                        "cooking_time": recipe.get("cooking_time"),
                        "shared_ingredient_count": recipe.get("shared"),
                        "steps": detail.get("steps") if detail else None,
                        "ingredients": ings,
                    })
                return {"recipe_info": {
                    "data": recipe_data,
                    "search_type": "similar",
                    "reference_menu": base["name"],
                    "reference_id": base["id"],
                }}

        # === 시나리오 1: 인기 레시피 ===
        if is_popular and not menu and not ingredients:
            recipes = get_popular_recipes(limit=5)
            for recipe in recipes:
                recipe_data.append({
                    "menu": recipe["name"],
                    "id": recipe["id"],
                    "servings": recipe.get("servings"),
                    "difficulty": recipe.get("difficulty"),
                    "view_count": recipe.get("view_count"),
                })
            return {"recipe_info": {"data": recipe_data, "search_type": "popular"}}

        # === 시나리오 2: 조건 기반 추천 ===
        if conditions:
            recipes = recommend_recipes(
                kind=conditions.get("kind"),
                difficulty=conditions.get("difficulty"),
                servings=conditions.get("servings"),
                cooking_method=conditions.get("cooking_method"),
                limit=5,
            )
            for recipe in recipes:
                recipe_data.append({
                    "menu": recipe["name"],
                    "id": recipe["id"],
                    "servings": recipe.get("servings"),
                    "difficulty": recipe.get("difficulty"),
                    "kind": recipe.get("kind"),
                    "cooking_time": recipe.get("cooking_time"),
                    "view_count": recipe.get("view_count"),
                })
            return {"recipe_info": {"data": recipe_data, "search_type": "condition", "conditions": conditions}}

        # === 시나리오 3: 재료 제외 검색 (알레르기 대응) ===
        if exclude and menu:
            recipes = get_recipes_excluding_ingredient(menu, exclude, limit=5)
            for recipe in recipes:
                ings = get_recipe_ingredients(recipe["id"])
                recipe_data.append({
                    "menu": recipe["name"],
                    "id": recipe["id"],
                    "servings": recipe.get("servings"),
                    "difficulty": recipe.get("difficulty"),
                    "ingredients": ings,
                    "excluded": exclude,
                })
            return {"recipe_info": {"data": recipe_data, "search_type": "exclude", "excluded": exclude}}

        # === 시나리오 4: 대체재 제안 ===
        if is_alternative and ingredients:
            target_ingredient = ingredients[0]
            recipes_with = get_recipes_by_ingredient(target_ingredient, limit=5)
            recipes_without_data = []
            for recipe in recipes_with:
                keyword = recipe["name"][:2]
                alt_recipes = get_recipes_excluding_ingredient(keyword, target_ingredient, limit=2)
                for alt in alt_recipes:
                    alt_ings = get_recipe_ingredients(alt["id"])
                    recipes_without_data.append({
                        "menu": alt["name"],
                        "id": alt["id"],
                        "servings": alt.get("servings"),
                        "ingredients": alt_ings,
                    })
            if recipes_without_data:
                return {"recipe_info": {"data": recipes_without_data[:5], "search_type": "alternative", "replaced": target_ingredient}}
            for recipe in recipes_with[:3]:
                ings = get_recipe_ingredients(recipe["id"])
                recipe_data.append({
                    "menu": recipe["name"],
                    "id": recipe["id"],
                    "ingredients": ings,
                })
            return {"recipe_info": {"data": recipe_data, "search_type": "by_ingredient", "note": f"{target_ingredient} 대체재 검색 시도했으나 결과 부족"}}

        # === 시나리오 5: 다중 재료 AND 검색 ===
        # menu가 같이 잡혔으면 시나리오 7(메뉴 검색)이 우선 — 메뉴 + 재료 둘 다일 때
        # ingredients만 있는 케이스로 한정 (조리단계 포함하는 시나리오 7 보존)
        if len(ingredients) >= 2 and not menu:
            recipes = get_recipes_by_multiple_ingredients(ingredients, limit=5)
            if recipes:
                for recipe in recipes:
                    ings = get_recipe_ingredients(recipe["id"])
                    recipe_data.append({
                        "menu": recipe["name"],
                        "id": recipe["id"],
                        "servings": recipe.get("servings"),
                        "difficulty": recipe.get("difficulty"),
                        "ingredients": ings,
                    })
                return {"recipe_info": {"data": recipe_data, "search_type": "multi_ingredient"}}
            recipes = get_recipes_by_ingredient(ingredients[0], limit=5)
            for recipe in recipes:
                recipe_data.append({
                    "menu": recipe["name"],
                    "id": recipe["id"],
                    "servings": recipe.get("servings"),
                })
            note = f"'{', '.join(ingredients)}' 전체 포함 레시피 없음, '{ingredients[0]}' 기준 검색"
            return {"recipe_info": {"data": recipe_data, "search_type": "single_ingredient_fallback", "note": note}}

        # === 시나리오 6: 단일 재료 기반 검색 ===
        if ingredients and not menu:
            recipes = get_recipes_by_ingredient(ingredients[0], limit=5)
            for recipe in recipes:
                recipe_data.append({
                    "menu": recipe["name"],
                    "id": recipe["id"],
                    "servings": recipe.get("servings"),
                    "difficulty": recipe.get("difficulty"),
                    "view_count": recipe.get("view_count"),
                })
            return {"recipe_info": {"data": recipe_data, "search_type": "by_ingredient"}}

        # === 시나리오 7: 메뉴 키워드 검색 + 재료 상세 + 조리단계 (기본) ===
        if menu:
            recipes = search_recipes(menu, limit=10)

            # ★ 원본 사용자 쿼리 추출 — preprocessor가 menu를 정규화(예: "마라김치찌개"→"김치찌개")
            #    하기 때문에, modifier(마라) 정보를 살리려면 원본 쿼리에서 가져와야 함.
            user_query = ""
            messages = state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                user_query = getattr(last_msg, "content", "") or ""

            # 의도 단어 제거해서 핵심 키워드만 남김
            clean_query = re.sub(
                r'(레시피|알려줘|만드는\s*법|조리법|추천|어떻게|만들기|만들고\s*싶어|알고\s*싶어|보여줘)',
                '',
                user_query,
            ).strip()
            clean_query_compact = re.sub(r'\s+', '', clean_query)
            menu_compact = re.sub(r'\s+', '', menu)

            # 원본 쿼리가 menu보다 더 긴 키워드를 포함하면 → modifier가 있다
            # 예: clean_query="마라김치찌개", menu="김치찌개" → modifier="마라"
            has_modifier = (
                clean_query_compact
                and menu_compact
                and clean_query_compact != menu_compact
                and len(clean_query_compact) > len(menu_compact)
            )

            # 1) DB 결과가 비었거나
            # 2) menu가 결과에 매칭 안 되거나
            # 3) modifier가 있으면 (preprocessor가 정규화한 케이스)
            # → graph_relation_context 호출
            exact_match = any(
                menu_compact in re.sub(r'\s+', '', r.get("name", ""))
                for r in recipes
            )
            need_graph_context = (not recipes) or (not exact_match) or has_modifier

            if need_graph_context:
                # menu 대신 원본 쿼리(modifier 포함)로 graph_context 호출
                graph_query = clean_query if has_modifier else menu
                graph_rows = build_graph_relation_context(graph_query, limit=3)
                if graph_rows:
                    if not recipes:
                        return {"recipe_info": {
                            "data": graph_rows,
                            "search_type": "graph_relation",
                            "note": f"'{graph_query}' 정확 매칭 없음, 그래프 관계로 조합 근거 제공",
                        }}
                    # 아래 일반 처리 후 마지막에 graph_rows 부착

            if conditions:
                filtered = []
                for r in recipes:
                    match = True
                    if conditions.get("difficulty") and r.get("difficulty") != conditions["difficulty"]:
                        match = False
                    if conditions.get("servings") and r.get("servings") != conditions["servings"]:
                        match = False
                    if match:
                        filtered.append(r)
                recipes = filtered if filtered else recipes
            # 같은 메뉴명 중복 제거 — [:3] 이전에 수행
            # search_recipes_smart는 score DESC로 가져오므로,
            # 같은 이름 중에선 자동으로 1등(점수 최고)이 채택됨.
            # 그 후 다른 변형 메뉴들이 2~3번에 들어와 다양성 확보.
            # 예: ["김치찌개"(A,점수1위), "김치찌개"(B), "김치찌개"(C), "차돌김치찌개", "돼지목살김치찌개"]
            #     → 중복 제거 → ["김치찌개"(A), "차돌김치찌개", "돼지목살김치찌개"]
            #     → [:3] → 그대로 3개 ✅
            seen_names = set()
            unique_recipes = []
            for r in recipes:
                if r["name"] not in seen_names:
                    seen_names.add(r["name"])
                    unique_recipes.append(r)
            recipes = unique_recipes[:3]

            for recipe in recipes:
                ings = get_recipe_ingredients(recipe["id"])
                detail = get_recipe_detail(recipe["id"])

                # 주재료 1개에 대한 대체재 후보 미리 조회 (report_generator의
                # '💡 비싼 재료 → 대체 제안' 섹션에서 활용).
                # Neo4j 쿼리 1회 추가 — 레시피 3개 × 1회 = 3회. 테스트 단계에서는 감수.
                main_ing = _pick_main_ingredient(ings)
                substitute_suggestions = None
                if main_ing:
                    try:
                        subs = suggest_substitute_ingredient(
                            recipe["name"], main_ing, limit=3
                        )
                        if subs:
                            substitute_suggestions = {
                                "target": main_ing,
                                "candidates": subs,
                            }
                    except Exception:
                        substitute_suggestions = None

                recipe_data.append({
                    "menu": recipe["name"],
                    "id": recipe["id"],
                    "servings": recipe.get("servings"),
                    "difficulty": recipe.get("difficulty"),
                    "cooking_time": recipe.get("cooking_time"),
                    "cooking_method": detail.get("cooking_method") if detail else None,
                    "description": detail.get("description") if detail else None,
                    "steps": detail.get("steps") if detail else None,
                    "ingredients": ings,
                    "substitute_suggestions": substitute_suggestions,
                })

            # 부분 매칭만 됐을 때 OR modifier가 있을 때 — graph_relation 컨텍스트 같이 반환
            # ("마라김치찌개" → 김치찌개 3개 + 마라 그래프 컨텍스트로 짬뽕)
            result_payload = {
                "data": recipe_data,
                "search_type": "keyword",
            }
            if need_graph_context:
                graph_query = clean_query if has_modifier else menu
                graph_rows = build_graph_relation_context(graph_query, limit=2)
                if graph_rows:
                    result_payload["graph_context"] = graph_rows
                    result_payload["search_type"] = "keyword_with_graph_context"
                    result_payload["note"] = (
                        f"'{graph_query}' 정확 매칭은 없음. "
                        f"base 메뉴('{menu}') + modifier 그래프 정보를 짬뽕해서 답변 권장."
                    )
                    result_payload["original_query"] = graph_query

                    # 'ai_new_menu_suggestions' — 사용자가 실제 입력한 modifier만 사용.
                    # ★ graph_rows의 modifier_menus는 substring 오매칭(마라→고구마라떼)이
                    #    섞일 수 있어 신메뉴 후보로는 쓰지 않는다.
                    #    대신 원본 쿼리에서 base 메뉴를 뺀 나머지를 modifier로 본다.
                    #    예: clean_query="마라김치찌개", menu="김치찌개" → modifier="마라"
                    modifier_text = ""
                    if has_modifier:
                        modifier_text = clean_query_compact.replace(menu_compact, "").strip()
                    if modifier_text:
                        result_payload["ai_new_menu_suggestions"] = [{
                            "name": graph_query,
                            "base_menu": menu,
                            "modifier": modifier_text,
                            "combo_label": f"{menu} + {modifier_text}",
                        }]
            return {"recipe_info": result_payload}

        # === 최종 폴백: 인기 레시피 ===
        recipes = get_popular_recipes(limit=5)
        for recipe in recipes:
            recipe_data.append({
                "menu": recipe["name"],
                "id": recipe["id"],
                "servings": recipe.get("servings"),
                "view_count": recipe.get("view_count"),
            })
        return {"recipe_info": {"data": recipe_data, "search_type": "popular_fallback"}}

    except Exception as e:
        return {
            "recipe_info": {},
            "error_log": existing_errors + [f"recipe_search: {str(e)}"],
        }
