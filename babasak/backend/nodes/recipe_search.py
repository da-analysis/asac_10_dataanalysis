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
        # "김치찌개랑 비슷한 거", "이거랑 유사한 레시피"
        # 기준 메뉴 1개로 base recipe 잡고 → find_similar_recipes
        if is_similar and reference_menu:
            base_recipes = search_recipes(reference_menu, limit=1)
            if base_recipes:
                base = base_recipes[0]
                similar = find_similar_recipes(base["id"], limit=5, min_shared=2)
                for recipe in similar:
                    ings = get_recipe_ingredients(recipe["id"])
                    recipe_data.append({
                        "menu": recipe["name"],
                        "id": recipe["id"],
                        "servings": recipe.get("servings"),
                        "difficulty": recipe.get("difficulty"),
                        "shared_ingredient_count": recipe.get("shared"),
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
        if len(ingredients) >= 2:
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

            # 빈 결과 → 없는 메뉴(예: "마라김치찌개")
            # → build_graph_relation_context로 base + modifier 조합 컨텍스트 생성
            if not recipes:
                graph_rows = build_graph_relation_context(menu, limit=3)
                if graph_rows:
                    return {"recipe_info": {
                        "data": graph_rows,
                        "search_type": "graph_relation",
                        "note": f"'{menu}' 정확 매칭 없음, 그래프 관계로 조합 근거 제공",
                    }}

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
                recipes = filtered[:3] if filtered else recipes[:3]
            else:
                recipes = recipes[:3]
            for recipe in recipes:
                ings = get_recipe_ingredients(recipe["id"])
                detail = get_recipe_detail(recipe["id"])
                recipe_data.append({
                    "menu": recipe["name"],
                    "id": recipe["id"],
                    "servings": recipe.get("servings"),
                    "difficulty": recipe.get("difficulty"),
                    "cooking_time": recipe.get("cooking_time"),
                    "cooking_method": detail.get("cooking_method") if detail else None,
                    "description": detail.get("description") if detail else None,
                    # 신규: 조리단계 (report_generator가 답변에 포함하도록)
                    "steps": detail.get("steps") if detail else None,
                    "ingredients": ings,
                })
            return {"recipe_info": {"data": recipe_data, "search_type": "keyword"}}

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
