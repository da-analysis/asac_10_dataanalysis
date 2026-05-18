"""
Neo4j 레시피 검색 노드.
기존 agent.py의 neo4j_graph_query 로직을 포팅하여
db.py의 8개 함수를 모두 활용.

지원 시나리오:
  1. 메뉴 키워드 검색 + 재료 상세
  2. 단일 재료 기반 레시피 역추적
  3. 다중 재료 AND 검색
  4. 재료 제외 검색 (알레르기 대응)
  5. 조건 기반 추천 (난이도, 인분, 종류, 조리법)
  6. 인기 레시피 폴백
  7. 대체재 제안 (특정 재료가 들어간 레시피 → 해당 재료 제외 레시피)
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
)

# 폴백용 키워드 리스트 (preprocessor 실패 시)
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


def _extract_keywords_from_query(query: str) -> dict:
    """원본 쿼리에서 메뉴/재료 키워드 직접 추출 (preprocessor fallback용)"""
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

    # ingredient 정규화: str → list
    if isinstance(ingredient, str):
        ingredients = [ingredient]
    elif isinstance(ingredient, list):
        ingredients = ingredient
    else:
        ingredients = []

    # fallback: entities 못 잡았으면 원본 쿼리에서 직접 추출
    if not menu and not ingredients and not is_popular and not conditions:
        query = state["messages"][-1].content if state.get("messages") else ""
        fallback = _extract_keywords_from_query(query)
        menu = fallback.get("menu")
        ingredients = fallback.get("ingredients", [])

    # fallback 후에도 키워드 없으면 인기 레시피로 폴백
    if not menu and not ingredients and not conditions:
        is_popular = True

    recipe_data = []

    try:
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
            # "두부 대신 쓸 수 있는 거" → 두부 제외 레시피 검색
            target_ingredient = ingredients[0]
            recipes_with = get_recipes_by_ingredient(target_ingredient, limit=5)
            recipes_without_data = []
            for recipe in recipes_with:
                # 해당 재료 없이 만들 수 있는 유사 레시피 찾기
                keyword = recipe["name"][:2]  # 레시피명 앞 2글자
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
            # 폴백: 해당 재료가 들어간 레시피라도 반환
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
            # AND 검색 결과 없으면 첫 재료로 폴백
            recipes = get_recipes_by_ingredient(ingredients[0], limit=5)
            for recipe in recipes:
                recipe_data.append({
                    "menu": recipe["name"],
                    "id": recipe["id"],
                    "servings": recipe.get("servings"),
                })
            return {"recipe_info": {"data": recipe_data, "search_type": "single_ingredient_fallback", "note": f"'{', '.join(ingredients)}' 전체 포함 레시피 없음, '{ingredients[0]}' 기준 검색"}}

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

        # === 시나리오 7: 메뉴 키워드 검색 + 재료 상세 (기본) ===
        if menu:
            recipes = search_recipes(menu, limit=3)
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
