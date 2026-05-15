from backend.db import search_recipes, get_recipe_ingredients, get_recipes_by_ingredient

# 원본 쿼리에서 메뉴/재료 키워드를 추출하기 위한 참조 리스트
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


def _extract_keywords_from_query(query: str) -> dict:
    """
    원본 쿼리에서 직접 메뉴/재료 키워드를 문자열 매칭으로 추출.
    preprocessor가 entity를 못 잡았을 때의 fallback용.
    """
    found_menu = None
    found_ingredient = None

    for menu in _KNOWN_MENUS:
        if menu in query:
            found_menu = menu
            break
    # 오타/줄임말 처리
    if not found_menu:
        _ALIASES = {
            '김찌': '김치찌개', '된찌': '된장찌개', '순찌': '순두부찌개',
            '부찌': '부대찌개', '김치찌게': '김치찌개', '된장찌게': '된장찌개',
            '돈까쓰': '돈까스', '떡볶히': '떡볶이',
        }
        for alias, real in _ALIASES.items():
            if alias in query:
                found_menu = real
                break

    for ing in _KNOWN_INGREDIENTS:
        if ing in query:
            found_ingredient = ing
            break

    return {'menu': found_menu, 'ingredient': found_ingredient}


def recipe_search_node(state: dict) -> dict:
    """
    backend/db.py의 Neo4j 함수를 활용하여 레시피/재료 정보를 조회합니다.
    entities가 비어있으면 원본 쿼리에서 키워드를 직접 추출하여 검색합니다 (fallback).
    """
    entities = state.get("entities", {})
    menu = entities.get("menu")
    ingredient = entities.get("ingredient")

    # === Fallback: entities가 비어있으면 원본 쿼리에서 직접 추출 ===
    if not menu and not ingredient:
        query = state["messages"][-1].content if state.get("messages") else ""
        fallback = _extract_keywords_from_query(query)
        menu = fallback.get("menu")
        ingredient = fallback.get("ingredient")

    recipe_data = []

    try:
        if menu:
            recipes = search_recipes(menu, limit=3)
            for recipe in recipes:
                ingredients = get_recipe_ingredients(recipe["id"])
                recipe_data.append({
                    "menu": recipe["name"],
                    "servings": recipe.get("servings"),
                    "difficulty": recipe.get("difficulty"),
                    "ingredients": ingredients
                })
        elif ingredient:
            recipes = get_recipes_by_ingredient(ingredient, limit=5)
            for recipe in recipes:
                recipe_data.append({
                    "menu": recipe["name"],
                    "id": recipe["id"],
                    "servings": recipe.get("servings"),
                })
    except Exception as e:
        return {"recipe_info": {}, "error_log": [f"recipe_search: {str(e)}"]}

    return {"recipe_info": {"data": recipe_data}}
