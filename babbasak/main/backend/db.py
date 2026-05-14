import os
from neo4j import GraphDatabase

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def get_session():
    """database 지정하여 세션 생성"""
    database = os.getenv("NEO4J_DATABASE", "e76ed54b")
    return get_driver().session(database=database)


# ============================================================
# 재료 관련
# ============================================================

def get_all_ingredients() -> list[dict]:
    """전체 재료 목록 (카테고리 포함)"""
    with get_session() as session:
        result = session.run("""
            MATCH (i:Ingredient)
            RETURN i.name AS name, i.lv1 AS category, i.lv2 AS subcategory
            ORDER BY i.name
        """)
        return [r.data() for r in result]


def get_ingredient_by_category(category: str) -> list[dict]:
    """카테고리별 재료 조회 (예: '채소류', '양념·기름')"""
    with get_session() as session:
        result = session.run("""
            MATCH (i:Ingredient)
            WHERE i.lv1 = $category
            RETURN i.name AS name, i.lv1 AS category, i.lv2 AS subcategory
            ORDER BY i.name
        """, category=category)
        return [r.data() for r in result]


# ============================================================
# 레시피 관련
# ============================================================

def search_recipes(keyword: str, limit: int = 5) -> list[dict]:
    """키워드로 레시피 검색 (점수 기반 정렬)"""
    with get_session() as session:
        result = session.run("""
            MATCH (r:Recipe)
            WHERE r.name CONTAINS $keyword
            WITH r, (r.view_count + r.recommend_count * 100 + r.scrap_count * 50) AS score
            RETURN r.rcp_sno AS id, r.name AS name, r.servings AS servings,
                   r.difficulty AS difficulty, r.cooking_time AS cooking_time,
                   r.kind AS kind, r.cooking_method AS cooking_method,
                   r.view_count AS view_count, score
            ORDER BY score DESC
            LIMIT $limit
        """, keyword=keyword, limit=limit)
        return [r.data() for r in result]


def get_recipe_ingredients(rcp_sno: int) -> list[dict]:
    """특정 레시피의 재료 + 수량 조회"""
    with get_session() as session:
        result = session.run("""
            MATCH (r:Recipe {rcp_sno: $rcp_sno})-[c:CONTAINS]->(i:Ingredient)
            RETURN i.name AS name, c.quantity AS quantity,
                   i.lv1 AS category, i.lv2 AS subcategory
            ORDER BY i.lv1, i.name
        """, rcp_sno=rcp_sno)
        return [r.data() for r in result]


def get_recipe_detail(rcp_sno: int) -> dict | None:
    """레시피 상세 정보"""
    with get_session() as session:
        result = session.run("""
            MATCH (r:Recipe {rcp_sno: $rcp_sno})
            RETURN r.rcp_sno AS id, r.name AS name, r.title AS title,
                   r.servings AS servings, r.difficulty AS difficulty,
                   r.cooking_time AS cooking_time, r.cooking_method AS cooking_method,
                   r.kind AS kind, r.situation AS situation,
                   r.main_ingredient AS main_ingredient,
                   r.view_count AS view_count, r.recommend_count AS recommend_count,
                   r.scrap_count AS scrap_count, r.description AS description,
                   r.image_url AS image_url
        """, rcp_sno=rcp_sno)
        record = result.single()
        return record.data() if record else None


# ============================================================
# 재료 기반 레시피 검색
# ============================================================

def get_recipes_by_ingredient(ingredient_name: str, limit: int = 10) -> list[dict]:
    """특정 재료가 들어간 레시피 조회"""
    with get_session() as session:
        result = session.run("""
            MATCH (r:Recipe)-[c:CONTAINS]->(i:Ingredient {name: $name})
            WITH r, (r.view_count + r.recommend_count * 100 + r.scrap_count * 50) AS score
            RETURN DISTINCT r.rcp_sno AS id, r.name AS name, r.difficulty AS difficulty,
                   r.servings AS servings, r.view_count AS view_count, score
            ORDER BY score DESC
            LIMIT $limit
        """, name=ingredient_name, limit=limit)
        return [r.data() for r in result]


def get_recipes_by_multiple_ingredients(ingredients: list[str], limit: int = 10) -> list[dict]:
    """여러 재료가 모두 들어간 레시피 조회"""
    with get_session() as session:
        result = session.run("""
            MATCH (r:Recipe)
            WHERE ALL(ing IN $ingredients WHERE (r)-[:CONTAINS]->(:Ingredient {name: ing}))
            WITH r, (r.view_count + r.recommend_count * 100 + r.scrap_count * 50) AS score
            RETURN r.rcp_sno AS id, r.name AS name, r.difficulty AS difficulty,
                   r.servings AS servings, r.view_count AS view_count, score
            ORDER BY score DESC
            LIMIT $limit
        """, ingredients=ingredients, limit=limit)
        return [r.data() for r in result]


def get_recipes_excluding_ingredient(keyword: str, exclude: str, limit: int = 10) -> list[dict]:
    """특정 재료를 제외한 레시피 조회 (알레르기 대응)"""
    with get_session() as session:
        result = session.run("""
            MATCH (r:Recipe)
            WHERE r.name CONTAINS $keyword
            AND NOT (r)-[:CONTAINS]->(:Ingredient {name: $exclude})
            WITH r, (r.view_count + r.recommend_count * 100 + r.scrap_count * 50) AS score
            RETURN r.rcp_sno AS id, r.name AS name, r.difficulty AS difficulty,
                   r.servings AS servings, score
            ORDER BY score DESC
            LIMIT $limit
        """, keyword=keyword, exclude=exclude, limit=limit)
        return [r.data() for r in result]


# ============================================================
# 조건 기반 추천
# ============================================================

def recommend_recipes(
    kind: str = None,
    difficulty: str = None,
    servings: str = None,
    cooking_method: str = None,
    limit: int = 10
) -> list[dict]:
    """조건 기반 레시피 추천 (종류, 난이도, 인분, 조리법)"""
    conditions = []
    params = {"limit": limit}

    if kind:
        conditions.append("r.kind = $kind")
        params["kind"] = kind
    if difficulty:
        conditions.append("r.difficulty = $difficulty")
        params["difficulty"] = difficulty
    if servings:
        conditions.append("r.servings = $servings")
        params["servings"] = servings
    if cooking_method:
        conditions.append("r.cooking_method = $cooking_method")
        params["cooking_method"] = cooking_method

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    with get_session() as session:
        result = session.run(f"""
            MATCH (r:Recipe)
            {where_clause}
            WITH r, (r.view_count + r.recommend_count * 100 + r.scrap_count * 50) AS score
            RETURN r.rcp_sno AS id, r.name AS name, r.difficulty AS difficulty,
                   r.servings AS servings, r.kind AS kind,
                   r.cooking_time AS cooking_time, r.view_count AS view_count, score
            ORDER BY score DESC
            LIMIT $limit
        """, **params)
        return [r.data() for r in result]


# ============================================================
# 인기 레시피
# ============================================================

def get_popular_recipes(limit: int = 10) -> list[dict]:
    """인기 레시피 top N"""
    with get_session() as session:
        result = session.run("""
            MATCH (r:Recipe)
            WITH r, (r.view_count + r.recommend_count * 100 + r.scrap_count * 50) AS score
            RETURN r.rcp_sno AS id, r.name AS name, r.difficulty AS difficulty,
                   r.servings AS servings, r.view_count AS view_count, score
            ORDER BY score DESC
            LIMIT $limit
        """, limit=limit)
        return [r.data() for r in result]


# ============================================================
# DB 통계
# ============================================================

def get_db_stats() -> dict:
    """DB 전체 통계"""
    with get_session() as session:
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(*) AS cnt
            ORDER BY cnt DESC
        """)
        stats = {r["label"]: r["cnt"] for r in result}

        rel_result = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
        stats["총 관계 수"] = rel_result.single()["cnt"]

        return stats


# ============================================================
# 챗봇 컨텍스트 (agent.py에서 호출)
# ============================================================

def get_chatbot_context(keyword: str) -> list[dict]:
    """
    agent.py의 recipe_db_expert에서 호출하는 함수.
    키워드로 레시피 검색 후 재료 목록까지 포함하여 반환.
    """
    recipes = search_recipes(keyword, limit=5)
    if not recipes:
        return []

    results = []
    for recipe in recipes:
        ingredients = get_recipe_ingredients(recipe["id"])
        ing_names = [ing["name"] for ing in ingredients]
        results.append({
            "menu": recipe["name"],
            "margin": 0,
            "ingredients": ing_names,
            "difficulty": recipe.get("difficulty"),
            "servings": recipe.get("servings"),
            "cooking_time": recipe.get("cooking_time"),
            "view_count": recipe.get("view_count"),
        })

    return results
