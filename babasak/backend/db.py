import os
from neo4j import GraphDatabase

_driver = None
_KNOWN_MENUS = []
_KNOWN_INGREDIENTS = []
_DICT_LOADED = False


def get_driver():
    """Neo4j 드라이버 싱글톤.

    환경변수에서 접속정보를 받음. Databricks Apps는 app.yml에서 secrets 주입,
    로컬 개발은 .env / shell export로 주입. 기본값 박지 않음.
    NEO4J_USERNAME과 NEO4J_USER 둘 다 호환 (ETL은 USERNAME, 일부 환경은 USER 사용).
    """
    global _driver
    if _driver is None:
        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER")
        password = os.environ.get("NEO4J_PASSWORD")
        if not uri or not user or password is None:
            raise RuntimeError(
                "Neo4j 환경변수 누락. "
                "NEO4J_URI / NEO4J_USERNAME(또는 NEO4J_USER) / NEO4J_PASSWORD 가 모두 설정돼야 함."
            )
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


# ============================================================
# 사전 로딩 / 토크나이저
# ============================================================

def _load_dictionaries(menu_limit=2000, ing_limit=1500):
    """빈도 상위 메뉴명/재료명을 Neo4j에서 읽어 메모리에 캐싱.
    긴 이름부터 매칭해야 '청양고추'가 '고추'로 먼저 잡히지 않음."""
    global _KNOWN_MENUS, _KNOWN_INGREDIENTS, _DICT_LOADED
    if _DICT_LOADED:
        return

    with get_driver().session() as session:
        menus = session.run("""
            MATCH (r:Recipe)
            WHERE r.name IS NOT NULL AND size(r.name) <= 20
            WITH r.name AS name, count(*) AS cnt
            ORDER BY cnt DESC LIMIT $limit
            RETURN name
        """, limit=menu_limit)
        _KNOWN_MENUS = [m["name"] for m in menus if m["name"]]

        ings = session.run("""
            MATCH ()-[:CONTAINS]->(i:Ingredient)
            WITH i.name AS name, count(*) AS cnt
            ORDER BY cnt DESC LIMIT $limit
            RETURN name
        """, limit=ing_limit)
        _KNOWN_INGREDIENTS = [i["name"] for i in ings if i["name"]]

    _KNOWN_MENUS.sort(key=len, reverse=True)
    _KNOWN_INGREDIENTS.sort(key=len, reverse=True)
    _DICT_LOADED = True


def _tokenize(query):
    """쿼리에서 알려진 메뉴/재료 키워드 추출.
    예: '명란김치찌개 매콤한 거' → menu=['김치찌개'], ing=['명란']"""
    _load_dictionaries()
    remaining = query
    menu_tokens = []
    ing_tokens = []
    for menu in _KNOWN_MENUS:
        if menu in remaining:
            menu_tokens.append(menu)
            remaining = remaining.replace(menu, " ")
    for ing in _KNOWN_INGREDIENTS:
        if ing in remaining:
            ing_tokens.append(ing)
            remaining = remaining.replace(ing, " ")
    return menu_tokens, ing_tokens


# ============================================================
# 점수 공식 — 인기도 기반, 모든 검색에서 통일
# ============================================================
_SCORE_EXPR = "(coalesce(r.view_count,0) + coalesce(r.recommend_count,0)*100 + coalesce(r.scrap_count,0)*50)"


# ============================================================
# 1. 통합 검색 — search_recipes_smart (핵심 함수)
# ============================================================

def search_recipes_smart(query, limit=3):
    """4단계 fallback으로 레시피 검색.

    1단계: 이름에 query 그대로 부분 매칭
    2단계: query 토큰화 → 메뉴 매칭 + 재료 토큰 일치 가중
    3단계: 메뉴 토큰 하나만으로 부분 매칭
    4단계: 끝까지 비면 인기 레시피 fallback
    """
    results = _search_by_name(query, limit)
    if len(results) >= limit:
        return results

    menu_tokens, ing_tokens = _tokenize(query)

    if menu_tokens:
        token_results = _search_by_tokens(menu_tokens, ing_tokens, limit)
        seen = {r["id"] for r in results}
        for r in token_results:
            if r["id"] not in seen and len(results) < limit:
                results.append(r)
                seen.add(r["id"])
        if len(results) >= limit:
            return results

    if menu_tokens and len(results) < limit:
        fallback = _search_by_name(menu_tokens[0], limit)
        seen = {r["id"] for r in results}
        for r in fallback:
            if r["id"] not in seen and len(results) < limit:
                r["match_type"] = "partial_token"
                results.append(r)

    if not results:
        results = get_popular_recipes(limit)
        for r in results:
            r["match_type"] = "popular_fallback"

    return results


def _search_by_name(keyword, limit):
    """1단계: 이름 부분 매칭. 인기도 정렬."""
    with get_driver().session() as session:
        result = session.run(f"""
            MATCH (r:Recipe)
            WHERE r.name CONTAINS $kw
            WITH r, {_SCORE_EXPR} AS score
            RETURN r.rcp_sno AS id, r.name AS name,
                   r.servings AS servings, r.difficulty AS difficulty,
                   r.kind AS kind, score
            ORDER BY score DESC
            LIMIT $limit
        """, kw=keyword, limit=limit)
        rows = []
        for r in result:
            d = r.data()
            d["match_type"] = "name_match"
            rows.append(d)
        return rows


def _search_by_tokens(menu_tokens, ing_tokens, limit):
    """2단계: 메뉴 토큰 매칭 + 재료 토큰 일치당 +1000점 가중."""
    if not menu_tokens:
        return []

    with get_driver().session() as session:
        result = session.run(f"""
            MATCH (r:Recipe)
            WHERE ANY(t IN $menus WHERE r.name CONTAINS t)
            OPTIONAL MATCH (r)-[:CONTAINS]->(i:Ingredient)
            WHERE i.name IN $ings
            WITH r, count(DISTINCT i) AS ing_hits, {_SCORE_EXPR} AS base_score
            WITH r, base_score + ing_hits * 1000 AS score, ing_hits
            RETURN r.rcp_sno AS id, r.name AS name,
                   r.servings AS servings, r.difficulty AS difficulty,
                   r.kind AS kind, score, ing_hits
            ORDER BY score DESC
            LIMIT $limit
        """, menus=menu_tokens, ings=ing_tokens, limit=limit)
        rows = []
        for r in result:
            d = r.data()
            d["match_type"] = "token_match"
            rows.append(d)
        return rows


# ============================================================
# 2. 레시피 상세/재료
# ============================================================

def get_recipe_ingredients(rcp_sno):
    """레시피의 재료 + 수량 조회."""
    with get_driver().session() as session:
        result = session.run("""
            MATCH (r:Recipe {rcp_sno: $rcp_sno})-[c:CONTAINS]->(i:Ingredient)
            RETURN i.name AS name, c.quantity AS quantity,
                   i.lv1 AS category, i.lv2 AS subcategory
            ORDER BY i.lv1, i.name
        """, rcp_sno=rcp_sno)
        return [r.data() for r in result]


def get_recipe_detail(rcp_sno):
    """레시피 상세 정보 (조리단계 r.steps 포함)."""
    with get_driver().session() as session:
        result = session.run("""
            MATCH (r:Recipe {rcp_sno: $rcp_sno})
            RETURN r.rcp_sno AS id, r.name AS name, r.title AS title,
                   r.servings AS servings, r.difficulty AS difficulty,
                   r.cooking_time AS cooking_time, r.cooking_method AS cooking_method,
                   r.kind AS kind, r.situation AS situation,
                   r.main_ingredient AS main_ingredient,
                   r.view_count AS view_count, r.recommend_count AS recommend_count,
                   r.scrap_count AS scrap_count, r.description AS description,
                   r.image_url AS image_url, r.steps AS steps
        """, rcp_sno=rcp_sno)
        record = result.single()
        return record.data() if record else None


# ============================================================
# 3. 재료 기반 검색
# ============================================================

def get_recipes_by_ingredient(ingredient_name, limit=3):
    """특정 재료가 들어간 레시피 (인기순)."""
    with get_driver().session() as session:
        result = session.run(f"""
            MATCH (r:Recipe)-[:CONTAINS]->(i:Ingredient {{name: $name}})
            WITH r, {_SCORE_EXPR} AS score
            RETURN DISTINCT r.rcp_sno AS id, r.name AS name,
                   r.difficulty AS difficulty, r.servings AS servings,
                   r.view_count AS view_count, score
            ORDER BY score DESC LIMIT $limit
        """, name=ingredient_name, limit=limit)
        return [r.data() for r in result]


def get_recipes_by_multiple_ingredients(ingredients, limit=3):
    """여러 재료가 모두 들어간 레시피."""
    with get_driver().session() as session:
        result = session.run(f"""
            MATCH (r:Recipe)
            WHERE ALL(ing IN $ingredients WHERE (r)-[:CONTAINS]->(:Ingredient {{name: ing}}))
            WITH r, {_SCORE_EXPR} AS score
            RETURN r.rcp_sno AS id, r.name AS name,
                   r.difficulty AS difficulty, r.servings AS servings,
                   r.view_count AS view_count, score
            ORDER BY score DESC LIMIT $limit
        """, ingredients=ingredients, limit=limit)
        return [r.data() for r in result]


def get_recipes_excluding_ingredient(keyword, exclude, limit=3):
    """특정 재료를 제외한 레시피 (알레르기 대응)."""
    with get_driver().session() as session:
        result = session.run(f"""
            MATCH (r:Recipe)
            WHERE r.name CONTAINS $keyword
            AND NOT (r)-[:CONTAINS]->(:Ingredient {{name: $exclude}})
            WITH r, {_SCORE_EXPR} AS score
            RETURN r.rcp_sno AS id, r.name AS name,
                   r.difficulty AS difficulty, r.servings AS servings, score
            ORDER BY score DESC LIMIT $limit
        """, keyword=keyword, exclude=exclude, limit=limit)
        return [r.data() for r in result]


# ============================================================
# 4. 유사 레시피 추천 (재료 공유도 기반)
# ============================================================

def find_similar_recipes(rcp_sno, limit=3, min_shared=2):
    """주어진 레시피와 재료를 공유하는 유사 레시피 추천.
    min_shared: 최소 공통 재료 수 (기본 2개)."""
    with get_driver().session() as session:
        result = session.run(f"""
            MATCH (base:Recipe {{rcp_sno: $rcp_sno}})-[:CONTAINS]->(i:Ingredient)
            MATCH (other:Recipe)-[:CONTAINS]->(i)
            WHERE other.rcp_sno <> base.rcp_sno
            WITH other, count(DISTINCT i) AS shared,
                 {_SCORE_EXPR.replace('r.', 'other.')} AS pop
            WHERE shared >= $min_shared
            RETURN other.rcp_sno AS id, other.name AS name,
                   other.difficulty AS difficulty, other.servings AS servings,
                   shared, pop AS score
            ORDER BY shared DESC, pop DESC
            LIMIT $limit
        """, rcp_sno=rcp_sno, min_shared=min_shared, limit=limit)
        return [r.data() for r in result]


# ============================================================
# 5. 인기 레시피
# ============================================================

def get_popular_recipes(limit=3):
    """전체 인기 레시피 top N."""
    with get_driver().session() as session:
        result = session.run(f"""
            MATCH (r:Recipe)
            WITH r, {_SCORE_EXPR} AS score
            RETURN r.rcp_sno AS id, r.name AS name,
                   r.difficulty AS difficulty, r.servings AS servings,
                   r.view_count AS view_count, score
            ORDER BY score DESC LIMIT $limit
        """, limit=limit)
        return [r.data() for r in result]


# ============================================================
# 6. 조건 기반 추천
# ============================================================

def recommend_recipes(
    kind=None,
    difficulty=None,
    servings=None,
    cooking_method=None,
    limit=3,
):
    """조건(종류/난이도/인분/조리법) 기반 추천."""
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

    with get_driver().session() as session:
        result = session.run(f"""
            MATCH (r:Recipe)
            {where_clause}
            WITH r, {_SCORE_EXPR} AS score
            RETURN r.rcp_sno AS id, r.name AS name,
                   r.difficulty AS difficulty, r.servings AS servings,
                   r.kind AS kind, r.cooking_time AS cooking_time,
                   r.view_count AS view_count, score
            ORDER BY score DESC LIMIT $limit
        """, **params)
        return [r.data() for r in result]

# ============================================================
# Legacy alias (기존 import 호환)
# ============================================================

def search_recipes(keyword, limit=5):
    """기존 search_recipes 인터페이스 유지 — 내부적으로 smart 검색 사용."""
    return search_recipes_smart(keyword, limit=limit)
def search_recipes(keyword, limit=5):
    """기존 search_recipes 인터페이스 유지 — 내부적으로 smart 검색 사용."""
    return search_recipes_smart(keyword, limit=limit)