import os
from neo4j import GraphDatabase

_driver = None
_KNOWN_MENUS = []       # [신규] 빈도 상위 메뉴명 캐시 (Neo4j에서 로드)
_KNOWN_INGREDIENTS = [] # [신규] 빈도 상위 재료명 캐시
_DICT_LOADED = False


def get_driver():
    """[유지] Neo4j 드라이버 싱글톤. 환경변수로 접속정보 주입."""
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI", "neo4j+s://e76ed54b.databases.neo4j.io")
        user = os.getenv("NEO4J_USER", "e76ed54b")
        password = os.getenv("NEO4J_PASSWORD", "")
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


# ============================================================
# 사전 로딩 (첫 호출 시 1회)
# ============================================================

def _load_dictionaries(menu_limit=2000, ing_limit=1500):
    global _KNOWN_MENUS, _KNOWN_INGREDIENTS, _DICT_LOADED
    if _DICT_LOADED:
        return

    with get_driver().session() as session:
        # 메뉴: 인기 순 + name이 너무 길지 않은 것
        menus = session.run("""
            MATCH (r:Recipe)
            WHERE r.name IS NOT NULL AND size(r.name) <= 20
            WITH r.name AS name, count(*) AS cnt
            ORDER BY cnt DESC LIMIT $limit
            RETURN name
        """, limit=menu_limit)
        _KNOWN_MENUS = [m["name"] for m in menus if m["name"]]

        # 재료: 사용 빈도 순
        ings = session.run("""
            MATCH ()-[:CONTAINS]->(i:Ingredient)
            WITH i.name AS name, count(*) AS cnt
            ORDER BY cnt DESC LIMIT $limit
            RETURN name
        """, limit=ing_limit)
        _KNOWN_INGREDIENTS = [i["name"] for i in ings if i["name"]]

    # 긴 이름 우선 매칭
    _KNOWN_MENUS.sort(key=len, reverse=True)
    _KNOWN_INGREDIENTS.sort(key=len, reverse=True)
    _DICT_LOADED = True


def _tokenize(query):
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
# [변경] 이전 버전: 각 함수마다 (r.view_count + r.recommend_count*100 + r.scrap_count*50) 하드코딩
#       지금 버전: 상수로 빼고 coalesce(...,0) 감싸서 NULL 안전
# ============================================================
_SCORE_EXPR = "(coalesce(r.view_count,0) + coalesce(r.recommend_count,0)*100 + coalesce(r.scrap_count,0)*50)"


# ============================================================
# 1. 통합 검색 — search_recipes_smart (핵심 함수)
# ============================================================

def search_recipes_smart(query, limit=3):
    # 1단계: 이름 부분 매칭
    results = _search_by_name(query, limit)
    if len(results) >= limit:
        return results

    # 2단계: 토큰 분해
    menu_tokens, ing_tokens = _tokenize(query)

    if menu_tokens:
        token_results = _search_by_tokens(menu_tokens, ing_tokens, limit)
        # 1단계와 합치되 중복 제거
        seen = {r["id"] for r in results}
        for r in token_results:
            if r["id"] not in seen and len(results) < limit:
                results.append(r)
                seen.add(r["id"])
        if len(results) >= limit:
            return results

    # 3단계: 메뉴 토큰만 (재료 가중 없이)
    if menu_tokens and len(results) < limit:
        fallback = _search_by_name(menu_tokens[0], limit)
        seen = {r["id"] for r in results}
        for r in fallback:
            if r["id"] not in seen and len(results) < limit:
                r["match_type"] = "partial_token"
                results.append(r)

    # 4단계: 끝까지 비어있으면 인기
    if not results:
        results = get_popular_recipes(limit)
        for r in results:
            r["match_type"] = "popular_fallback"

    return results


def _search_by_name(keyword, limit):
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
    if not menu_tokens:
        return []

    with get_driver().session() as session:
        # 메뉴 토큰 중 하나라도 이름에 포함
        # 재료 토큰이 CONTAINS로 연결돼 있으면 += 50점 (재료당)
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
    with get_driver().session() as session:
        result = session.run("""
            MATCH (r:Recipe {rcp_sno: $rcp_sno})-[c:CONTAINS]->(i:Ingredient)
            RETURN i.name AS name, c.quantity AS quantity,
                   i.lv1 AS category, i.lv2 AS subcategory
            ORDER BY i.lv1, i.name
        """, rcp_sno=rcp_sno)
        return [r.data() for r in result]


def get_recipe_detail(rcp_sno):
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
    with get_driver().session() as session:
        result = session.run(f"""
            MATCH (base:Recipe {{rcp_sno: $rcp_sno}})-[:CONTAINS]->(i:Ingredient)
            MATCH (other:Recipe)-[:CONTAINS]->(i)
            WHERE other.rcp_sno <> base.rcp_sno
            WITH other, count(DISTINCT i) AS shared, {_SCORE_EXPR.replace('r.', 'other.')} AS pop
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
# 6. 조건 기반 추천 (recipe_search.py가 import하므로 유지)
# ============================================================

def recommend_recipes(
    kind=None,
    difficulty=None,
    servings=None,
    cooking_method=None,
    limit=3,
):
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
# Legacy alias (기존 코드 호환)
# ============================================================

def search_recipes(keyword, limit=5):
    return search_recipes_smart(keyword, limit=limit)
    """
    return search_recipes_smart(keyword, limit=limit)
