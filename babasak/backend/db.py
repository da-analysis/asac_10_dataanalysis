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
    """[신규] Neo4j에서 빈도 상위 메뉴명/재료명을 메모리에 캐싱.

    이전 버전: 사전 자체가 없음. 검색은 r.name CONTAINS keyword 단일 쿼리뿐.
    변경 의도: 멘토 피드백 — '명란김치찌개' 같이 없는 메뉴를 토큰으로 분해해
              '명란' + '김치찌개'로 부분 매칭하기 위함. 그래프 RAG에서 자연스러움.
    긴 이름부터 매칭해야 '청양고추'가 '고추'로 먼저 잡히지 않음."""
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
    """[신규] 쿼리에서 알려진 메뉴/재료 키워드 추출.

    예: '명란김치찌개 매콤한 거' → menu=['김치찌개'], ing=['명란']
    이전 버전: 토큰화 자체가 없었음. preprocessor가 LLM으로 entity 추출했고
              실패하면 그냥 빈 응답 반환.
    반환: (menu_tokens, ingredient_tokens)"""
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
    """[신규] 3단계 fallback으로 레시피 검색.

    이전 search_recipes()는 단순히 r.name CONTAINS keyword 한 줄.
    없는 메뉴면 빈 결과 → 챗봇이 '못 찾았다'만 답함.

    멘토 피드백 반영:
      - 없는 메뉴도 유사한 거 찾아주기
      - LIKE 형식 말고 그래프 traverse로 재료 공유까지 보기
      - 우선순위 명확하게 (정확매칭 > 토큰+재료 > 토큰만 > 인기)

    1단계: 이름에 query 그대로 포함 (정확/부분)
    2단계: query를 사전으로 토큰 분해. 메뉴 토큰 매칭 + 재료 토큰 일치 가중(+1000/재료)
    3단계: 메뉴 토큰 하나만으로 부분 매칭
    4단계: 그래도 없으면 인기 레시피 fallback (match_type='popular_fallback')

    반환: list of {id, name, servings, difficulty, kind, score, match_type}
    """
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
    """[신규] 1단계: 이름 부분 매칭. 인기도 점수 정렬.
    이전 search_recipes()의 본문이 사실상 이 함수.
    match_type='name_match' 라벨 추가."""
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
    """[신규] 2단계: 메뉴 토큰 매칭 + 재료 토큰 일치 가중.

    Cypher 핵심:
      - 메뉴 토큰 중 하나라도 이름에 포함 (ANY)
      - 그 레시피의 CONTAINS Ingredient 중 ing_tokens 일치하는 갯수(ing_hits)를 셈
      - 점수 = 인기점수 + ing_hits * 1000
      이렇게 하면 '명란김치찌개' 검색 시 '명란'까지 들어간 김치찌개가 가장 위로.
    """
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
    """[유지] 레시피의 재료 + 수량 조회. 시그니처/Cypher 그대로.
    Neo4j에는 CONTAINS 관계에 c.quantity 속성이 들어있음 (silver.ingredients_final 기반)."""
    with get_driver().session() as session:
        result = session.run("""
            MATCH (r:Recipe {rcp_sno: $rcp_sno})-[c:CONTAINS]->(i:Ingredient)
            RETURN i.name AS name, c.quantity AS quantity,
                   i.lv1 AS category, i.lv2 AS subcategory
            ORDER BY i.lv1, i.name
        """, rcp_sno=rcp_sno)
        return [r.data() for r in result]


def get_recipe_detail(rcp_sno):
    """[변경] 레시피 상세 정보 반환.
    이전 버전 대비 추가: r.steps (조리단계). ETL에서 cooking_steps를 Recipe.steps로 적재함.
    챗봇이 '조리법 알려줘' 답할 때 사용."""
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
    """[변경] 특정 재료가 들어간 레시피 (인기순).
    이전: limit 기본값 10. 지금: 3 (멘토 피드백 — top 3 노출)."""
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
    """[변경] 여러 재료가 모두 들어간 레시피.
    이전: limit 10 → 지금: 3."""
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
    """[변경] 특정 재료를 제외한 레시피 (알레르기 대응).
    이전: limit 10 → 지금: 3."""
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
    """[신규] 주어진 레시피와 재료를 공유하는 유사 레시피 추천.

    이전 버전: 이런 함수 없었음. 챗봇이 '비슷한 거 보여줘' 답을 못 함.
    멘토 피드백: '레시피의 재료를 가져오는 게 3번과 4번에서 되어야' = 재료 그래프 활용.

    Cypher 핵심:
      MATCH (base)-[:CONTAINS]->(i)<-[:CONTAINS]-(other)
      base와 other가 같은 Ingredient i를 공유하는 패턴
      shared(공통 재료 수) DESC, 인기 DESC 정렬

    - min_shared: 최소 공통 재료 수 (기본 2개) — 너무 약한 매칭 차단
    """
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
    """[변경] 전체 인기 레시피 top N.
    이전: limit 10 → 지금: 3. 점수 공식은 _SCORE_EXPR로 통일."""
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
# Legacy alias (기존 코드 호환)
# ============================================================

def search_recipes(keyword, limit=5):
    """[변경] 기존 코드 호환용. 시그니처는 그대로지만 내부는 search_recipes_smart 호출.

    이전 동작: r.name CONTAINS keyword 단일 쿼리 (못 찾으면 빈 결과)
    지금 동작: 3단계 fallback (정확 → 토큰 → 부분 → 인기)
    => recipe_search_node.py 등 기존 호출자가 그대로 import해도 깨지지 않음.
    """
    return search_recipes_smart(keyword, limit=limit)
