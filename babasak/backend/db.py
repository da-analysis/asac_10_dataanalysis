import os
import re
from neo4j import GraphDatabase

_driver = None
_KNOWN_MENUS = []
_KNOWN_INGREDIENTS = []
_DICT_LOADED = False

_SCORE_EXPR = "(coalesce(r.view_count,0) + coalesce(r.recommend_count,0)*100 + coalesce(r.scrap_count,0)*50)"

_QUERY_STOPWORDS = [
    "레시피", "요리", "메뉴", "음식", "추천", "알려줘", "찾아줘",
    "들어간", "들어가는", "넣은", "넣는", "사용한", "사용하는",
    "포함", "있는", "으로", "로", "만든", "만들",
]

_INGREDIENT_ALIASES = {
    "달걀": "계란",
    "돼지": "돼지고기",
    "돈육": "돼지고기",
    "소": "소고기",
    "우육": "소고기",
    "닭": "닭고기",
    "치킨": "닭고기",
}


def _compact(text):
    if not text:
        return ""
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", str(text))


def _strip_query_words(text):
    cleaned = str(text or "")
    for word in sorted(_QUERY_STOPWORDS, key=len, reverse=True):
        cleaned = cleaned.replace(word, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _merge_results(base, extra, limit):
    seen = {r.get("id") for r in base}
    for row in extra:
        if row.get("id") not in seen and len(base) < limit:
            base.append(row)
            seen.add(row.get("id"))
    return base


def get_driver():
    """Neo4j 드라이버 싱글톤."""
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


def get_session():
    """NEO4J_DATABASE가 있으면 사용하고, 없으면 기본 DB를 사용."""
    database = os.environ.get("NEO4J_DATABASE")
    if database:
        return get_driver().session(database=database)
    return get_driver().session()


# ============================================================
# 사전 로딩 / 토크나이저
# ============================================================

def _load_dictionaries(menu_limit=2000, ing_limit=5000):
    """빈도 상위 메뉴명/재료명을 Neo4j에서 읽어 메모리에 캐싱."""
    global _KNOWN_MENUS, _KNOWN_INGREDIENTS, _DICT_LOADED
    if _DICT_LOADED:
        return

    with get_session() as session:
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

    _KNOWN_MENUS = sorted(set(_KNOWN_MENUS), key=len, reverse=True)
    _KNOWN_INGREDIENTS = sorted(set(_KNOWN_INGREDIENTS), key=len, reverse=True)
    _DICT_LOADED = True


def _tokenize(query):
    """쿼리에서 알려진 메뉴/재료 키워드 추출."""
    _load_dictionaries()
    remaining = query or ""
    compact_remaining = _compact(remaining)
    menu_tokens = []
    ing_tokens = []

    for menu in _KNOWN_MENUS:
        if menu in remaining:
            menu_tokens.append(menu)
            remaining = remaining.replace(menu, " ")

    for ing in _KNOWN_INGREDIENTS:
        ing_compact = _compact(ing)
        if not ing_compact:
            continue
        if ing in remaining or ing_compact in compact_remaining:
            ing_tokens.append(ing)
            remaining = remaining.replace(ing, " ")
            compact_remaining = compact_remaining.replace(ing_compact, " ")

    return list(dict.fromkeys(menu_tokens)), list(dict.fromkeys(ing_tokens))


def _resolve_ingredient_candidates(text, limit=5):
    """자연어/롱테일 재료 표현을 DB의 Ingredient.name 후보로 변환."""
    _load_dictionaries()
    raw = text or ""
    cleaned = _strip_query_words(raw)
    raw_compact = _compact(raw)
    cleaned_compact = _compact(cleaned)
    candidates = []

    alias = _INGREDIENT_ALIASES.get(cleaned_compact) or _INGREDIENT_ALIASES.get(raw_compact)
    if alias:
        candidates.append(alias)

    # exact 우선
    for ing in _KNOWN_INGREDIENTS:
        ing_compact = _compact(ing)
        if ing_compact and ing_compact in {raw_compact, cleaned_compact}:
            candidates.append(ing)

    # substring fallback: "당근손가락길이만큼" -> "당근"
    for ing in _KNOWN_INGREDIENTS:
        ing_compact = _compact(ing)
        if not ing_compact or len(ing_compact) < 2:
            continue
        if ing_compact in raw_compact or ing_compact in cleaned_compact:
            candidates.append(ing)
        elif len(cleaned_compact) >= 2 and cleaned_compact in ing_compact:
            candidates.append(ing)
        if len(dict.fromkeys(candidates)) >= limit:
            break

    return list(dict.fromkeys(candidates))[:limit]


def _extract_graph_parts(query):
    """없는 메뉴를 조합하기 위한 base menu + modifier ingredient 추출.

    예: "마라김치찌개" -> base_menus=["김치찌개"], modifier_terms=["마라"],
        modifier_ingredients=["마라소스", ...]
    """
    _load_dictionaries()
    raw = query or ""
    raw_compact = _compact(raw)
    remaining_compact = raw_compact

    base_menus = []
    for menu in _KNOWN_MENUS:
        menu_compact = _compact(menu)
        if menu_compact and menu_compact in remaining_compact:
            base_menus.append(menu)
            remaining_compact = remaining_compact.replace(menu_compact, " ")

    modifier_terms = [
        term for term in re.split(r"\s+", remaining_compact.strip())
        if len(term) >= 2
    ]

    modifier_ingredients = []
    for term in modifier_terms:
        modifier_ingredients.extend(_resolve_ingredient_candidates(term, limit=3))

    # "마라 김치찌개"처럼 공백이 있는 쿼리도 잡기 위해 원문 토큰도 한 번 더 본다.
    for token in re.split(r"\s+", _strip_query_words(raw)):
        token_compact = _compact(token)
        if len(token_compact) >= 2 and token_compact not in [_compact(m) for m in base_menus]:
            modifier_ingredients.extend(_resolve_ingredient_candidates(token_compact, limit=2))

    return {
        "base_menus": list(dict.fromkeys(base_menus)),
        "modifier_terms": list(dict.fromkeys(modifier_terms)),
        "modifier_ingredients": list(dict.fromkeys(modifier_ingredients)),
    }


def get_related_ingredients(ingredient_names, limit=12):
    """특정 재료와 같은 레시피에 자주 등장하는 재료 관계."""
    if not ingredient_names:
        return []

    with get_session() as session:
        result = session.run("""
            MATCH (r:Recipe)-[:CONTAINS]->(seed:Ingredient)
            WHERE seed.name IN $names
            MATCH (r)-[:CONTAINS]->(co:Ingredient)
            WHERE NOT co.name IN $names
            WITH co.name AS name, co.lv1 AS category, count(DISTINCT r) AS recipe_count
            ORDER BY recipe_count DESC
            LIMIT $limit
            RETURN name, category, recipe_count
        """, names=ingredient_names, limit=limit)
        return [r.data() for r in result]


def build_graph_relation_context(query, limit=3):
    """DB에 정확한 레시피가 없을 때 Neo4j 관계로 조합 근거를 만든다.

    이 함수는 완성 레시피를 창작하지 않는다.
    Neo4j에서 base recipe, modifier ingredient, co-occurring ingredient 관계만 반환한다.
    """
    parts = _extract_graph_parts(query)
    base_menus = parts["base_menus"]
    modifier_ingredients = parts["modifier_ingredients"]

    if not base_menus:
        return []

    base_menu = base_menus[0]
    base_recipes = _search_by_name(base_menu, limit)
    if not base_recipes:
        return []

    related = get_related_ingredients(modifier_ingredients, limit=12)

    modifier_recipes = []
    for ing in modifier_ingredients[:3]:
        modifier_recipes.extend(get_recipes_by_ingredient(ing, limit=2))

    source_blocks = []
    for recipe in base_recipes:
        ingredients = get_recipe_ingredients(recipe["id"])
        source_blocks.append({
            "id": recipe.get("id"),
            "name": recipe.get("name"),
            "servings": recipe.get("servings"),
            "difficulty": recipe.get("difficulty"),
            "ingredients": [
                f"{ing.get('name')}: {ing.get('quantity')}"
                if ing.get("quantity") else ing.get("name")
                for ing in ingredients
            ],
        })

    relation_lines = [
        f"[조합 요청] {query}",
        f"[기본 메뉴] {base_menu}",
    ]
    if modifier_ingredients:
        relation_lines.append(f"[추가 재료 후보] {', '.join(modifier_ingredients[:5])}")
    if related:
        relation_lines.append(
            "[추가 재료와 같이 자주 나오는 재료] "
            + ", ".join(f"{r['name']}({r['recipe_count']})" for r in related[:10])
        )
    relation_lines.append(
        "[기본 레시피 재료] "
        + " / ".join(
            f"{src['name']}: {', '.join(src['ingredients'][:12])}"
            for src in source_blocks[:2]
        )
    )
    if modifier_recipes:
        seen = []
        for recipe in modifier_recipes:
            if recipe.get("name") not in seen:
                seen.append(recipe.get("name"))
        relation_lines.append("[추가 재료 사용 레시피 예시] " + ", ".join(seen[:5]))

    return [{
        "mode": "graph_relation_context",
        "source": "neo4j",
        "status": "exact_recipe_not_found",
        "menu": query,
        "margin": 0,
        "ingredients": relation_lines,
        "base_menu": base_menu,
        "modifier_terms": parts["modifier_terms"],
        "modifier_ingredients": modifier_ingredients,
        "related_ingredients": related,
        "source_recipes": source_blocks,
        "modifier_recipes": modifier_recipes[:5],
    }]


# ============================================================
# 1. 통합 검색
# ============================================================

def search_recipes_smart(query, limit=3, fallback_popular=False):
    """레시피명/재료명 기반 통합 검색.

    기존 코드와 다른 핵심:
    - 재료만 있는 쿼리도 검색함: "두부 들어간 요리" -> 두부 레시피
    - 기본적으로 인기 레시피 fallback을 하지 않음.
      그래야 DB에 없을 때 챗봇이 "없다"고 판단할 수 있음.
    """
    results = _search_by_name(query, limit)
    if len(results) >= limit:
        return results

    menu_tokens, ing_tokens = _tokenize(query)

    if menu_tokens:
        token_results = _search_by_tokens(menu_tokens, ing_tokens, limit)
        results = _merge_results(results, token_results, limit)
        if len(results) >= limit:
            return results

        fallback = _search_by_name(menu_tokens[0], limit)
        for row in fallback:
            row["match_type"] = "partial_token"
        results = _merge_results(results, fallback, limit)
        if len(results) >= limit:
            return results

    if ing_tokens:
        if len(ing_tokens) == 1:
            ingredient_results = get_recipes_by_ingredient(ing_tokens[0], limit)
        else:
            ingredient_results = get_recipes_by_multiple_ingredients(ing_tokens[:3], limit)
        results = _merge_results(results, ingredient_results, limit)

    if not results and fallback_popular:
        results = get_popular_recipes(limit)
        for row in results:
            row["match_type"] = "popular_fallback"

    return results


def _search_by_name(keyword, limit):
    """이름 부분 매칭. 인기도 정렬."""
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    with get_session() as session:
        result = session.run(f"""
            MATCH (r:Recipe)
            WHERE r.name CONTAINS $kw
            WITH r, {_SCORE_EXPR} AS score
            RETURN r.rcp_sno AS id, r.name AS name,
                   r.servings AS servings, r.difficulty AS difficulty,
                   r.kind AS kind, r.cooking_time AS cooking_time,
                   r.view_count AS view_count, score
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
    """메뉴 토큰 매칭 + 재료 토큰 일치당 +1000점 가중."""
    if not menu_tokens:
        return []

    with get_session() as session:
        result = session.run(f"""
            MATCH (r:Recipe)
            WHERE ANY(t IN $menus WHERE r.name CONTAINS t)
            OPTIONAL MATCH (r)-[:CONTAINS]->(i:Ingredient)
            WHERE i.name IN $ings
            WITH r, count(DISTINCT i) AS ing_hits, {_SCORE_EXPR} AS base_score
            WITH r, base_score + ing_hits * 1000 AS score, ing_hits
            RETURN r.rcp_sno AS id, r.name AS name,
                   r.servings AS servings, r.difficulty AS difficulty,
                   r.kind AS kind, r.cooking_time AS cooking_time,
                   r.view_count AS view_count, score, ing_hits
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
    with get_session() as session:
        result = session.run("""
            MATCH (r:Recipe {rcp_sno: $rcp_sno})-[c:CONTAINS]->(i:Ingredient)
            RETURN i.name AS name, coalesce(c.quantity, c.quantity_text, '') AS quantity,
                   i.lv1 AS category, i.lv2 AS subcategory
            ORDER BY i.lv1, i.name
        """, rcp_sno=rcp_sno)
        return [r.data() for r in result]


def get_recipe_detail(rcp_sno):
    """레시피 상세 정보."""
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
                   r.image_url AS image_url, r.steps AS steps
        """, rcp_sno=rcp_sno)
        record = result.single()
        return record.data() if record else None


# ============================================================
# 3. 재료 기반 검색
# ============================================================

def get_recipes_by_ingredient(ingredient_name, limit=3):
    """특정 재료가 들어간 레시피. 자연어 표현도 처리."""
    candidates = _resolve_ingredient_candidates(ingredient_name, limit=5)
    if not candidates:
        return []

    with get_session() as session:
        result = session.run(f"""
            MATCH (r:Recipe)-[:CONTAINS]->(i:Ingredient)
            WHERE i.name IN $names
            WITH r, collect(DISTINCT i.name) AS matched_ingredients, {_SCORE_EXPR} AS score
            RETURN DISTINCT r.rcp_sno AS id, r.name AS name,
                   r.difficulty AS difficulty, r.servings AS servings,
                   r.cooking_time AS cooking_time, r.kind AS kind,
                   r.view_count AS view_count, score, matched_ingredients
            ORDER BY score DESC LIMIT $limit
        """, names=candidates, limit=limit)
        rows = [r.data() for r in result]
        for row in rows:
            row["match_type"] = "ingredient_match"
        return rows


def get_recipes_by_multiple_ingredients(ingredients, limit=3):
    """여러 재료가 모두 들어간 레시피."""
    resolved = []
    for ing in ingredients:
        candidates = _resolve_ingredient_candidates(ing, limit=1)
        if candidates:
            resolved.append(candidates[0])
    resolved = list(dict.fromkeys(resolved))
    if not resolved:
        return []

    with get_session() as session:
        result = session.run(f"""
            MATCH (r:Recipe)-[:CONTAINS]->(i:Ingredient)
            WHERE i.name IN $ingredients
            WITH r, collect(DISTINCT i.name) AS matched_ingredients, {_SCORE_EXPR} AS score
            WHERE size(matched_ingredients) = size($ingredients)
            RETURN r.rcp_sno AS id, r.name AS name,
                   r.difficulty AS difficulty, r.servings AS servings,
                   r.cooking_time AS cooking_time, r.kind AS kind,
                   r.view_count AS view_count, score, matched_ingredients
            ORDER BY score DESC LIMIT $limit
        """, ingredients=resolved, limit=limit)
        rows = [r.data() for r in result]
        for row in rows:
            row["match_type"] = "multi_ingredient_match"
        return rows


def get_recipes_excluding_ingredient(keyword, exclude, limit=3):
    """특정 재료를 제외한 레시피."""
    exclude_candidates = _resolve_ingredient_candidates(exclude, limit=1)
    exclude_name = exclude_candidates[0] if exclude_candidates else exclude

    with get_session() as session:
        result = session.run(f"""
            MATCH (r:Recipe)
            WHERE r.name CONTAINS $keyword
            AND NOT (r)-[:CONTAINS]->(:Ingredient {{name: $exclude}})
            WITH r, {_SCORE_EXPR} AS score
            RETURN r.rcp_sno AS id, r.name AS name,
                   r.difficulty AS difficulty, r.servings AS servings,
                   r.cooking_time AS cooking_time, r.kind AS kind,
                   r.view_count AS view_count, score
            ORDER BY score DESC LIMIT $limit
        """, keyword=keyword, exclude=exclude_name, limit=limit)
        return [r.data() for r in result]


# ============================================================
# 4. 유사 레시피 추천
# ============================================================

def find_similar_recipes(rcp_sno, limit=3, min_shared=2):
    """주어진 레시피와 재료를 공유하는 유사 레시피 추천."""
    with get_session() as session:
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
# 5. 인기/조건 추천
# ============================================================

def get_popular_recipes(limit=3):
    """전체 인기 레시피 top N."""
    with get_session() as session:
        result = session.run(f"""
            MATCH (r:Recipe)
            WITH r, {_SCORE_EXPR} AS score
            RETURN r.rcp_sno AS id, r.name AS name,
                   r.difficulty AS difficulty, r.servings AS servings,
                   r.cooking_time AS cooking_time, r.kind AS kind,
                   r.view_count AS view_count, score
            ORDER BY score DESC LIMIT $limit
        """, limit=limit)
        return [r.data() for r in result]


def recommend_recipes(
    kind=None,
    difficulty=None,
    servings=None,
    cooking_method=None,
    limit=3,
):
    """조건 기반 추천."""
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
            WITH r, {_SCORE_EXPR} AS score
            RETURN r.rcp_sno AS id, r.name AS name,
                   r.difficulty AS difficulty, r.servings AS servings,
                   r.kind AS kind, r.cooking_time AS cooking_time,
                   r.view_count AS view_count, score
            ORDER BY score DESC LIMIT $limit
        """, **params)
        return [r.data() for r in result]


# ============================================================
# 6. 챗봇/기존 import 호환
# ============================================================

def search_recipes(keyword, limit=5):
    """기존 search_recipes 인터페이스 유지."""
    return search_recipes_smart(keyword, limit=limit, fallback_popular=False)


def get_chatbot_context(keyword):
    """agent.py의 recipe_db_expert에서 호출하는 형태로 반환.

    동작 순서:
    1. 요청 문장 그대로 레시피명 검색
    2. 없으면 "마라김치찌개" 같은 조합형 메뉴를 Neo4j 관계 컨텍스트로 반환
    3. 그래도 없으면 재료/토큰 기반 검색
    """
    recipes = _search_by_name(keyword, limit=5)
    if not recipes:
        graph_rows = build_graph_relation_context(keyword, limit=3)
        if graph_rows:
            return graph_rows

    if not recipes:
        recipes = search_recipes_smart(keyword, limit=5, fallback_popular=False)

    if not recipes:
        return []

    results = []
    for recipe in recipes:
        ingredients = get_recipe_ingredients(recipe["id"])
        ing_names = []
        for ing in ingredients:
            name = ing.get("name")
            quantity = ing.get("quantity")
            ing_names.append(f"{name}: {quantity}" if quantity else name)

        results.append({
            "menu": recipe["name"],
            "margin": 0,
            "ingredients": ing_names,
            "difficulty": recipe.get("difficulty"),
            "servings": recipe.get("servings"),
            "cooking_time": recipe.get("cooking_time"),
            "view_count": recipe.get("view_count"),
            "match_type": recipe.get("match_type"),
            "matched_ingredients": recipe.get("matched_ingredients", []),
        })

    return results
