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
    안전화 규칙:
    - exact 매칭 우선
    - substring 매칭은 한 방향만:
      * "당근손가락길이만큼"(긴 쿼리) 안에 ing("당근")이 포함 → OK (수다 제거 케이스)
      * cleaned(쿼리)가 ing 안에 포함되는 역방향은 길이 차이가 작을 때만
        예) "마라"(2글자)가 "고구마라떼"(6글자) 안에 → 차이 너무 큼
            "다진마"가 "다진마늘" 안에 → 차이 1글자
    """
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

    # substring fallback
    for ing in _KNOWN_INGREDIENTS:
        ing_compact = _compact(ing)
        if not ing_compact or len(ing_compact) < 2:
            continue
        # 정방향: 쿼리 안에 재료명 포함 (수다 제거)
        if ing_compact in raw_compact or ing_compact in cleaned_compact:
            candidates.append(ing)
        # 역방향: 쿼리가 재료명 안에 포함되는 경우 — 길이 차이 1글자 이내만
        elif (len(cleaned_compact) >= 2
              and cleaned_compact in ing_compact
              and (len(ing_compact) - len(cleaned_compact)) <= 1):
            candidates.append(ing)
        if len(dict.fromkeys(candidates)) >= limit:
            break

    return list(dict.fromkeys(candidates))[:limit]


def _extract_graph_parts(query):
    """없는 메뉴를 조합하기 위한 base menu + modifier 추출 (재료 + 메뉴 둘 다).

    동작:
      1) _KNOWN_MENUS 매칭으로 base 메뉴들 추출 (가장 첫 매칭이 base)
      2) 남은 토큰을 modifier로 분류:
         - modifier가 _KNOWN_INGREDIENTS와 매칭 → modifier_ingredients
         - modifier가 _KNOWN_MENUS와 매칭 → modifier_menus
           (예: "김치된장찌개" → base="된장찌개", modifier_menu="김치찌개"
            → 김치찌개 재료까지 컨텍스트에 추가)

    예시:
      "마라김치찌개"   → base=["김치찌개"], modifier_terms=["마라"],
                          modifier_ingredients=[], modifier_menus=[]
                          ("마라" 매칭 실패 시 빈 list — 안전)
      "김치된장찌개"   → base=["된장찌개"], modifier_terms=["김치"],
                          modifier_ingredients=["김치"], modifier_menus=["김치찌개", "김치전", ...]
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
            # base는 1개만 잡고 나머지는 modifier로 (김치된장찌개의 김치찌개를 base로 두지 않게)
            break

    # 남은 토큰들 (공백 기준)
    modifier_terms = [
        term for term in re.split(r"\s+", remaining_compact.strip())
        if len(term) >= 2
    ]
    # 원문 토큰도 추가 (공백 있는 케이스)
    for token in re.split(r"\s+", _strip_query_words(raw)):
        token_compact = _compact(token)
        if (len(token_compact) >= 2
            and token_compact not in [_compact(m) for m in base_menus]
            and token_compact not in modifier_terms):
            modifier_terms.append(token_compact)

    modifier_ingredients = []
    modifier_menus = []
    base_compact_set = {_compact(m) for m in base_menus}

    for term in modifier_terms:
        # 재료 후보
        modifier_ingredients.extend(_resolve_ingredient_candidates(term, limit=3))
        # 메뉴 후보 (modifier가 _KNOWN_MENUS의 어떤 메뉴를 substring으로 포함)
        # 예: "김치" → 김치찌개, 김치전 등 매칭
        for menu in _KNOWN_MENUS:
            mc = _compact(menu)
            if not mc or mc in base_compact_set:
                continue
            if term in mc and len(modifier_menus) < 5:
                modifier_menus.append(menu)

    return {
        "base_menus": list(dict.fromkeys(base_menus)),
        "modifier_terms": list(dict.fromkeys(modifier_terms)),
        "modifier_ingredients": list(dict.fromkeys(modifier_ingredients)),
        "modifier_menus": list(dict.fromkeys(modifier_menus)),
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
    """DB에 정확한 레시피가 없을 때 Neo4j 관계로 조합 컨텍스트 생성.

    동작 흐름:
      1) _extract_graph_parts → base 메뉴 + modifier (재료/메뉴)
      2) base 메뉴의 인기 레시피 조회 (search_by_name)
      3) modifier_ingredients의 co-occurring 재료 조회
      4) modifier_menus가 있으면 그 메뉴들의 대표 재료도 컨텍스트에 추가
         예: "김치된장찌개" → base=된장찌개, modifier_menus=[김치찌개, 김치전]
              → 김치찌개의 재료 일부도 컨텍스트에 같이
      5) 모든 정보를 텍스트 라인으로 묶어서 LLM에 전달

    이 함수는 완성 레시피를 만들지 않음. 관계 정보만 반환 → LLM이 RAG로 답변 생성.
    """
    parts = _extract_graph_parts(query)
    base_menus = parts["base_menus"]
    modifier_ingredients = parts["modifier_ingredients"]
    modifier_menus = parts.get("modifier_menus", [])

    if not base_menus:
        return []

    base_menu = base_menus[0]
    base_recipes = _search_by_name(base_menu, limit)
    if not base_recipes:
        return []

    # modifier 재료와 같이 나오는 co-occurring 재료
    related = get_related_ingredients(modifier_ingredients, limit=12)

    # modifier 재료가 들어간 다른 레시피 예시
    modifier_recipes = []
    for ing in modifier_ingredients[:3]:
        modifier_recipes.extend(get_recipes_by_ingredient(ing, limit=2))

    # base 레시피들의 재료 컨텍스트
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

    # 신규: modifier가 메뉴인 경우 (김치된장찌개 → modifier_menus=[김치찌개])
    # 그 메뉴들의 대표 레시피 1개씩 재료 컨텍스트에 추가
    modifier_menu_blocks = []
    for mm in modifier_menus[:2]:
        mm_recipes = _search_by_name(mm, 1)
        if mm_recipes:
            mm_recipe = mm_recipes[0]
            mm_ings = get_recipe_ingredients(mm_recipe["id"])
            modifier_menu_blocks.append({
                "menu_name": mm,
                "recipe_name": mm_recipe.get("name"),
                "ingredients": [
                    f"{ing.get('name')}: {ing.get('quantity')}"
                    if ing.get("quantity") else ing.get("name")
                    for ing in mm_ings
                ],
            })

    relation_lines = [
        f"[조합 요청] {query}",
        f"[기본 메뉴] {base_menu}",
    ]
    if modifier_ingredients:
        relation_lines.append(f"[추가 재료 후보] {', '.join(modifier_ingredients[:5])}")
    if modifier_menus:
        relation_lines.append(f"[추가 메뉴 후보] {', '.join(modifier_menus[:5])}")
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
    if modifier_menu_blocks:
        relation_lines.append(
            "[추가 메뉴의 대표 재료 — 이걸 base 메뉴에 합쳐서 답변 구성 권장] "
            + " / ".join(
                f"{blk['menu_name']}: {', '.join(blk['ingredients'][:10])}"
                for blk in modifier_menu_blocks
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
        "modifier_menus": modifier_menus,
        "related_ingredients": related,
        "source_recipes": source_blocks,
        "modifier_recipes": modifier_recipes[:5],
        "modifier_menu_blocks": modifier_menu_blocks,
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
    """이름 매칭 — 짧은 쿼리는 prefix만, 긴 건 substring.

    [한국어 메뉴명 특성]
    띄어쓰기 없는 합성어 많음 → substring 매칭이 우연 매칭 일으킴.
      예) "장어" CONTAINS → "고추장어묵볶음"의 "장+어" 부분 매칭 ❌
          "마라" CONTAINS → "고구마라떼"의 "마+라" 부분 매칭 ❌

    [해결]
    - 1~2글자 쿼리: STARTS WITH 만 허용 (자연스러운 매칭만)
    - 3글자 이상 쿼리: CONTAINS 허용 (우연 매칭 위험 ↓)
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    # 길이 분기 — Cypher에서 처리
    cypher = f"""
        MATCH (r:Recipe)
        WHERE
          (size($kw) <= 2 AND r.name STARTS WITH $kw)
          OR (size($kw) >= 3 AND r.name CONTAINS $kw)
        WITH r, {_SCORE_EXPR} AS score
        RETURN r.rcp_sno AS id, r.name AS name,
               r.servings AS servings, r.difficulty AS difficulty,
               r.kind AS kind, r.cooking_time AS cooking_time,
               r.view_count AS view_count, score
        ORDER BY score DESC
        LIMIT $limit
    """
    with get_session() as session:
        result = session.run(cypher, kw=keyword, limit=limit)
        rows = []
        for r in result:
            d = r.data()
            d["match_type"] = "name_match"
            rows.append(d)
        return rows


def _search_by_tokens(menu_tokens, ing_tokens, limit):
    """메뉴 토큰 매칭 + 재료 토큰 일치당 +1000점 가중.

    _search_by_name과 같은 길이 분기 룰 적용 (짧은 토큰은 prefix만).
    """
    if not menu_tokens:
        return []

    with get_session() as session:
        result = session.run(f"""
            MATCH (r:Recipe)
            WHERE ANY(t IN $menus WHERE
              (size(t) <= 2 AND r.name STARTS WITH t)
              OR (size(t) >= 3 AND r.name CONTAINS t)
            )
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

# ── 재료 노이즈 필터 (발표용 가벼운 버전, _DROP_NOISE_INGREDIENTS=False로 끄기 가능) ──
# 근본 재파싱(silver) 대신, 그래프에서 재료를 꺼낼 때 명백한 파편만 거른다.
# 표시 단계 필터라 ETL/silver는 안 건드리며, 모든 하위(원가·가격·답변)에 동일 적용됨.
_DROP_NOISE_INGREDIENTS = True
_NOISE_TOKENS = ("국그릇", "그릇", "한그릇")  # 수량/그릇이 재료명에 붙은 파편 (예: '신김치국그릇')


def _filter_noise_ingredients(rows):
    """재료 row 리스트에서 명백한 노이즈/중복 제거. 전부 걸러지면 원본 유지(안전)."""
    if not _DROP_NOISE_INGREDIENTS:
        return rows
    cleaned, seen = [], set()
    for r in rows:
        name = (r.get("name") or "").strip()
        if not name:
            continue
        if any(tok in name for tok in _NOISE_TOKENS):
            continue          # '신김치국그릇' 같은 파편 제거
        if name in seen:
            continue          # 같은 이름 중복 제거
        seen.add(name)
        cleaned.append(r)
    return cleaned if cleaned else rows   # 전부 걸러지면 원본 유지(빈 재료 사고 방지)


def get_recipe_ingredients(rcp_sno):
    """레시피의 재료 + 수량 조회."""
    with get_session() as session:
        result = session.run("""
            MATCH (r:Recipe {rcp_sno: $rcp_sno})-[c:CONTAINS]->(i:Ingredient)
            RETURN i.name AS name, coalesce(c.quantity, c.quantity_text, '') AS quantity,
                   i.lv1 AS category, i.lv2 AS subcategory
            ORDER BY i.lv1, i.name
        """, rcp_sno=rcp_sno)
        return _filter_noise_ingredients([r.data() for r in result])


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
            WHERE (
              (size($keyword) <= 2 AND r.name STARTS WITH $keyword)
              OR (size($keyword) >= 3 AND r.name CONTAINS $keyword)
            )
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
# 4. 대체재 추천 (그래프 기반 RAG)
# ============================================================

def suggest_substitute_ingredient(menu, missing_ingredient, limit=5):
    """주어진 메뉴에서 특정 재료를 대체할 만한 재료를 그래프 관계로 추천.

    cost_calculator가 이 함수의 결과(후보 5개)를 받고,
    가격사전과 매칭해 원가가 가장 낮은 대체재를 고르는 식으로 사용 가능.

    동작 순서:
      1. missing_ingredient의 lv1(대분류)를 찾는다 (예: 돼지고기 → '육류·계란')
      2. 해당 메뉴를 가진 레시피들의 재료 중,
         같은 lv1이면서 missing이 아닌 재료를 빈도순으로 뽑는다.

    예: suggest_substitute_ingredient("김치찌개", "돼지고기")
        → [
            {"name": "햄",   "category": "육류·계란", "recipe_count": 23},
            {"name": "참치", "category": "어패류",   "recipe_count": 18},
            ...
          ]

    자연어 표현도 처리: "고기" → "돼지고기"로 자동 변환 후 검색
    """
    candidates = _resolve_ingredient_candidates(missing_ingredient, limit=1)
    if not candidates:
        return []
    missing = candidates[0]

    with get_session() as session:
        # missing의 lv1(대분류)·lv2(소분류) 확인
        meta = session.run(
            "MATCH (i:Ingredient {name: $name}) RETURN i.lv1 AS lv1, i.lv2 AS lv2 LIMIT 1",
            name=missing,
        ).single()
        if not meta or not meta["lv1"]:
            return []
        missing_lv1 = meta["lv1"]
        missing_lv2 = meta["lv2"]

        # 넓은 풀: 같은 lv1 재료 전체 + 전체 등장빈도(recipe_count) + 해당 메뉴 등장수(menu_count)
        # (메뉴를 '하드 필터'로 쓰면 같은 계열 재료가 그 메뉴 레시피에 없을 때 누락되므로,
        #  메뉴는 가산점으로만 쓰고 풀은 lv1 전체에서 가져온다)
        pool = session.run("""
            MATCH (r:Recipe)-[:CONTAINS]->(alt:Ingredient)
            WHERE alt.lv1 = $missing_lv1 AND alt.name <> $missing
            WITH alt.name AS name, alt.lv1 AS lv1, alt.lv2 AS lv2,
                 count(DISTINCT r) AS recipe_count,
                 count(DISTINCT CASE WHEN r.name CONTAINS $menu THEN r END) AS menu_count
            ORDER BY recipe_count DESC
            LIMIT 60
            RETURN name, lv1, lv2, recipe_count, menu_count
        """, menu=menu, missing=missing, missing_lv1=missing_lv1).data()

        # 이 메뉴의 대표 main_ingredient 조회 (제육볶음 → '돼지고기')
        # → missing이 이 요리의 핵심 재료인지 판정하는 데 사용
        mi_row = session.run("""
            MATCH (r:Recipe)
            WHERE r.name CONTAINS $menu AND r.main_ingredient IS NOT NULL
            RETURN r.main_ingredient AS mi, count(*) AS c
            ORDER BY c DESC LIMIT 1
        """, menu=menu).single()
        menu_main = (mi_row["mi"] if mi_row else "") or ""

    # ── 핵심 재료(core) 판정 ──
    # 제육볶음의 돼지고기처럼, missing이 그 요리의 '정체성' 재료면
    # 계열 밖(스팸·참치)으로는 절대 대체하지 않는다(없으면 추천 안 함).
    # 신호: ① 메뉴명이 missing을 포함(돼지고기김치찌개) ② 대표 main_ingredient와 같은 계열
    def _same_family(a, b):
        if not a or not b:
            return False
        if a in b or b in a:
            return True
        return len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2]

    is_core = (bool(menu) and missing in menu) or _same_family(missing, menu_main)

    # ── 요리 정체성 점수로 재정렬 ──
    # 같은 재료 계열(같은 동물/소분류)이 위로 오게. 스팸·참치처럼 계열 신호 0인 건 뒤로.
    def _family_score(name, lv2):
        s = 0
        if missing in name or name in missing:        # 돼지고기 ↔ 돼지고기목살
            s += 4
        if len(missing) >= 2 and len(name) >= 2 and missing[:2] == name[:2]:  # 돼지.. 공유
            s += 3
        if missing_lv2 and lv2 and lv2 == missing_lv2:  # 같은 소분류
            s += 2
        return s

    ranked = []
    for row in pool:
        fs = _family_score(row["name"], row.get("lv2"))
        ranked.append((fs, row.get("menu_count", 0), row.get("recipe_count", 0), row))
    # 계열성 > 메뉴 등장수 > 전체 빈도 순
    ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

    # 계열 신호 있는 것 우선.
    family = [r for r in ranked if r[0] > 0]
    if is_core:
        # 핵심 재료(제육볶음의 돼지고기)는 계열 밖(스팸·참치)으로 대체 금지.
        # 같은 계열(돼지 부위)만 허용. 없으면 빈 결과 → 엉뚱한 대체 안 함.
        chosen = family[:limit]
    else:
        # 보조 재료는 계열 우선, 없으면 빈도순으로라도 채움.
        chosen = (family if family else ranked)[:limit]
    return [{"name": r[3]["name"], "category": r[3]["lv1"],
             "recipe_count": r[3]["recipe_count"]} for r in chosen]


def get_menu_main_ingredient(menu):
    """메뉴의 대표 main_ingredient(레시피 작성자 표기)를 반환. 핵심재료 판정용.

    예: '제육볶음' → '돼지고기',  '김치찌개' → '김치'(보통)
    cost_calculator가 '핵심 재료는 대체 대상에서 제외'하는 데 사용.
    """
    if not menu:
        return ""
    with get_session() as session:
        row = session.run("""
            MATCH (r:Recipe)
            WHERE r.name CONTAINS $menu AND r.main_ingredient IS NOT NULL
            RETURN r.main_ingredient AS mi, count(*) AS c
            ORDER BY c DESC LIMIT 1
        """, menu=menu).single()
        return (row["mi"] if row else "") or ""


# ============================================================
# 5. 유사 레시피 추천
# ============================================================

def find_similar_recipes(rcp_sno, limit=3, min_shared=2):
    """주어진 레시피와 재료를 공유하는 유사 레시피 추천."""
    #기준 레시피 1개를 잡고
    # → 그 레시피와 재료가 많이 겹치는 다른 레시피 찾기

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
