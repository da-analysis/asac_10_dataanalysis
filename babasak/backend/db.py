import os
import re
from neo4j import GraphDatabase

_driver = None
_KNOWN_MENUS = []
_KNOWN_INGREDIENTS = []
_DICT_LOADED = False

_SCORE_EXPR = (
    "coalesce(r.view_count,0) + "
    "coalesce(r.recommend_count,0) * 100 + "
    "coalesce(r.scrap_count,0) * 50"
)

_QUERY_WORDS = [
    "레시피", "요리", "메뉴", "음식", "추천", "알려줘", "찾아줘",
    "들어간", "들어가는", "넣은", "넣는", "사용한", "사용하는",
    "포함", "있는", "으로 만든", "로 만든", "만든", "만들",
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
    for word in sorted(_QUERY_WORDS, key=len, reverse=True):
        cleaned = cleaned.replace(word, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _dedupe(values):
    return list(dict.fromkeys([v for v in values if v]))


def _first_text(*values):
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _recipe_display_name(recipe):
    """User-facing recipe name.

    r.rcp_sno/id is kept only for internal lookups. Chatbot rows should expose
    title/name instead of recipe ids.
    """
    return _first_text(recipe.get("title"), recipe.get("name")) or "레시피"


def get_driver():
    """Neo4j driver singleton."""
    global _driver
    if _driver is None:
        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER")
        password = os.environ.get("NEO4J_PASSWORD")
        if not uri or not user or password is None:
            raise RuntimeError(
                "Neo4j 환경변수 누락: "
                "NEO4J_URI / NEO4J_USERNAME(또는 NEO4J_USER) / NEO4J_PASSWORD"
            )
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def get_session():
    """Use NEO4J_DATABASE when present; otherwise use the default database."""
    database = os.environ.get("NEO4J_DATABASE")
    if database:
        return get_driver().session(database=database)
    return get_driver().session()


# ============================================================
# Dictionaries / query parsing
# ============================================================

def _load_dictionaries(menu_limit=3000, ing_limit=8000):
    """Load recipe-name and ingredient dictionaries from the graph."""
    global _KNOWN_MENUS, _KNOWN_INGREDIENTS, _DICT_LOADED
    if _DICT_LOADED:
        return

    with get_session() as session:
        menu_rows = session.run("""
            MATCH (r:Recipe)
            WHERE r.name IS NOT NULL AND size(r.name) <= 25
            WITH r.name AS name, count(*) AS cnt
            ORDER BY cnt DESC
            LIMIT $limit
            RETURN name
        """, limit=menu_limit)
        _KNOWN_MENUS = [row["name"] for row in menu_rows if row["name"]]

        ing_rows = session.run("""
            MATCH (:Recipe)-[:CONTAINS]->(i:Ingredient)
            WHERE i.name IS NOT NULL
            WITH i.name AS name, count(*) AS cnt
            ORDER BY cnt DESC
            LIMIT $limit
            RETURN name
        """, limit=ing_limit)
        _KNOWN_INGREDIENTS = [row["name"] for row in ing_rows if row["name"]]

    _KNOWN_MENUS = sorted(set(_KNOWN_MENUS), key=len, reverse=True)
    _KNOWN_INGREDIENTS = sorted(set(_KNOWN_INGREDIENTS), key=len, reverse=True)
    _DICT_LOADED = True


def extract_graph_query_parts(query):
    """Split a query into base recipe/menu tokens and modifier ingredients.

    Example:
    - "명란김치찌개" -> base_menus=["김치찌개"], modifier_ingredients=["명란"]
    - "두부 들어간 요리" -> base_menus=[], modifier_ingredients=["두부"]
    """
    _load_dictionaries()
    raw = query or ""
    raw_compact = _compact(_strip_query_words(raw))
    remaining_compact = raw_compact

    base_menus = []
    for menu in _KNOWN_MENUS:
        menu_compact = _compact(menu)
        if menu_compact and menu_compact in remaining_compact:
            base_menus.append(menu)
            remaining_compact = remaining_compact.replace(menu_compact, " ")
            break

    modifier_ingredients = []
    ingredient_scan_text = remaining_compact if base_menus else raw_compact
    for ing in _KNOWN_INGREDIENTS:
        ing_compact = _compact(ing)
        if not ing_compact:
            continue
        if ing_compact in ingredient_scan_text:
            modifier_ingredients.append(ing)
            ingredient_scan_text = ingredient_scan_text.replace(ing_compact, " ")

    if not modifier_ingredients:
        alias = _INGREDIENT_ALIASES.get(raw_compact)
        if alias:
            modifier_ingredients.append(alias)

    remaining_terms = [
        term for term in re.split(r"\s+", ingredient_scan_text.strip())
        if len(term) >= 2
    ]

    return {
        "query": raw,
        "base_menus": _dedupe(base_menus),
        "modifier_ingredients": _dedupe(modifier_ingredients),
        "remaining_terms": _dedupe(remaining_terms),
    }


def resolve_ingredient_candidates(text, limit=5):
    """Resolve a messy ingredient phrase to Ingredient.name candidates."""
    _load_dictionaries()
    raw_compact = _compact(_strip_query_words(text))
    candidates = []

    alias = _INGREDIENT_ALIASES.get(raw_compact)
    if alias:
        candidates.append(alias)

    for ing in _KNOWN_INGREDIENTS:
        ing_compact = _compact(ing)
        if ing_compact and ing_compact == raw_compact:
            candidates.append(ing)

    for ing in _KNOWN_INGREDIENTS:
        ing_compact = _compact(ing)
        if not ing_compact or len(ing_compact) < 2:
            continue
        if ing_compact in raw_compact or raw_compact in ing_compact:
            candidates.append(ing)
        if len(_dedupe(candidates)) >= limit:
            break

    return _dedupe(candidates)[:limit]


# ============================================================
# Core relation queries
# ============================================================

def search_recipes_by_name(keyword, limit=5):
    """Recipe-name candidate search. Popularity is only a recipe tie-breaker."""
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    with get_session() as session:
        rows = session.run(f"""
            MATCH (r:Recipe)
            WHERE r.name = $keyword OR r.name CONTAINS $keyword
            WITH r,
                 CASE WHEN r.name = $keyword THEN 1 ELSE 0 END AS exact_hit,
                 {_SCORE_EXPR} AS popularity_score
            RETURN r.rcp_sno AS id, r.name AS name, r.title AS title,
                   r.servings AS servings, r.difficulty AS difficulty,
                   r.cooking_time AS cooking_time, r.kind AS kind,
                   r.cooking_method AS cooking_method,
                   r.view_count AS view_count,
                   r.steps AS steps,
                   popularity_score,
                   exact_hit,
                   CASE WHEN exact_hit = 1 THEN 'exact_name' ELSE 'name_contains' END AS match_type
            ORDER BY exact_hit DESC, popularity_score DESC
            LIMIT $limit
        """, keyword=keyword, limit=limit)
        return [row.data() for row in rows]


def get_recipe_ingredients(rcp_sno):
    """Ingredients of a recipe. Uses directed Recipe -> Ingredient edge."""
    with get_session() as session:
        rows = session.run("""
            MATCH (r:Recipe {rcp_sno: $rcp_sno})-[c:CONTAINS]->(i:Ingredient)
            WITH i, c,
                 CASE
                   WHEN c.quantity_text IS NOT NULL AND trim(toString(c.quantity_text)) <> ''
                     THEN c.quantity_text
                   WHEN c.quantity IS NOT NULL AND trim(toString(c.quantity)) <> ''
                     THEN c.quantity
                   WHEN c.quantity_count IS NOT NULL
                     THEN toString(c.quantity_count) + coalesce(c.quantity_unit, '')
                   ELSE ''
                 END AS display_quantity
            RETURN i.name AS name,
                   display_quantity AS quantity,
                   c.quantity_text AS quantity_text,
                   c.quantity_g AS quantity_g,
                   c.quantity_count AS quantity_count,
                   c.quantity_unit AS quantity_unit,
                   i.lv1 AS category,
                   i.lv2 AS subcategory
            ORDER BY i.lv1, i.name
        """, rcp_sno=rcp_sno)
        return [row.data() for row in rows]


def get_recipe_detail(rcp_sno):
    """Recipe metadata including cooking steps."""
    with get_session() as session:
        row = session.run("""
            MATCH (r:Recipe {rcp_sno: $rcp_sno})
            RETURN r.rcp_sno AS id,
                   r.name AS name,
                   r.title AS title,
                   r.servings AS servings,
                   r.difficulty AS difficulty,
                   r.cooking_time AS cooking_time,
                   r.cooking_method AS cooking_method,
                   r.kind AS kind,
                   r.situation AS situation,
                   r.main_ingredient AS main_ingredient,
                   r.view_count AS view_count,
                   r.recommend_count AS recommend_count,
                   r.scrap_count AS scrap_count,
                   r.description AS description,
                   r.image_url AS image_url,
                   r.steps AS steps
        """, rcp_sno=rcp_sno).single()
        return row.data() if row else None


def get_recipes_by_ingredient(ingredient_name, limit=5):
    """Recipes containing an ingredient. Popularity is a tie-breaker, not relation score."""
    candidates = resolve_ingredient_candidates(ingredient_name, limit=5)
    if not candidates:
        return []

    with get_session() as session:
        rows = session.run(f"""
            MATCH (i:Ingredient)<-[:CONTAINS]-(r:Recipe)
            WHERE i.name IN $names
            WITH r, collect(DISTINCT i.name) AS matched_ingredients,
                 count(DISTINCT i) AS relation_score,
                 {_SCORE_EXPR} AS popularity_score
            RETURN r.rcp_sno AS id, r.name AS name, r.title AS title,
                   r.servings AS servings, r.difficulty AS difficulty,
                   r.cooking_time AS cooking_time, r.kind AS kind,
                   r.view_count AS view_count,
                   r.steps AS steps,
                   matched_ingredients,
                   relation_score,
                   popularity_score,
                   'ingredient_relation' AS match_type
            ORDER BY relation_score DESC, popularity_score DESC
            LIMIT $limit
        """, names=candidates, limit=limit)
        return [row.data() for row in rows]


def get_recipes_by_multiple_ingredients(ingredients, limit=5):
    """Recipes containing all requested ingredients."""
    resolved = []
    for ing in ingredients or []:
        resolved.extend(resolve_ingredient_candidates(ing, limit=1))
    resolved = _dedupe(resolved)
    if not resolved:
        return []

    with get_session() as session:
        rows = session.run(f"""
            MATCH (i:Ingredient)<-[:CONTAINS]-(r:Recipe)
            WHERE i.name IN $names
            WITH r, collect(DISTINCT i.name) AS matched_ingredients,
                 {_SCORE_EXPR} AS popularity_score
            WHERE size(matched_ingredients) = size($names)
            RETURN r.rcp_sno AS id, r.name AS name, r.title AS title,
                   r.servings AS servings, r.difficulty AS difficulty,
                   r.cooking_time AS cooking_time, r.kind AS kind,
                   r.view_count AS view_count,
                   r.steps AS steps,
                   matched_ingredients,
                   size(matched_ingredients) AS relation_score,
                   popularity_score,
                   'multi_ingredient_relation' AS match_type
            ORDER BY relation_score DESC, popularity_score DESC
            LIMIT $limit
        """, names=resolved, limit=limit)
        return [row.data() for row in rows]


def get_related_ingredients(ingredient_names, limit=12):
    """Ingredients frequently co-occurring with the given ingredients.

    This is an ingredient relation query, so view_count is intentionally not used.
    """
    names = _dedupe(ingredient_names or [])
    if not names:
        return []

    with get_session() as session:
        rows = session.run("""
            MATCH (seed:Ingredient)<-[:CONTAINS]-(r:Recipe)-[:CONTAINS]->(co:Ingredient)
            WHERE seed.name IN $names AND NOT co.name IN $names
            WITH co.name AS name, co.lv1 AS category, count(DISTINCT r) AS co_recipe_count
            RETURN name, category, co_recipe_count
            ORDER BY co_recipe_count DESC
            LIMIT $limit
        """, names=names, limit=limit)
        return [row.data() for row in rows]


def find_similar_recipes_by_ingredients(seed_ingredients, exclude_ids=None, limit=5):
    """Graph similarity by shared ingredients, not simple LIKE search."""
    seed_ingredients = _dedupe(seed_ingredients or [])
    exclude_ids = exclude_ids or []
    if not seed_ingredients:
        return []

    with get_session() as session:
        rows = session.run(f"""
            MATCH (r:Recipe)-[:CONTAINS]->(i:Ingredient)
            WHERE i.name IN $seed_ingredients
              AND NOT r.rcp_sno IN $exclude_ids
            WITH r, collect(DISTINCT i.name) AS shared_ingredients
            MATCH (r)-[:CONTAINS]->(all_i:Ingredient)
            WITH r, shared_ingredients,
                 count(DISTINCT all_i) AS recipe_ingredient_count,
                 size(shared_ingredients) AS shared_count,
                 size($seed_ingredients) AS seed_count,
                 {_SCORE_EXPR} AS popularity_score
            WITH r, shared_ingredients, recipe_ingredient_count, shared_count, popularity_score,
                 seed_count + recipe_ingredient_count - shared_count AS union_count
            WITH r, shared_ingredients, recipe_ingredient_count, shared_count, popularity_score,
                 CASE
                   WHEN union_count = 0 THEN 0.0
                   ELSE toFloat(shared_count) / toFloat(union_count)
                 END AS graph_similarity
            RETURN r.rcp_sno AS id, r.name AS name, r.title AS title,
                   r.servings AS servings, r.difficulty AS difficulty,
                   r.cooking_time AS cooking_time, r.kind AS kind,
                   r.view_count AS view_count,
                   r.steps AS steps,
                   shared_ingredients,
                   shared_count,
                   recipe_ingredient_count,
                   graph_similarity,
                   popularity_score,
                   'ingredient_similarity' AS match_type
            ORDER BY graph_similarity DESC, shared_count DESC, popularity_score DESC
            LIMIT $limit
        """, seed_ingredients=seed_ingredients, exclude_ids=exclude_ids, limit=limit)
        return [row.data() for row in rows]


def find_similar_recipes(rcp_sno, limit=5, min_shared=2):
    """Recipes similar to a given recipe by shared ingredients."""
    ingredients = [ing["name"] for ing in get_recipe_ingredients(rcp_sno)]
    rows = find_similar_recipes_by_ingredients(ingredients, exclude_ids=[rcp_sno], limit=limit)
    return [row for row in rows if row.get("shared_count", 0) >= min_shared]


# ============================================================
# GraphRAG context
# ============================================================

def build_graph_relation_context(query, limit=3):
    """Return relation context only. No recipe creation happens here."""
    parts = extract_graph_query_parts(query)
    base_menus = parts["base_menus"]
    modifier_ingredients = parts["modifier_ingredients"]

    base_recipes = []
    for menu in base_menus[:1]:
        base_recipes.extend(search_recipes_by_name(menu, limit=limit))
    base_recipes = base_recipes[:limit]

    modifier_recipes = []
    for ing in modifier_ingredients[:3]:
        modifier_recipes.extend(get_recipes_by_ingredient(ing, limit=2))

    if not base_recipes and modifier_ingredients:
        return {
            "mode": "ingredient_recipe_relation",
            "source": "neo4j",
            "query": query,
            "modifier_ingredients": modifier_ingredients,
            "recipes": get_recipes_by_multiple_ingredients(modifier_ingredients[:3], limit=limit)
                       if len(modifier_ingredients) > 1
                       else get_recipes_by_ingredient(modifier_ingredients[0], limit=limit),
            "related_ingredients": get_related_ingredients(modifier_ingredients, limit=12),
        }

    if not base_recipes:
        return {}

    base_ids = [recipe["id"] for recipe in base_recipes if recipe.get("id") is not None]
    base_ingredients = []
    source_recipes = []
    for recipe in base_recipes:
        detail = get_recipe_detail(recipe["id"]) or {}
        ingredients = get_recipe_ingredients(recipe["id"])
        ing_names = [ing["name"] for ing in ingredients if ing.get("name")]
        base_ingredients.extend(ing_names)
        source_recipes.append({
            "id": recipe.get("id"),
            "name": _recipe_display_name({**recipe, **detail}),
            "recipe_name": recipe.get("name") or detail.get("name"),
            "recipe_title": recipe.get("title") or detail.get("title"),
            "servings": recipe.get("servings"),
            "difficulty": recipe.get("difficulty"),
            "cooking_time": recipe.get("cooking_time"),
            "steps": detail.get("steps") or recipe.get("steps"),
            "ingredients": [
                f"{ing.get('name')}: {ing.get('quantity')}"
                if ing.get("quantity") else ing.get("name")
                for ing in ingredients
            ],
        })

    seed_ingredients = _dedupe(modifier_ingredients + base_ingredients)
    similar_recipes = find_similar_recipes_by_ingredients(
        seed_ingredients,
        exclude_ids=base_ids,
        limit=limit,
    )
    related_ingredients = get_related_ingredients(modifier_ingredients, limit=12)

    summary_lines = [
        "GRAPH_CONTEXT_ONLY",
        f"[query] {query}",
        f"[base_menus] {', '.join(base_menus) if base_menus else '-'}",
        f"[modifier_ingredients] {', '.join(modifier_ingredients) if modifier_ingredients else '-'}",
        "[source_recipes] "
        + " / ".join(
            f"{src['name']}: {', '.join(src['ingredients'][:12])}"
            for src in source_recipes[:2]
        ),
    ]
    if related_ingredients:
        summary_lines.append(
            "[co_occurring_ingredients] "
            + ", ".join(
                f"{row['name']}({row['co_recipe_count']})"
                for row in related_ingredients[:10]
            )
        )
    if similar_recipes:
        summary_lines.append(
            "[similar_recipes_by_shared_ingredients] "
            + ", ".join(
                f"{row['name']}({row['shared_count']} shared)"
                for row in similar_recipes[:5]
            )
        )

    return {
        "mode": "graph_relation_context",
        "source": "neo4j",
        "status": "exact_recipe_not_found",
        "query": query,
        "base_menus": base_menus,
        "modifier_ingredients": modifier_ingredients,
        "source_recipes": source_recipes,
        "modifier_recipes": modifier_recipes[:5],
        "related_ingredients": related_ingredients,
        "similar_recipes": similar_recipes,
        "summary_lines": summary_lines,
    }


# ============================================================
# Compatibility functions used by agent.py
# ============================================================

def search_recipes(keyword, limit=5):
    """Existing import compatibility."""
    return search_recipes_by_name(keyword, limit=limit)


def get_chatbot_context(keyword):
    """Context consumed by recipe_db_expert.

    Exact/name matches return recipe rows. Missing composite recipes return
    Neo4j relation context rows, not LLM-created recipes.
    """
    direct_rows = search_recipes_by_name(keyword, limit=5)
    if direct_rows:
        return _format_recipe_rows_for_agent(direct_rows)

    context = build_graph_relation_context(keyword, limit=3)
    if not context:
        return []

    if context.get("mode") == "ingredient_recipe_relation":
        return _format_recipe_rows_for_agent(context.get("recipes", []))

    return [{
        "mode": "graph_relation_context",
        "source": "neo4j",
        "source_label": "Neo4j DB",
        "menu": keyword,
        "margin": 0,
        "ingredients": context["summary_lines"],
        "steps": None,
        "graph_context": context,
    }]


def _format_recipe_rows_for_agent(recipes):
    rows = []
    for recipe in recipes:
        detail = get_recipe_detail(recipe["id"]) or {}
        ingredients = get_recipe_ingredients(recipe["id"])
        ing_texts = [
            f"{ing.get('name')}: {ing.get('quantity')}"
            if ing.get("quantity") else ing.get("name")
            for ing in ingredients
        ]
        rows.append({
            "menu": _recipe_display_name({**recipe, **detail}),
            "recipe_name": recipe.get("name") or detail.get("name"),
            "recipe_title": recipe.get("title") or detail.get("title"),
            "margin": 0,
            "ingredients": ing_texts,
            "source": "neo4j",
            "source_label": "Neo4j DB",
            "difficulty": recipe.get("difficulty") or detail.get("difficulty"),
            "servings": recipe.get("servings") or detail.get("servings"),
            "cooking_time": recipe.get("cooking_time") or detail.get("cooking_time"),
            "cooking_method": recipe.get("cooking_method") or detail.get("cooking_method"),
            "view_count": recipe.get("view_count"),
            "steps": detail.get("steps") or recipe.get("steps"),
            "description": detail.get("description"),
            "match_type": recipe.get("match_type"),
            "matched_ingredients": recipe.get("matched_ingredients", []),
            "graph_similarity": recipe.get("graph_similarity"),
            "shared_ingredients": recipe.get("shared_ingredients", []),
        })
    return rows


def get_recipes_excluding_ingredient(keyword, exclude, limit=5):
    candidates = resolve_ingredient_candidates(exclude, limit=1)
    exclude_name = candidates[0] if candidates else exclude
    with get_session() as session:
        rows = session.run(f"""
            MATCH (r:Recipe)
            WHERE r.name CONTAINS $keyword
              AND NOT (r)-[:CONTAINS]->(:Ingredient {{name: $exclude}})
            WITH r, {_SCORE_EXPR} AS popularity_score
            RETURN r.rcp_sno AS id, r.name AS name, r.title AS title,
                   r.servings AS servings, r.difficulty AS difficulty,
                   r.cooking_time AS cooking_time, r.kind AS kind,
                   r.view_count AS view_count,
                   r.steps AS steps,
                   popularity_score,
                   'exclude_ingredient' AS match_type
            ORDER BY popularity_score DESC
            LIMIT $limit
        """, keyword=keyword, exclude=exclude_name, limit=limit)
        return [row.data() for row in rows]


def recommend_recipes(kind=None, difficulty=None, servings=None, cooking_method=None, limit=5):
    """Simple property filter kept for compatibility; Databricks can own richer catalog filters."""
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
        rows = session.run(f"""
            MATCH (r:Recipe)
            {where_clause}
            WITH r, {_SCORE_EXPR} AS popularity_score
            RETURN r.rcp_sno AS id, r.name AS name, r.title AS title,
                   r.servings AS servings, r.difficulty AS difficulty,
                   r.cooking_time AS cooking_time, r.kind AS kind,
                   r.view_count AS view_count,
                   r.steps AS steps,
                   popularity_score,
                   'property_filter' AS match_type
            ORDER BY popularity_score DESC
            LIMIT $limit
        """, **params)
        return [row.data() for row in rows]


def get_popular_recipes(limit=5):
    """Popularity ranking for selecting among multiple existing recipe variants."""
    with get_session() as session:
        rows = session.run(f"""
            MATCH (r:Recipe)
            WITH r, {_SCORE_EXPR} AS popularity_score
            RETURN r.rcp_sno AS id, r.name AS name, r.title AS title,
                   r.servings AS servings, r.difficulty AS difficulty,
                   r.cooking_time AS cooking_time, r.kind AS kind,
                   r.view_count AS view_count,
                   r.steps AS steps,
                   popularity_score,
                   'popular_recipe' AS match_type
            ORDER BY popularity_score DESC
            LIMIT $limit
        """, limit=limit)
        return [row.data() for row in rows]
