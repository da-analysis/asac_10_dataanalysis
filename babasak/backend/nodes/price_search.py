"""
Genie Space로 KAMIS 도매가를 조회하는 노드.

흐름:
  1. 입력 재료 목록을 catalog.resolve_many()로 정규화하여 4그룹으로 분리
       - matched       : (db_name, db_unit)로 정확명 쿼리 ─ Genie/direct_sql 대상
       - recipe_direct : ingredient_recipe B2B 유통가 직접 사용 ─ Genie 건너뜀
       - passthrough   : alias 미등록('unmapped') ─ 원본 이름 그대로 자유 쿼리
       - skip          : 카탈로그에 없는 게 명백('ambiguous', 'not_in_catalog')
                         Genie를 거치지 않고 곧장 unavailable로 분류
  2. recipe_direct는 즉시 structured_prices에 투입 (Genie 호출 불필요)
  3. matched/passthrough만 배치로 묶어 Genie에 병렬 호출
  4. Genie 실패 시 direct_sql + recipe B2B 폴백 후 최종 unavailable만 missing_price_search로
"""
import os
import time
import re
import pandas as pd
import mlflow
from concurrent.futures import ThreadPoolExecutor
from mlflow.entities import SpanType
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

from backend.debug_log import archive
from backend.catalog import resolve_many, ResolveResult, get_recipe_prices_for_items

GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "01f148e5845f1f68843892ceb53abd32")

# Genie 한 번에 조회할 재료 수 제한 (작게 잡을수록 SQL 생성 안정성 ↑)
_GENIE_BATCH_SIZE = 7
# Genie 동시 호출 워커 수 (Databricks rate limit 고려)
_GENIE_MAX_WORKERS = 5
# 배치가 FAILED로 실패했을 때 항목별 단건 재시도 최대 동시 워커 수
_GENIE_RETRY_MAX_WORKERS = 2

# WorkspaceClient 싱글턴 — 매 호출마다 재생성하지 않고 재사용
_ws_client: WorkspaceClient | None = None


def _get_client() -> WorkspaceClient:
    """WorkspaceClient 싱글턴 반환. 인증 핸드셰이크를 1회만 수행."""
    global _ws_client
    if _ws_client is None:
        _ws_client = WorkspaceClient()
    return _ws_client


# negation 패턴: Genie가 "X는 조회되지 않았습니다" 식으로 말할 때 잡기 위함
_NEGATION_PATTERNS = [
    re.compile(r"조회되지\s*않"),
    re.compile(r"확인되지\s*않"),
    re.compile(r"데이터가\s*없"),
    re.compile(r"제공할\s*수\s*없"),
    re.compile(r"찾을\s*수\s*없"),
    re.compile(r"존재하지\s*않"),
    re.compile(r"기록이\s*없"),
]

# 가격 표기 패턴 — "1,200원", "₩1,200", "1.2kg당 3,000원" 등
_PRICE_PATTERN = re.compile(r"(?:₩|\d[\d,]{2,}\s*원)|(?:\d+\s*(?:kg|g|ml|L|개)\s*당?\s*\d[\d,]+)")

# --- 빠른 응답과 일관성을 위한 인메모리 캐시 (용량 제한 및 TTL 적용) ---
_batch_cache = {}
_MAX_CACHE_SIZE = 100       # 최대 100개의 배치 묶음만 기억
_CACHE_TTL_SECONDS = 3600   # 1시간(3600초)이 지나면 만료


def _get_warehouse_id(w: WorkspaceClient) -> str | None:
    """direct_sql 폴백용 warehouse 조회. databricks_db.py / catalog.py와 동일 패턴."""
    wh_id = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if wh_id:
        return wh_id
    for wh in w.warehouses.list():
        if wh.state and wh.state.value in ("RUNNING", "STARTING"):
            return wh.id
    warehouses = list(w.warehouses.list())
    return warehouses[0].id if warehouses else None


def _direct_sql_query(targets: list[tuple[str, str, str]]) -> dict:
    """카탈로그 (재료명, 단위) 조합으로 statement_execution을 직접 호출.

    Genie 우회 — LLM SQL 생성을 건너뛰고 결정적인 WHERE 절로 최근 30일 평균 도매가 조회.
    재질의도 실패한 matched 항목의 마지막 폴백으로 사용한다.

    Returns:
        {"text": "재료명: ₩가격/kg ..." 형식 문자열, "found": [회복된 input_name], "error": Optional[str]}
        결과가 없거나 실패하면 found=[]로 반환.
    """
    if not targets:
        return {"text": "", "found": [], "error": None}

    # WHERE 절 동적 생성: (재료명='X' AND 단위='Y') OR ...
    where_clauses = []
    db_to_input: dict[tuple[str, str], str] = {}
    for input_name, db_name, db_unit in targets:
        # SQL 인젝션 방지: 작은따옴표 escape
        safe_name = (db_name or "").replace("'", "''")
        safe_unit = (db_unit or "").replace("'", "''")
        where_clauses.append(f"(`재료명` = '{safe_name}' AND `단위` = '{safe_unit}')")
        db_to_input[(db_name, db_unit)] = input_name

    sql = f"""
SELECT `재료명`, `단위`, ROUND(AVG(`가격`)) AS `평균가격`, COUNT(*) AS `행수`
FROM silver.ingredient.ingredient
WHERE ({" OR ".join(where_clauses)})
  AND `날짜` >= DATE_SUB(CURRENT_DATE(), 30)
  AND `가격` IS NOT NULL
GROUP BY `재료명`, `단위`
""".strip()

    try:
        host = os.environ.get("DATABRICKS_HOST")
        token = os.environ.get("DATABRICKS_TOKEN")
        w = WorkspaceClient(host=host, token=token) if host and token else WorkspaceClient()
        warehouse_id = _get_warehouse_id(w)
        if not warehouse_id:
            return {"text": "", "found": [], "error": "warehouse_not_found"}

        resp = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=sql,
            wait_timeout="50s",
        )
        if not resp.status or resp.status.state != StatementState.SUCCEEDED:
            err_msg = resp.status.error.message if resp.status and resp.status.error else "unknown"
            return {"text": "", "found": [], "error": f"sql_failed: {err_msg}"}

        # 결과 파싱: 재료명, 단위 → 평균가격 (cost_calculator의 정규식이 잡도록 "재료명: ₩X/kg" 포맷)
        lines: list[str] = []
        found_inputs: list[str] = []
        for row in (resp.result.data_array or []):
            db_name = str(row[0] or "").strip()
            db_unit = str(row[1] or "").strip()
            try:
                avg_price = int(float(row[2])) if row[2] is not None else None
            except (TypeError, ValueError):
                avg_price = None
            if avg_price is None or avg_price <= 0:
                continue
            input_name = db_to_input.get((db_name, db_unit), db_name)
            # cost_calculator._PRICE_LINE_PATTERNS이 "재료명: ... NNN원/kg" 형식을 잡음
            unit_lower = db_unit.lower().replace(" ", "")
            kg_match = re.match(r"^(\d+(?:\.\d+)?)kg$", unit_lower)
            g_match = re.match(r"^(\d+(?:\.\d+)?)g$", unit_lower)
            if kg_match:
                kg_val = float(kg_match.group(1))
                price_per_kg = int(avg_price / kg_val) if kg_val > 0 else avg_price
                lines.append(
                    f"{input_name}: 약 ₩{price_per_kg:,}/kg "
                    f"(KAMIS direct_sql, {db_name}/{db_unit} 평균)"
                )
            elif g_match:
                g_val = float(g_match.group(1))
                price_per_kg = int(avg_price * 1000 / g_val) if g_val > 0 else avg_price
                lines.append(
                    f"{input_name}: 약 ₩{price_per_kg:,}/kg "
                    f"(KAMIS direct_sql, {db_name}/{db_unit} → kg 환산)"
                )
            else:
                lines.append(
                    f"{input_name}: 약 ₩{avg_price:,}/{db_unit} "
                    f"(KAMIS direct_sql, {db_name}/{db_unit} 평균)"
                )
            found_inputs.append(input_name)

        return {
            "text": "\n".join(lines),
            "found": found_inputs,
            "error": None,
            "sql": sql,
        }
    except Exception as e:
        return {"text": "", "found": [], "error": f"exception: {str(e)}"}


def _ask_genie(question: str, conversation_id: str = None) -> dict:
    w = _get_client()
    result = {"text": None, "sql": None, "dataframe": None, "conversation_id": None}

    if conversation_id is None:
        response = w.genie.start_conversation_and_wait(
            space_id=GENIE_SPACE_ID, content=question)
        result["conversation_id"] = response.conversation_id
    else:
        response = w.genie.create_message_and_wait(
            space_id=GENIE_SPACE_ID, conversation_id=conversation_id, content=question)
        result["conversation_id"] = conversation_id

    if response.attachments:
        text_parts = []
        for att in response.attachments:
            if att.text and att.text.content:
                text_parts.append(att.text.content)
            if att.query and att.query.query:
                result["sql"] = att.query.query
            if att.attachment_id:
                try:
                    qr = w.genie.get_message_query_result(
                        space_id=GENIE_SPACE_ID,
                        conversation_id=result["conversation_id"],
                        message_id=response.id,
                        attachment_id=att.attachment_id)
                    if qr.columns and qr.data_array:
                        result["dataframe"] = pd.DataFrame(
                            qr.data_array, columns=[c.name for c in qr.columns])
                except Exception:
                    pass
        if text_parts:
            result["text"] = "\n".join(text_parts)

    return result


# Genie 구조적 응답 파싱 패턴
_FOUND_LIST_RE = re.compile(
    r"조회(?:된|되는)?\s*재료(?:는|가|로)?\s*[(\[]?(.+?)(?:입니다|이며|이다|입\b|이고|\.)",
    re.DOTALL,
)
_NOT_FOUND_LIST_RE = re.compile(
    r"나머지\s*재료\s*[(\[]?(.+?)(?:[)\]]|는\s*DB|는\s*'없음'|는\s*\"없음\"|는\s*데이터|입니다)",
    re.DOTALL,
)


def _detect_unavailable(ingredients: list, genie_text: str) -> list:
    """요청한 재료 vs Genie 응답을 비교하여 누락된 재료 반환."""
    if not genie_text or not ingredients:
        return list(ingredients)

    explicit_found = set()
    for fm in _FOUND_LIST_RE.finditer(genie_text):
        found_text = re.sub(r"\*+", "", fm.group(1))
        for ing in ingredients:
            if ing in found_text:
                explicit_found.add(ing)

    explicit_not_found = set()
    for nm in _NOT_FOUND_LIST_RE.finditer(genie_text):
        nf_text = nm.group(1)
        for ing in ingredients:
            if ing in nf_text:
                explicit_not_found.add(ing)

    if explicit_found:
        return [ing for ing in ingredients if ing not in explicit_found]
    if explicit_not_found:
        return [ing for ing in ingredients if ing in explicit_not_found]

    unavailable = []
    for ing in ingredients:
        if ing not in genie_text:
            unavailable.append(ing)
            continue
        positive_evidence = False
        for match in re.finditer(re.escape(ing), genie_text):
            start = max(0, match.start() - 30)
            end = min(len(genie_text), match.end() + 80)
            window = genie_text[start:end]
            has_negation = any(p.search(window) for p in _NEGATION_PATTERNS)
            has_price = bool(_PRICE_PATTERN.search(window))
            if has_price and not has_negation:
                positive_evidence = True
                break
        if not positive_evidence:
            unavailable.append(ing)
    return unavailable


def _build_catalog_query(targets: list[tuple[str, str, str]]) -> str:
    """(input_name, db_name, db_unit) 튜플 목록으로 정확명 쿼리 생성."""
    lines = []
    for input_name, db_name, db_unit in targets:
        lines.append(f"- 재료명='{db_name}', 단위='{db_unit}'  (사용자 입력: '{input_name}')")
    return (
        "아래는 silver.ingredient.ingredient 테이블에 확실히 존재하는 (재료명, 단위) 조합입니다. "
        "각 항목에 대해 정확히 그 재료명/단위로 WHERE 절을 작성하여 최근 도매가를 조회해줘. "
        "다른 단위로 대체하거나 LIKE 검색하지 말고 명시된 조건만 사용해줘. "
        "반드시 WHERE 재료명 IN ('A', 'B', 'C') 형식으로 단 1개의 SQL 쿼리를 작성해서 모든 재료를 한 번에 조회해줘. "
        "재료별로 개별 쿼리를 실행하지 마. "
        "응답은 반드시 다음 형식을 포함해줘:\n"
        "  '조회된 재료는 X, Y, Z입니다.'\n"
        "  '나머지 재료(A, B, C)는 DB에 데이터가 없어 없음으로 분류됩니다.'\n"
        "그 다음 각 재료별 가격을 자세히 알려줘.\n"
        + "\n".join(lines)
    )


def _build_passthrough_query(batch_items: list[str]) -> str:
    """alias 미등록 재료를 위한 자유 텍스트 쿼리."""
    items_str = "', '".join(batch_items)
    return (
        "다음 재료들에 대해서만 최근 도매가를 조회해줘. "
        "반드시 실제 DB에 있는 데이터만 보고해줘. DB에 없는 재료는 '없음'으로 표시해줘. 추정값 사용 금지. "
        f"반드시 WHERE 재료명 IN ('{items_str}') 형식으로 단 1개의 SQL 쿼리를 작성해서 모든 재료를 한 번에 조회해줘. "
        "재료별로 개별 쿼리를 실행하지 마. "
        "응답은 반드시 다음 형식을 포함해줘:\n"
        "  '조회된 재료는 X, Y, Z입니다.'\n"
        "  '나머지 재료(A, B, C)는 DB에 데이터가 없어 없음으로 분류됩니다.'\n"
        "그 다음 각 재료별 가격을 자세히 알려줘. "
        f"재료 목록: {', '.join(batch_items)}"
    )


def price_search_node(state: dict) -> dict:
    """Genie Space로 도매가 조회. 조회 후 누락 재료를 unavailable 필드로 반환."""
    entities = state.get("entities", {})
    recipe_info = state.get("recipe_info", {})
    loop_count = state.get("loop_count", 0)

    ingredients = []
    if recipe_info and recipe_info.get("data"):
        for recipe in recipe_info["data"]:
            for ing in recipe.get("ingredients", []):
                name = ing.get("name", "") if isinstance(ing, dict) else str(ing)
                if name:
                    ingredients.append(name)
    elif entities.get("ingredient"):
        ingredients = [entities["ingredient"]] if isinstance(entities["ingredient"], str) else entities["ingredient"]
    elif entities.get("menu"):
        ingredients = [entities["menu"]] if isinstance(entities["menu"], str) else entities["menu"]

    if not ingredients:
        user_query = state["messages"][-1].content if state.get("messages") else ""
        if user_query:
            try:
                with mlflow.start_span(name="genie_freeform_query", span_type=SpanType.RETRIEVER) as span:
                    span.set_inputs({"question": user_query})
                    result = _ask_genie(user_query)
                    span.set_outputs({
                        "text": result.get("text"),
                        "has_sql": bool(result.get("sql")),
                        "has_dataframe": result.get("dataframe") is not None,
                    })
                if result["text"] or result["dataframe"] is not None:
                    price_data = {"source": "genie", "data": [user_query]}
                    if result["text"]:
                        price_data["text"] = result["text"]
                    if result["sql"]:
                        price_data["sql"] = result["sql"]
                    if result["dataframe"] is not None:
                        price_data["table"] = result["dataframe"].to_string(index=False, max_rows=15)
                    return {"price_info": price_data}
            except Exception:
                pass
        return {"price_info": {"source": "genie", "data": [], "note": "조회할 재료 없음"}}

    ingredients = list(dict.fromkeys(ingredients))

    # 재료별 trace_id — 모든 archive에 동일 형식으로 박아 grep 추적용.
    trace_ids = {ing: f"{loop_count}:{ing}" for ing in ingredients}

    # ─── 카탈로그 기반 그룹 분리 (4그룹) ────────────────────────
    # resolve_many()는 각 재료를 5가지 status로 분류한다:
    #   matched         → 정확명 쿼리 그룹 (catalog_targets)
    #   recipe_matched  → B2B 유통가 직접 사용 (recipe_direct) ★ NEW
    #   unmapped        → alias 미등록, 자유 쿼리로 시도 (passthrough)
    #   ambiguous       → 자동 매칭 금지, Genie 건너뛰고 곧장 unavailable (skip)
    #   not_in_catalog  → alias 매핑은 있는데 카탈로그엔 없음 (skip)
    resolved: list[ResolveResult] = resolve_many(ingredients)
    resolved_by_input: dict[str, ResolveResult] = {r.input_name: r for r in resolved}

    catalog_targets: list[tuple[str, str, str]] = []  # (input, db_name, db_unit)
    recipe_direct: list[str] = []  # recipe_matched → B2B 가격 직접 사용, Genie 건너뜀
    passthrough: list[str] = []
    skip_unavailable: list[str] = []
    for r in resolved:
        if r.status == "matched" and r.db_name and r.db_unit:
            catalog_targets.append((r.input_name, r.db_name, r.db_unit))
        elif r.status == "recipe_matched":
            recipe_direct.append(r.input_name)
        elif r.status == "unmapped":
            passthrough.append(r.input_name)
        else:
            # ambiguous / not_in_catalog → Genie를 거치지 않고 곧장 unavailable로
            skip_unavailable.append(r.input_name)

    archive("price_search.input", {
        "ingredients": ingredients,
        "num_ingredients": len(ingredients),
        "batch_size": _GENIE_BATCH_SIZE,
        "trace_ids": trace_ids,
        "groups": {
            "catalog_targets": [t[0] for t in catalog_targets],
            "recipe_direct": recipe_direct,
            "passthrough": passthrough,
            "skip_unavailable": skip_unavailable,
        },
        "resolved": [
            {"input": r.input_name, "status": r.status,
             "db_name": r.db_name, "db_unit": r.db_unit, "reason": r.reason}
            for r in resolved
        ],
    })

    try:
        # ─── 배치 계획 ───────────────────────────────────────
        batches: list[tuple[str, list]] = []
        for i in range(0, len(catalog_targets), _GENIE_BATCH_SIZE):
            batches.append(("catalog", catalog_targets[i:i + _GENIE_BATCH_SIZE]))
        for i in range(0, len(passthrough), _GENIE_BATCH_SIZE):
            batches.append(("passthrough", passthrough[i:i + _GENIE_BATCH_SIZE]))

        archive("price_search.batch_plan", {
            "num_batches": len(batches),
            "num_recipe_direct": len(recipe_direct),
            "num_skip": len(skip_unavailable),
            "max_workers": _GENIE_MAX_WORKERS,
            "plan": [
                {"mode": mode, "size": len(items),
                 "items": [it[0] if mode == "catalog" else it for it in items]}
                for mode, items in batches
            ],
        })

        def _process_batch(idx_batch: tuple) -> tuple:
            """배치 1개를 Genie에 질의. (idx, mode, result|None, input_names, err|None)."""
            idx, (mode, items) = idx_batch
            if mode == "catalog":
                query = _build_catalog_query(items)
                input_names = [t[0] for t in items]
            else:
                query = _build_passthrough_query(items)
                input_names = items

            cache_key = tuple(sorted(input_names))
            if cache_key in _batch_cache:
                cached_time, cached_result = _batch_cache[cache_key]
                if time.time() - cached_time < _CACHE_TTL_SECONDS:
                    return (idx, mode, cached_result, input_names, None)
                else:
                    del _batch_cache[cache_key]

            archive("price_search.genie_query", {
                "batch_index": idx,
                "mode": mode,
                "batch_items": input_names,
                "trace_ids": [trace_ids.get(n, n) for n in input_names],
                "query": query,
            })
            try:
                with mlflow.start_span(
                    name=f"genie_batch_{idx}_{mode}_[{','.join(input_names)[:60]}]",
                    span_type=SpanType.RETRIEVER,
                ) as span:
                    span.set_inputs({"batch_index": idx, "mode": mode,
                                      "batch_items": input_names, "question": query})
                    result = _ask_genie(query)
                    span.set_outputs({
                        "text": result.get("text"),
                        "has_sql": bool(result.get("sql")),
                        "has_dataframe": result.get("dataframe") is not None,
                    })

                if result.get("text") or result.get("dataframe") is not None:
                    if len(_batch_cache) >= _MAX_CACHE_SIZE:
                        oldest_key = next(iter(_batch_cache))
                        del _batch_cache[oldest_key]
                    _batch_cache[cache_key] = (time.time(), result)

                return (idx, mode, result, input_names, None)
            except Exception as batch_exc:
                return (idx, mode, None, input_names, str(batch_exc))

        # ── 병렬 호출 ──
        all_texts: list[str] = []
        all_sqls: list[str] = []
        all_tables: list[str] = []
        failed_batch_items: list[str] = []

        if batches:
            with ThreadPoolExecutor(max_workers=min(len(batches), _GENIE_MAX_WORKERS)) as ex:
                batch_results = list(ex.map(_process_batch, enumerate(batches)))
            batch_results.sort(key=lambda r: r[0])

            for idx, mode, result, batch_items, err in batch_results:
                if err:
                    archive("price_search.batch_failed", {
                        "batch_index": idx,
                        "mode": mode,
                        "batch_items": batch_items,
                        "trace_ids": [trace_ids.get(n, n) for n in batch_items],
                        "error": err,
                    })
                    failed_batch_items.extend(batch_items)
                    continue

                archive("price_search.genie_response", {
                    "batch_index": idx,
                    "mode": mode,
                    "batch_items": batch_items,
                    "trace_ids": [trace_ids.get(n, n) for n in batch_items],
                    "has_text": bool(result.get("text")),
                    "has_sql": bool(result.get("sql")),
                    "has_dataframe": result.get("dataframe") is not None,
                    "text_preview": (result.get("text") or "")[:300],
                })

                if result.get("text"):
                    all_texts.append(result["text"])
                if result.get("sql"):
                    all_sqls.append(result["sql"])
                if result.get("dataframe") is not None:
                    all_tables.append(result["dataframe"].to_string(index=False, max_rows=15))

        # ─── 실패 배치 단건 재시도 ────────────────────────────
        retry_recovered: list[str] = []
        catalog_input_map = {t[0]: t for t in catalog_targets}
        retry_targets = [i for i in failed_batch_items if i in catalog_input_map]
        skip_retry_passthrough = [i for i in failed_batch_items if i not in catalog_input_map]
        if skip_retry_passthrough:
            archive("price_search.retry_skipped_passthrough", {
                "items": skip_retry_passthrough,
                "reason": "passthrough_goes_to_naver",
            })
        if retry_targets:

            def _retry_single(item: str) -> tuple:
                """단건 재시도(catalog 전용). (item, result|None, err|None)."""
                query = _build_catalog_query([catalog_input_map[item]])
                item_mode = "catalog"
                archive("price_search.retry_single_attempt", {
                    "item": item,
                    "mode": item_mode,
                    "trace_id": trace_ids.get(item, item),
                })
                try:
                    with mlflow.start_span(
                        name=f"genie_retry_{item_mode}_[{item}]",
                        span_type=SpanType.RETRIEVER,
                    ) as span:
                        span.set_inputs({"item": item, "mode": item_mode, "question": query})
                        r = _ask_genie(query)
                        span.set_outputs({
                            "text": r.get("text"),
                            "has_sql": bool(r.get("sql")),
                            "has_dataframe": r.get("dataframe") is not None,
                        })
                    return (item, r, None)
                except Exception as retry_exc:
                    return (item, None, str(retry_exc))

            with ThreadPoolExecutor(
                max_workers=min(len(retry_targets), _GENIE_RETRY_MAX_WORKERS)
            ) as ex:
                retry_results = list(ex.map(_retry_single, retry_targets))

            still_failed_after_retry: list[str] = []
            for item, r, err in retry_results:
                if err or not r:
                    archive("price_search.retry_single_failed", {
                        "item": item,
                        "trace_id": trace_ids.get(item, item),
                        "error": err or "no_result",
                    })
                    still_failed_after_retry.append(item)
                    continue
                archive("price_search.retry_single_success", {
                    "item": item,
                    "trace_id": trace_ids.get(item, item),
                    "has_text": bool(r.get("text")),
                    "has_sql": bool(r.get("sql")),
                    "has_dataframe": r.get("dataframe") is not None,
                    "text_preview": (r.get("text") or "")[:200],
                })
                retry_recovered.append(item)
                if r.get("text"):
                    all_texts.append(r["text"])
                if r.get("sql"):
                    all_sqls.append(r["sql"])
                if r.get("dataframe") is not None:
                    all_tables.append(r["dataframe"].to_string(index=False, max_rows=15))

            failed_batch_items = still_failed_after_retry + skip_retry_passthrough
        else:
            failed_batch_items = skip_retry_passthrough

        price_data = {"source": "genie", "data": ingredients}
        if all_texts:
            price_data["text"] = "\n\n".join(all_texts)
        if all_sqls:
            price_data["sql"] = "\n---\n".join(all_sqls)
        if all_tables:
            price_data["table"] = "\n---\n".join(all_tables)

        # ─── recipe_matched 재료 B2B 가격 즉시 투입 ────────────
        # Genie를 거치지 않고 ingredient_recipe의 유통가를 structured_prices에 직접 삽입.
        # cost_calculator가 structured_prices를 1순위로 참조하므로 이 재료들은 확정 가격.
        if recipe_direct:
            structured_prices = price_data.get("structured_prices", {})
            recipe_text_lines = []
            for ing_name in recipe_direct:
                r = resolved_by_input.get(ing_name)
                if r and r.recipe_info:
                    ri_info = r.recipe_info
                    price = ri_info.get("price", 0)
                    unit = ri_info.get("unit", "")
                    product_name = ri_info.get("product_name", "")
                    unit_numeric = ri_info.get("unit_numeric")
                    unit_text = (ri_info.get("unit_text") or "").lower().strip()
                    price_per_kg = None
                    if unit_numeric and unit_numeric > 0:
                        if unit_text in ("g", "그램"):
                            price_per_kg = int(price * 1000 / unit_numeric)
                        elif unit_text in ("kg", "킬로그램"):
                            price_per_kg = int(price / unit_numeric)
                        elif unit_text in ("ml", "밀리리터"):
                            price_per_kg = int(price * 1000 / unit_numeric)
                        elif unit_text in ("l", "리터"):
                            price_per_kg = int(price / unit_numeric)
                    structured_prices[ing_name] = {
                        "price_per_kg": price_per_kg,
                        "confidence": "high",
                        "unit_hint": f"{unit} (B2B 유통가, 상품: {product_name})",
                        "note": "ingredient_recipe B2B 가격 직접 사용",
                    }
                    if price_per_kg:
                        recipe_text_lines.append(
                            f"{ing_name}: 약 ₩{price_per_kg:,}/kg (B2B 유통가, {product_name} {unit})"
                        )
                    else:
                        recipe_text_lines.append(
                            f"{ing_name}: ₩{price:,}/{unit} (B2B 유통가, {product_name})"
                        )
            if structured_prices:
                price_data["structured_prices"] = structured_prices
            if recipe_text_lines:
                existing_text = price_data.get("text", "")
                price_data["text"] = (
                    existing_text + "\n\n[ingredient_recipe B2B 유통가]\n"
                    + "\n".join(recipe_text_lines)
                )
            archive("price_search.recipe_direct_applied", {
                "items": recipe_direct,
                "num_priced": len(structured_prices),
            })

        # ─── unavailable 판정 ─────────────────────────────────
        queried_input_names = [t[0] for t in catalog_targets] + passthrough
        judge_targets: list[str] = []
        for name in queried_input_names:
            judge_targets.append(name)
            r = resolved_by_input.get(name)
            if r and r.db_name and r.db_name not in judge_targets:
                judge_targets.append(r.db_name)

        genie_response_text = (price_data.get("text") or "") + "\n" + (price_data.get("table") or "")
        missing_set = set(_detect_unavailable(judge_targets, genie_response_text))

        unavailable: list[str] = []
        for name in queried_input_names:
            r = resolved_by_input.get(name)
            in_response_by_input = name not in missing_set
            in_response_by_db = bool(r and r.db_name and r.db_name not in missing_set)
            if not (in_response_by_input or in_response_by_db):
                unavailable.append(name)

        # 실패 배치는 무조건 unavailable
        for item in failed_batch_items:
            if item not in unavailable:
                unavailable.append(item)

        # ─── 카탈로그 재질의 ──────────────────────────────────
        catalog_input_set = {t[0] for t in catalog_targets}
        recoverable = [ing for ing in unavailable if ing in catalog_input_set]
        requery_recovered: list[str] = []
        if recoverable:
            requery_targets: list[tuple[str, str, str]] = [
                (ing, resolved_by_input[ing].db_name, resolved_by_input[ing].db_unit)
                for ing in recoverable
            ]
            requery_query = _build_catalog_query(requery_targets)
            archive("price_search.catalog_requery_attempt", {
                "recoverable": recoverable,
                "trace_ids": [trace_ids.get(ing, ing) for ing in recoverable],
                "targets": [
                    {"input": ing, "db_name": db_name, "db_unit": db_unit}
                    for ing, db_name, db_unit in requery_targets
                ],
            })
            try:
                with mlflow.start_span(
                    name=f"genie_catalog_requery_[{','.join(recoverable)[:60]}]",
                    span_type=SpanType.RETRIEVER,
                ) as span:
                    span.set_inputs({"recoverable": recoverable, "question": requery_query})
                    requery_result = _ask_genie(requery_query)
                    span.set_outputs({
                        "text": requery_result.get("text"),
                        "has_sql": bool(requery_result.get("sql")),
                        "has_dataframe": requery_result.get("dataframe") is not None,
                    })

                requery_text = requery_result.get("text") or ""
                if requery_result.get("dataframe") is not None:
                    requery_text += "\n" + requery_result["dataframe"].to_string(index=False, max_rows=15)

                requery_judge_targets: list[str] = []
                for ing in recoverable:
                    requery_judge_targets.append(ing)
                    db_name = resolved_by_input[ing].db_name
                    if db_name and db_name not in requery_judge_targets:
                        requery_judge_targets.append(db_name)
                requery_missing_set = set(_detect_unavailable(requery_judge_targets, requery_text))

                found_now: list[str] = []
                for ing in recoverable:
                    r = resolved_by_input[ing]
                    in_resp_by_input = ing not in requery_missing_set
                    in_resp_by_db = bool(r.db_name and r.db_name not in requery_missing_set)
                    if in_resp_by_input or in_resp_by_db:
                        found_now.append(ing)

                archive("price_search.catalog_requery_result", {
                    "recoverable": recoverable,
                    "found_now": found_now,
                    "still_missing": [ing for ing in recoverable if ing not in found_now],
                    "text_preview": requery_text[:300],
                })

                if found_now:
                    requery_recovered = list(found_now)
                    unavailable = [ing for ing in unavailable if ing not in found_now]
                    if requery_result.get("text"):
                        existing_text = price_data.get("text", "")
                        price_data["text"] = (
                            existing_text + "\n\n[카탈로그 재질의 추가 조회]\n" + requery_result["text"]
                        )
                    if requery_result.get("sql"):
                        existing_sql = price_data.get("sql", "")
                        price_data["sql"] = (
                            existing_sql + "\n---\n" + requery_result["sql"] if existing_sql
                            else requery_result["sql"]
                        )
                    if requery_result.get("dataframe") is not None:
                        existing_table = price_data.get("table", "")
                        requery_table = requery_result["dataframe"].to_string(index=False, max_rows=15)
                        price_data["table"] = (
                            existing_table + "\n---\n" + requery_table if existing_table
                            else requery_table
                        )
            except Exception as requery_exc:
                archive("price_search.catalog_requery_error", {
                    "error": str(requery_exc),
                    "recoverable": recoverable,
                })

        # ─── direct_sql_fallback ──────────────────────────────
        direct_sql_recovered: list[str] = []
        unrecoverable_matched = [ing for ing in unavailable if ing in catalog_input_set]
        if unrecoverable_matched:
            direct_targets: list[tuple[str, str, str]] = [
                (ing, resolved_by_input[ing].db_name, resolved_by_input[ing].db_unit)
                for ing in unrecoverable_matched
            ]
            archive("price_search.direct_sql_attempt", {
                "items": unrecoverable_matched,
                "trace_ids": [trace_ids.get(ing, ing) for ing in unrecoverable_matched],
                "targets": [
                    {"input": ing, "db_name": db_name, "db_unit": db_unit}
                    for ing, db_name, db_unit in direct_targets
                ],
            })
            try:
                with mlflow.start_span(
                    name=f"direct_sql_[{','.join(unrecoverable_matched)[:60]}]",
                    span_type=SpanType.RETRIEVER,
                ) as span:
                    span.set_inputs({"items": unrecoverable_matched, "targets": direct_targets})
                    direct_result = _direct_sql_query(direct_targets)
                    span.set_outputs({
                        "found": direct_result.get("found"),
                        "error": direct_result.get("error"),
                        "text_preview": (direct_result.get("text") or "")[:200],
                    })

                direct_found = direct_result.get("found") or []
                archive("price_search.direct_sql_result", {
                    "items": unrecoverable_matched,
                    "found": direct_found,
                    "still_missing": [ing for ing in unrecoverable_matched if ing not in direct_found],
                    "error": direct_result.get("error"),
                    "text_preview": (direct_result.get("text") or "")[:300],
                })

                if direct_found:
                    direct_sql_recovered = list(direct_found)
                    unavailable = [ing for ing in unavailable if ing not in direct_found]
                    if direct_result.get("text"):
                        existing_text = price_data.get("text", "")
                        price_data["text"] = (
                            existing_text + "\n\n[KAMIS direct_sql 폴백 조회]\n"
                            + direct_result["text"]
                        )
                    if direct_result.get("sql"):
                        existing_sql = price_data.get("sql", "")
                        price_data["sql"] = (
                            existing_sql + "\n---\n" + direct_result["sql"] if existing_sql
                            else direct_result["sql"]
                        )
            except Exception as direct_exc:
                archive("price_search.direct_sql_error", {
                    "error": str(direct_exc),
                    "items": unrecoverable_matched,
                })

        # ─── recipe B2B 최종 폴백 ─────────────────────────────
        # direct_sql에서도 회복 안 된 항목 + passthrough 실패분에 대해
        # ingredient_recipe에서 B2B 유통가를 마지막으로 시도.
        # 이렇게 하면 Naver로 넘어가는 재료를 최소화.
        recipe_fallback_recovered: list[str] = []
        recipe_fallback_candidates = [
            ing for ing in unavailable if ing not in skip_unavailable
        ]
        if recipe_fallback_candidates:
            recipe_fallback_result = get_recipe_prices_for_items(recipe_fallback_candidates)
            if recipe_fallback_result:
                recipe_fallback_recovered = list(recipe_fallback_result.keys())
                unavailable = [ing for ing in unavailable if ing not in recipe_fallback_recovered]
                # structured_prices에 추가
                structured_prices = price_data.get("structured_prices", {})
                recipe_fb_lines = []
                for ing_name, info in recipe_fallback_result.items():
                    ppk = info.get("price_per_kg")
                    structured_prices[ing_name] = {
                        "price_per_kg": ppk,
                        "confidence": "medium",
                        "unit_hint": f"{info.get('unit', '')} (B2B 유통가 폴백, {info.get('product_name', '')})",
                        "note": "Genie/direct_sql 실패 후 recipe B2B 폴백",
                    }
                    if ppk:
                        recipe_fb_lines.append(
                            f"{ing_name}: 약 ₩{ppk:,}/kg (B2B 유통가 폴백, {info.get('product_name', '')} {info.get('unit', '')})"
                        )
                    else:
                        recipe_fb_lines.append(
                            f"{ing_name}: ₩{info.get('price', 0):,}/{info.get('unit', '')} (B2B 유통가 폴백)"
                        )
                if structured_prices:
                    price_data["structured_prices"] = structured_prices
                if recipe_fb_lines:
                    existing_text = price_data.get("text", "")
                    price_data["text"] = (
                        existing_text + "\n\n[recipe B2B 유통가 폴백]\n"
                        + "\n".join(recipe_fb_lines)
                    )
                archive("price_search.recipe_fallback_applied", {
                    "candidates": recipe_fallback_candidates,
                    "recovered": recipe_fallback_recovered,
                    "still_unavailable": [ing for ing in recipe_fallback_candidates if ing not in recipe_fallback_recovered],
                })

        # skip 그룹 합류 (Genie를 거치지 않은 ambiguous/not_in_catalog)
        for item in skip_unavailable:
            if item not in unavailable:
                unavailable.append(item)

        if unavailable:
            price_data["unavailable"] = unavailable
            price_data["unavailable_resolved"] = {
                ing: {
                    "status": resolved_by_input[ing].status,
                    "db_name": resolved_by_input[ing].db_name,
                    "db_unit": resolved_by_input[ing].db_unit,
                }
                for ing in unavailable if ing in resolved_by_input
            }

        # missing_price_search가 같은 trace_id를 재사용하도록 전달
        price_data["trace_ids"] = trace_ids

        archive("price_search.output", {
            "requested": ingredients,
            "unavailable": unavailable,
            "num_batches_used": len(batches),
            "num_recipe_direct": len(recipe_direct),
            "num_skipped_to_unavailable": len(skip_unavailable),
            "num_failed_batches": len(failed_batch_items),
            "num_retry_recovered": len(retry_recovered),
            "retry_recovered": retry_recovered,
            "num_requery_recovered": len(requery_recovered),
            "requery_recovered": requery_recovered,
            "num_direct_sql_recovered": len(direct_sql_recovered),
            "direct_sql_recovered": direct_sql_recovered,
            "num_recipe_fallback_recovered": len(recipe_fallback_recovered),
            "recipe_fallback_recovered": recipe_fallback_recovered,
            "has_text": bool(price_data.get("text")),
            "has_table": bool(price_data.get("table")),
            "text_preview": (price_data.get("text") or "")[:300],
            "unavailable_by_status": {
                status: [ing for ing in unavailable
                         if ing in resolved_by_input and resolved_by_input[ing].status == status]
                for status in ("matched", "recipe_matched", "ambiguous", "not_in_catalog", "unmapped")
            },
        })
        return {"price_info": price_data}

    except Exception as e:
        archive("price_search.error", {"error": str(e), "ingredients": ingredients})
        return {"price_info": {"source": "error", "data": [], "error": str(e)}, "error_log": [f"price_search: {str(e)}"]}
