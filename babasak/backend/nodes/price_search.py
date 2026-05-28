"""
Genie Space로 KAMIS 도매가를 조회하는 노드.

흐름:
  1. 입력 재료 목록을 catalog.resolve_many()로 정규화하여 3그룹으로 분리
       - matched     : (db_name, db_unit)로 정확명 쿼리 ─ 1차에서 명시 쿼리로 던짐
       - passthrough : alias 미등록('unmapped') ─ 원본 이름 그대로 자유 쿼리
       - skip        : 카탈로그에 없는 게 명백('ambiguous', 'not_in_catalog')
                       Genie를 거치지 않고 곧장 unavailable로 분류
  2. matched/passthrough만 배치로 묶어 Genie에 병렬 호출
  3. Genie unavailable 판정 + 실패 배치를 합쳐 missing_price_search로 넘김
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
from backend.catalog import resolve_many, ResolveResult

GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "01f148e5845f1f68843892ceb53abd32")

# Genie 한 번에 조회할 재료 수 제한 (작게 잡을수록 SQL 생성 안정성 ↑)
_GENIE_BATCH_SIZE = 7
# Genie 동시 호출 워커 수 (Databricks rate limit 고려)
_GENIE_MAX_WORKERS = 5

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

# WorkspaceClient 재사용을 위한 전역 변수 (커넥션 풀링 효과)
_workspace_client = None

def _get_workspace_client():
    global _workspace_client
    if _workspace_client is None:
        _workspace_client = WorkspaceClient()
    return _workspace_client


# --- 빠른 응답과 일관성을 위한 인메모리 캐시 (용량 제한 및 TTL 적용) ---
_batch_cache = {}
_MAX_CACHE_SIZE = 100       # 최대 100개의 배치 묶음만 기억
_CACHE_TTL_SECONDS = 3600   # 1시간(3600초)이 지나면 만료

def _simplify_ingredient(name: str) -> str | None:
    """구체적 부위·수식어 제거 후 단순화된 재료명 반환. 이미 단순하면 None.

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
            # 단위가 kg/g 외(개/마리/포기 등)면 그대로 ₩표기로만 보고하되 cost_calculator의
            # _PER_PIECE_GRAMS 매칭에 의존하지 않는 흐름은 ₩/kg가 가장 안전. 단위에 'kg'가
            # 포함되면 그 단위로 환산하고, 아니면 raw 단위로 표기.
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
                # 개/마리/포기 등 — cost_calculator가 _PER_PIECE_GRAMS로 환산 시도하므로
                # 단가는 그대로 표기하되 단위 정보를 같이 노출
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
    """요청한 재료 vs Genie 응답을 비교하여 누락된 재료 반환.

    전략:
    1) Genie가 "조회된 재료는 X, Y, Z" 식으로 명시한 경우 → 그 외는 모두 unavailable
    2) "나머지 재료(A, B, C)는 ... 없" 패턴이 있으면 그것도 활용
    3) 명시적 패턴 없으면 기존 윈도우 기반 polling으로 폴백
    """
    if not genie_text or not ingredients:
        return list(ingredients)

    # ── Step 1: "조회된 재료는 X, Y, Z" 명시적 추출 ──
    explicit_found = set()
    for fm in _FOUND_LIST_RE.finditer(genie_text):
        found_text = re.sub(r"\*+", "", fm.group(1))  # markdown ** 제거
        for ing in ingredients:
            if ing in found_text:
                explicit_found.add(ing)

    # ── Step 2: "나머지 재료(A, B, C)는 ... 없" 명시적 추출 ──
    explicit_not_found = set()
    for nm in _NOT_FOUND_LIST_RE.finditer(genie_text):
        nf_text = nm.group(1)
        for ing in ingredients:
            if ing in nf_text:
                explicit_not_found.add(ing)

    # ── Step 3: 명시적 found 리스트가 있으면 그게 진실 ──
    if explicit_found:
        return [ing for ing in ingredients if ing not in explicit_found]

    # ── Step 4: 명시적 not_found 리스트만 있으면 그것만 unavailable ──
    if explicit_not_found:
        return [ing for ing in ingredients if ing in explicit_not_found]

    # ── Step 5: 폴백 — 기존 윈도우 기반 polling ──
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
    """(input_name, db_name, db_unit) 튜플 목록으로 정확명 쿼리 생성.

    Genie가 LIKE 검색이나 다른 단위 대체 없이 정확히 명시된 (재료명, 단위)로만
    WHERE 절을 만들도록 유도한다.
    """
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
    # missing_price_search 노드도 같은 형식을 사용한다.
    trace_ids = {ing: f"{loop_count}:{ing}" for ing in ingredients}

    # ─── 카탈로그 기반 그룹 분리 ────────────────────────────────
    # resolve_many()는 각 재료를 4가지 status로 분류한다:
    #   matched         → 정확명 쿼리 그룹 (catalog_targets)
    #   unmapped        → alias 미등록, 자유 쿼리로 시도 (passthrough)
    #   ambiguous       → 자동 매칭 금지, Genie 건너뛰고 곧장 unavailable (skip)
    #   not_in_catalog  → alias 매핑은 있는데 카탈로그엔 없음 (skip)
    resolved: list[ResolveResult] = resolve_many(ingredients)
    resolved_by_input: dict[str, ResolveResult] = {r.input_name: r for r in resolved}

    catalog_targets: list[tuple[str, str, str]] = []  # (input, db_name, db_unit)
    passthrough: list[str] = []
    skip_unavailable: list[str] = []
    for r in resolved:
        if r.status == "matched" and r.db_name and r.db_unit:
            catalog_targets.append((r.input_name, r.db_name, r.db_unit))
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
        # 정확명 그룹과 자유 그룹은 쿼리 형식이 달라 별도 배치로 분리.
        # 그룹 내에서만 _GENIE_BATCH_SIZE 단위로 묶는다.
        # batches는 (mode, items) 형태로 통일. mode는 "catalog" | "passthrough".
        batches: list[tuple[str, list]] = []
        for i in range(0, len(catalog_targets), _GENIE_BATCH_SIZE):
            batches.append(("catalog", catalog_targets[i:i + _GENIE_BATCH_SIZE]))
        for i in range(0, len(passthrough), _GENIE_BATCH_SIZE):
            batches.append(("passthrough", passthrough[i:i + _GENIE_BATCH_SIZE]))

        archive("price_search.batch_plan", {
            "num_batches": len(batches),
            "num_skip": len(skip_unavailable),
            "max_workers": _GENIE_MAX_WORKERS,
            "plan": [
                {"mode": mode, "size": len(items),
                 "items": [it[0] if mode == "catalog" else it for it in items]}
                for mode, items in batches
            ],
        })

        def _build_query(batch_items: list[str]) -> str:
            return (
                "다음 재료들의 최근 도매가를 정확히 조회해줘. "
                "규칙 1: 실제 DB에 있는 데이터만 사용하고 추정값은 절대 금지. "
                "규칙 2: 응답의 시작 부분에 반드시 아래와 똑같은 형식의 문장을 포함해줘:\n"
                "조회된 재료는 X, Y, Z입니다.\n"
                "나머지 재료(A, B, C)는 데이터가 없습니다.\n"
                "규칙 3: 그 다음 각 재료별 가격 정보를 알려줘. "
                f"재료 목록: {', '.join(batch_items)}"
            )

        def _process_batch(idx_batch: tuple) -> tuple:
            """배치 1개를 Genie에 질의. (idx, result_dict, batch, err_str|None) 반환."""
            idx, batch = idx_batch
            
            # --- 1. 캐시 확인 (반복 질문 시 즉시 반환) ---
            cache_key = tuple(sorted(batch))
            if cache_key in _batch_cache:
                cached_time, cached_result = _batch_cache[cache_key]
                if time.time() - cached_time < _CACHE_TTL_SECONDS:
                    return (idx, cached_result, batch, None)
                else:
                    del _batch_cache[cache_key]  # 시간이 지나 만료된 캐시 삭제

            query = _build_query(batch)
            archive("price_search.genie_query", {"batch_index": idx, "batch_items": batch, "query": query})
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
                
                # --- 2. 검색 성공 시 캐시에 결과 저장 ---
                if result.get("text") or result.get("dataframe") is not None:
                    if len(_batch_cache) >= _MAX_CACHE_SIZE:
                        # 용량 초과 시 가장 오래된 항목(FIFO) 삭제
                        oldest_key = next(iter(_batch_cache))
                        del _batch_cache[oldest_key]
                    _batch_cache[cache_key] = (time.time(), result)

                return (idx, result, batch, None)
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
        # 배치 단위 호출이 MessageStatus.FAILED 등으로 통째로 실패한 경우,
        # 그 안의 항목을 1개씩 분리하여 다시 호출. 1건 처리 시 SQL 생성 자유도가
        # 줄어 성공률이 올라가는 효과를 기대.
        # mode는 catalog_targets에 포함된 입력명이면 'catalog', 아니면 'passthrough'.
        retry_recovered: list[str] = []
        if failed_batch_items:
            catalog_input_map = {t[0]: t for t in catalog_targets}

            def _retry_single(item: str) -> tuple:
                """단건 재시도. (item, result|None, err|None)."""
                if item in catalog_input_map:
                    query = _build_catalog_query([catalog_input_map[item]])
                    item_mode = "catalog"
                else:
                    query = _build_passthrough_query([item])
                    item_mode = "passthrough"
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
                max_workers=min(len(failed_batch_items), _GENIE_RETRY_MAX_WORKERS)
            ) as ex:
                retry_results = list(ex.map(_retry_single, failed_batch_items))

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

            # 단건 재시도로 성공한 항목은 실패 목록에서 제외 → 후속 단계에 영향 없도록
            failed_batch_items = still_failed_after_retry

        price_data = {"source": "genie", "data": ingredients}
        if all_texts:
            price_data["text"] = "\n\n".join(all_texts)
        if all_sqls:
            price_data["sql"] = "\n---\n".join(all_sqls)
        if all_tables:
            price_data["table"] = "\n---\n".join(all_tables)

        # ─── unavailable 판정 ─────────────────────────────────
        # 1) skip 그룹은 처음부터 unavailable
        # 2) Genie를 거친 재료 중 응답에 없는 것도 unavailable
        # 3) 배치 실패분도 unavailable
        # 판정은 input_name + db_name 둘 다로 — Genie가 어느 표기로 응답해도 잡힘.
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
        # 'matched' 상태였는데 Genie 1차에서 unavailable로 분류된 재료를 한 번 더 질의.
        # 카탈로그에 (db_name, db_unit)이 확실히 존재하므로 Genie의 LLM 비결정성으로
        # SQL을 잘못 생성했을 가능성이 큼. 명시적 WHERE 조건을 자연어로 강하게 유도하여
        # 재시도하고, 성공한 항목은 unavailable에서 제거.
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

                # 판정은 input_name + db_name 둘 다로 — Genie가 어느 표기로 응답해도 잡힘
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

                # 재질의로 회복된 재료는 unavailable에서 제거하고, Genie 응답 텍스트는 누적
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
        # matched 였는데 Genie 1차 + 카탈로그 재질의에도 회복 안 된 항목을 마지막으로 시도.
        # statement_execution으로 (재료명, 단위) WHERE 절을 직접 작성해 평균 가격 조회.
        # Genie의 LLM SQL 생성을 완전히 우회하므로 결정적이며, 카탈로그에 데이터가 실제로
        # 존재한다면 거의 항상 회복 가능.
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

        # skip 그룹 합류 (Genie를 거치지 않은 ambiguous/not_in_catalog)
        for item in skip_unavailable:
            if item not in unavailable:
                unavailable.append(item)

        if unavailable:
            price_data["unavailable"] = unavailable
            # missing_price_search 검증에 활용할 resolve 정보 첨부
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
            "num_skipped_to_unavailable": len(skip_unavailable),
            "num_failed_batches": len(failed_batch_items),
            "num_retry_recovered": len(retry_recovered),
            "retry_recovered": retry_recovered,
            "num_requery_recovered": len(requery_recovered),
            "requery_recovered": requery_recovered,
            "num_direct_sql_recovered": len(direct_sql_recovered),
            "direct_sql_recovered": direct_sql_recovered,
            "has_text": bool(price_data.get("text")),
            "has_table": bool(price_data.get("table")),
            "text_preview": (price_data.get("text") or "")[:300],
            "unavailable_by_status": {
                status: [ing for ing in unavailable
                         if ing in resolved_by_input and resolved_by_input[ing].status == status]
                for status in ("matched", "ambiguous", "not_in_catalog", "unmapped")
            },
        })
        return {"price_info": price_data}

    except Exception as e:
        archive("price_search.error", {"error": str(e), "ingredients": ingredients})
        return {"price_info": {"source": "error", "data": [], "error": str(e)}, "error_log": [f"price_search: {str(e)}"]}
