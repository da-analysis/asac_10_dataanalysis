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
import re
import pandas as pd
import mlflow
from concurrent.futures import ThreadPoolExecutor
from mlflow.entities import SpanType
from databricks.sdk import WorkspaceClient

from backend.debug_log import archive
from backend.catalog import resolve_many, ResolveResult

GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "01f148e5845f1f68843892ceb53abd32")

# Genie 한 번에 조회할 재료 수 제한 (작게 잡을수록 SQL 생성 안정성 ↑)
_GENIE_BATCH_SIZE = 5
# Genie 동시 호출 워커 수 (Databricks rate limit 고려)
_GENIE_MAX_WORKERS = 3

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


def _ask_genie(question: str, conversation_id: str = None) -> dict:
    w = WorkspaceClient()
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
        "응답은 반드시 다음 형식을 포함해줘:\n"
        "  '조회된 재료는 X, Y, Z입니다.'\n"
        "  '나머지 재료(A, B, C)는 DB에 데이터가 없어 없음으로 분류됩니다.'\n"
        "그 다음 각 재료별 가격을 자세히 알려줘.\n"
        + "\n".join(lines)
    )


def _build_passthrough_query(batch_items: list[str]) -> str:
    """alias 미등록 재료를 위한 자유 텍스트 쿼리 (기존 방식)."""
    return (
        "다음 재료들에 대해서만 최근 도매가를 조회해줘. "
        "반드시 실제 DB에 있는 데이터만 보고해줘. DB에 없는 재료는 '없음'으로 표시해줘. 추정값 사용 금지. "
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

        def _process_batch(idx_batch: tuple) -> tuple:
            """배치 1개를 Genie에 질의. (idx, mode, result|None, batch, err|None)."""
            idx, (mode, items) = idx_batch
            if mode == "catalog":
                query = _build_catalog_query(items)
                input_names = [t[0] for t in items]
            else:
                query = _build_passthrough_query(items)
                input_names = items
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
