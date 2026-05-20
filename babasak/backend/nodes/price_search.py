import os
import re
import pandas as pd
import mlflow
from concurrent.futures import ThreadPoolExecutor
from mlflow.entities import SpanType
from databricks.sdk import WorkspaceClient

from backend.debug_log import archive

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

# 재료명 단순화: 수식어·부위명 제거 패턴
_SIMPLIFY_PREFIX_RE = re.compile(r"^(국내산|수입산|냉동|신선|유기농|무농약|깐|손질|데친|절인|생)\s+")

# 연결된 재료명에서 추출할 알려진 prefix (Neo4j 데이터 품질 문제 대응)
_KNOWN_PREFIXES = [
    "돼지고기", "소고기", "닭고기", "오리고기", "한우",
    "양파", "대파", "쪽파", "당근", "감자", "고구마", "양배추", "배추", "무",
    "마늘", "생강", "고추", "청양고추", "버섯", "두부",
    "김치", "신김치", "깍두기", "젓갈",
    "간장", "된장", "고추장", "설탕", "소금", "후추",
    "참기름", "들기름", "식용유", "올리고당", "물엿", "매실원액",
    "통깨", "깨소금", "소주", "맛술",
]


def _simplify_ingredient(name: str) -> str | None:
    """구체적 부위·수식어 제거 후 단순화된 재료명 반환. 이미 단순하면 None.

    예: "돼지고기 앞다리살" → "돼지고기"
        "돼지고기목살앞다리살" → "돼지고기" (연결된 이름)
        "국내산 한우" → "한우"
        "대파" → None (단순화 불필요)
    """
    cleaned = _SIMPLIFY_PREFIX_RE.sub("", name.strip())
    parts = cleaned.split()
    if len(parts) > 1:
        return parts[0]
    if cleaned != name.strip():
        return cleaned
    # 공백 없이 연결된 이름: 알려진 prefix가 있으면 추출
    for prefix in _KNOWN_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            return prefix
    return None


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


def price_search_node(state: dict) -> dict:
    """Genie Space로 도매가 조회. 조회 후 누락 재료를 unavailable 필드로 반환."""
    entities = state.get("entities", {})
    recipe_info = state.get("recipe_info", {})

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
    archive("price_search.input", {
        "ingredients": ingredients,
        "num_ingredients": len(ingredients),
        "batch_size": _GENIE_BATCH_SIZE,
    })

    try:
        # ── 배치 처리: 재료를 _GENIE_BATCH_SIZE개씩 나눠 Genie 호출 (병렬) ──
        all_texts = []
        all_sqls = []
        all_tables = []
        failed_batch_items: list[str] = []  # 배치 실패 시 해당 재료들을 unavailable로 보냄
        batches = [ingredients[i:i + _GENIE_BATCH_SIZE] for i in range(0, len(ingredients), _GENIE_BATCH_SIZE)]
        archive("price_search.batch_plan", {
            "num_batches": len(batches),
            "batch_sizes": [len(b) for b in batches],
            "max_workers": _GENIE_MAX_WORKERS,
        })

        def _build_query(batch_items: list[str]) -> str:
            return (
                "다음 재료들에 대해서만 최근 도매가를 조회해줘. "
                "반드시 실제 DB에 있는 데이터만 보고해줘. DB에 없는 재료는 '없음'으로 표시해줘. 추정값 사용 금지. "
                "응답은 반드시 다음 형식을 포함해줘:\n"
                "  '조회된 재료는 X, Y, Z입니다.'\n"
                "  '나머지 재료(A, B, C)는 DB에 데이터가 없어 없음으로 분류됩니다.'\n"
                "그 다음 각 재료별 가격을 자세히 알려줘. "
                f"재료 목록: {', '.join(batch_items)}"
            )

        def _process_batch(idx_batch: tuple) -> tuple:
            """배치 1개를 Genie에 질의. (idx, result_dict, batch, err_str|None) 반환."""
            idx, batch = idx_batch
            query = _build_query(batch)
            archive("price_search.genie_query", {"batch_index": idx, "batch_items": batch, "query": query})
            try:
                with mlflow.start_span(name=f"genie_batch_{idx}_[{','.join(batch)[:60]}]", span_type=SpanType.RETRIEVER) as span:
                    span.set_inputs({"batch_index": idx, "batch_items": batch, "question": query})
                    result = _ask_genie(query)
                    span.set_outputs({
                        "text": result.get("text"),
                        "has_sql": bool(result.get("sql")),
                        "has_dataframe": result.get("dataframe") is not None,
                    })
                return (idx, result, batch, None)
            except Exception as batch_exc:
                return (idx, None, batch, str(batch_exc))

        # ── 병렬 호출 ──
        with ThreadPoolExecutor(max_workers=min(len(batches), _GENIE_MAX_WORKERS)) as ex:
            batch_results = list(ex.map(_process_batch, enumerate(batches)))

        # idx 순으로 정렬 후 응답 처리 (출력 순서 보존)
        batch_results.sort(key=lambda r: r[0])
        for idx, result, batch, err in batch_results:
            if err:
                archive("price_search.batch_failed", {
                    "batch_index": idx,
                    "batch_items": batch,
                    "error": err,
                })
                failed_batch_items.extend(batch)
                continue

            archive("price_search.genie_response", {
                "batch_index": idx,
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

        # [핵심] 누락 재료 감지 — 모든 배치 응답을 합쳐서 판정
        genie_response_text = (price_data.get("text") or "") + "\n" + (price_data.get("table") or "")
        unavailable = _detect_unavailable(ingredients, genie_response_text)

        # 실패한 배치의 재료는 무조건 unavailable로 (Naver 폴백 대상)
        if failed_batch_items:
            for item in failed_batch_items:
                if item not in unavailable:
                    unavailable.append(item)
            archive("price_search.failed_batches_to_unavailable", {
                "failed_items": failed_batch_items,
                "total_unavailable": len(unavailable),
            })

        # ── 부분 매칭 재시도: "돼지고기 앞다리살" → "돼지고기" 등으로 단순화 후 재검색 ──
        if unavailable:
            simplify_pairs = [(ing, _simplify_ingredient(ing)) for ing in unavailable]
            retry_candidates = [(orig, simp) for orig, simp in simplify_pairs if simp]
            if retry_candidates:
                simplified_names = list(dict.fromkeys(simp for _, simp in retry_candidates))
                retry_query = (
                    "다음 재료들에 대해 최근 도매가를 조회해줘. "
                    "정확한 품목이 없으면 가장 유사한 품목으로 대체 조회해줘. "
                    "추정값 사용 금지 — DB에 없으면 '없음'으로 표시. "
                    f"재료 목록: {', '.join(simplified_names)}"
                )
                archive("price_search.partial_match_attempt", {
                    "pairs": [(o, s) for o, s in retry_candidates],
                    "simplified_query": simplified_names,
                })
                try:
                    with mlflow.start_span(name=f"genie_partial_match_[{','.join(simplified_names)[:60]}]", span_type=SpanType.RETRIEVER) as span:
                        span.set_inputs({"simplified_query": simplified_names, "question": retry_query})
                        retry_result = _ask_genie(retry_query)
                        span.set_outputs({"text": retry_result.get("text")})
                    retry_text = retry_result.get("text") or ""
                    if retry_result.get("dataframe") is not None:
                        retry_text += "\n" + retry_result["dataframe"].to_string(index=False, max_rows=10)
                    still_missing = _detect_unavailable(simplified_names, retry_text)
                    found_simplified = set(simplified_names) - set(still_missing)
                    archive("price_search.partial_match_result", {
                        "simplified_queried": simplified_names,
                        "found": list(found_simplified),
                        "still_missing": still_missing,
                        "text_preview": retry_text[:300],
                    })
                    if found_simplified and retry_result.get("text"):
                        existing = price_data.get("text", "")
                        price_data["text"] = existing + f"\n\n[부분 매칭 추가 조회]\n{retry_result['text']}"
                    new_unavailable = []
                    for orig, simp in simplify_pairs:
                        if simp and simp in found_simplified:
                            pass  # 부분 매칭으로 찾음
                        else:
                            new_unavailable.append(orig)
                    unavailable = new_unavailable
                except Exception as retry_exc:
                    archive("price_search.partial_match_error", {"error": str(retry_exc)})

        if unavailable:
            price_data["unavailable"] = unavailable

        archive("price_search.output", {
            "requested": ingredients,
            "unavailable": unavailable,
            "num_batches_used": len(batches),
            "has_text": bool(price_data.get("text")),
            "has_table": bool(price_data.get("table")),
            "text_preview": (price_data.get("text") or "")[:300],
        })
        return {"price_info": price_data}

    except Exception as e:
        archive("price_search.error", {"error": str(e), "ingredients": ingredients})
        return {"price_info": {"source": "error", "data": [], "error": str(e)}, "error_log": [f"price_search: {str(e)}"]}
