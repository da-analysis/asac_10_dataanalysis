import os

import mlflow
from mlflow.entities import SpanType
from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage

from backend.search_backends import naver_search
from backend.debug_log import archive

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatDatabricks(endpoint="databricks-gpt-5-4-mini", temperature=0.3)
    return _llm


def _split_naver_by_ingredient(unavailable: list[str], naver_text: str) -> tuple[list[str], list[str]]:
    """네이버 결과를 재료별로 파싱하여 찾은 것과 못 찾은 것을 분리.

    Returns: (found_lines, still_missing_ingredients)
    """
    found_lines = []
    still_missing = []
    for ing in unavailable:
        matched = False
        for line in naver_text.splitlines():
            if line.startswith(ing + ":") and "약 ₩" in line:
                found_lines.append(line)
                matched = True
                break
        if not matched:
            still_missing.append(ing)
    return found_lines, still_missing


def missing_price_search_node(state: dict) -> dict:
    """
    price_search에서 누락된 재료의 가격을 보충 검색합니다.
    1순위: 네이버 쇼핑 검색 (SEARCH_BACKEND=naver, 기본값) — title 핵심토큰 검증 포함
    2순위: LLM 기반 추정 (네이버에서 못 찾은 재료 최종 폴백)
    결과를 price_info에 병합하여 반환.
    """
    price_info = state.get("price_info", {})
    unavailable = price_info.get("unavailable", [])
    loop_count = state.get("loop_count", 0)
    # 재료별 trace_id — price_search가 만든 게 있으면 그대로, 없으면 같은 형식으로 생성.
    # 한 재료의 전체 흐름을 동일 ID로 grep할 수 있게 형식 통일.
    upstream_trace_ids = price_info.get("trace_ids", {}) if isinstance(price_info, dict) else {}
    trace_ids = {ing: upstream_trace_ids.get(ing, f"{loop_count}:{ing}") for ing in unavailable}
    archive("missing_price_search.input", {
        "unavailable": unavailable,
        "num_unavailable": len(unavailable),
        "trace_ids": trace_ids,
    })

    if not unavailable:
        archive("missing_price_search.output", {"reason": "no_unavailable_skipped"})
        return {}  # 누락 재료 없으면 패스

    # ── 1순위: 네이버 쇼핑 검색 ──
    backend = os.getenv("SEARCH_BACKEND", "naver").lower()
    archive("missing_price_search.backend_selected", {"backend": backend})
    if backend == "naver":
        try:
            items_text = ", ".join(unavailable)
            with mlflow.start_span(name=f"naver_search_[{items_text[:60]}]", span_type=SpanType.TOOL) as span:
                span.set_inputs({"ingredients": unavailable, "query": items_text})
                naver_text = naver_search(items_text)
                span.set_outputs({"result_preview": naver_text[:500]})
            archive("missing_price_search.naver_attempt", {
                "result_preview": naver_text[:400],
            })
            # 재료별 검증: 찾은 것과 못 찾은 것 분리
            found_lines, still_missing = _split_naver_by_ingredient(unavailable, naver_text)
            archive("missing_price_search.naver_validation", {
                "found": found_lines,
                "still_missing": still_missing,
            })
            updated_price_info = dict(price_info)
            if found_lines:
                found_text = "\n".join(found_lines)
                existing_text = updated_price_info.get("text", "")
                updated_price_info["text"] = (
                    existing_text + "\n\n[누락 재료 - 네이버 쇼핑 시세]\n" + found_text
                    + "\n※ 네이버 쇼핑 소매가 기준, KAMIS DB에 없는 재료에 한해 참고용으로 제공됩니다."
                )
                updated_price_info["estimation_source"] = "naver"
            # 여전히 못 찾은 재료만 다음 폴백으로 전달
            unavailable = still_missing
            updated_price_info["unavailable"] = still_missing
            if not still_missing:
                archive("missing_price_search.output", {"source": "naver", "found_lines": found_lines})
                return {"price_info": updated_price_info}
            # 일부만 찾은 경우 — price_info 업데이트 후 LLM 폴백으로 계속
            price_info = updated_price_info
        except Exception as e:
            archive("missing_price_search.naver_error", {"error": str(e)})
            state.setdefault("error_log", []).append(f"missing_price_search.naver: {str(e)}")

    # ── 2순위: LLM 기반 추정 (네이버 실패/비활성 시 최종 폴백) ──
    sys_prompt = """당신은 한국 식재료 도매 시세 전문가입니다.
아래 재료들의 대략적인 도매 가격을 추정해주세요.

규칙:
1. 각 재료별로 "재료명: 추정가격 (단위)" 형식으로 답변
2. 가격은 2024-2025년 기준 한국 도매시장/식자재마트 기준
3. 양념류는 1회 사용량 기준으로 환산 (예: 간장 1큰술 = 약 50원)
4. 정확하지 않아도 됨. 대략적 추정치로 충분
5. 마지막에 "[LLM 추정가]" 라고 표기

예시:
- 국간장: 약 80원 (1큰술 기준) [LLM 추정가]
- 미림: 약 100원 (1큰술 기준) [LLM 추정가]
- 참기름: 약 150원 (1작은술 기준) [LLM 추정가]
"""

    user_prompt = f"다음 재료들의 도매 가격을 추정해주세요:\n{chr(10).join(f'- {ing}' for ing in unavailable)}"

    try:
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ]
        with mlflow.start_span(name=f"llm_estimate_[{','.join(unavailable)[:60]}]", span_type=SpanType.LLM) as span:
            span.set_inputs({"ingredients": unavailable, "user_prompt": user_prompt})
            response = _get_llm().invoke(messages)
            estimated_text = response.content
            span.set_outputs({"estimated_preview": str(estimated_text)[:500]})

        # 기존 price_info에 추정 결과 병합
        updated_price_info = dict(price_info)
        existing_text = updated_price_info.get("text", "")
        updated_price_info["text"] = existing_text + "\n\n[누락 재료 LLM 추정]\n" + estimated_text
        updated_price_info["estimated_prices"] = estimated_text
        updated_price_info["estimation_source"] = "llm"
        # unavailable을 비워서 router가 다시 이 노드로 보내지 않게 함
        updated_price_info["unavailable"] = []

        archive("missing_price_search.output", {
            "source": "llm",
            "trace_ids": [trace_ids.get(i, i) for i in unavailable],
            "result_preview": estimated_text[:300],
        })
        return {"price_info": updated_price_info}

    except Exception as e:
        # 실패해도 unavailable을 비워서 무한루프 방지
        updated_price_info = dict(price_info)
        updated_price_info["unavailable"] = []
        updated_price_info["estimated_prices"] = f"추정 실패: {str(e)}"
        archive("missing_price_search.error", {"error": str(e)})
        return {"price_info": updated_price_info, "error_log": [f"missing_price_search: {str(e)}"]}
