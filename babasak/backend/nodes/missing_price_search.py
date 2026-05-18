from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatDatabricks(endpoint="databricks-gpt-5-4-mini", temperature=0.3)
    return _llm


def missing_price_search_node(state: dict) -> dict:
    """
    price_search에서 누락된 재료의 가격을 LLM으로 추정합니다.
    Genie DB에 없는 양념류/가공식품 등의 대략적 시세를 LLM이 추정.
    결과를 price_info에 병합하여 반환.
    """
    price_info = state.get("price_info", {})
    unavailable = price_info.get("unavailable", [])

    if not unavailable:
        return {}  # 누락 재료 없으면 패스

    # LLM에게 누락 재료 가격 추정 요청
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
        response = _get_llm().invoke(messages)
        estimated_text = response.content

        # 기존 price_info에 추정 결과 병합
        updated_price_info = dict(price_info)
        existing_text = updated_price_info.get("text", "")
        updated_price_info["text"] = existing_text + "\n\n[누락 재료 LLM 추정]\n" + estimated_text
        updated_price_info["estimated_prices"] = estimated_text
        # unavailable을 비워서 router가 다시 이 노드로 보내지 않게 함
        updated_price_info["unavailable"] = []

        return {"price_info": updated_price_info}

    except Exception as e:
        # 실패해도 unavailable을 비워서 무한루프 방지
        updated_price_info = dict(price_info)
        updated_price_info["unavailable"] = []
        updated_price_info["estimated_prices"] = f"추정 실패: {str(e)}"
        return {"price_info": updated_price_info, "error_log": [f"missing_price_search: {str(e)}"]}
