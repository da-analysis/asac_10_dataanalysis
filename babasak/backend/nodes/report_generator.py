from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# lazy init
_llm_report = None

def _get_llm_report():
    global _llm_report
    if _llm_report is None:
        _llm_report = ChatDatabricks(endpoint="databricks-gpt-5-4-mini", temperature=0.7)
    return _llm_report

SYSTEM_PROMPT = """당신은 소상공인(식당 운영자)을 위한 요리 비서 '바바삭'입니다.
아래 제공된 정보만을 활용하여 사용자의 질문에 답변하세요.

[답변 작성 규칙]
1. 제공된 정보 범위 내에서만 답변하세요. 추측이나 외부 지식을 덧붙이지 마세요.
2. 간결하고 핵심만 전달하세요.
3. 마크다운 형식(볼드, 리스트 등)을 활용해 가독성을 높이세요.
4. 답변은 한국어로 작성하세요.
"""


def report_generator_node(state: dict) -> dict:
    """
    수집된 정보(레시피, 가격 등)를 LLM으로 종합하여 자연스러운 최종 답변을 생성합니다.
    """
    if not state.get("is_valid", False):
        final_answer = "해당 질문은 요리 레시피 및 식재료 원가/가격 조회와 관련이 없습니다. 식당 운영 및 메뉴 원가 관련 질문을 해주세요."
        return {
            "final_report": final_answer,
            "messages": [AIMessage(content=final_answer)]
        }

    recipe_info = state.get("recipe_info", {})
    price_info = state.get("price_info", {})
    entities = state.get("entities", {})
    user_query = state["messages"][0].content if state.get("messages") else ""

    cost_info = state.get("cost_info", {})

    # 컨텍스트 조합
    context_parts = []
    if recipe_info and recipe_info.get("data"):
        context_parts.append(f"[레시피/재료 정보]\n{recipe_info['data']}")
    if price_info:
        context_parts.append(f"[가격/시세 정보]\n{price_info}")
    if cost_info and cost_info.get("analysis"):
        context_parts.append(f"[원가 분석]\n{cost_info['analysis']}")

    context = "\n\n".join(context_parts) if context_parts else "관련 정보를 찾지 못했습니다."

    user_prompt = f"""사용자 질문: {user_query}

{context}

위 정보만 활용하여 간결하게 답변하세요. 정보에 없는 내용은 언급하지 마세요."""

    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        response = _get_llm_report().invoke(messages)
        final_answer = response.content
    except Exception as e:
        # LLM 호출 실패 시 기본 포맷으로 폴백
        final_answer = f"요청하신 {entities} 관련 답변입니다.\n\n"
        if recipe_info and recipe_info.get("data"):
            final_answer += f"[레시피 정보]\n{recipe_info['data']}\n\n"
        if price_info:
            final_answer += f"[가격 정보]\n{price_info}\n\n"
        final_answer += f"\n(자연어 요약 생성 중 오류 발생: {e})"

    return {
        "final_report": final_answer,
        "messages": [AIMessage(content=final_answer)]
    }
