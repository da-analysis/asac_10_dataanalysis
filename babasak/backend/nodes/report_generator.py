from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def _get_last_human_query(messages: list) -> str:
    """메시지 리스트에서 마지막 HumanMessage의 content를 반환"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

# lazy init
_llm_report = None

def _get_llm_report():
    global _llm_report
    if _llm_report is None:
        _llm_report = ChatDatabricks(endpoint="databricks-gpt-5-4-mini", temperature=0.7)
    return _llm_report


def _format_price_info(price_info: dict) -> str:
    """
    price_info dict를 LLM이 읽기 좋은 텍스트로 변환.
    raw dict 대신 핵심 데이터(table, text)만 추출하여 전달.
    """
    if not price_info or not isinstance(price_info, dict):
        return str(price_info)

    parts = []

    # 테이블 데이터가 가장 구체적인 정보
    if price_info.get("table"):
        parts.append(price_info["table"])

    # Genie의 텍스트 요약
    if price_info.get("text"):
        parts.append(price_info["text"])

    # 누락 재료 정보
    if price_info.get("unavailable"):
        parts.append(f"시세 DB에 없는 재료: {', '.join(price_info['unavailable'])}")

    return "\n".join(parts) if parts else str(price_info)


SYSTEM_PROMPT = (
    "당신은 소상공인을 위한 물가 연동형 메뉴 추천 AI '바바삭' 입니다. "
    "사용자 메시지에 [업종: X, 지역: Y] 형태의 정보가 있으면 해당 업종과 지역에 맞춘 답변을 제공하세요. "
    "아래 정보를 기반으로 사용자 질문에 답변하세요. "
    "답변 형식 규칙: "
    "- 제공된 데이터에 구체적인 수치(가격, 단위, 등급 등)가 있으면 반드시 그 수치를 답변에 포함하세요. 요약만 하지 말고 실제 가격을 보여주세요. "
    "- 소고기(한우) 가격 정보를 제시할 때는 반드시 등급 순서(1++ -> 1+ -> 1)로 정리하여 표시하세요. "
    "- 도매 단위에서 실사용량으로 환산한 가격은 '환산가격' 또는 '사용량 기준 환산가'로 명시하세요. "
    "- 이모지, 장식용 특수문자는 사용하지 마세요. 깔끔한 텍스트로만 답변하세요. "
    "- web_search_price 결과를 사용할 경우 반드시 출처를 가격 옆에 표기하고, "
    "'시세 DB에 없는 재료로, 웹 검색 결과를 참고한 가격입니다'라고 명시하세요. "
    "- 음식, 요리, 식재료와 전혀 관련 없는 것의 가격을 물어보면 '식재료에 대한 질문만 답변드릴 수 있습니다'라고 안내하세요. "
    "답변은 항상 한국어로 작성하세요."
)


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
    user_query = _get_last_human_query(state.get("messages", []))

    cost_info = state.get("cost_info", {})

    # 컨텍스트 조합 — LLM이 읽기 좋은 형태로 정리
    context_parts = []
    if recipe_info and recipe_info.get("data"):
        context_parts.append(f"[레시피/재료 정보]\n{recipe_info['data']}")
    if price_info:
        formatted_price = _format_price_info(price_info)
        context_parts.append(f"[가격/시세 정보]\n{formatted_price}")
    if cost_info and cost_info.get("analysis"):
        context_parts.append(f"[원가 분석]\n{cost_info['analysis']}")

    context = "\n\n".join(context_parts) if context_parts else "관련 정보를 찾지 못했습니다."

    user_prompt = f"""사용자 질문: {user_query}

{context}

위 정보를 활용하여 답변하세요. 가격 데이터가 있으면 구체적인 수치(원/kg, 등급 등)를 반드시 포함하세요."""

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
            final_answer += f"[가격 정보]\n{_format_price_info(price_info)}\n\n"
        final_answer += f"\n(자연어 요약 생성 중 오류 발생: {e})"

    return {
        "final_report": final_answer,
        "messages": [AIMessage(content=final_answer)]
    }
