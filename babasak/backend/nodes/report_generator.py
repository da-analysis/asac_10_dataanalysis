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


SYSTEM_PROMPT = (
    "당신은 소상공인을 위한 물가 연동형 메뉴 추천 AI '바바삭' 입니다. "
    "사용자 메시지에 [업종: X, 지역: Y] 형태의 정보가 있으면 해당 업종과 지역에 맞춘 답변을 제공하세요. "
    "주어진 도구(tools)를 활용하여 질문에 답하세요. "
    "도구 호출 규칙: "
    "① 레시피만 요청 시: recipe_db_expert로 먼저 조회하고, 없으면 creative_generator로 생성하세요. 가격 도구는 호출하지 마세요. "
    "② 레시피와 원가를 함께 요청 시: recipe_db_expert → (없으면) creative_generator로 레시피를 먼저 생성한 뒤, "
    "식재료 전체(주재료, 채소, 양념류 포함)를 '재료명: 수량' 형식으로 price_expert에 넘겨 원가를 조회하세요. "
    "원가는 전체 패키지 가격이 아닌 실제 사용량 기준 비례 원가로 계산해야 합니다. "
    "price_expert에서 결과가 없는 재료는 web_search_price로 검색하세요. "
    "② 식재료 가격·시세·원가 조회만 요청 시: price_expert로 먼저 조회하고, 데이터가 없으면 web_search_price로 검색하세요. "
    "web_search_price 결과를 사용할 경우 반드시 도구 결과에 포함된 출처(예: 네이버 쇼핑, CJ프레시웨이, 에이스식자재몰 등)를 그대로 가격 옆에 표기하고, '시세 DB에 없는 재료로, 웹 검색 결과를 참고한 가격입니다'라고 명시하세요. "
    "음식, 요리, 식재료와 전혀 관련 없는 것(가전제품, 자동차, 부동산 등)의 가격을 물어보면 '식재료에 대한 질문만 답변드릴 수 있습니다'라고 안내하세요. "
    "커피 원두, 밀가루, 설탕, 버터 등 식음료 재료는 모두 식재료로 간주하고 답변하세요. "
    "답변은 항상 한국어로, 핵심만 간결하게 작성하세요."


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
