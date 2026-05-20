from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from backend.debug_log import archive

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
    "\n\n"
    "[입력 데이터 활용 규칙] "
    "당신은 아래 단계로 수집된 정보를 받게 됩니다. **모든 정보를 빠짐없이 활용**하여 종합 답변을 작성하세요.\n"
    "1. [레시피/재료 정보] — Neo4j DB에서 조회한 레시피와 재료 목록\n"
    "2. [가격/시세 정보] — Genie(KAMIS DB)에서 조회한 도매가, 그리고 누락 재료를 보충 검색한 결과\n"
    "   - `text` 안에 '[누락 재료 - 네이버 쇼핑 시세]' 섹션이 있으면 그 안의 가격들도 모두 답변에 포함하세요. "
    "이때 가격 옆에 '(네이버 쇼핑)'이라고 출처를 표기하고, 'KAMIS DB에 없어 네이버 쇼핑 시세를 참고한 가격입니다'라고 명시하세요.\n"
    "   - `text` 안에 '[누락 재료 LLM 추정]' 섹션이 있으면 그 가격들도 포함하고, '(LLM 추정가)'라고 표기하세요.\n"
    "   - `estimated_prices` 필드가 있으면 같은 데이터입니다. 텍스트와 중복되지 않게 한 번만 보여주세요.\n"
    "3. [원가 분석] — 사용량 기준 비례 원가 계산 결과\n"
    "\n"
    "[답변 작성 규칙] "
    "- 정보에 없는 내용은 절대 만들어내지 마세요.\n"
    "- 가격 데이터가 KAMIS, 네이버, LLM 추정 세 출처에서 섞여 올 수 있습니다. 각 가격 옆에 출처를 명확히 표기하세요.\n"
    "- '데이터가 조회되지 않았다'고 단정하기 전에, 입력 데이터 안에 그 재료에 대한 네이버 시세나 LLM 추정가가 있는지 반드시 다시 확인하세요.\n"
    "- **레시피가 여러 개일 경우(인기 1위, 2위, 3위 등) 반드시 각각을 분리해서 표시하세요. 재료나 원가를 합치거나 평균내지 마세요.** "
    "사용자가 가장 적합한 레시피를 직접 선택할 수 있도록 인기 순위 순서대로 나열하세요.\n"
    "- 음식·요리·식재료와 무관한 것(가전제품, 자동차, 부동산 등) 질문이면 '식재료에 대한 질문만 답변드릴 수 있습니다'라고 안내하세요.\n"
    "- 커피 원두, 밀가루, 설탕, 버터 등 식음료 재료는 모두 식재료로 간주하세요.\n"
    "- 답변은 항상 한국어로, 핵심만 간결하게 작성하세요."
)


def report_generator_node(state: dict) -> dict:
    """
    수집된 정보(레시피, 가격 등)를 LLM으로 종합하여 자연스러운 최종 답변을 생성합니다.
    """
    archive("report_generator.input", {
        "is_valid": state.get("is_valid", False),
        "has_recipe": bool(state.get("recipe_info")),
        "has_price": bool(state.get("price_info")),
        "has_cost": bool(state.get("cost_info")),
        "loop_count": state.get("loop_count"),
    })

    if not state.get("is_valid", False):
        final_answer = "해당 질문은 요리 레시피 및 식재료 원가/가격 조회와 관련이 없습니다. 식당 운영 및 메뉴 원가 관련 질문을 해주세요."
        archive("report_generator.output", {"reason": "not_valid", "answer_preview": final_answer[:200]})
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

    archive("report_generator.output", {
        "answer_preview": final_answer[:400],
        "answer_length": len(final_answer),
    })
    return {
        "final_report": final_answer,
        "messages": [AIMessage(content=final_answer)]
    }
