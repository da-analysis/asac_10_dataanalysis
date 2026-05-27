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
    "당신은 소상공인을 위한 물가 연동형 메뉴 추천 AI '바바삭'입니다. "
    "사용자 메시지에 [업종: X, 지역: Y] 형태가 있으면 그 맥락에 맞춰 답하세요.\n"
    "\n"
    # ─── 핵심 원칙 (정확성 1순위) ───
    "[핵심 원칙]\n"
    "1. **DB 데이터 우선.** 입력 데이터에 있는 값은 그대로 사용. 절대 변형/요약하지 마세요.\n"
    "2. **DB에 없는 건 LLM이 보완해도 됨.** 단, 그 부분 옆에 반드시 '(LLM 추정)' 또는 '(LLM 일반 지식)'을 표기하세요.\n"
    "3. **출처 명시.** 모든 값에 어디서 가져왔는지 표기 (DB / LLM 추정 / KAMIS / 네이버 등).\n"
    "4. **DB 값을 LLM 지식으로 덮어쓰지 마세요.** DB가 우선, 부족할 때만 LLM이 채움.\n"
    "\n"
    # ─── 입력 데이터 ───
    "[입력 데이터]\n"
    "1. [레시피/재료 정보] — Neo4j DB의 레시피 목록. 각 레시피는 `menu`, `ingredients`(재료+수량), `steps`(조리단계), `difficulty`, `cooking_time` 등 필드 포함.\n"
    "2. [가격/시세 정보] — Genie(KAMIS DB) 도매가 + 누락 재료 보충 데이터.\n"
    "   - '[누락 재료 - 네이버 쇼핑 시세]' 섹션 → 답변에 포함, 가격 옆에 '(네이버 쇼핑)' 표기.\n"
    "   - '[누락 재료 LLM 추정]' 섹션 또는 `estimated_prices` 필드 → '(LLM 추정가)' 표기.\n"
    "3. [원가 분석] — 사용량 기준 비례 계산 결과.\n"
    "\n"
    # ─── 답변 형식 (일관성) ───
    "[답변 형식 — 메뉴 질문일 때 이 구조로]\n"
    "## 🍲 {메뉴명} 레시피 N개\n"
    "\n"
    "### 1) {레시피명}\n"
    "- 분량 / 난이도 / 조리시간\n"
    "- **재료**\n"
    "  - {재료}: {수량}\n"
    "  - ...\n"
    "- **조리 단계**\n"
    "  1. ...\n"
    "  2. ...\n"
    "- **원가** (있을 때만): {1인분 X원} — 비싼 재료: {재료명}\n"
    "\n"
    "### 2) ... (레시피 여러 개면 1), 2), 3)으로 분리. 재료/원가 절대 합치지 마세요.)\n"
    "\n"
    # ─── 출처 표기 ───
    "[출처 표기 표준]\n"
    "- KAMIS: `1,200원/kg (INGR)`\n"
    "- 네이버: `1,500원 (네이버 쇼핑, INGR 미수록)`\n"
    "- LLM 추정: `약 2,000원 (LLM 추정가, 참고용)`\n"
    "\n"
    # ─── 조리단계 규칙 ───
    "[조리단계]\n"
    "- `steps` 필드가 있으면 그 텍스트를 1, 2, 3... **그대로** 답변에 포함. DB 데이터를 절대 변형/요약하지 마세요.\n"
    "- `steps`가 없거나 비어 있으면 LLM 일반 지식으로 조리법을 작성해도 OK. 단 제목에 '(LLM 일반 지식, DB 미수록)' 명시.\n"
    "- 절대 DB의 steps를 무시하고 자기 지식으로 덮어쓰지 마세요.\n"
    "\n"
    # ─── 의도(intent)별 톤 ───
    "[의도별 답변 톤]\n"
    "- `recipe_only` → 조리법 위주. 가격/원가 섹션 생략 또는 매우 짧게.\n"
    "- `cost_analysis` → 원가 표 위주. 조리단계는 요약만.\n"
    "- `price_only` → 재료 가격만. 레시피 본문 생략.\n"
    "- 의도 정보가 없으면 모든 정보 종합.\n"
    "\n"
    # ─── 기타 ───
    "[기타]\n"
    "- 음식·식재료와 무관한 질문(가전·자동차·부동산 등)이면 '식재료 관련만 답변드릴 수 있습니다'라고 안내.\n"
    "- 커피 원두, 밀가루, 설탕, 버터 등 식음료 재료는 식재료로 간주.\n"
    "- 한국어로, 핵심만 간결하게. 불필요한 인사말/주의사항 생략."
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
    rewritten_query = state.get("rewritten_query", "")
    intent = entities.get("intent", "")

    cost_info = state.get("cost_info", {})

    # 컨텍스트 조합 — LLM이 읽기 좋은 형태로 정리
    context_parts = []

    # ★ preprocessor 분석 결과를 맨 앞에 배치 (답변 포커스 결정)
    if rewritten_query and rewritten_query != user_query:
        context_parts.append(f"[분석된 의도]\n질문 해석: {rewritten_query}\n의도 유형: {intent}")

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

위 정보를 활용하여 답변하세요. [분석된 의도]가 있으면 그 해석에 맞춰 답변의 초점을 잡으세요. 가격 데이터가 있으면 구체적인 수치(원/kg, 등급 등)를 반드시 포함하세요."""

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

    archive("report_generator.output", {
        "answer_preview": final_answer[:400],
        "answer_length": len(final_answer),
    })
    return {
        "final_report": final_answer,
        "messages": [AIMessage(content=final_answer)]
    }
