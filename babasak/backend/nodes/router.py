from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage


def _get_last_human_query(messages: list) -> str:
    """메시지 리스트에서 마지막 HumanMessage의 content를 반환"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


MAX_LOOP_COUNT = 5

_llm_router = None

def _get_llm_router():
    global _llm_router
    if _llm_router is None:
        _llm_router = ChatDatabricks(endpoint="databricks-gpt-5-4-mini", temperature=0.1)
    return _llm_router


# === 원가 관련 키워드 ===
_COST_KEYWORDS = ["원가", "마진", "판매가", "비용", "수익", "1인분 가격"]
_PRICE_KEYWORDS = ["가격", "시세", "도매가", "얼마", "단가"]
_RECIPE_KEYWORDS = ["레시피", "재료", "만드는 법", "조리법", "만들기"]
# === 신규: 제외/대체재/조건/인기 키워드 ===
_EXCLUDE_KEYWORDS = ["제외", "빼고", "없는", "알레르기", "못 먹"]
_ALTERNATIVE_KEYWORDS = ["대신", "대체", "바꿀", "대용"]
_CONDITION_KEYWORDS = ["초급", "쉬운", "간단", "중급", "인분", "국/탕", "찌개", "볶음", "구이"]
_POPULAR_KEYWORDS = ["인기", "top", "best", "추천순", "랭킹"]


def _detect_intent(query: str) -> str | None:
    """쿼리에서 의도 키워드를 감지하여 힌트 반환"""
    # 제외/대체재/조건/인기 → recipe 의도
    for kw in _EXCLUDE_KEYWORDS + _ALTERNATIVE_KEYWORDS:
        if kw in query:
            return "recipe"
    for kw in _CONDITION_KEYWORDS + _POPULAR_KEYWORDS:
        if kw in query:
            return "recipe"
    for kw in _COST_KEYWORDS:
        if kw in query:
            return "cost"
    for kw in _PRICE_KEYWORDS:
        if kw in query:
            return "price"
    for kw in _RECIPE_KEYWORDS:
        if kw in query:
            return "recipe"
    return None


def _rule_based_route(state: dict, query: str) -> str | None:
    """
    규칙 기반 라우팅. 명확한 경우 LLM 호출 없이 결정.
    None 반환 시 LLM fallback.
    """
    recipe_info = state.get("recipe_info", {})
    price_info = state.get("price_info", {})
    cost_info = state.get("cost_info", {})
    entities = state.get("entities", {})

    has_recipe = bool(recipe_info)
    has_price = bool(price_info)
    has_cost = bool(cost_info)
    has_unavailable = bool(price_info.get("unavailable")) if isinstance(price_info, dict) else False

    intent = _detect_intent(query)

    # ★ 신규 규칙: 제외/대체재/조건/인기 의도는 직접 recipe_search로
    if not has_recipe:
        if entities.get("exclude") or entities.get("is_alternative"):
            return "recipe_search"
        if entities.get("conditions"):
            return "recipe_search"
        if entities.get("is_popular"):
            return "recipe_search"

    # 규칙 1: 레시피 필요한데 없으면 → recipe_search
    if not has_recipe and intent in ("recipe", "cost"):
        return "recipe_search"

    # 규칙 2: 가격만 묻는 질문 + 가격 없으면 → price_search
    if not has_price and intent == "price":
        return "price_search"

    # ★ [Fix] 규칙 3 (기존 규칙 8): 레시피만 묻는 질문 + 레시피 있음 → report_generator
    #   가격/원가 의도가 없으면 바로 답변 생성 (price_search 불필요)
    if has_recipe and intent == "recipe":
        return "report_generator"

    # 규칙 4 (기존 규칙 3): 레시피 있고 가격 없으면 → price_search
    #   (여기까지 왔다면 intent가 "cost" 이거나 None → 가격 조회 필요)
    if has_recipe and not has_price:
        return "price_search"

    # 규칙 5: 가격 있고 누락 재료 있으면 → missing_price_search
    if has_price and has_unavailable:
        return "missing_price_search"

    # 규칙 6: 레시피+가격 있고 원가 미완 + 원가 의도 → cost_calculator
    if has_recipe and has_price and not has_cost and intent == "cost":
        return "cost_calculator"

    # 규칙 7: 모든 정보 수집 완료 → report_generator
    if has_recipe and has_price and has_cost:
        return "report_generator"

    # 규칙 8: 가격만 묻는 질문 + 가격 있음 → report_generator
    if has_price and intent == "price":
        return "report_generator"

    return None  # 판단 불가 → LLM fallback


def router_node(state: dict) -> dict:
    """다음 액션을 결정하는 라우터. 규칙 기반 우선, 애매하면 LLM fallback."""
    loop_count = state.get("loop_count", 0) + 1

    if not state.get("is_valid", False):
        return {"next_action": "report_generator", "loop_count": loop_count}

    if loop_count > MAX_LOOP_COUNT:
        return {"next_action": "report_generator", "loop_count": loop_count}

    query = _get_last_human_query(state.get("messages", []))

    # === 1단계: 규칙 기반 라우팅 (LLM 호출 없음) ===
    rule_result = _rule_based_route(state, query)
    if rule_result:
        # 중복 호출 방지
        last_action = state.get("next_action", "")
        if rule_result == last_action and rule_result != "report_generator":
            return {"next_action": "report_generator", "loop_count": loop_count}
        return {"next_action": rule_result, "loop_count": loop_count}

    # === 2단계: LLM fallback (규칙으로 판단 불가한 경우만) ===
    recipe_info = state.get("recipe_info", {})
    price_info = state.get("price_info", {})
    cost_info = state.get("cost_info", {})

    has_recipe = bool(recipe_info)
    has_price = bool(price_info)
    has_cost = bool(cost_info)
    has_unavailable = bool(price_info.get("unavailable")) if isinstance(price_info, dict) else False

    sys_prompt = f"""당신은 요리 어시스턴트의 라우터입니다.
반드시 아래 다섯 가지 중 하나만 출력하세요: recipe_search, price_search, missing_price_search, cost_calculator, report_generator

[현재 상태]
- 레시피/재료 정보: {"수집됨" if has_recipe else "없음"}
- 시세/가격 정보: {"수집됨" if has_price else "없음"}
- 누락 재료 존재: {"있음" if has_unavailable else "없음/미확인"}
- 원가 계산 완료: {"완료" if has_cost else "미완료"}

[라우팅 규칙]
1. 레시피/재료가 필요한데 없으면 → recipe_search
2. 가격/시세가 필요한데 없으면 → price_search
3. 가격 조회 후 누락 재료가 있으면 → missing_price_search
4. 레시피+가격 있고 원가 계산이 안 되었으면 → cost_calculator
5. 모든 정보가 충분하면 → report_generator
"""

    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"사용자 질문: {query}")
    ]

    response = _get_llm_router().invoke(messages)
    decision = response.content.strip().lower()

    valid = ["recipe_search", "price_search", "missing_price_search", "cost_calculator", "report_generator"]
    matched = "report_generator"
    for v in valid:
        if v in decision:
            matched = v
            break

    # 중복 호출 방지
    last_action = state.get("next_action", "")
    if matched == last_action and matched != "report_generator":
        matched = "report_generator"

    return {"next_action": matched, "loop_count": loop_count}


def route_edge(state: dict) -> str:
    return state.get("next_action", "report_generator")
