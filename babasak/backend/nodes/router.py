from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage

from backend.debug_log import archive


def _get_last_human_query(messages: list) -> str:
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


_NEEDS_PRICE_INTENTS = {"cost_analysis", "price_inquiry"}
_RECIPE_ONLY_INTENTS = {"recipe_only", "recommendation", "alternative"}


def _rule_based_route(state: dict, query: str) -> str | None:
    recipe_info = state.get("recipe_info", {})
    price_info = state.get("price_info", {})
    cost_info = state.get("cost_info", {})
    entities = state.get("entities", {})

    has_recipe = bool(recipe_info.get("data") if isinstance(recipe_info, dict) else recipe_info)
    has_price = bool(price_info)
    has_cost = bool(cost_info)
    has_unavailable = bool(price_info.get("unavailable")) if isinstance(price_info, dict) else False

    intent = entities.get("intent", "general")

    #신규 규칙: 제외/대체재/조건/인기 의도는 직접 recipe_search로
    if not has_recipe:
        if entities.get("exclude") or entities.get("is_alternative"):
            return "recipe_search"
        if entities.get("conditions"):
            return "recipe_search"
        if entities.get("is_popular"):
            return "recipe_search"

    #규칙 0: fallback — preprocessor가 엔티티 추출 실패 시 바로 report
    if intent == "fallback":
        return "report_generator"

    # 규칙 1: 레시피 필요한데 없으면 → recipe_search
    if not has_recipe and intent in ("recipe_only", "cost_analysis", "recommendation", "alternative"):
        return "recipe_search"

    # 규칙 2: 가격/비교 질문 + 가격 없으면 → price_search
    if not has_price and intent == "price_inquiry":
        return "price_search"

    # 규칙 2-1: 분석형 질문 — 재료 추출 없이 원문 그대로 Genie freeform 경로
    # preprocessor가 analytics로 분류 시 menu=None, ingredients=None을 보장하므로
    # price_search_node의 freeform path(_ask_genie(user_query))가 자동으로 작동한다.
    if intent == "analytics":
        if not has_price:
            return "price_search"
        return "report_generator"

    # ★ 규칙 3 (C4 핵심 수정): 레시피 있을 때, intent에 따라 분기
    if has_recipe and not has_price:
        if intent in _NEEDS_PRICE_INTENTS:
            return "price_search"
        # recipe_only, recommendation, alternative, general → 가격 불필요, 바로 report
        return "report_generator"

    # 규칙 4: 가격 있고 누락 재료 있으면 → missing_price_search
    if has_price and has_unavailable:
        return "missing_price_search"

    # 규칙 5: 레시피+가격 있고 원가 미완 + 원가 의도 → cost_calculator
    if has_recipe and has_price and not has_cost and intent == "cost_analysis":
        return "cost_calculator"

    # 규칙 6: 모든 정보 수집 완료 → report_generator
    if has_recipe and has_price and has_cost:
        return "report_generator"

    # 규칙 7: 가격만 묻는 질문 + 가격 있음 → report_generator
    if has_price and intent == "price_inquiry":
        return "report_generator"

    # 규칙 8: 레시피만 묻는 질문 + 레시피 있음 → report_generator
    if has_recipe and intent in ("recipe_only", "recommendation", "alternative"):
        return "report_generator"

    return None


def router_node(state: dict) -> dict:
    loop_count = state.get("loop_count", 0) + 1

    archive("router.input", {
        "loop_count": loop_count,
        "is_valid": state.get("is_valid", False),
        "has_recipe_info": bool(state.get("recipe_info")),
        "has_price_info": bool(state.get("price_info")),
        "has_cost_info": bool(state.get("cost_info")),
        "last_action": state.get("next_action", ""),
        "intent": state.get("entities", {}).get("intent", "unknown"),
        "unavailable_count": len(state.get("price_info", {}).get("unavailable", []) or []) if isinstance(state.get("price_info"), dict) else 0,
    })

    if not state.get("is_valid", False):
        archive("router.output", {"next_action": "report_generator", "reason": "not_valid"})
        return {"next_action": "report_generator", "loop_count": loop_count}

    if loop_count > MAX_LOOP_COUNT:
        archive("router.output", {"next_action": "report_generator", "reason": "max_loop"})
        return {"next_action": "report_generator", "loop_count": loop_count}

    query = _get_last_human_query(state.get("messages", []))

    # 1단계: 규칙 기반 라우팅
    rule_result = _rule_based_route(state, query)
    if rule_result:
        last_action = state.get("next_action", "")
        if rule_result == last_action and rule_result != "report_generator":
            archive("router.output", {"next_action": "report_generator", "reason": "duplicate_action", "would_repeat": rule_result})
            return {"next_action": "report_generator", "loop_count": loop_count}
        archive("router.output", {"next_action": rule_result, "reason": "rule_based"})
        return {"next_action": rule_result, "loop_count": loop_count}

    # 2단계: LLM fallback
    recipe_info = state.get("recipe_info", {})
    price_info = state.get("price_info", {})
    cost_info = state.get("cost_info", {})

    has_recipe = bool(recipe_info.get("data") if isinstance(recipe_info, dict) else recipe_info)
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
    duplicate = False
    if matched == last_action and matched != "report_generator":
        matched = "report_generator"
        duplicate = True

    archive("router.output", {
        "next_action": matched,
        "reason": "llm_fallback",
        "llm_decision_raw": decision[:100],
        "duplicate_forced_to_report": duplicate,
    })
    return {"next_action": matched, "loop_count": loop_count}


def route_edge(state: dict) -> str:
    return state.get("next_action", "report_generator")
