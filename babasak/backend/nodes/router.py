from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage

MAX_LOOP_COUNT = 5

_llm_router = None

def _get_llm_router():
    global _llm_router
    if _llm_router is None:
        _llm_router = ChatDatabricks(endpoint="databricks-gpt-5-4-mini", temperature=0.1)
    return _llm_router

def router_node(state: dict) -> dict:
    """다음 액션을 결정하는 라우터. MAX_LOOP_COUNT 초과 시 강제 종료."""
    loop_count = state.get("loop_count", 0) + 1

    if not state.get("is_valid", False):
        return {"next_action": "report_generator", "loop_count": loop_count}

    if loop_count > MAX_LOOP_COUNT:
        return {"next_action": "report_generator", "loop_count": loop_count}

    query = state["messages"][0].content
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
    for v in valid:
        if v in decision:
            return {"next_action": v, "loop_count": loop_count}

    return {"next_action": "report_generator", "loop_count": loop_count}

def route_edge(state: dict) -> str:
    return state.get("next_action", "report_generator")
