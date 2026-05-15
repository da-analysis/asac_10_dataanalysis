from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage

MAX_LOOP_COUNT = 3

# lazy init
_llm_router = None

def _get_llm_router():
    global _llm_router
    if _llm_router is None:
        _llm_router = ChatDatabricks(endpoint="databricks-gpt-5-4-mini", temperature=0.1)
    return _llm_router

def router_node(state: dict) -> dict:
    """
    현재 수집된 상태와 사용자의 질문을 바탕으로 다음 액션을 결정하는 노드입니다.
    순환(Loop) 구조를 지원하되, MAX_LOOP_COUNT 초과 시 강제로 report_generator로 이동합니다.
    """
    loop_count = state.get("loop_count", 0) + 1

    if not state.get("is_valid", False):
        return {"next_action": "report_generator", "loop_count": loop_count}

    # 순환 횟수 초과 시 강제 종료
    if loop_count > MAX_LOOP_COUNT:
        return {"next_action": "report_generator", "loop_count": loop_count}
        
    query = state['messages'][0].content
    recipe_info = state.get('recipe_info', {})
    price_info = state.get('price_info', {})
    
    has_recipe = bool(recipe_info)
    has_price = bool(price_info)
    
    sys_prompt = f"""당신은 요리 어시스턴트의 지능형 라우터(Router)입니다.
사용자의 질문과 현재까지 수집된 정보를 확인하고, 다음 단계로 어느 노드를 실행해야 할지 결정하세요.
반드시 아래 세 가지 문자열 중 하나만 출력하세요: recipe_search, price_search, report_generator

[현재 수집된 정보 상태]
- 레시피/재료 정보: {'수집됨' if has_recipe else '없음'}
- 시세/가격 정보: {'수집됨' if has_price else '없음'}

[라우팅 규칙]
1. 질문에서 레시피나 재료를 요구하는데 아직 '레시피/재료 정보'가 없다면 'recipe_search'를 출력하세요.
2. 질문에서 가격, 원가, 시세를 요구하는데 아직 '시세/가격 정보'가 없다면 'price_search'를 출력하세요.
   (단, 둘 다 요구할 경우 순서는 상관없으나 아직 안 모은 정보의 노드를 우선 출력하세요.)
3. 사용자의 질문에 답하기 위한 정보가 모두 수집되었거나, 더 이상 검색할 필요가 없다면 'report_generator'를 출력하세요.
"""
    
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"사용자 질문: {query}")
    ]
    
    response = _get_llm_router().invoke(messages)
    decision = response.content.strip()
    
    # 안전장치 (원하는 출력이 아닐 경우 보정)
    if "recipe_search" in decision:
        decision = "recipe_search"
    elif "price_search" in decision:
        decision = "price_search"
    else:
        decision = "report_generator"
        
    return {"next_action": decision, "loop_count": loop_count}

def route_edge(state: dict) -> str:
    """
    단순 조건부 엣지 함수입니다. router_node가 결정한 next_action을 반환합니다.
    """
    return state.get("next_action", "report_generator")
