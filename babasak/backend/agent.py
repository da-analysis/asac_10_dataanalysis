from langgraph.graph import StateGraph, END
from backend.state import AgentState
from backend.nodes.preprocessor import preprocessor_node
from backend.nodes.router import router_node, route_edge
from backend.nodes.recipe_search import recipe_search_node
from backend.nodes.price_search import price_search_node
from backend.nodes.report_generator import report_generator_node

# 그래프 초기화
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("preprocessor", preprocessor_node)
workflow.add_node("router", router_node)
workflow.add_node("recipe_search", recipe_search_node)
workflow.add_node("price_search", price_search_node)
workflow.add_node("report_generator", report_generator_node)

# 엣지 연결
workflow.set_entry_point("preprocessor")

# preprocessor -> router
workflow.add_edge("preprocessor", "router")

# 라우터 조건부 분기 (순환 구조의 핵심)
workflow.add_conditional_edges(
    "router",
    route_edge,
    {
        "recipe_search": "recipe_search",
        "price_search": "price_search",
        "report_generator": "report_generator",
    }
)

# 서브 노드 작업 완료 후 다시 라우터로 돌아와서 추가 정보가 필요한지 판단 (순환)
workflow.add_edge("recipe_search", "router")
workflow.add_edge("price_search", "router")

# 종료점 연결
workflow.add_edge("report_generator", END)

# 컴파일
culinary_app = workflow.compile()


def get_chatbot():
    """FastAPI routers/chatbot.py에서 호출하는 진입점"""
    return culinary_app
