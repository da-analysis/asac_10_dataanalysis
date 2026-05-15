"""
LangGraph 노드 순환 그래프 조립.

흐름:
  START → preprocessor → router ─┬→ recipe_search → router (순환)
                                  ├→ price_search  → router (순환)
                                  └→ report_generator → END
"""
from langgraph.graph import StateGraph, START, END
from backend.state import AgentState
from backend.nodes.preprocessor import preprocessor_node
from backend.nodes.router import router_node, route_edge
from backend.nodes.recipe_search import recipe_search_node
from backend.nodes.price_search import price_search_node
from backend.nodes.report_generator import report_generator_node


def build_graph():
    """노드 순환 그래프를 빌드하여 컴파일된 앱을 반환합니다."""
    workflow = StateGraph(AgentState)

    # 노드 등록
    workflow.add_node("preprocessor", preprocessor_node)
    workflow.add_node("router", router_node)
    workflow.add_node("recipe_search", recipe_search_node)
    workflow.add_node("price_search", price_search_node)
    workflow.add_node("report_generator", report_generator_node)

    # 엣지: START → preprocessor → router
    workflow.add_edge(START, "preprocessor")
    workflow.add_edge("preprocessor", "router")

    # 조건부 엣지: router → (recipe_search | price_search | report_generator)
    workflow.add_conditional_edges(
        "router",
        route_edge,
        {
            "recipe_search": "recipe_search",
            "price_search": "price_search",
            "report_generator": "report_generator",
        },
    )

    # 순환 엣지: recipe_search / price_search → router (다시 판단)
    workflow.add_edge("recipe_search", "router")
    workflow.add_edge("price_search", "router")

    # 종료: report_generator → END
    workflow.add_edge("report_generator", END)

    return workflow.compile()


# 싱글턴: 앱 전체에서 한 번만 컴파일
_graph = None


def get_graph():
    """컴파일된 그래프 싱글턴을 반환합니다."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
