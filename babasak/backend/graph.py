"""
LangGraph 노드 순환 그래프 조립.

흐름:
  START → preprocessor → router ─┬→ recipe_search        → router (순환)
                                  ├→ price_search         → router (순환)
                                  ├→ missing_price_search → router (순환)
                                  ├→ cost_calculator      → router (순환)
                                  └→ report_generator     → END
"""
import os
import traceback
import mlflow
import mlflow.langchain
from langgraph.graph import StateGraph, START, END
from backend.state import AgentState
from backend.nodes.preprocessor import preprocessor_node

# MLflow Tracing — LangGraph/ChatDatabricks 호출을 자동 캡처.
# Apps 환경에선 DATABRICKS_HOST/credential이 자동 주입되므로 set_experiment만 지정.
_MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "/Shared/babasak-traces")
try:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    mlflow.langchain.autolog()
    print(f"[graph] mlflow tracing enabled: experiment={_MLFLOW_EXPERIMENT}", flush=True)
except Exception as _e:
    print(f"[graph] mlflow setup FAILED: {type(_e).__name__}: {_e}", flush=True)
    print(traceback.format_exc(), flush=True)
from backend.nodes.router import router_node, route_edge
from backend.nodes.recipe_search import recipe_search_node
from backend.nodes.price_search import price_search_node
from backend.nodes.missing_price_search import missing_price_search_node
from backend.nodes.cost_calculator import cost_calculator_node
from backend.nodes.report_generator import report_generator_node


def build_graph():
    """노드 순환 그래프를 빌드하여 컴파일된 앱을 반환합니다."""
    workflow = StateGraph(AgentState)

    workflow.add_node("preprocessor", preprocessor_node)
    workflow.add_node("router", router_node)
    workflow.add_node("recipe_search", recipe_search_node)
    workflow.add_node("price_search", price_search_node)
    workflow.add_node("missing_price_search", missing_price_search_node)
    workflow.add_node("cost_calculator", cost_calculator_node)
    workflow.add_node("report_generator", report_generator_node)

    workflow.add_edge(START, "preprocessor")
    workflow.add_edge("preprocessor", "router")

    workflow.add_conditional_edges(
        "router",
        route_edge,
        {
            "recipe_search": "recipe_search",
            "price_search": "price_search",
            "missing_price_search": "missing_price_search",
            "cost_calculator": "cost_calculator",
            "report_generator": "report_generator",
        },
    )

    workflow.add_edge("recipe_search", "router")
    workflow.add_edge("price_search", "router")
    workflow.add_edge("missing_price_search", "router")
    workflow.add_edge("cost_calculator", "router")
    workflow.add_edge("report_generator", END)

    return workflow.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
