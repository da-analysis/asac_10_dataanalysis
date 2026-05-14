import operator
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    is_valid: bool
    entities: Dict[str, Any]
    rewritten_query: str
    recipe_info: Dict[str, Any]
    price_info: Dict[str, Any]
    final_report: str
    error_log: List[str]
    next_action: str
