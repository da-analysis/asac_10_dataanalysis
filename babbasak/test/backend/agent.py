from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from backend.search_backends import search as _external_search
from databricks_langchain import ChatDatabricks
from databricks.sdk import WorkspaceClient
import mlflow
from mlflow.entities import SpanType

MODEL_NAME = "databricks-claude-sonnet-4-6"
GENIE_SPACE_ID = "01f148e5845f1f68843892ceb53abd32"
_llm = None
_mlflow_ready = False

SYSTEM_PROMPT = (
    "당신은 소상공인을 위한 물가 연동형 메뉴 추천 AI '바바삭' 입니다. "
    "사용자 메시지에 [업종: X, 지역: Y] 형태의 정보가 있으면 해당 업종과 지역에 맞춘 답변을 제공하세요. "
    "주어진 도구(tools)를 활용하여 질문에 답하세요. "
    "도구 호출 규칙: "
    "① 레시피만 요청 시: recipe_db_expert로 먼저 조회하고, 없으면 creative_generator로 생성하세요. 가격 도구는 호출하지 마세요. "
    "② 레시피와 원가를 함께 요청 시: recipe_db_expert → (없으면) creative_generator로 레시피를 먼저 생성한 뒤, "
    "식재료 전체(주재료, 채소, 양념류 포함)를 '재료명: 수량' 형식으로 price_expert에 넘겨 원가를 조회하세요. "
    "원가는 전체 패키지 가격이 아닌 실제 사용량 기준 비례 원가로 계산해야 합니다. "
    "price_expert에서 결과가 없는 재료는 web_search_price로 검색하세요. "
    "② 식재료 가격·시세·원가 조회만 요청 시: price_expert로 먼저 조회하고, 데이터가 없으면 web_search_price로 검색하세요. "
    "web_search_price 결과를 사용할 경우 반드시 도구 결과에 포함된 출처(예: 네이버 쇼핑, CJ프레시웨이, 에이스식자재몰 등)를 그대로 가격 옆에 표기하고, '시세 DB에 없는 재료로, 웹 검색 결과를 참고한 가격입니다'라고 명시하세요. "
    "음식, 요리, 식재료와 전혀 관련 없는 것(가전제품, 자동차, 부동산 등)의 가격을 물어보면 '식재료에 대한 질문만 답변드릴 수 있습니다'라고 안내하세요. "
    "커피 원두, 밀가루, 설탕, 버터 등 식음료 재료는 모두 식재료로 간주하고 답변하세요. "
    "답변은 항상 한국어로, 핵심만 간결하게 작성하세요."
)


def _setup_mlflow():
    global _mlflow_ready
    if not _mlflow_ready:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment("/Shared/LangGraph_Chatbot")
        _mlflow_ready = True


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatDatabricks(model=MODEL_NAME, temperature=0, max_tokens=2048)
    return _llm


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def recipe_db_expert(query: str) -> str:
    """사용자가 요청한 요리의 레시피가 기존 데이터베이스에 존재하는지 검색합니다."""
    from backend.db import get_chatbot_context

    _setup_mlflow()
    with mlflow.start_span(name="vector_db_search", span_type=SpanType.RETRIEVER) as span:
        span.set_inputs({"query": query})
        try:
            rows = get_chatbot_context(query)
            if rows:
                items = "\n".join(
                    f"- {r['menu']} (마진: {r['margin']}%, 재료: {', '.join(r['ingredients'])})" for r in rows
                )
                result = f"DB 검색 결과:\n{items}"
            else:
                result = "DB 검색 결과: 요청하신 레시피가 데이터베이스에 존재하지 않습니다."
        except Exception as e:
            result = f"DB 검색 실패: {e}"
        span.set_outputs({"result": result})
    return result


@tool
def creative_generator(dish_name: str) -> str:
    """DB에 레시피가 없을 때 새로운 레시피를 창작하고 필요 식재료를 추출합니다."""
    prompt = (
        f"당신은 창의적인 요리 전문가입니다. '{dish_name}'의 레시피를 작성하세요.\n"
        "가격·단가·비용은 절대 포함하지 마세요. 아래 형식으로 상세하게 답변하세요.\n\n"
        "**레시피 이름**: (요리명)\n\n"
        "**식재료** (2인분 기준):\n"
        "- 재료명: 수량 (단위 명시, 예: 돼지고기 목살 300g, 마라소스 3큰술)\n"
        "(주재료, 채소, 양념류 모두 빠짐없이 나열)\n\n"
        "**조리 단계** (각 단계를 2~3문장으로 상세히 설명):\n"
        "1. 재료 손질: ...\n"
        "2. 밑간/양념장 준비: ...\n"
        "3. 조리: ...\n"
        "4. 마무리: ...\n\n"
        "**요리 팁**: 맛을 높이는 1~2가지 핵심 팁"
    )
    tool_llm = ChatDatabricks(model=MODEL_NAME, temperature=0.7)
    response = tool_llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def price_expert(ingredients_text: str) -> str:
    """식재료의 도매 시세 가격을 Databricks Genie로 조회합니다."""
    question = (
        f"다음 식재료들의 오늘자 도매 시세를 구해줘. "
        f"각 재료는 레시피에 표기된 수량 기준으로 실제 사용량만큼의 가격을 계산해줘 "
        f"(예: 마라소스 50g이면 kg당 단가 × 0.05). 전체 패키지 가격이 아닌 사용량 비례 원가로 합산해줘:\n{ingredients_text}"
    )
    _setup_mlflow()
    with mlflow.start_span(name="genie_price_calc", span_type=SpanType.TOOL) as span:
        span.set_inputs({"question": question})
        try:
            w = WorkspaceClient()
            response = w.genie.start_conversation_and_wait(space_id=GENIE_SPACE_ID, content=question)
            # TextAttachment에 실제 가격 상세 내용이 담겨있음
            text_parts = []
            if response.attachments:
                for att in response.attachments:
                    if att.text and att.text.content:
                        text_parts.append(att.text.content)
            result = "\n".join(text_parts) if text_parts else (response.content or "조회 결과 없음")
        except Exception as e:
            result = f"원가 조회 실패: {e}"
        span.set_outputs({"result": result})
    return result


@tool
def web_search_price(query: str) -> str:
    """price_expert에서 데이터를 찾지 못했을 때 외부에서 식재료 시세를 검색합니다."""
    return _external_search(query)


def _build_graph():
    _setup_mlflow()
    tools = [recipe_db_expert, creative_generator, price_expert, web_search_price]
    llm_with_tools = _get_llm().bind_tools(tools)

    def agent_node(state: AgentState):
        safe_msgs = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in state["messages"]:
            if not isinstance(msg, SystemMessage):
                safe_msgs.append(msg)
        response = llm_with_tools.invoke(safe_msgs)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    return workflow.compile()


# 챗봇 첫 호출 시 한 번만 컴파일
_chatbot = None


def get_chatbot():
    global _chatbot
    if _chatbot is None:
        _chatbot = _build_graph()
    return _chatbot
