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
    "③ 레시피 그래프 DB 상세 조회 시: neo4j_graph_query를 사용하세요. "
    "이 도구는 Neo4j 그래프 데이터베이스에서 레시피-재료 관계를 조회합니다. "
    "재료 기반 검색, 난이도/인분/종류 조건 추천, 알레르기 재료 제외 검색이 가능합니다. "
    "recipe_db_expert로 결과가 부족하면 neo4j_graph_query로 추가 검색하세요. "
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


@tool
def neo4j_graph_query(query: str) -> str:
    """Neo4j 그래프 DB에서 레시피와 재료 관계를 조회합니다.

    사용 상황:
    - 레시피 검색: "김치찌개", "된장찌개" 등 요리명
    - 재료 기반 검색: "두부가 들어간 요리", "소고기 요리"
    - 조건 기반 추천: "초급 난이도", "2인분", "국/탕 종류"
    - 재료 제외: "돼지고기 빼고 김치찌개" (알레르기 대응)
    - 인기 레시피: "인기 레시피 top 5"
    - 특정 레시피 재료: "김치찌개에 뭐가 들어가?"

    query에 자연어로 요청하면 적절한 그래프 쿼리를 실행합니다.
    결과는 인기순(조회수+추천+스크랩)으로 정렬됩니다."""
    from backend.db import (
        search_recipes,
        get_recipe_ingredients,
        get_recipes_by_ingredient,
        get_recipes_excluding_ingredient,
        recommend_recipes,
        get_popular_recipes,
    )

    try:
        q = query.strip()

        # 인기 레시피
        if any(kw in q for kw in ['인기', 'top', 'best', '많이', '추천순']):
            results = get_popular_recipes(limit=5)
            return f"인기 레시피 top 5:\n{_format_recipes(results)}"

        # 재료 제외 (알레르기)
        if any(kw in q for kw in ['제외', '빼고', '없는', '알레르기', '못 먹']):
            parts = q.split()
            exclude_keywords = ['제외', '빼고', '없는', '알레르기', '못', '먹']
            keyword = ""
            exclude = ""
            for i, part in enumerate(parts):
                if any(ek in part for ek in exclude_keywords):
                    exclude = parts[i - 1] if i > 0 else ""
                    keyword = " ".join(
                        p for p in parts
                        if p != exclude and not any(ek in p for ek in exclude_keywords)
                    )
                    break
            if keyword and exclude:
                results = get_recipes_excluding_ingredient(keyword, exclude, limit=5)
                return f"'{exclude}' 제외 '{keyword}' 레시피:\n{_format_recipes(results)}"

        # 재료 기반 검색
        ingredient_keywords = ['들어간', '포함', '사용한', '넣은', '으로 만든', '로 만든']
        if any(kw in q for kw in ingredient_keywords):
            ingredient = q
            for kw in ingredient_keywords + ['요리', '레시피', '메뉴', '이', '가', ' ']:
                ingredient = ingredient.replace(kw, '')
            ingredient = ingredient.strip()
            if ingredient:
                results = get_recipes_by_ingredient(ingredient, limit=5)
                if results:
                    return f"'{ingredient}' 들어간 레시피:\n{_format_recipes(results)}"

        # 조건 기반 추천
        kind = None
        difficulty = None
        servings = None
        cooking_method = None
        condition_match = False

        kind_map = {
            '국/탕': '국/탕', '국': '국/탕', '탕': '국/탕', '찌개': '찌개류',
            '반찬': '메인반찬', '밑반찬': '밑반찬', '디저트': '디저트',
            '면': '면/만두', '볶음': '볶음', '구이': '구이',
        }
        for k, v in kind_map.items():
            if k in q:
                kind = v
                condition_match = True
                break

        if '초급' in q or '쉬운' in q:
            difficulty = '초급'
            condition_match = True
        elif '중급' in q:
            difficulty = '중급'
            condition_match = True

        for s in ['1인분', '2인분', '3인분', '4인분', '5인분', '6인분이상']:
            if s in q:
                servings = s
                condition_match = True
                break

        method_map = {'볶음': '볶기', '끓이': '끓이기', '굽': '굽기', '찜': '찌기', '튀김': '튀기기'}
        for k, v in method_map.items():
            if k in q:
                cooking_method = v
                condition_match = True
                break

        if condition_match:
            results = recommend_recipes(
                kind=kind, difficulty=difficulty,
                servings=servings, cooking_method=cooking_method, limit=5
            )
            conditions_str = ", ".join(filter(None, [kind, difficulty, servings, cooking_method]))
            return f"조건({conditions_str}) 추천 레시피:\n{_format_recipes(results)}"

        # 기본: 키워드 검색 + 1등 재료 포함
        results = search_recipes(q, limit=5)
        if results:
            top = results[0]
            ingredients = get_recipe_ingredients(top['id'])
            ing_text = ", ".join(f"{i['name']}({i['quantity']})" for i in ingredients[:15])

            output = f"🥇 1등 레시피: {top['name']}\n"
            output += f"   난이도: {top.get('difficulty', '-')}, 인분: {top.get('servings', '-')}, "
            output += f"조리시간: {top.get('cooking_time', '-')}\n"
            output += f"   재료: {ing_text}\n\n"

            if len(results) > 1:
                output += "📋 다른 레시피도 있어요:\n"
                for r in results[1:]:
                    output += f"   - {r['name']} (난이도: {r.get('difficulty', '-')}, {r.get('servings', '-')})\n"

            return output

        return "검색 결과가 없습니다."

    except Exception as e:
        return f"Neo4j 조회 실패: {str(e)}"


def _format_recipes(results: list[dict]) -> str:
    if not results:
        return "검색 결과가 없습니다."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r.get('name', '-')} "
            f"(난이도: {r.get('difficulty', '-')}, "
            f"{r.get('servings', '-')}, "
            f"조회수: {r.get('view_count', 0):,})"
        )
    return "\n".join(lines)


def _build_graph():
    _setup_mlflow()
    tools = [recipe_db_expert, neo4j_graph_query, creative_generator, price_expert, web_search_price]
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


_chatbot = None


def get_chatbot():
    global _chatbot
    if _chatbot is None:
        _chatbot = _build_graph()
    return _chatbot
