# Databricks notebook source
# MAGIC %md
# MAGIC # Neo4j 메뉴 추천 에이전트
# MAGIC HTTP API 기반 LangGraph 에이전트 (FastMCP 미사용)

# COMMAND ----------

# MAGIC %pip install langchain-databricks langgraph langchain-core --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Neo4j HTTP API 설정

# COMMAND ----------

import requests
import base64

NEO4J_HOST = "e76ed54b.databases.neo4j.io"
NEO4J_DATABASE = "e76ed54b"
NEO4J_USERNAME = "e76ed54b"
NEO4J_PASSWORD = dbutils.secrets.get(scope="neo4j-scope", key="neo4j_password")
NEO4J_AUTH = base64.b64encode(f"{NEO4J_USERNAME}:{NEO4J_PASSWORD}".encode()).decode()

# 연결 테스트
r = requests.post(
    f"https://{NEO4J_HOST}/db/{NEO4J_DATABASE}/query/v2",
    json={"statement": "MATCH (n) RETURN count(n) AS cnt"},
    headers={"Authorization": f"Basic {NEO4J_AUTH}", "Content-Type": "application/json", "Accept": "application/json"},
    timeout=30
)
print(f"Status: {r.status_code}")
print(f"Response: {r.text[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 도구 + LLM + 에이전트 설정

# COMMAND ----------

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_databricks import ChatDatabricks
from langgraph.graph import StateGraph, MessagesState, START, END

# --- Neo4j 도구 (HTTP API) ---
@tool
def query_neo4j(cypher_query: str) -> str:
    """Neo4j 그래프 데이터베이스에 Cypher 쿼리를 실행합니다.
    메뉴, 재료, 레시피 관계를 조회할 때 사용합니다."""
    try:
        r = requests.post(
            f"https://{NEO4J_HOST}/db/{NEO4J_DATABASE}/query/v2",
            json={"statement": cypher_query},
            headers={
                "Authorization": f"Basic {NEO4J_AUTH}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            timeout=30
        )
        return r.text
    except Exception as e:
        return f"에러: {str(e)}"

# --- LLM ---
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", temperature=0)

tools = [query_neo4j]
tool_map = {"query_neo4j": query_neo4j}
llm_with_tools = llm.bind_tools(tools)

# --- 시스템 프롬프트 ---
MENU_SYSTEM_PROMPT = """You are a Korean menu recommendation assistant.
You MUST answer ONLY based on data from the Neo4j database.
NEVER use your own knowledge. If no results, say "데이터에서 찾을 수 없습니다".

ALWAYS use query_neo4j tool to execute Cypher queries.

Node types: Recipe (rcp_sno, name, cooking_method, situation, kind, servings, difficulty, cooking_time, view_count, recommend_count, scrap_count), Ingredient (name, lv1, lv2)
Relationship: (Recipe)-[CONTAINS {quantity}]->(Ingredient)

IMPORTANT: 같은 이름의 레시피가 여러 개 있을 수 있음. 반드시 아래 2단계 패턴을 사용할 것.

Step 1 - Top 5 레시피 조회 (점수순):
MATCH (r:Recipe) WHERE r.name CONTAINS '김치찌개'
WITH r, (r.view_count + r.recommend_count * 100 + r.scrap_count * 50) AS score
RETURN r.rcp_sno, r.name, r.servings, r.difficulty, r.cooking_time, score
ORDER BY score DESC LIMIT 5

Step 2 - 1등 레시피의 재료만 조회 (rcp_sno 사용):
MATCH (r:Recipe {rcp_sno: 여기에1등번호})-[c:CONTAINS]->(i:Ingredient)
RETURN i.name, c.quantity

응답 형식:
1. Step 1으로 top 5 조회
2. Step 2로 1등 재료 조회
3. 1등 레시피 상세 소개 (재료 + 수량)
4. "다른 레시피도 있어요!" 로 2~5등 간단 소개 (이름, 인분, 난이도만)

Other Example Cypher:
- 특정 재료 포함: MATCH (r:Recipe)-[:CONTAINS]->(i:Ingredient {name: '두부'}) RETURN DISTINCT r.name, r.view_count ORDER BY r.view_count DESC LIMIT 5
- 인기순: MATCH (r:Recipe) RETURN r.name, r.view_count ORDER BY r.view_count DESC LIMIT 5
- DB 통계: MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt ORDER BY cnt DESC

Answer in Korean."""

# --- LangGraph 에이전트 ---
def call_llm(state: MessagesState):
    messages = [SystemMessage(content=MENU_SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}

def call_tools(state: MessagesState):
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        print(f"  🔧 [HTTP API] {tc['name']}({str(tc['args'])[:80]})")
        result = tool_map[tc["name"]].invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}

def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

graph = StateGraph(MessagesState)
graph.add_node("agent", call_llm)
graph.add_node("tools", call_tools)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
menu_agent = graph.compile()

print("메뉴 에이전트 준비 완료!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 테스트

# COMMAND ----------

def ask_menu(question):
    print(f"\n{'='*60}")
    print(f"❓ {question}")
    print(f"{'='*60}")
    result = menu_agent.invoke({"messages": [HumanMessage(content=question)]})
    print(f"\n💡 {result['messages'][-1].content}")

ask_menu("김치찌개 레시피 알려줘")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 추가 테스트 (필요시 주석 해제)

# COMMAND ----------

#ask_menu("두부가 들어가는 레시피 5개 알려줘")
#ask_menu("초급 난이도 레시피 추천해줘")
ask_menu("가장 인기 있는 레시피 top 5")
#ask_menu("2인분 국/탕 요리 추천해줘")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 팀 통합용 함수
# MAGIC 백엔드 LangGraph 라우터에서 아래 함수를 import하여 사용

# COMMAND ----------

def neo4j_node(state: MessagesState) -> dict:
    """
    Neo4j 그래프 DB 조회 노드
    
    사용법 (팀 라우터에서):
        graph.add_node("neo4j", neo4j_node)
    """
    messages = [SystemMessage(content=MENU_SYSTEM_PROMPT)] + state["messages"]
    
    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            return {"messages": [response]}
        
        for tc in response.tool_calls:
            result = tool_map[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    
    return {"messages": [messages[-1]]}

print("✅ neo4j_node 함수 준비 완료 — 팀 라우터에서 사용 가능")
