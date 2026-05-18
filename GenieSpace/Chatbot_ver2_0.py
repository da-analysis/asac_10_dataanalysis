# Databricks notebook source
# MAGIC %md
# MAGIC # 🍳 Chatbot_ver1.0 — LangGraph 요리 어시스턴트 (Genie Space 연동)
# MAGIC
# MAGIC `bind_tools` + `tools_condition` 기반 LLM 자율 도구 선택 패턴.
# MAGIC - **recipe_db_expert**: 레시피 DB 검색
# MAGIC - **creative_generator**: 레시피 창작 (내부 LLM 호출)
# MAGIC - **price_expert**: Databricks Genie Space를 통한 도매 시세 조회

# COMMAND ----------

# MAGIC %pip install -q langgraph langchain databricks-langchain langchain_core pydantic "mlflow[databricks]>=3.2.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import time
import requests
import pandas as pd

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from databricks_langchain import ChatDatabricks
import mlflow
from mlflow.entities import SpanType

# ========== MLflow Tracing ==========
mlflow.langchain.autolog()
try:
    _current_user = spark.sql("SELECT current_user()").first()[0]
except Exception:
    _current_user = "rimmyeb@gmail.com"
mlflow.set_experiment(f"/Users/{_current_user}/LangGraph_Chatbot_v1")

MODEL_NAME = "databricks-claude-opus-4-6"
llm_main = ChatDatabricks(model=MODEL_NAME, temperature=0.3, max_tokens=2048)
llm_creative = ChatDatabricks(model=MODEL_NAME, temperature=0.7, max_tokens=1024)

# Genie Space ID (환경변수로 관리)
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "")

# ========== 1. State 정의 ==========
class AgentState(MessagesState):
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 도구 정의

# COMMAND ----------

# DBTITLE 1,ask_genie() — Genie Space REST API 연동 함수
# ========== Genie Space REST API 연동 함수 ==========

def ask_genie(
    question: str,
    space_id: str,
    conversation_id: str = None,
    timeout: int = 120,
    verbose: bool = True,
) -> dict:
    """
    Genie API에 질문을 보내고 결과를 반환하는 함수.

    Args:
        question: 질문 또는 Genie가 되물었을 때의 답변
        space_id: Genie Space ID
        conversation_id: 기존 대화 ID (없으면 새 대화 시작, 있으면 후속 메시지)
        timeout: 폴링 타임아웃(초)
        verbose: 진행 상황 출력 여부

    Returns:
        {
            "conversation_id": str,
            "message_id": str,
            "status": str,           # "COMPLETED" or "FAILED"
            "text": str | None,      # Genie 텍스트 응답 (모든 텍스트 attachment 합침)
            "sql": str | None,       # 생성된 SQL
            "dataframe": DataFrame | None,  # 쿼리 결과
        }
    """
    host = spark.conf.get("spark.databricks.workspaceUrl")
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    base_url = f"https://{host}/api/2.0/genie/spaces/{space_id}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # ========== 1. 메시지 전송 ==========
    if conversation_id is None:
        # 새 대화 시작
        if verbose:
            print(f"\U0001f195 새 대화: {question}")
        resp = requests.post(
            f"{base_url}/start-conversation",
            headers=headers,
            json={"content": question},
        )
        resp.raise_for_status()
        data = resp.json()
        conversation_id = data["conversation"]["id"]
        message_id = data["message"]["id"]
    else:
        # 후속 메시지 (답변 또는 추가 질문)
        if verbose:
            print(f"\U0001f504 후속 응답: {question}")
        resp = requests.post(
            f"{base_url}/conversations/{conversation_id}/messages",
            headers=headers,
            json={"content": question},
        )
        resp.raise_for_status()
        message_id = resp.json()["id"]

    # ========== 2. 폴링 (COMPLETED 될 때까지 대기) ==========
    poll_url = f"{base_url}/conversations/{conversation_id}/messages/{message_id}"
    start = time.time()
    while time.time() - start < timeout:
        msg = requests.get(poll_url, headers=headers).json()
        status = msg.get("status")
        if verbose:
            print(f"  \u23f3 {status}")
        if status in ("COMPLETED", "FAILED"):
            break
        time.sleep(3)
    else:
        raise TimeoutError(f"{timeout}\ucd08 \ub0b4 \uc644\ub8cc\ub418\uc9c0 \uc54a\uc558\uc2b5\ub2c8\ub2e4.")

    # ========== 3. 결과 파싱 ==========
    result = {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "status": status,
        "text": None,
        "sql": None,
        "dataframe": None,
    }

    if status != "COMPLETED" or not msg.get("attachments"):
        if verbose:
            print(f"  \u274c {msg.get('error', status)}")
        return result

    # 모든 텍스트 attachment를 합침 
    # (띄어쓰기 대신 누적)
    text_parts = []
    for att in msg["attachments"]:
        if att.get("text"):
            text_parts.append(att["text"]["content"])
            if verbose:
                print(f"\n\U0001f4ac {att['text']['content']}")
        if att.get("query"):
            result["sql"] = att["query"]["query"]
            if verbose:
                print(f"\n\U0001f4ca SQL:\n{att['query']['query']}")
        if att.get("attachment_id"):
            qr_url = f"{base_url}/conversations/{conversation_id}/messages/{message_id}/query-result/{att['attachment_id']}"
            qr = requests.get(qr_url, headers=headers).json()
            if qr.get("columns") and qr.get("data_array"):
                result["dataframe"] = pd.DataFrame(
                    qr["data_array"],
                    columns=[c["name"] for c in qr["columns"]],
                )
    
    # 텍스트 합침
    if text_parts:
        result["text"] = "\n".join(text_parts)

    return result

print("\u2705 ask_genie() REST API \ud568\uc218 \uc815\uc758 \uc644\ub8cc")

# COMMAND ----------

# DBTITLE 1,도구 정의
# ========== 2-1. 레시피 DB 검색 도구 ==========
@tool
def recipe_db_expert(query: str) -> str:
    """사용자가 요청한 요리의 레시피가 기존 데이터베이스에 존재하는지 검색합니다. 레시피를 물어보면 가장 먼저 이 도구를 써서 확인해야 합니다."""
    with mlflow.start_span(name="vector_db_search", span_type=SpanType.RETRIEVER) as span:
        span.set_inputs({"query": query})
        result = "DB 검색 결과: 요청하신 레시피가 데이터베이스에 존재하지 않습니다."
        span.set_outputs({"result": result})
    return result

# ========== 2-2. 레시피 창작 도구 ==========
@tool
def creative_generator(dish_name: str) -> str:
    """(비용 주의: 내부 LLM 호출 발생) DB에 레시피가 존재하지 않는다는 결과가 나왔을 때만 사용합니다.
    새로운 레시피를 창작하고 필요 식재료를 세팅합니다. 입력값은 창작할 요리의 이름입니다."""
    
    prompt = f"""당신은 창의적인 요리 전문가입니다. '{dish_name}'의 새로운 레시피를 창작하고, 식재료 목록을 추출하세요.
    1. 레시피 이름
    2. 필요 식재료 목록 (분량 포함)
    3. 간단한 조리 단계
    한국어로 답변하세요."""
    
    response = llm_creative.invoke([HumanMessage(content=prompt)])
    return response.content

# ========== 2-3. 도매 시세 조회 도구 (Genie Space REST API) ==========
# "데이터 없음" 감지 키워드
_NO_DATA_KEYWORDS = ["존재하지 않", "데이터가 없", "제공할 수 없", "조회할 수 없", "해당하는 데이터"]


def _is_no_data_response(genie_result: dict) -> bool:
    """결과가 '데이터 없음'인지 판별"""
    # DataFrame이 None이거나 비어있으면
    if genie_result["dataframe"] is None or (
        genie_result["dataframe"] is not None and genie_result["dataframe"].empty
    ):
        # 텍스트에 '데이터 없음' 키워드가 있으면
        text = genie_result.get("text") or ""
        if any(kw in text for kw in _NO_DATA_KEYWORDS):
            return True
    return False


@tool
def price_expert(user_question: str) -> str:
    """식재료의 도매 시세 가격을 알아야 할 때 작동하는 Databricks Genie 계산기입니다.
    입력값은 사용자의 원래 질문을 그대로 전달해야 합니다. 절대 요약하거나 식재료만 추출하지 마세요.
    예시: '오늘자 소고기 등심 100g 가격 알려줘' → 그대로 전달"""
    
    with mlflow.start_span(name="genie_price_calc", span_type=SpanType.TOOL) as span:
        span.set_inputs({"question": user_question})
        
        genie_space_id = os.getenv("GENIE_SPACE_ID", "")
        if not genie_space_id:
            result = "\u26a0\ufe0f Genie Space ID가 설정되지 않았습니다. 환경변수 GENIE_SPACE_ID를 설정한 뒤 다시 시도하세요."
            span.set_outputs({"error": result})
            return result
        
        try:
            # ========== 1차 호출: 사용자 질문 그대로 ==========
            genie_result = ask_genie(
                question=user_question,
                space_id=genie_space_id,
                verbose=True,
            )
            
            # 되물음 처리: Genie가 SQL 없이 텍스트만 보내고 '?'가 있으면 후속 응답
            if genie_result["status"] == "COMPLETED" and not genie_result["sql"] and genie_result["text"] and "?" in genie_result["text"]:
                followup = f"원래 질문 맥락에 맞춰서 알아서 판단해줘. 원본 질문: {user_question}"
                genie_result = ask_genie(
                    question=followup,
                    space_id=genie_space_id,
                    conversation_id=genie_result["conversation_id"],
                    verbose=True,
                )
            
            # ========== 2차 호출: 데이터 없음 Fallback ==========
            if genie_result["status"] == "COMPLETED" and _is_no_data_response(genie_result):
                print("\n\U0001f504 [데이터 없음 감지] → 대체 단위 조회 시도...")
                fallback_q = f"그러면 위 질문에서 요청한 식재료가 실제로 어떤 단위로 판매되는지 보여줘. 존재하는 단위 기준으로 평균 가격, 조사 건수를 알려줘."
                fallback_result = ask_genie(
                    question=fallback_q,
                    space_id=genie_space_id,
                    conversation_id=genie_result["conversation_id"],
                    verbose=True,
                )
                # Fallback 성공 시 결과 교체
                if fallback_result["status"] == "COMPLETED" and not _is_no_data_response(fallback_result):
                    genie_result = fallback_result
            
            # ========== 결과 조합 ==========
            parts = []
            if genie_result["text"]:
                parts.append(genie_result["text"])
            if genie_result["sql"]:
                parts.append(f"\n[근거 SQL]\n{genie_result['sql']}")
            if genie_result["dataframe"] is not None:
                df_summary = genie_result["dataframe"].to_string(index=False, max_rows=10)
                parts.append(f"\n[조회 결과]\n{df_summary}")
            
            result = "\n".join(parts) if parts else "\u26a0\ufe0f Genie가 텍스트 응답을 반환하지 않았습니다."
            
        except Exception as e:
            result = f"\u26a0\ufe0f 도매시세 조회에 실패했습니다 (사유: {type(e).__name__}: {str(e)[:200]}). 가격 정보를 제공할 수 없습니다."
        
        span.set_outputs({"result": result})
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Genie Space ID 설정
# MAGIC
# MAGIC 아래 셀에서 `GENIE_SPACE_ID` 환경변수를 설정합니다.
# MAGIC Genie Space ID는 Genie 페이지 URL에서 확인할 수 있습니다.

# COMMAND ----------

# Genie Space ID를 직접 설정 (테스트 시 아래 주석을 해제하고 실제 ID를 입력하세요)
os.environ["GENIE_SPACE_ID"] = "01f148e5845f1f68843892ceb53abd32"

# 현재 설정된 Genie Space ID 확인
current_genie_id = os.getenv("GENIE_SPACE_ID", "")
if current_genie_id:
    print(f"✅ Genie Space ID 설정됨: {current_genie_id}")
else:
    print("⚠️ GENIE_SPACE_ID가 설정되지 않았습니다. 위의 셀에서 설정하거나 클러스터 환경변수에 추가하세요.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 라우팅 및 그래프 설정

# COMMAND ----------

tools = [recipe_db_expert, creative_generator, price_expert]
llm_with_tools = llm_main.bind_tools(tools)

def agent_node(state: AgentState):
    sys_msg = SystemMessage(content="""당신은 요리 어시스턴트 팀장입니다. 도구를 활용해 사용자의 요리 관련 질문에 답변하세요.

[도구 우선순위]
1. recipe_db_expert — 레시피 DB 검색 (항상 가장 먼저 호출)
2. creative_generator — DB에 없을 때 레시피 창작 (비용 높음, 필요시에만)
3. price_expert — 도매 시세 조회 (Genie Space 연동, 식재료 가격 필요 시)

[규칙]
- 레시피 질문 시 반드시 recipe_db_expert를 먼저 호출합니다.
- DB에 없을 경우에만 creative_generator를 사용합니다.
- 가격 관련 질문에는 price_expert를 사용합니다.
- 모든 도구 결과를 수집한 뒤 최종 답변을 작성합니다.
- 한국어로 답변합니다.

[price_expert 호출 시 질문 보강 규칙]
사용자 질문을 price_expert에 전달할 때, 다음 규칙에 따라 질문을 보강하세요:
- 소고기(한우) 관련 가격 질문: 반드시 "등급별 평균가격, 최저가, 최고가"를 포함하여 전달하세요. 소고기는 등급(1++등급, 1+등급, 1등급)이 가격의 핵심 차별 요인입니다.
  예: 사용자 "등심 100g 가격" → price_expert에 "소고기 등심 100g 등급별 평균가격, 최저가, 최고가 알려줘" 전달
  예: 사용자 "한우 갈비 얼마야" → price_expert에 "한우 갈비 등급별 평균가격, 최저가, 최고가 알려줘" 전달
- 사용자가 명시적으로 "지역별", "시도별"을 요청한 경우에만 지역 기준으로 전달합니다.
- 소고기 외 다른 식재료는 원본 질문을 그대로 전달합니다.

[가격 답변 포맷 규칙]
price_expert 도구 결과를 바탕으로 최종 답변을 작성할 때 다음 형식을 따르세요:

1. 소고기 등급별 가격 데이터가 있을 때:
   - 1++등급 → 1+등급 → 1등급 순서로 나열합니다.
   - 각 등급별 평균가격, 최저가, 최고가를 보여줍니다.
   - 예:
     "1++등급: 평균 14,710원 (최저 12,000원 ~ 최고 18,500원)
     1+등급: 평균 12,290원 (최저 9,800원 ~ 최고 15,000원)
     1등급: 평균 10,630원 (최저 8,200원 ~ 최고 13,500원)"

2. 요청한 단위의 데이터가 없을 때 (대체 단위 데이터가 반환된 경우):
   - 먼저 "해당 식재료는 [X] 단위로 판매되지 않고, 주로 [Y] 단위로 판매됩니다"라고 안내합니다.
   - 지역별 실제 판매 단위 가격을 나열합니다.
   - 예:
     "최근 90일간 수도권(서울, 경기) 마트에서는 대파가 1kg 단위로 판매되지 않고, 주로 1단(묶음) 단위로 판매되고 있습니다.

     경기: 1단 평균 가격 3,210원 (조사 16건)
     서울: 1단 평균 가격 2,390원 (조사 8건)

     따라서 1kg 단위 가격 정보는 제공되지 않으며, 실제 판매 단위 기준 가격만 확인할 수 있습니다."

3. 환산 가격인 경우:
   - "환산 가격"임을 명시합니다.
   - 예: "고등어는 100g 단위로 조사되며, 1kg 환산 시 약 6,250원입니다."

4. 불필요한 이모지나 장식적 표현을 사용하지 마세요. 간결하고 사실적으로 답변합니다.""")
    
    safe_msgs = [sys_msg] + [msg for msg in state["messages"] if not isinstance(msg, SystemMessage)]
        
    response = llm_with_tools.invoke(safe_msgs)
    return {"messages": [response]}

# === 개별 도구 실행용 래퍼 함수 ===
def execute_recipe_db_expert(state: AgentState):
    return ToolNode([recipe_db_expert]).invoke(state)

def execute_creative_generator(state: AgentState):
    return ToolNode([creative_generator]).invoke(state)

def execute_price_expert(state: AgentState):
    return ToolNode([price_expert]).invoke(state)

# === 라우터 로직 ===
def custom_router(state: AgentState):
    last_message = state["messages"][-1]
    
    # tool_calls가 없으면 END로 이동
    if not getattr(last_message, "tool_calls", None):
        return END
        
    # 여러 도구가 동시에 호출될 수 있으므로, 리스트로 노드 이름을 리턴하여 병렬 처리
    return [tc["name"] for tc in last_message.tool_calls]

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)

# 개별 노드 추가
workflow.add_node("recipe_db_expert", execute_recipe_db_expert)
workflow.add_node("creative_generator", execute_creative_generator)
workflow.add_node("price_expert", execute_price_expert)

workflow.add_edge(START, "agent")

# 모든 도구 노드에서 다시 agent로 돌아오도록 설정
workflow.add_edge("recipe_db_expert", "agent")
workflow.add_edge("creative_generator", "agent")
workflow.add_edge("price_expert", "agent")

# 조건부 엣지 설정
workflow.add_conditional_edges("agent", custom_router, {
    "recipe_db_expert": "recipe_db_expert",
    "creative_generator": "creative_generator",
    "price_expert": "price_expert",
    END: END
})

chatbot_app = workflow.compile()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 테스트 구동

# COMMAND ----------

user_input = "2026년 5월 12일 감자 가격 알려줘"

print(f"🙋‍♂️ [사용자 질문]: {user_input}\n" + "=" * 60)

initial_state = {"messages": [HumanMessage(content=user_input)]}

for state in chatbot_app.stream(
    initial_state,
    stream_mode="values",
    config={"recursion_limit": 15}
):
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            print(f"🤖 [에이전트 판단]: '{tool_call['name']}' 도구 호출")
            
    elif isinstance(last_message, ToolMessage):
        print(f"  🛠️ [도구 응답 ({last_message.name})]: {last_message.content[:150]}...")
        
    elif isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None) == []:
        print("\n💬 [최종 답변]:\n" + last_message.content)

# COMMAND ----------

# DBTITLE 1,후속 질문 (멀티턴)
# Cell 13 실행 후, 이전 대화 이력을 이어받아 후속 질문
follow_up = "어디서 사야 가장 저렴해?"  # ← 여기에 후속 질문 입력

print(f"\U0001f504 [후속 질문]: {follow_up}\n" + "=" * 60)

# 이전 대화 이력(state)에 후속 질문 추가
follow_state = {"messages": state["messages"] + [HumanMessage(content=follow_up)]}

for s in chatbot_app.stream(
    follow_state,
    stream_mode="values",
    config={"recursion_limit": 15}
):
    last_message = s["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            print(f"\U0001f916 [에이전트 판단]: '{tool_call['name']}' 도구 호출")
            
    elif isinstance(last_message, ToolMessage):
        print(f"  \U0001f6e0\ufe0f [도구 응답 ({last_message.name})]: {last_message.content[:150]}...")
        
    elif isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None) == []:
        print("\n\U0001f4ac [최종 답변]:\n" + last_message.content)

# 다음 후속 질문을 위해 state 갱신
state = s

# COMMAND ----------

# Cell 14 실행 후, 이전 대화 이력을 이어받아 후속 질문
follow_up = "다른 단위 기준으로는 얼마야?"  # ← 여기에 후속 질문 입력

print(f"\U0001f504 [후속 질문]: {follow_up}\n" + "=" * 60)

# 이전 대화 이력(state)에 후속 질문 추가
follow_state = {"messages": state["messages"] + [HumanMessage(content=follow_up)]}

for s in chatbot_app.stream(
    follow_state,
    stream_mode="values",
    config={"recursion_limit": 15}
):
    last_message = s["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            print(f"\U0001f916 [에이전트 판단]: '{tool_call['name']}' 도구 호출")
            
    elif isinstance(last_message, ToolMessage):
        print(f"  \U0001f6e0\ufe0f [도구 응답 ({last_message.name})]: {last_message.content[:150]}...")
        
    elif isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None) == []:
        print("\n\U0001f4ac [최종 답변]:\n" + last_message.content)

# 다음 후속 질문을 위해 state 갱신
state = s

# COMMAND ----------

# Cell 15 실행 후, 이전 대화 이력을 이어받아 후속 질문
follow_up = "당근은 얼마야?"  # ← 여기에 후속 질문 입력

print(f"\U0001f504 [후속 질문]: {follow_up}\n" + "=" * 60)

# 이전 대화 이력(state)에 후속 질문 추가
follow_state = {"messages": state["messages"] + [HumanMessage(content=follow_up)]}

for s in chatbot_app.stream(
    follow_state,
    stream_mode="values",
    config={"recursion_limit": 15}
):
    last_message = s["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            print(f"\U0001f916 [에이전트 판단]: '{tool_call['name']}' 도구 호출")
            
    elif isinstance(last_message, ToolMessage):
        print(f"  \U0001f6e0\ufe0f [도구 응답 ({last_message.name})]: {last_message.content[:150]}...")
        
    elif isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None) == []:
        print("\n\U0001f4ac [최종 답변]:\n" + last_message.content)

# 다음 후속 질문을 위해 state 갱신
state = s
