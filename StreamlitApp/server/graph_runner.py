from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from handlers.genie_sales_handler import GenieSALESHandler
from handlers.genie_license_handler import GenieLICENSEHandler
from handlers.genie_100_handler import Genie100Handler
from handlers.local_rag_handler import LocalRAGHandler
from handlers.router import LLMRouter
from handlers.fallback_handler import fallback_example_node
from dotenv import load_dotenv
import time
import pandas as pd
import streamlit as st
import os, sys
sys.path.insert(0, "../handlers")
load_dotenv()

# OpenAI API Key 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 시스템 클래스 인스턴스 정의
genie_sales_api = GenieSALESHandler(
    workspace=os.environ["DATABRICKS_WORKSPACE"],
    token=os.environ["DATABRICKS_TOKEN_SALE"],
    space_id=os.environ["DATABRICKS_SPACE_ID_SALE"]
)

genie_license_api = GenieLICENSEHandler(
    workspace=os.environ["DATABRICKS_WORKSPACE"],
    token=os.environ["DATABRICKS_TOKEN_LICENSE"],
    space_id=os.environ["DATABRICKS_SPACE_ID_LICENSE"]
)

genie_100_api = Genie100Handler(
    workspace=os.environ["DATABRICKS_WORKSPACE"],
    token=os.environ["DATABRICKS_TOKEN_100"],
    space_id=os.environ["DATABRICKS_SPACE_ID_100"]
)

rag_api = LocalRAGHandler(
    bucket=os.environ["BUCKET_NAME"],
    key=os.environ["BUCKET_KEY_XML"],
    pdf_prefix=os.environ.get("BUCKET_PREFIX_PDF")
)

router = LLMRouter()

class GraphState(TypedDict, total=False):
    question: str
    history: list[str]
    route: Literal["A", "B", "C", "D", "X"]
    response: str
    response_df: pd.DataFrame
    description: str

def question_node(state: GraphState) -> GraphState:
    return {
        "question": state.get("question", ""),
        "history": state.get("history", [])
    }

def classify_node(state: GraphState) -> GraphState:
    question = state["question"]
    history = state.get("history", [])

    context = "\n.".join(history[-3:])

    route = router.route(question, context=context)
    normalized_route = route.strip().upper()
    print(f"[DEBUG] 정규화된 경로: {normalized_route}")
    return {**state, "route": normalized_route}

def genie_sales_node(state: GraphState) -> GraphState:
    try:
        question = state["question"]
        if "genie_sales_conversation_id" not in st.session_state:
            print("[DEBUG] [SALES] 대화 새로 시작")
            result = genie_sales_api.start_conversation(question)
            st.session_state["genie_sales_conversation_id"] = result["conversation_id"]
        else:
            print("[DEBUG] [SALES] 이전 대화 계속 사용")
            result = genie_sales_api.ask_followup(st.session_state["genie_sales_conversation_id"], question)

        conversation_id = result["conversation_id"]
        message_id = result["message_id"]
        print(f"[DEBUG] [SALES] conversation_id: {conversation_id}, message_id: {message_id}")

        for i in range(30):
            print(f"[DEBUG] [SALES] polling {i+1}/30 ...")
            time.sleep(2)
            message = genie_sales_api.get_query_info(conversation_id, message_id)
            attachments = message.get("attachments", [])
            status_val = message.get("status", "")
            print("[DEBUG] [SALES] 현재 상태:", status_val)
            if status_val in ("SUCCEEDED", "COMPLETED") and attachments:
                print("[DEBUG] [SALES] 쿼리 성공!")
                break
        else:
            return {**state, "response": "❗쿼리 결과를 가져오는 데 실패했습니다."}

        attachment = attachments[0]
        query_block = attachment.get("query")
        text_block = attachment.get("text", {}).get("content")

        if not query_block:
            return {**state, "response": text_block or "답변은 생성됐지만 실행 가능한 쿼리는 없었습니다."}

        attachment_id = attachment["attachment_id"]
        description = query_block.get("description", None)

        result_data = genie_sales_api.get_query_result(conversation_id, message_id, attachment_id)
        data_array = result_data.get("statement_response", {}).get("result", {}).get("data_array", [])
        columns_schema = result_data.get("statement_response", {}).get("manifest", {}).get("schema", {}).get("columns", [])
        column_names = [col.get("name", f"col{i}") for i, col in enumerate(columns_schema)]

        if data_array and column_names:
            df = pd.DataFrame(data_array, columns=column_names)
            return {**state, "response_df": df, "description": description}
        else:
            return {**state, "response": "데이터가 비어있습니다.", "description": description}

    except Exception as e:
        print("[ERROR] [SALES] 예외 발생:", str(e))
        return {**state, "response": f"❗데이터 처리 중 오류가 발생했습니다.\\n({str(e)})"}

def genie_license_node(state: GraphState) -> GraphState:
    try:
        question = state["question"]
        if "genie_license_conversation_id" not in st.session_state:
            print("[DEBUG] [LICENSE] 대화 새로 시작")
            result = genie_license_api.start_conversation(question)
            st.session_state["genie_license_conversation_id"] = result["conversation_id"]
        else:
            print("[DEBUG] [LICENSE] 이전 대화 계속 사용")
            result = genie_license_api.ask_followup(st.session_state["genie_license_conversation_id"], question)

        conversation_id = result["conversation_id"]
        message_id = result["message_id"]
        print(f"[DEBUG] [LICENSE] conversation_id: {conversation_id}, message_id: {message_id}")

        for i in range(30):
            print(f"[DEBUG] [LICENSE] polling {i+1}/30 ...")
            time.sleep(2)
            message = genie_license_api.get_query_info(conversation_id, message_id)
            attachments = message.get("attachments", [])
            status_val = message.get("status", "")
            print("[DEBUG] [LICENSE] 현재 상태:", status_val)
            if status_val in ("SUCCEEDED", "COMPLETED") and attachments:
                print("[DEBUG] [LICENSE] 쿼리 성공!")
                break
        else:
            return {**state, "response": "❗쿼리 결과를 가져오는 데 실패했습니다."}

        attachment = attachments[0]
        query_block = attachment.get("query")
        text_block = attachment.get("text", {}).get("content")

        if not query_block:
            return {**state, "response": text_block or "답변은 생성됐지만 실행 가능한 쿼리는 없었습니다."}

        attachment_id = attachment["attachment_id"]
        description = query_block.get("description", None)

        result_data = genie_license_api.get_query_result(conversation_id, message_id, attachment_id)
        data_array = result_data.get("statement_response", {}).get("result", {}).get("data_array", [])
        columns_schema = result_data.get("statement_response", {}).get("manifest", {}).get("schema", {}).get("columns", [])
        column_names = [col.get("name", f"col{i}") for i, col in enumerate(columns_schema)]

        if data_array and column_names:
            df = pd.DataFrame(data_array, columns=column_names)
            return {**state, "response_df": df, "description": description}
        else:
            return {**state, "response": "데이터가 비어있습니다.", "description": description}

    except Exception as e:
        print("[ERROR] [LICENSE] 예외 발생:", str(e))
        return {**state, "response": f"❗LICENSE 처리 중 오류 발생: {str(e)}"} # 기존 오류 메시지 유지

def genie_100_node(state: GraphState) -> GraphState:
    try:
        question = state["question"]
        if "genie_100_conversation_id" not in st.session_state:
            print("[DEBUG] [100] 대화 새로 시작")
            result = genie_100_api.start_conversation(question)
            st.session_state["genie_100_conversation_id"] = result["conversation_id"]
        else:
            print("[DEBUG] [100] 이전 대화 계속 사용")
            result = genie_100_api.ask_followup(st.session_state["genie_100_conversation_id"], question)

        conversation_id = result["conversation_id"]
        message_id = result["message_id"]
        print(f"[DEBUG] [100] conversation_id: {conversation_id}, message_id: {message_id}")

        for i in range(30):
            print(f"[DEBUG] [100] polling {i+1}/30 ...")
            time.sleep(2)
            message = genie_100_api.get_query_info(conversation_id, message_id)
            attachments = message.get("attachments", [])
            status_val = message.get("status", "")
            print("[DEBUG] [100] 현재 상태:", status_val)
            if status_val in ("SUCCEEDED", "COMPLETED") and attachments:
                print("[DEBUG] [100] 쿼리 성공!")
                break
        else:
            return {**state, "response": "❗쿼리 결과를 가져오는 데 실패했습니다."}

        attachment = attachments[0]
        query_block = attachment.get("query")
        text_block = attachment.get("text", {}).get("content")

        if not query_block:
            return {**state, "response": text_block or "답변은 생성됐지만 실행 가능한 쿼리는 없었습니다."}

        attachment_id = attachment["attachment_id"]
        description = query_block.get("description", None)

        result_data = genie_100_api.get_query_result(conversation_id, message_id, attachment_id)
        data_array = result_data.get("statement_response", {}).get("result", {}).get("data_array", [])
        columns_schema = result_data.get("statement_response", {}).get("manifest", {}).get("schema", {}).get("columns", [])
        column_names = [col.get("name", f"col{i}") for i, col in enumerate(columns_schema)]

        if data_array and column_names:
            df = pd.DataFrame(data_array, columns=column_names)
            return {**state, "response_df": df, "description": description}
        else:
            return {**state, "response": "데이터가 비어있습니다.", "description": description}

    except Exception as e:
        print("[ERROR] [100] 예외 발생:", str(e))
        return {**state, "response": f"❗데이터 처리 중 오류가 발생했습니다.\n({str(e)})"} 

# D 시스템 처리
def rag_node(state: GraphState) -> GraphState:
    answer, meta = rag_api.ask(state["question"])
    print("[DEBUG][RAG] answer:", answer)
    print("[DEBUG][RAG] meta:", meta)
    return {**state, "response": answer}

# fallback 처리 노드 (분류 실패 시 예시 안내)
def fallback_node(state: GraphState) -> GraphState:
    return {
        **state,
        "response": fallback_example_node(state["question"])["response"]
    }

def response_node(state: GraphState) -> GraphState:
    question = state.get("question")
    history = state.get("history", []) + [question]

    print("✅ [DEBUG] response_node 실행됨")
    print("🧾 [DEBUG] response_node 진입 시 history:", state.get("history"))
    print("🆕 [DEBUG] response_node 종료 시 history (현재 질문 추가됨):", history)

    return {
        "response_df": state.get("response_df"),
        "response": state.get("response"),
        "description": state.get("description"),
        "history": history
    }
# ─────────────────────────────────────────────────────────────
# 4. LangGraph 구성

builder = StateGraph(GraphState)
builder.set_entry_point("question_node")

builder.add_node("question_node", question_node)
builder.add_node("classify_node", classify_node)
builder.add_node("genie_sales_node", genie_sales_node)
builder.add_node("genie_license_node", genie_license_node)
builder.add_node("genie_100_node", genie_100_node)
builder.add_node("rag_node", rag_node)
builder.add_node("respond_node", response_node)
builder.add_node("fallback_node", fallback_node)
builder.add_edge("question_node", "classify_node")

builder.add_conditional_edges(
    "classify_node",
    lambda state: state["route"].strip().upper(),
    {
        "A": "genie_sales_node",
        "B": "genie_license_node",
        "C": "genie_100_node",
        "D": "rag_node",
        "X": "fallback_node",
    }
)

builder.add_edge("genie_sales_node", "respond_node")
builder.add_edge("genie_license_node", "respond_node")
builder.add_edge("genie_100_node", "respond_node")
builder.add_edge("rag_node", "respond_node")
builder.add_edge("fallback_node", "respond_node")
builder.set_finish_point("respond_node") # 최종 종료 지점

graph = builder.compile()

# ─────────────────────────────────────────────────────────────
# 5. 실행 함수

def run_chatbot(question: str, history: list[str] = None) -> dict:
    if history is None:
        history = []

    print("🧪[DEBUG] run_chatbot 전달된 히스토리:", history)

    output = graph.invoke({
        "question": question,
        "history": history
    })

    print("🧪[DEBUG] LangGraph 최종 응답 결과 (output):", output)

    return {
        "response": output.get("response"),
        "response_df": output.get("response_df"),
        "description": output.get("description"),
        "route": output.get("route")
    }

if __name__ == "__main__":

    current_history = []
    while True:
        user_input = input("질문을 입력하세요 (종료하려면 'exit' 입력): ")
        if user_input.lower() == 'exit':
            break
        try:
            print(f"\n[INFO] 현재 __main__ 환경에서는 Streamlit의 session_state를 사용할 수 없어 Genie 노드의 대화 ID 관리에 문제가 있을 수 있습니다.")
            print(f"[INFO] 정확한 테스트는 Streamlit 앱을 통해 진행해주세요.")
            
            if not hasattr(st, 'session_state'):
                class MockSessionState(dict):
                    def __getattr__(self, key):
                        return self.get(key)
                    def __setattr__(self, key, value):
                        self[key] = value
                st.session_state = MockSessionState()


            result = run_chatbot(user_input, history=current_history)

            if result.get("response_df") is not None:
                print("\n[📊 데이터프레임 결과]")
                print(result["response_df"])
            elif result.get("response"):
                print("\n[💬 텍스트 응답]")
                print(result["response"])
            else:
                print("\n❗답변이 없습니다.")

        except Exception as e:
            print(f"[ERROR] __main__ 실행 중 오류: {e}")
            print(f"[INFO] 이는 __main__ 환경에서의 st.session_state 제한 때문일 수 있습니다.")