from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from backend.graph import get_graph

router = APIRouter(tags=["chatbot"])


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    industry: str = ""
    region: str = ""


@router.post("/chat")
def chat(req: ChatRequest):
    profile = f"[업종: {req.industry}, 지역: {req.region}]" if req.industry and req.region else ""
    user_message = f"{profile}\n{req.message}".strip() if profile else req.message

    try:
        # 대화 히스토리 구성
        history_messages = []
        for msg in req.history:
            if msg.get("role") == "user":
                history_messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                history_messages.append(AIMessage(content=msg.get("content", "")))

        # 노드 순환 그래프 실행
        graph = get_graph()
        result = graph.invoke(
            {"messages": history_messages + [HumanMessage(content=user_message)]},
            config={"recursion_limit": 15},
        )

        # 최종 답변 추출
        response = result.get("final_report", "")
        if not response and result.get("messages"):
            response = result["messages"][-1].content

        # ═══ [차트 HTML 추출] ═══════════════════════════════════════
        # price_search_node가 price_info["chart_html"]에 차트를 저장함.
        # 차트가 있으면 프론트엔드로 함께 전달.
        # 차트를 비활성화하려면 chart_utils.py의 ENABLE_CHART = False로 변경.
        # ═══════════════════════════════════════════════════════════════
        chart_html = None
        price_info = result.get("price_info")
        if isinstance(price_info, dict):
            chart_html = price_info.get("chart_html")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # chart_html이 있으면 함께 반환, 없으면 기존과 동일하게 response만
    resp_body = {"response": response}
    if chart_html:
        resp_body["chart_html"] = chart_html
    return resp_body
