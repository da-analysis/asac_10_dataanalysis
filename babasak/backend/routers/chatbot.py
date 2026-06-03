import json
import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from backend.graph import get_graph

router = APIRouter(tags=["chatbot"])


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    industry: str = ""
    region: str = ""


def _build_inputs(req: ChatRequest):
    """공통: 요청으로부터 graph 입력을 구성합니다."""
    profile = f"[업종: {req.industry}, 지역: {req.region}]" if req.industry and req.region else ""
    user_message = f"{profile}\n{req.message}".strip() if profile else req.message

    history_messages = []
    for msg in req.history:
        if msg.get("role") == "user":
            history_messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            history_messages.append(AIMessage(content=msg.get("content", "")))

    return {"messages": history_messages + [HumanMessage(content=user_message)]}


# ═══════════════════════════════════════════════════════════════
# 기존 동기 엔드포인트 (하위 호환)
# ═══════════════════════════════════════════════════════════════
@router.post("/chat")
async def chat(req: ChatRequest):
    try:
        graph = get_graph()
        result = await graph.ainvoke(
            _build_inputs(req),
            config={"recursion_limit": 15},
        )

        response = result.get("final_report", "")
        if not response and result.get("messages"):
            response = result["messages"][-1].content

        chart_html = None
        price_info = result.get("price_info")
        if isinstance(price_info, dict):
            chart_html = price_info.get("chart_html")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    resp_body = {"response": response}
    if chart_html:
        resp_body["chart_html"] = chart_html
    return resp_body


# ═══════════════════════════════════════════════════════════════
# 스트리밍 엔드포인트 (SSE) — timeout 문제 해결
# ═══════════════════════════════════════════════════════════════
@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    graph = get_graph()
    inputs = _build_inputs(req)

    async def event_generator():
        final_response = ""
        chart_html = None

        try:
            async for event in graph.astream_events(
                inputs,
                config={"recursion_limit": 15},
                version="v2",
            ):
                kind = event["event"]

                # LLM 토큰 스트리밍 (report_generator의 출력만 전달)
                if kind == "on_chat_model_stream":
                    # metadata.langgraph_node로 노드 이름을 확인 (parent_ids는 UUID 리스트라 사용 불가)
                    node_name = event.get("metadata", {}).get("langgraph_node", "")
                    if node_name == "report_generator":
                        chunk = event["data"]["chunk"]
                        content = chunk.content if hasattr(chunk, "content") else str(chunk)
                        if content:
                            final_response += content
                            yield f"data: {json.dumps({'type': 'token', 'content': content}, ensure_ascii=False)}\n\n"

                # 노드 완료 이벤트 (진행 상태 전송)
                elif kind == "on_chain_end":
                    node_name = event.get("name", "")
                    if node_name in [
                        "recipe_search", "price_search",
                        "missing_price_search", "cost_calculator",
                        "report_generator",
                    ]:
                        yield f"data: {json.dumps({'type': 'step', 'node': node_name, 'status': 'done'}, ensure_ascii=False)}\n\n"

                    # 노드 출력에서 price_info 추출 (price_search / cost_calculator 가 실제로 price_info를 반환)
                    # report_generator의 출력은 AIMessage이므로 여기서 꺼낼 수 없음
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        price_info = output.get("price_info")
                        if isinstance(price_info, dict) and price_info.get("chart_html"):
                            chart_html = price_info["chart_html"]

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

        # 완료 메시지 (차트 포함)
        done_payload = {"type": "done", "content": final_response}
        if chart_html:
            done_payload["chart_html"] = chart_html
        yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 비활성화
        },
    )
