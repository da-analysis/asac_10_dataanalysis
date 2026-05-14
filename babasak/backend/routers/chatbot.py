from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from backend.agent import get_chatbot

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
        history_messages = []
        for msg in req.history:
            if msg.get("role") == "user":
                history_messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                history_messages.append(AIMessage(content=msg.get("content", "")))

        chatbot = get_chatbot()
        result = chatbot.invoke({"messages": history_messages + [HumanMessage(content=user_message)]})
        response = result["messages"][-1].content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"response": response}
