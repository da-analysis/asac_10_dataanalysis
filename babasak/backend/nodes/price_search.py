import os
import pandas as pd
from databricks.sdk import WorkspaceClient

GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "01f148e5845f1f68843892ceb53abd32")

_NO_DATA_KEYWORDS = ["존재하지 않", "데이터가 없", "제공할 수 없", "조회할 수 없", "해당하는 데이터", "찾을 수 없"]


def _is_no_data_response(result: dict) -> bool:
    text = result.get("text") or ""
    if any(kw in text for kw in _NO_DATA_KEYWORDS):
        return True
    if result.get("dataframe") is not None and result["dataframe"].empty:
        return True
    return False


def _ask_genie(question: str, conversation_id: str = None) -> dict:
    w = WorkspaceClient()
    result = {"text": None, "sql": None, "dataframe": None, "conversation_id": None}

    if conversation_id is None:
        response = w.genie.start_conversation_and_wait(
            space_id=GENIE_SPACE_ID, content=question)
        result["conversation_id"] = response.conversation_id
    else:
        response = w.genie.create_message_and_wait(
            space_id=GENIE_SPACE_ID, conversation_id=conversation_id, content=question)
        result["conversation_id"] = conversation_id

    if response.attachments:
        text_parts = []
        for att in response.attachments:
            if att.text and att.text.content:
                text_parts.append(att.text.content)
            if att.query and att.query.query:
                result["sql"] = att.query.query
            if att.attachment_id:
                try:
                    qr = w.genie.get_message_query_result(
                        space_id=GENIE_SPACE_ID,
                        conversation_id=result["conversation_id"],
                        message_id=response.id,
                        attachment_id=att.attachment_id)
                    if qr.columns and qr.data_array:
                        result["dataframe"] = pd.DataFrame(
                            qr.data_array, columns=[c.name for c in qr.columns])
                except Exception:
                    pass
        if text_parts:
            result["text"] = "\n".join(text_parts)

    return result


def _detect_unavailable(ingredients: list, genie_text: str) -> list:
    """요청한 재료 vs Genie 응답을 비교하여 누락된 재료 반환."""
    if not genie_text or not ingredients:
        return ingredients
    unavailable = []
    for ing in ingredients:
        if ing not in genie_text:
            unavailable.append(ing)
    return unavailable


def price_search_node(state: dict) -> dict:
    """Genie Space로 도매가 조회. 조회 후 누락 재료를 unavailable 필드로 반환."""
    entities = state.get("entities", {})
    recipe_info = state.get("recipe_info", {})

    ingredients = []
    if recipe_info and recipe_info.get("data"):
        for recipe in recipe_info["data"]:
            for ing in recipe.get("ingredients", []):
                name = ing.get("name", "") if isinstance(ing, dict) else str(ing)
                if name:
                    ingredients.append(name)
    elif entities.get("ingredient"):
        ingredients = [entities["ingredient"]] if isinstance(entities["ingredient"], str) else entities["ingredient"]
    elif entities.get("menu"):
        ingredients = [entities["menu"]] if isinstance(entities["menu"], str) else entities["menu"]

    if not ingredients:
        user_query = state["messages"][-1].content if state.get("messages") else ""
        if user_query:
            try:
                result = _ask_genie(user_query)
                if result["text"] or result["dataframe"] is not None:
                    price_data = {"source": "genie", "data": [user_query]}
                    if result["text"]:
                        price_data["text"] = result["text"]
                    if result["sql"]:
                        price_data["sql"] = result["sql"]
                    if result["dataframe"] is not None:
                        price_data["table"] = result["dataframe"].to_string(index=False, max_rows=15)
                    return {"price_info": price_data}
            except Exception:
                pass
        return {"price_info": {"source": "genie", "data": [], "note": "조회할 재료 없음"}}

    ingredients = list(dict.fromkeys(ingredients))

    try:
        query = f"{', '.join(ingredients[:10])}의 최근 도매가 알려줘"
        result = _ask_genie(query)

        if not result["sql"] and result["text"] and "?" in result["text"]:
            followup = f"원래 질문 맥락에 맞춰서 판단해줘. 원본 질문: {query}"
            result = _ask_genie(followup, conversation_id=result["conversation_id"])

        if _is_no_data_response(result):
            fallback_q = "위 질문에서 요청한 식재료가 실제로 어떤 단위로 판매되는지 보여줘."
            fallback_result = _ask_genie(fallback_q, conversation_id=result["conversation_id"])
            if not _is_no_data_response(fallback_result):
                result = fallback_result

        price_data = {}
        price_data["source"] = "genie"
        price_data["data"] = ingredients

        if result["text"]:
            price_data["text"] = result["text"]
        if result["sql"]:
            price_data["sql"] = result["sql"]
        if result["dataframe"] is not None:
            price_data["table"] = result["dataframe"].to_string(index=False, max_rows=15)

        # [핵심] 누락 재료 감지
        genie_response_text = (result.get("text") or "") + (price_data.get("table") or "")
        unavailable = _detect_unavailable(ingredients, genie_response_text)
        if unavailable:
            price_data["unavailable"] = unavailable

        return {"price_info": price_data}

    except Exception as e:
        return {"price_info": {"source": "error", "data": [], "error": str(e)}, "error_log": [f"price_search: {str(e)}"]}
