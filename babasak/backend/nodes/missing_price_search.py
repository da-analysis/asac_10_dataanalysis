import os
from databricks.sdk import WorkspaceClient

GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "01f148e5845f1f68843892ceb53abd32")

def _find_missing(recipe_info: dict, price_info: dict) -> list[str]:
    """recipe_info 재료 중 price_info에 없는 것을 찾음"""
    ingredients = []
    if recipe_info and recipe_info.get("data"):
        for recipe in recipe_info["data"]:
            for ing in recipe.get("ingredients", []):
                name = ing.get("name", "") if isinstance(ing, dict) else str(ing)
                if name:
                    ingredients.append(name)
    if not ingredients:
        return []
    price_text = str(price_info) if price_info else ""
    return list({ing for ing in ingredients if ing not in price_text})


def missing_price_search_node(state: dict) -> dict:
    """price_search에서 누락된 재료를 개별 Genie 재조회합니다."""
    recipe_info = state.get("recipe_info", {})
    price_info = state.get("price_info", {})
    missing = _find_missing(recipe_info, price_info)

    if not missing:
        return {}  # 누락 없음

    w = WorkspaceClient()
    found = []
    still_missing = []

    for ing in missing[:5]:
        try:
            resp = w.genie.start_conversation_and_wait(
                space_id=GENIE_SPACE_ID, content=f"{ing} 최근 도매가 알려줘"
            )
            parts = []
            if resp.attachments:
                for att in resp.attachments:
                    if att.text and att.text.content:
                        parts.append(att.text.content)
            text = "\n".join(parts)
            if text and "존재하지 않" not in text and "데이터가 없" not in text:
                found.append(f"- {ing}: {text}")
            else:
                still_missing.append(ing)
        except Exception:
            still_missing.append(ing)

    updated = dict(price_info) if price_info else {}
    if found:
        updated["supplementary"] = "\n".join(found)
    if still_missing:
        updated["unavailable"] = still_missing
    return {"price_info": updated}
