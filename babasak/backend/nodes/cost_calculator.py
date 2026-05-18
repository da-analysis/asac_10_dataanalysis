from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatDatabricks(endpoint="databricks-gpt-5-4-mini", temperature=0.1)
    return _llm


def _extract_ingredients(recipe_info: dict) -> str:
    """recipe_info에서 재료명+수량만 추출하여 간결한 문자열로 반환"""
    if not recipe_info:
        return "정보 없음"

    data = recipe_info.get("data", recipe_info)
    if isinstance(data, list):
        lines = []
        for recipe in data:
            if isinstance(recipe, dict):
                for ing in recipe.get("ingredients", []):
                    name = ing.get("name", "")
                    amount = ing.get("amount", "")
                    if name:
                        lines.append(f"- {name}: {amount}" if amount else f"- {name}")
        return "\n".join(lines) if lines else str(data)[:500]
    return str(data)[:500]


def _extract_prices(price_info: dict) -> str:
    """price_info에서 단가 정보만 추출"""
    if not price_info:
        return "정보 없음"

    parts = []
    if price_info.get("text"):
        parts.append(price_info["text"])
    if price_info.get("table"):
        parts.append(price_info["table"])
    return "\n".join(parts) if parts else str(price_info)[:500]


def cost_calculator_node(state: dict) -> dict:
    """레시피 재료 + 가격 데이터를 바탕으로 사용량 기준 원가를 계산합니다."""
    recipe_info = state.get("recipe_info", {})
    price_info = state.get("price_info", {})

    if not recipe_info or not price_info:
        return {"cost_info": {"error": "레시피 또는 가격 정보 부족"}}

    # 필요한 필드만 추출 (토큰 절약)
    ingredients_text = _extract_ingredients(recipe_info)
    prices_text = _extract_prices(price_info)

    prompt = f"""다음 재료와 시세로 원가를 계산하세요.

[재료 및 사용량]
{ingredients_text}

[시세]
{prices_text}

[계산 규칙]
1. 각 재료의 레시피 사용량 기준 비례 원가 계산
2. 시세 없는 재료는 "시세 미확인" 표시
3. 총 원가 합산 → 1인분 원가 산출
4. 마진 30% 기준 권장 판매가 제시

**재료별 원가** (표 형식)
**총 원가**: X원
**1인분 원가**: Y원
**권장 판매가 (마진 30%)**: Z원"""

    try:
        messages = [
            SystemMessage(content="당신은 식당 원가 분석 전문가입니다. 정확한 수치로 계산하세요."),
            HumanMessage(content=prompt)
        ]
        result = _get_llm().invoke(messages).content
    except Exception as e:
        result = f"원가 계산 실패: {str(e)}"

    return {"cost_info": {"analysis": result}}
