from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatDatabricks(endpoint="databricks-gpt-5-4-mini", temperature=0.1)
    return _llm


def cost_calculator_node(state: dict) -> dict:
    """레시피 재료 + 가격 데이터를 바탕으로 사용량 기준 원가를 계산합니다."""
    recipe_info = state.get("recipe_info", {})
    price_info = state.get("price_info", {})

    if not recipe_info or not price_info:
        return {"cost_info": {"error": "레시피 또는 가격 정보 부족"}}

    prompt = f"""다음 레시피의 재료 정보와 시세 데이터를 바탕으로 원가를 계산해주세요.

[레시피 재료 정보]
{recipe_info}

[시세/가격 데이터]
{price_info}

[계산 규칙]
1. 각 재료의 레시피 사용량 파악
2. 시세 단가(kg당, 개당 등) 기준 실제 사용량의 비례 원가 계산
3. 시세 없는 재료는 "시세 미확인" 표시
4. 총 원가 합산 후 1인분 원가 산출
5. 마진 30% 기준 권장 판매가 제시

아래 형식으로 답변:
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
