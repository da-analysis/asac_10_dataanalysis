from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage

from backend.debug_log import archive

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatDatabricks(endpoint="databricks-gpt-5-4-mini", temperature=0.1)
    return _llm


def _extract_ingredients(recipe_info: dict) -> str:
    """recipe_info에서 레시피별로 재료명+수량을 분리해서 반환.

    각 레시피를 인기 순위(데이터 정렬 순서)로 섹션 구분하여
    cost_calculator LLM이 레시피 단위로 원가를 계산할 수 있게 함.
    """
    if not recipe_info:
        return "정보 없음"

    data = recipe_info.get("data", recipe_info)
    if isinstance(data, list):
        sections = []
        for idx, recipe in enumerate(data, start=1):
            if not isinstance(recipe, dict):
                continue
            menu_name = recipe.get("menu") or recipe.get("name") or f"레시피 {idx}"
            header = f"### [인기 {idx}위] {menu_name}"
            lines = [header]
            for ing in recipe.get("ingredients", []):
                name = ing.get("name", "")
                # Neo4j db.py는 "quantity"로 반환, 옛 코드는 "amount"였음 — 둘 다 호환
                amount = ing.get("quantity") or ing.get("amount") or ""
                if name:
                    lines.append(f"- {name}: {amount}" if amount else f"- {name}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections) if sections else str(data)[:500]
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

    archive("cost_calculator.input", {
        "has_recipe": bool(recipe_info),
        "has_price": bool(price_info),
        "price_source": price_info.get("estimation_source") if isinstance(price_info, dict) else None,
    })

    if not recipe_info or not price_info:
        archive("cost_calculator.output", {"reason": "insufficient_data", "error": "레시피 또는 가격 정보 부족"})
        return {"cost_info": {"error": "레시피 또는 가격 정보 부족"}}

    # 필요한 필드만 추출 (토큰 절약)
    ingredients_text = _extract_ingredients(recipe_info)
    prices_text = _extract_prices(price_info)

    prompt = f"""다음은 사용자가 요청한 메뉴와 관련된 **인기 순 레시피 여러 개**입니다.
**각 레시피를 독립적으로** 원가 계산하세요. 절대 레시피끼리 재료를 합산하지 마세요.

[레시피별 재료 및 사용량]
{ingredients_text}

[시세 (모든 레시피 공통)]
{prices_text}

[단위 변환표 — 반드시 이 값을 사용]
- 1큰술 = 15g = 15ml = 0.015kg
- 1작은술 = 5g = 5ml = 0.005kg
- 1스푼 = 1큰술 = 15g
- 1컵 = 200g = 200ml = 0.2kg
- 1꼬집 = 약 1g
- 1개 = 재료마다 다름 (양파 1개 ≈ 200g, 대파 1대 ≈ 100g, 청양고추 1개 ≈ 7g, 애호박 1개 ≈ 250g 등 상식적 추정)

[계산 규칙]
1. 레시피마다 별도 섹션으로 작성 (인기 순위 유지)
2. **시세가 kg 단위로 주어지면 사용량(g)을 0.001 곱해서 kg으로 변환 후 단가 곱셈**
   예: 맛술 1큰술 = 15g = 0.015kg, 단가 3,268원/kg → 원가 = 3,268 × 0.015 = 약 49원
3. 시세 자체가 없는 재료만 "시세 미확인" 표시
4. **"약간/적당량/1줌" 같은 정성적 표현은 1g~5g로 추정해서 계산** (정확하지 않아도 계산하기, 미확인 처리 금지)
5. 레시피별로 총 원가 → 1인분 원가 → 마진 30% 권장 판매가 산출
6. **절대 여러 레시피의 재료를 합치지 말 것** (각각 다른 변형 레시피임)

[출력 형식 — 각 레시피마다 반복]
## [인기 N위] 레시피명
**재료별 원가** (표: 재료 / 사용량 / 단가 / 원가)
**총 원가**: X원
**1인분 원가**: Y원
**권장 판매가 (마진 30%)**: Z원

마지막에 인기 순위별 원가 비교 요약 한 줄 추가."""

    try:
        messages = [
            SystemMessage(content="당신은 식당 원가 분석 전문가입니다. 정확한 수치로 계산하세요."),
            HumanMessage(content=prompt)
        ]
        result = _get_llm().invoke(messages).content
        archive("cost_calculator.output", {"success": True, "result_preview": result[:400]})
    except Exception as e:
        result = f"원가 계산 실패: {str(e)}"
        archive("cost_calculator.error", {"error": str(e)})

    return {"cost_info": {"analysis": result}}
