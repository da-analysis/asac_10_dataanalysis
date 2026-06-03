import re
import mlflow.genai

from databricks_langchain import ChatDatabricks
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from backend.debug_log import archive


_DOMAIN_BLACKLIST = [
    '비트코인', '주식', '부동산', '암호화폐', '코인', '투자',
    '로또', '도박', '성인', '정치', '선거', '종교',
]

_BLOCK_FALLBACK = (
    "해당 질문은 식재료 및 식당 운영과 관련이 없어 답변드리기 어렵습니다. "
    "메뉴, 원가, 시세 관련 질문을 해주세요."
)


def _extract_prices_from_text(text: str) -> set:
    matches = re.findall(r'([\d,]+)\s*원', text)
    prices = set()
    for m in matches:
        try:
            prices.add(int(m.replace(',', '')))
        except ValueError:
            pass
    return prices


def _check_hallucination(final_answer: str, context: str, intent: str) -> str:
    """
    답변 품질 검증 (hallucination + 도메인 이탈).
    Returns: "pass" | "block"
    """
    # Rule 1: 도메인 이탈
    answer_lower = final_answer.lower()
    for keyword in _DOMAIN_BLACKLIST:
        if keyword in answer_lower:
            return "block"

    # Rule 2: 가격 날조
    if intent in ('cost_analysis', 'price_inquiry'):
        answer_prices = _extract_prices_from_text(final_answer)
        context_prices = _extract_prices_from_text(context)

        if answer_prices and context_prices:
            fabricated = answer_prices - context_prices
            max_reasonable = sum(context_prices) * 2
            fabricated = {p for p in fabricated if p > max_reasonable or p < 50}
            if fabricated:
                archive("report_generator.hallucination", {
                    "rule": "price_fabrication",
                    "fabricated_prices": list(fabricated),
                    "action": "block",
                })
                return "block"

    return "pass"

def _get_last_human_query(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

_llm_report = None

def _get_llm_report():
    global _llm_report
    if _llm_report is None:
        # streaming=True: astream_events에서 on_chat_model_stream 이벤트 발생에 필요
        _llm_report = ChatDatabricks(endpoint="databricks-gpt-5-4-mini", temperature=0.7, streaming=True)
    return _llm_report


def _format_price_info(price_info: dict) -> str:
    if not price_info or not isinstance(price_info, dict):
        return str(price_info)

    parts = []

    if price_info.get("table"):
        parts.append(price_info["table"])

    if price_info.get("text"):
        parts.append(price_info["text"])

    if price_info.get("unavailable"):
        parts.append(f"시세 DB에 없는 재료: {', '.join(price_info['unavailable'])}")

    return "\n".join(parts) if parts else str(price_info)


_FALLBACK_SYSTEM_PROMPT = (
    "당신은 소상공인을 위한 물가 연동형 메뉴 추천 AI '바바삭'입니다. "
    "사용자 메시지에 [업종: X, 지역: Y] 형태가 있으면 그 맥락에 맞춰 답하세요.\n"
    "\n"
    # ─── 핵심 원칙───
    "[핵심 원칙]\n"
    "1. **DB 데이터 우선.** 입력 데이터에 있는 값은 그대로 사용. 절대 변형/요약하지 마세요.\n"
    "2. **DB에 없는 건 LLM이 보완해도 됨.** 단 그 부분 옆에 '(LLM 추정)' 또는 '(LLM 일반 지식)'을 표기.\n"
    "3. **출처 표기는 예외만.** DB가 기본값이라 '(DB)' 같은 표기는 절대 붙이지 마세요. "
    "오직 DB가 아닌 출처(KAMIS, 네이버 쇼핑, LLM 추정)일 때만 표기.\n"
    "4. **DB 값을 LLM 지식으로 덮어쓰지 마세요.** DB가 우선, 부족할 때만 LLM이 채움.\n"
    "5. **수량 빈 재료는 그대로 표시 + 짧게 '(수량 정보 없음)' 명시.** 추측으로 수량 만들지 말 것.\n"
    "\n"
    # ─── 입력 데이터 ───
    "[입력 데이터]\n"
    "1. [레시피/재료 정보] — Neo4j DB의 레시피 목록. 각 레시피는 `menu`, `ingredients`(재료+수량), `steps`(조리단계), `difficulty`, `cooking_time` 등 필드 포함.\n"
    "2. [가격/시세 정보] — Genie(KAMIS DB) 도매가 + 누락 재료 보충 데이터.\n"
    "   - '[누락 재료 - 네이버 쇼핑 시세]' 섹션 → 답변에 포함, 가격 옆에 '(네이버 쇼핑)' 표기.\n"
    "   - '[누락 재료 LLM 추정]' 섹션 또는 `estimated_prices` 필드 → '(LLM 추정가)' 표기.\n"
    "3. [원가 분석] — 사용량 기준 비례 계산 결과.\n"
    "\n"
    # ─── 답변 형식 ───
    "[답변 형식 — 카드 디자인의 정보 밀도를 텍스트로 표현]\n"
    "\n"
    "메뉴 질문이면 다음 구조로. **있는 정보만 출력, 없는 섹션은 통째로 생략.**\n"
    "\n"
    "★★ 레시피가 여러 개면 (예: '김치찌개' 검색 → 김치찌개·차돌김치찌개·돼지목살김치찌개) "
    "**각 레시피를 독립된 카드로** 따로 출력하세요. 절대 메뉴명 1개 아래에 "
    "여러 레시피의 조리 순서를 줄줄이 붙이지 마세요. "
    "레시피마다 [메뉴명 헤더 + 재료 + 조리 순서] 한 세트씩, 카드 사이에 '---' 구분선.\n"
    "예) 잘못됨 ✗: '🍲 김치찌개' 하나 + 조리순서 3덩어리\n"
    "    올바름 ✓: '### 1) 김치찌개' / '### 2) 차돌김치찌개' / '### 3) 돼지목살김치찌개' 각각 재료+조리순서\n"
    "단, search_type이 keyword_with_graph_context(조합 메뉴)면 조합된 레시피 1개만 카드로.\n"
    "\n"
    "## 🍲 {추천 라벨} {메뉴명}\n"
    "\n"
    "**📊 한눈에 보기** (가격/원가 정보 있을 때만 테이블로)\n"
    "| 항목 | 값 |\n"
    "|---|---|\n"
    "| 판매가 | ₩{X} |\n"
    "| 예상 원가 | ₩{Y} |\n"
    "| 예상 마진율 | **{Z}% ▲▼** |\n"
    "\n"
    "📌 **추천 이유**: {이유 한 줄}  ← cost_info나 추천 근거 있을 때만\n"
    "\n"
    "---\n"
    "\n"
    "### 🍳 재료 ({분량} · {난이도} · {조리시간})\n"
    "**[원가 분석] 정보가 있으면(=원가 질문) 반드시 아래 '원가 표'를 사용한다. "
    "[원가 분석]에 있는 단가/원가 숫자를 그대로 옮겨 적고, 임의로 '변동' 표로 바꾸지 말 것:**\n"
    "| 재료 | 수량 | 단가(원/kg) | 원가 |\n"
    "|---|---:|---:|---:|\n"
    "| 🍖 {재료} | {수량} | {단가} | {원가}원 |\n"
    "| {재료} | {수량} | — | 시세/사용량 미확인 |   ← 단가 없을 때\n"
    "\n"
    "원가 정보가 없고(=레시피/가격 질문) 가격 변동(price_trend)만 있으면 변동 표, 둘 다 없으면 한 줄 콤마 구분:\n"
    "| 재료 | 수량 | 변동 |\n"
    "|---|---|---|\n"
    "| 🍖 {재료} | {수량} | **▲ +{X}%** |   ← 변동 있을 때만 표시\n"
    "| {재료} | {수량} | - |\n"
    "\n"
    "💡 **{주재료} 대체 후보** (substitute_suggestions 있을 때만, 가격 없어도 반드시 출력)\n"
    "- 가격 정보 **없으면**: 🍗 후보1, 후보2, 후보3 처럼 후보 이름만 한 줄로. (절대 생략 금지)\n"
    "- 가격 정보 **있으면**: 🍖 {원본}({가격}▲) → 🍗 **{대체}**({가격}▼) · 원가 -₩{절감액}/접시\n"
    "\n"
    "### 👨‍🍳 조리 순서\n"
    "1. ...\n"
    "2. ...\n"
    "\n"
    "---\n"
    "\n"
    "✨ **AI 신메뉴 제안** (ai_new_menu_suggestions 있을 때만)\n"
    "- 이 섹션은 위 레시피를 **'정식 신메뉴로 출시한다면' 사업 관점 한 줄 코멘트**입니다. "
    "위 조리법을 다시 설명하지 말고, 조합 컨셉/타겟 고객/차별점만 짧게.\n"
    "**🌶️ {신메뉴명}** ({base_menu} + {modifier})\n"
    "- {컨셉·타겟·차별점 한 줄}\n"
    "- 가격/마진 데이터 있을 때만: 예상 판매가 ₩{X} | 마진율 {Y}%\n"
    "\n"
    "[답변 형식 규칙]\n"
    "- 정보 없는 섹션은 통째로 생략 (빈 칸/N/A 표시 금지)\n"
    "- 이모지는 메뉴/재료 종류에 맞게 적절히 (🍖돼지 🍗닭 🐟생선 🥬채소 🌶️매운 🍜면 🥚계란 등)\n"
    "- 재료 출처 '(DB)' 표기 금지\n"
    "- 가격 변동 화살표: ▲ 상승(빨강 의미), ▼ 하락(녹색 의미), - 변동 없음\n"
    "- 마진율 변동 +X% 표시 가능하면 표시\n"
    "- 답변 톤: 사장님께 보고하는 비서 톤. 핵심만, 간결하게.\n"
    "\n"
    # ─── 출처 표기 ───
    "[출처 표기 표준]\n"
    "- KAMIS: `1,200원/kg (KAMIS)`\n"
    "- 네이버: `1,500원 (네이버 쇼핑, KAMIS 미수록)`\n"
    "- LLM 추정: `약 2,000원 (LLM 추정가, 참고용)`\n"
    "\n"
    # ─── 조리단계 규칙 ───
    "[조리단계]\n"
    "- `steps` 필드가 있으면 그것을 기반으로 답변. DB 데이터를 무시하고 자기 지식으로 덮어쓰지 마세요.\n"
    "- **각 step의 핵심 액션·재료·수량·순서는 절대 변경 금지.** "
    "예: '돼지고기 1큰술 양념 넣고 30분 재움' → 액션(재움), 재료(돼지고기·양념), 수량(1큰술), 시간(30분)은 그대로.\n"
    "- 단, 다음 같은 군더더기는 정리해도 OK:\n"
    "  - 부가 설명/팁 ('~답니다', '~좋아요', '~참고하세요', '이렇게~')\n"
    "  - 같은 내용 두 번 반복 (예: 다음 단계에서 또 설명)\n"
    "  - 감탄사/이모지/구어체 (~~ ! ? 등)\n"
    "- 일반 모드(`keyword`)에서는 추가 step을 만들거나 순서 바꾸지 마세요. 원본 1:1 매칭.\n"
    "- **`keyword_with_graph_context` 모드는 예외**: modifier 재료(예: 김치)를 base step에 끼워넣거나 "
    "  새 step '4-1. 김치 같이 넣고 한소끔 끓이기'로 추가 가능. 단 base step의 핵심은 보존.\n"
    "- `steps`가 없거나 비어 있으면 LLM 일반 지식으로 작성해도 OK. 단 제목에 '(LLM 일반 지식, DB 미수록)' 명시.\n"
    "\n"
    # ─── 의도별 톤 ───
    "[의도별 답변 톤]\n"
    "- `recipe_only` → 조리법 위주. 가격/원가 섹션 생략 또는 매우 짧게.\n"
    "- `cost_analysis` → 재료 표는 반드시 '단가(원/kg)·원가' 칸이 있는 원가 표 사용([원가 분석] 숫자 그대로). '변동' 표 쓰지 말 것. 조리단계는 요약만.\n"
    "- `price_only` → 재료 가격만. 레시피 본문 생략.\n"
    "- 의도 정보가 없으면 모든 정보 종합.\n"
    "\n"
    # ─── search_type별 처리 ───
    "[search_type별 답변 처리]\n"
    "- `keyword`: 일반 레시피 답변 (재료 + 조리단계 그대로)\n"
    "- `keyword_with_graph_context`: 정확 매칭 없는 메뉴 (예: '김치된장찌개', '마라김치찌개').\n"
    "  **재료 + 조리단계 둘 다 짬뽕**해야 합니다:\n"
    "  ① 재료 = base 메뉴 재료 + modifier 메뉴/재료의 핵심 재료를 합쳐서 표시.\n"
    "  ② 조리단계 = base 메뉴의 steps를 큰 틀로 두되, **modifier 재료를 자연스러운 step에 끼워넣어라.**\n"
    "     예: 된장찌개 step '된장 풀고 호박/감자/양파 넣고 끓이기'\n"
    "         → '된장 풀고 호박/감자/양파/**김치** 넣고 끓이기'\n"
    "     또는 새 step 추가: '4-1. 김치를 같이 넣고 한소끔 더 끓여줍니다 (김치된장찌개 조합)'\n"
    "  ③ base steps의 액션/순서는 보존. 다만 modifier 재료 끼워넣는 건 OK.\n"
    "  ④ 답변 제목에 '(레시피 조합 제안)' 명시.\n"
    "  ⑤ 마지막 줄에 짬뽕 근거 한 줄: "
    "    '※ 김치는 4단계에서 같이 넣는 방식이 자연스럽습니다' 같이.\n"
    "- `graph_relation`: base 메뉴와 modifier만 있을 때. 그래프 컨텍스트로 조합 답변.\n"
    "- `substitute`: **대체 재료 후보 리스트만**. 조리단계/조리법 절대 만들지 마세요. "
    "출력은 후보 재료 이름 + 카테고리 + DB 등장 빈도만. "
    "예시: '돼지고기 대체 후보: 햄(육류·계란, 23회), 참치(어패류, 18회), ...'\n"
    "- `similar`: **유사 레시피 N개**. 각 레시피 이름 + 인분 + 난이도 정도만 짧게. "
    "조리단계는 첫 레시피만 또는 생략. 사용자가 원하면 그 중 골라서 자세히 물어보게.\n"
    "- `popular` / `condition` / `by_ingredient`: 짧게 레시피 리스트.\n"
    "\n"
    # ─── 기타 ───
    "[기타]\n"
    "- 음식·식재료와 무관한 질문(가전·자동차·부동산 등)이면 '식재료 관련만 답변드릴 수 있습니다'라고 안내.\n"
    "- 커피 원두, 밀가루, 설탕, 버터 등 식음료 재료는 식재료로 간주.\n"
    "- 한국어로, 핵심만 간결하게. 불필요한 인사말/주의사항 생략."
)

try:
    SYSTEM_PROMPT = mlflow.genai.load_prompt(
        "prompts:/gold.ai_agent.report_generator@production"
    ).template
except Exception:
    SYSTEM_PROMPT = _FALLBACK_SYSTEM_PROMPT


_STEP_NOISE_RE = re.compile(r'^[\s\d.\-)]*$')
_STEP_NOISE_KEYWORDS = ("손질법 레시피", "레시피 보기", "바로가기", "더보기")


def _clean_steps_for_card(steps_text) -> list:
    """조리단계 문자열 → 노이즈 제거한 리스트 (카드용, 규칙 기반 가벼운 정리)."""
    if not steps_text:
        return []
    out = []
    for raw in str(steps_text).split("\n"):
        s = re.sub(r'^\s*\d+\s*[.)]\s*', '', raw.strip()).strip()
        if not s or _STEP_NOISE_RE.match(s):
            continue
        if re.search(r'[가-힣]', s) is None:
            continue
        if any(k in s for k in _STEP_NOISE_KEYWORDS) or len(s) <= 2:
            continue
        # 괄호 안 개인후기 / 웃음 / 느낌표 남발 정리
        s = re.sub(r'\(\s*(?:전 |저[는도가]|제가)[^)]{0,40}?\)', '', s)
        s = re.sub(r'\s*[ㅋㅎ]{2,}', '', s)
        s = re.sub(r'!{2,}', '!', s).strip()
        if s:
            out.append(s)
    return out


_CARD_SEASONING_HINTS = (
    "고춧가루", "후춧가루", "후추", "설탕", "소금", "간장", "고추장", "된장", "쌈장",
    "참기름", "들기름", "식용유", "포도씨유", "올리브유", "마늘", "다진마늘", "생강",
    "다시다", "미원", "맛술", "미림", "식초", "물엿", "올리고당", "액젓", "굴소스", "깨",
)


def _card_estimate_qty(name: str) -> str:
    """수량 없는 재료의 카드 표시용 추정값. 양념류는 소량, 그 외 적당량."""
    if any(h in name for h in _CARD_SEASONING_HINTS):
        return "약 10g(추정)"
    return "약 80g(추정)"


def _build_card_data(recipe_info: dict, cost_info: dict) -> dict:
    """recipe_info + cost_info에서 카드 렌더용 dict 조립. 정보 없으면 {}."""
    if not isinstance(recipe_info, dict):
        return {}
    recipes_src = recipe_info.get("data") or []
    if not isinstance(recipes_src, list) or not recipes_src:
        return {}

    calc_by_menu = {}
    if isinstance(cost_info, dict):
        for c in (cost_info.get("calc_results") or []):
            if isinstance(c, dict) and c.get("menu"):
                calc_by_menu[c["menu"]] = c

    card_recipes = []
    for recipe in recipes_src:
        if not isinstance(recipe, dict):
            continue
        menu = recipe.get("menu") or recipe.get("name") or "이름없음"
        calc = calc_by_menu.get(menu, {})
        cost_items = {it["name"]: it for it in (calc.get("items") or [])
                      if isinstance(it, dict) and it.get("name")}

        has_cost = bool(calc.get("total_cost"))
        ings_out = []
        for ing in (recipe.get("ingredients") or []):
            if isinstance(ing, dict):
                iname = (ing.get("name") or "").strip()
                iqty = (ing.get("quantity") or ing.get("amount") or "").strip()
            else:
                iname, iqty = str(ing).strip(), ""
            if not iname:
                continue
            ci = cost_items.get(iname, {})

            if not iqty:
                if (ci.get("qty_reason") or "").startswith("default") and ci.get("grams"):
                    iqty = f"약 {int(ci['grams'])}g(추정)"
                else:
                    iqty = _card_estimate_qty(iname)
            ings_out.append({
                "name": iname, "quantity": iqty or None,
                "price_per_kg": ci.get("price_per_kg"), "cost": ci.get("cost"),
                "source": ci.get("source"),
            })

        total = calc.get("total_cost")
        card_recipes.append({
            "menu": menu,
            "servings": recipe.get("servings"),
            "difficulty": recipe.get("difficulty"),
            "cooking_time": recipe.get("cooking_time"),
            "has_cost": has_cost,                   
            "total_cost": total if total else None,
            "suggested_price": int(total / 0.7) if total else None,
            "ingredients": ings_out,
            "steps": _clean_steps_for_card(recipe.get("steps")),
            "substitute": calc.get("substitute"), 
        })

    new_menus = []
    for s in (recipe_info.get("ai_new_menu_suggestions") or []):
        if isinstance(s, dict) and s.get("name"):
            new_menus.append({"name": s.get("name"), "combo_label": s.get("combo_label")})

    card = {"recipes": card_recipes}
    if new_menus:
        card["new_menus"] = new_menus
    return card


async def report_generator_node(state: dict) -> dict:
    """
    수집된 정보(레시피, 가격 등)를 LLM으로 종합하여 자연스러운 최종 답변을 생성합니다.
    async def + ainvoke: astream_events에서 on_chat_model_stream 토큰 이벤트가
    스레드 경계 없이 안정적으로 전파됩니다.
    """
    archive("report_generator.input", {
        "is_valid": state.get("is_valid", False),
        "has_recipe": bool(state.get("recipe_info")),
        "has_price": bool(state.get("price_info")),
        "has_cost": bool(state.get("cost_info")),
        "loop_count": state.get("loop_count"),
    })

    if not state.get("is_valid", False):
        final_answer = "해당 질문은 요리 레시피 및 식재료 원가/가격 조회와 관련이 없습니다. 식당 운영 및 메뉴 원가 관련 질문을 해주세요."
        archive("report_generator.output", {"reason": "not_valid", "answer_preview": final_answer[:200]})
        return {
            "final_report": final_answer,
            "messages": [AIMessage(content=final_answer)]
        }

    recipe_info = state.get("recipe_info", {})
    price_info = state.get("price_info", {})
    entities = state.get("entities", {})
    user_query = _get_last_human_query(state.get("messages", []))
    rewritten_query = state.get("rewritten_query", "")
    intent = entities.get("intent", "")

    cost_info = state.get("cost_info", {})

    #LLM이 읽기 좋은 형태로 정리
    context_parts = []

    #preprocessor 분석 결과를 맨 앞에 배치
    if rewritten_query and rewritten_query != user_query:
        context_parts.append(f"[분석된 의도]\n질문 해석: {rewritten_query}\n의도 유형: {intent}")

    if recipe_info and recipe_info.get("data"):
        # 레시피별로 분리해서 컨텍스트 구성. steps는 별도 섹션으로 강조.
        recipes_data = recipe_info["data"]
        if isinstance(recipes_data, list) and recipes_data:
            recipe_blocks = []
            for idx, recipe in enumerate(recipes_data, 1):
                if not isinstance(recipe, dict):
                    recipe_blocks.append(f"[레시피 {idx}]\n{recipe}")
                    continue
                lines = [f"[레시피 {idx}: {recipe.get('menu', '이름없음')}]"]
                for k, v in recipe.items():
                    if k in ("steps", "substitute_suggestions"):
                        continue
                    if v is None or v == "":
                        continue
                    lines.append(f"- {k}: {v}")
                recipe_blocks.append("\n".join(lines))

                subs = recipe.get("substitute_suggestions")
                if subs and isinstance(subs, dict) and subs.get("candidates"):
                    target = subs.get("target", "주재료")
                    cand_lines = []
                    for c in subs["candidates"]:
                        if isinstance(c, dict):
                            cname = c.get("name") or c.get("ingredient") or ""
                            cfreq = c.get("freq") or c.get("count") or c.get("frequency")
                            ccat = c.get("lv1") or c.get("category") or ""
                            extra = []
                            if ccat:
                                extra.append(str(ccat))
                            if cfreq:
                                extra.append(f"{cfreq}회")
                            suffix = f" ({', '.join(extra)})" if extra else ""
                            cand_lines.append(f"{cname}{suffix}")
                        else:
                            cand_lines.append(str(c))
                    if cand_lines:
                        recipe_blocks.append(
                            f"[레시피 {idx} 주재료 '{target}' 대체 후보 — "
                            f"'💡 비싼 재료 → 대체 제안' 섹션에 활용 가능 (가격 정보 있으면 절감액도)]\n"
                            + ", ".join(cand_lines)
                        )

                steps = recipe.get("steps")
                if steps:
                    recipe_blocks.append(
                        f"[레시피 {idx} 조리단계 — 아래는 DB 원문(블로그 말투 포함).\n"
                        f"★★ 이 원문을 '레시피 책'처럼 깔끔한 명령형 단계로 **반드시 다시 써라.** "
                        f"핵심(조리 동작·재료·수량·시간·순서)은 100% 보존하되, 블로그 수다는 **남기지 말고 전부 삭제.**\n"
                        f"  ▸ 반드시 삭제: \n"
                        f"     · 괄호 안 개인 후기 — 예: '(전 다 사용했어요!)', '(저는 ~)' → 통째로 삭제\n"
                        f"     · 개인 경험담 — '저같은 경우는~', '이번에는 ~없어서', '저는 4번에 나눠 부었어요'\n"
                        f"     · 권유·잡담·말장난 — '~좋지요?', '~맛있죠?', '아시죠?', 'ㅋㅋ', 'ㅎㅎ', '!!', '식탁으로~'\n"
                        f"     · 인사·사족 — '~답니다', '~거든요', '참고하세요', '예쁘게 담아주세요'\n"
                        f"     · 의미 없는 짧은 파편 — '불끄기3분전', '한쪽으로 밀어두고' 처럼 동작이 불완전한 조각은 "
                        f"앞뒤 단계에 합치거나, 정 안 되면 삭제\n"
                        f"  ▸ 명령형으로: '~해주세요/~합니다' 정도로 통일. 각 단계는 한 동작.\n"
                        f"  ▸ 절대 새 단계를 지어내거나 외부 지식으로 보완하지 말 것(환각 금지). 원문 동작만 남겨 다듬는다.\n{steps}"
                    )
                else:
                    recipe_blocks.append(
                        f"[레시피 {idx} 조리단계 정보 없음 — LLM 일반 지식으로 보완하되 '(LLM 일반 지식, DB 미수록)' 표기 필수]"
                    )
            context_parts.append("[레시피/재료 정보]\n\n" + "\n\n".join(recipe_blocks))
        else:
            context_parts.append(f"[레시피/재료 정보]\n{recipes_data}")

        graph_context = recipe_info.get("graph_context")
        if graph_context:
            original_query = recipe_info.get("original_query", "조합 메뉴")
            graph_lines = [
                "[그래프 RAG 컨텍스트 — 정확 매칭 없는 메뉴를 base + modifier로 분해한 정보]",
                "",
                f"** 사용자가 '{original_query}' 라고 요청했으나 DB에 그 정확한 이름이 없음.**",
                "** 위에 있는 [레시피/재료 정보](= base 메뉴)와 아래 [추가 메뉴의 대표 재료](= modifier)를",
                "   짬뽕해서 답변하세요. **",
                "",
                "**짬뽕 룰 (재료 + 조리단계 둘 다):**",
                "1. 재료: base 메뉴 재료 전체 + modifier의 핵심 재료(예: 김치, 김치국물)를 같이 표시.",
                "2. 조리단계: base steps 큰 틀 유지. modifier 재료를 자연스러운 step에 끼워넣어라.",
                "   예: '된장 풀고 호박/감자/양파 넣기' → '된장 풀고 호박/감자/양파/김치 넣기'",
                "3. 답변 제목: '(레시피 조합 제안)' 표시.",
                "4. 마지막 줄에 짬뽕 근거: '※ 김치는 4단계에서 같이 넣는 게 자연스럽습니다' 같이.",
                "",
                "[관계 정보]",
            ]
            for block in graph_context:
                if isinstance(block, dict):
                    for line in block.get("ingredients", []):
                        graph_lines.append(line)
            context_parts.append("\n".join(graph_lines))

        ai_suggestions = recipe_info.get("ai_new_menu_suggestions")
        if ai_suggestions and isinstance(ai_suggestions, list):
            sugg_lines = [
                "[AI 신메뉴 제안 후보 — '✨ AI 신메뉴 제안' 섹션 카드로 출력하세요]",
                "각 카드: 신메뉴명 + (base 메뉴 + modifier 조합) + 트렌드 코멘트 한 줄.",
                "가격/마진 정보가 따로 없으면 판매가·마진율 줄은 생략하고 조합 컨셉만 제시.",
                "",
            ]
            for i, s in enumerate(ai_suggestions, 1):
                if not isinstance(s, dict):
                    continue
                name = s.get("name", "")
                combo = s.get("combo_label", "")
                sugg_lines.append(f"#{i}. {name}  (조합: {combo})")
            context_parts.append("\n".join(sugg_lines))
    if price_info:
        formatted_price = _format_price_info(price_info)
        context_parts.append(f"[가격/시세 정보]\n{formatted_price}")
    if cost_info and cost_info.get("analysis"):
        context_parts.append(f"[원가 분석]\n{cost_info['analysis']}")

    context = "\n\n".join(context_parts) if context_parts else "관련 정보를 찾지 못했습니다."

    user_prompt = f"""사용자 질문: {user_query}

{context}

위 정보를 활용하여 답변하세요. [분석된 의도]가 있으면 그 해석에 맞춰 답변의 초점을 잡으세요. 가격 데이터가 있으면 구체적인 수치(원/kg, 등급 등)를 반드시 포함하세요."""

    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        # ainvoke: async 컨텍스트에서 직접 실행되므로 콜백이 스레드 경계 없이 전파됨
        response = await _get_llm_report().ainvoke(messages)
        final_answer = response.content

        final_answer = re.sub(r'(\d+)\s*~\s*(\d+)', r'\1∼\2', final_answer)

        final_answer = re.sub(r'\(\s*(?:전 |저[는도가]|제가)[^)]{0,40}?\)', '', final_answer)
        final_answer = re.sub(r'\s*[ㅋㅎ]{2,}', '', final_answer)
        final_answer = re.sub(r'!{2,}', '!', final_answer)

        check_result = _check_hallucination(final_answer, context, intent)
        if check_result == "block":
            final_answer = _BLOCK_FALLBACK
            archive("report_generator.output", {"reason": "hallucination_blocked", "answer_preview": final_answer})
            return {
                "final_report": final_answer,
                "messages": [AIMessage(content=final_answer)]
            }
    except Exception as e:
        final_answer = f"요청하신 {entities} 관련 답변입니다.\n\n"
        if recipe_info and recipe_info.get("data"):
            final_answer += f"[레시피 정보]\n{recipe_info['data']}\n\n"
        if price_info:
            final_answer += f"[가격 정보]\n{_format_price_info(price_info)}\n\n"
        final_answer += f"\n(자연어 요약 생성 중 오류 발생: {e})"

    try:
        card_data = _build_card_data(recipe_info, cost_info)
    except Exception as _card_exc:
        card_data = {}
        archive("report_generator.card_error", {"error": str(_card_exc)})

    archive("report_generator.output", {
        "answer_preview": final_answer[:400],
        "answer_length": len(final_answer),
        "has_card": bool(card_data.get("recipes")) if card_data else False,
    })
    return {
        "final_report": final_answer,
        "card_data": card_data,
        "messages": [AIMessage(content=final_answer)]
    }
