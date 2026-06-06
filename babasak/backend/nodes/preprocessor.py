import re

import mlflow.genai
from langchain_core.messages import HumanMessage, SystemMessage

from backend.debug_log import archive

MENU_NAMES = [
    '김치찌개', '된장찌개', '불고기', '순두부찌개', '비빔밥',
    '제육볶음', '김치전', '떡볶이', '잡채', '오므라이스',
    '김밥', '부대찌개', '미역국', '소불고기덮밥', '치즈돈까스',
    '파전', '치킨카레', '삼계탕', '갈비탕', '냉면',
    '콩나물국', '마라탕', '부침개', '돈까스', '어묵국',
]

INGREDIENT_NAMES = [
    '돼지고기', '소고기', '닭고기', '두부', '김치',
    '양파', '대파', '감자', '당근', '애호박',
    '버섯', '고추', '마늘', '계란', '우유',
    '밀가루', '설탕', '간장', '된장', '고춧가루',
    '참기름', '식용유', '소금', '후추', '새우',
]

MENU_ALIASES = {
    '김찌': '김치찌개', '된찌': '된장찌개', '순찌': '순두부찌개',
    '부찌': '부대찌개', '김치찌게': '김치찌개', '된장찌게': '된장찌개',
    '돈까쓰': '돈까스', '떡볶히': '떡볶이', '제육': '제육볶음',
    '비빔': '비빔밥', '삼겹': '불고기', '순대국': '순두부찌개',
}


_INTENT_KEYWORDS = {
    'cost_analysis': ['원가', '마진', '판매가', '비용', '수익', '1인분 가격', '단가', '가격 책정'],
    'price_inquiry': ['도매가', '시세', '도매 시세', '식자재 가격'],
    'recipe_only': ['만드는 법', '조리법', '레시피', '만들기', '요리법', '조리 순서'],
    'alternative': ['대신', '대체', '바꿀', '대용', '다른 걸로'],
    'recommendation': ['추천', '뭐 먹', '어떤 게 좋'],
    'analytics': ['가격 변동', '변동폭', '가격 추이', '가장 비싼', '가장 싼', '가격 순위',
                  '가격 비교', '최고가', '최저가', '가격 상승', '가격 하락',
                  '얼마나 올랐', '얼마나 내렸', '많이 오른', '많이 내린'],
}


_INVALID_KEYWORDS = ['날씨', '주식', '비트코인', '뉴스', '영화', '여행', '수학', '코드', '번역', '노래', '게임', '택시']


_RATE_VALUE_RE = r"(?P<value>\d+(?:\.\d+)?)\s*(?:%|퍼센트|프로)"
_MARGIN_RATE_PATTERNS = [
    re.compile(r"(?:마진율|마진|수익률|이익률|매출총이익률)\s*(?:은|는|을|를|로|:)?\s*" + _RATE_VALUE_RE, re.IGNORECASE),
    re.compile(_RATE_VALUE_RE + r"\s*(?:마진율|마진|수익률|이익률|매출총이익률)", re.IGNORECASE),
]
_COST_RATIO_PATTERNS = [
    re.compile(r"(?:원가율|식재료비율|재료비율|푸드코스트|food\s*cost|cost\s*ratio)\s*(?:은|는|을|를|로|:)?\s*" + _RATE_VALUE_RE, re.IGNORECASE),
    re.compile(_RATE_VALUE_RE + r"\s*(?:원가율|식재료비율|재료비율|푸드코스트|food\s*cost|cost\s*ratio)", re.IGNORECASE),
]


def _rate_to_decimal(raw_value) -> float | None:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if value > 1:
        value = value / 100
    if 0 < value < 0.95:
        return round(value, 4)
    return None


def _detect_pricing_policy(query: str) -> dict:
    """Extract user pricing intent from text."""
    matches = []
    for source, patterns in (("user_margin", _MARGIN_RATE_PATTERNS), ("user_cost_ratio", _COST_RATIO_PATTERNS)):
        for pattern in patterns:
            for match in pattern.finditer(query or ""):
                rate = _rate_to_decimal(match.group("value"))
                if rate is None:
                    continue
                matches.append((match.start(), source, rate, match.group(0).strip()))

    if not matches:
        return {
            "target_margin_rate": None,
            "target_cost_ratio": None,
            "pricing_source": None,
            "pricing_text": None,
        }

    _, source, rate, pricing_text = sorted(matches, key=lambda item: item[0])[-1]
    if source == "user_margin":
        margin_rate = rate
        cost_ratio = round(1 - rate, 4)
    else:
        cost_ratio = rate
        margin_rate = round(1 - rate, 4)

    return {
        "target_margin_rate": margin_rate,
        "target_cost_ratio": cost_ratio,
        "pricing_source": source,
        "pricing_text": pricing_text,
    }


def _keyword_intent(query: str) -> str | None:
    """키워드 매칭으로 intent 1차 감지. 명확하면 LLM 스킵 가능."""
    priority = ['analytics', 'alternative', 'cost_analysis', 'price_inquiry', 'recipe_only', 'recommendation']
    for intent_name in priority:
        for kw in _INTENT_KEYWORDS[intent_name]:
            if kw in query:
                return intent_name
    return None


def _match_menu_from_query(query: str) -> str | None:
    """문자열 매칭으로 메뉴 추출 (정확명 → ALIASES 순서)"""
    for m in MENU_NAMES:
        if m in query:
            return m
    for alias, real in MENU_ALIASES.items():
        if alias in query:
            return real
    return None


def _match_ingredients_from_query(query: str) -> list[str]:
    """문자열 매칭으로 재료 추출 (복수 가능)"""
    return [ing for ing in INGREDIENT_NAMES if ing in query]


def _is_clearly_invalid(query: str) -> bool:
    """명백히 무관한 질문은 LLM 호출 없이 빠르게 거부"""
    return any(kw in query for kw in _INVALID_KEYWORDS)



_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from databricks_langchain import ChatDatabricks

        _llm = ChatDatabricks(endpoint="databricks-gpt-5-4-mini", temperature=0, max_tokens=200)
    return _llm


_FALLBACK_SYSTEM_PROMPT = """당신은 식당/요리 질문 분석 전문가입니다.
사용자의 질문을 분석하여 아래 형식으로 답하세요.

[분석 절차]
1. 질문을 3가지 관점으로 재해석합니다 (모호함 해소).
2. 가장 적합한 해석을 선택합니다.
3. 선택한 해석에서 정보를 추출합니다.

[출력 형식 — 반드시 이 형식만 사용]
해석1: <재해석 문장>
해석2: <재해석 문장>
해석3: <재해석 문장>
선택: <1|2|3>
유효: <예|아니오>
의도: <recipe_only|cost_analysis|price_inquiry|alternative|recommendation|analytics|general>
메뉴: <음식이름 또는 NONE>
재료: <재료1, 재료2 또는 NONE>
제외: <제외할 재료 또는 NONE>
조건: <난이도/인분/종류/조리법 또는 NONE>

[의도 분류 기준]
- recipe_only: 레시피, 만드는 법, 조리법, 재료 목록만 원할 때
- cost_analysis: 원가, 마진, 비용, 가격 책정, 수익률 분석
- price_inquiry: 특정 재료의 시세/도매가만 단순 조회
- alternative: 대체재, 다른 재료로 바꾸기
- recommendation: 메뉴 추천, 뭐 먹을까
- analytics: 가격 변동·추이·순위·비교 등 DB 전체 분석형 질문 (특정 메뉴/재료 지정 없음, 예: '가격이 가장 많이 오른 채소는?')
- general: 위 어디에도 해당 안 됨

[규칙]
- 식재료/요리/식당 운영과 무관한 질문이면 유효=아니오, 나머지는 NONE으로 채우세요.
- 줄임말/오타를 정확한 음식명으로 변환하세요 (예: 김찌→김치찌개, 제육→제육볶음).
- 재료가 여러 개면 쉼표로 구분하세요.

[예시]
질문: 김찌 좀 싸게 해먹으려면
해석1: 김치찌개를 저렴한 재료로 만드는 법
해석2: 김치찌개 원가를 절감하는 방법
해석3: 김치찌개 재료 중 저렴한 대체재 추천
선택: 2
유효: 예
의도: cost_analysis
메뉴: 김치찌개
재료: NONE
제외: NONE
조건: NONE

질문: 비 오는 날 간단한 거
해석1: 비 오는 날 먹기 좋은 간단한 메뉴 추천
해석2: 간단하게 만들 수 있는 탕/국 종류 추천
해석3: 비 오는 날 인기 있는 분식 메뉴
선택: 1
유효: 예
의도: recommendation
메뉴: NONE
재료: NONE
제외: NONE
조건: 초급"""

try:
    _UNIFIED_SYSTEM_PROMPT = mlflow.genai.load_prompt(
        "prompts:/gold.ai_agent.preprocessor@production"
    ).template
except Exception:
    _UNIFIED_SYSTEM_PROMPT = _FALLBACK_SYSTEM_PROMPT


def _llm_multiquery_extract(query: str) -> dict:
    """
    멀티쿼리 리트리버: 사용자 질문을 3가지로 재해석한 뒤
    가장 적합한 해석에서 엔티티를 추출.

    반환: {
        'is_valid': bool,
        'intent': str,
        'menu': str|None,
        'ingredients': list[str],
        'exclude': str|None,
        'conditions': dict|None,
        'rewritten': str|None,
        'interpretations': list[str],
    }
    """
    default = {
        'is_valid': True, 'intent': 'general', 'menu': None,
        'ingredients': [], 'exclude': None, 'conditions': None,
        'rewritten': None, 'interpretations': [],
    }

    try:
        messages = [
            SystemMessage(content=_UNIFIED_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        response = _get_llm().invoke(messages)
        text = response.content.strip()

        archive("preprocessor.multiquery_raw", {"query": query, "response": text})

        # ── 파싱 ──
        result = dict(default)
        interpretations = []
        selected_idx = 0

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("해석1:") or line.startswith("해석2:") or line.startswith("해석3:"):
                interpretations.append(line.split(":", 1)[1].strip())
            elif line.startswith("선택:"):
                try:
                    selected_idx = int(line.split(":")[1].strip()) - 1
                except (ValueError, IndexError):
                    pass
            elif line.startswith("유효:"):
                val = line.split(":")[1].strip()
                result['is_valid'] = val in ('예', '네', 'yes', 'true')
            elif line.startswith("의도:"):
                val = line.split(":")[1].strip().lower()
                valid_intents = ['recipe_only', 'cost_analysis', 'price_inquiry',
                                 'alternative', 'recommendation', 'analytics', 'general']
                result['intent'] = val if val in valid_intents else 'general'
            elif line.startswith("메뉴:"):
                val = line.split(":", 1)[1].strip()
                if val and val.upper() != "NONE" and len(val) <= 20:
                    result['menu'] = val
            elif line.startswith("재료:"):
                val = line.split(":", 1)[1].strip()
                if val and val.upper() != "NONE":
                    result['ingredients'] = [
                        ing.strip() for ing in val.split(",")
                        if ing.strip() and len(ing.strip()) <= 10
                    ]
            elif line.startswith("제외:"):
                val = line.split(":", 1)[1].strip()
                if val and val.upper() != "NONE":
                    result['exclude'] = val
            elif line.startswith("조건:"):
                val = line.split(":", 1)[1].strip()
                if val and val.upper() != "NONE":
                    result['conditions'] = _parse_conditions(val)

        result['interpretations'] = interpretations
        if 0 <= selected_idx < len(interpretations):
            result['rewritten'] = interpretations[selected_idx]

        archive("preprocessor.multiquery_parsed", {
            "query": query,
            "is_valid": result['is_valid'],
            "intent": result['intent'],
            "menu": result['menu'],
            "ingredients": result['ingredients'],
            "num_interpretations": len(interpretations),
            "selected": result['rewritten'],
        })

        return result

    except Exception as e:
        archive("preprocessor.multiquery_error", {"query": query, "error": str(e)})
        return default


def _parse_conditions(text: str) -> dict | None:
    """조건 텍스트를 구조화된 dict로 변환."""
    conditions = {}
    text_lower = text.lower()

    # 난이도
    if '초급' in text or '쉬운' in text or '간단' in text:
        conditions['difficulty'] = '초급'
    elif '중급' in text:
        conditions['difficulty'] = '중급'
    elif '고급' in text or '어려운' in text:
        conditions['difficulty'] = '고급'

    # 인분
    for s in ['1인분', '2인분', '3인분', '4인분', '5인분', '6인분이상']:
        if s in text:
            conditions['servings'] = s
            break

    # 종류
    kind_map = {'국/탕': '국/탕', '탕': '국/탕', '반찬': '메인반찬', '볶음': '볶음', '구이': '구이', '면': '면/만두'}
    for k, v in kind_map.items():
        if k in text:
            conditions['kind'] = v
            break

    return conditions if conditions else None




def _ground_menu(menu: str | None) -> str | None:
    """LLM이 추출한 메뉴명을 알려진 이름으로 정규화."""
    if not menu:
        return None
    # 정확히 알려진 메뉴면 그대로
    if menu in MENU_NAMES:
        return menu
    # ALIASES에 있으면 정규화
    if menu in MENU_ALIASES:
        return MENU_ALIASES[menu]
    # 부분 매칭 시도 (LLM이 "김치찌개요리" 같이 길게 쓸 수 있음)
    for m in MENU_NAMES:
        if m in menu or menu in m:
            return m
    # 알려지지 않은 메뉴도 허용 (Neo4j에 있을 수 있음)
    return menu


def _ground_ingredients(ingredients: list[str]) -> list[str]:
    """LLM이 추출한 재료명을 알려진 이름으로 정규화."""
    if not ingredients:
        return []
    grounded = []
    for ing in ingredients:
        # 정확히 알려진 재료면 그대로
        if ing in INGREDIENT_NAMES:
            grounded.append(ing)
            continue
        # 부분 매칭
        matched = False
        for known in INGREDIENT_NAMES:
            if known in ing or ing in known:
                grounded.append(known)
                matched = True
                break
        if not matched:
            grounded.append(ing)  # 알려지지 않은 재료도 허용
    return grounded


def _carry_from_history(messages: list) -> tuple[str | None, list[str]]:
    """이전 HumanMessage에서 메뉴/재료를 상속. AIMessage는 제외."""
    menu = None
    ingredients = []

    for prev_msg in reversed(messages[:-1]):
        if not isinstance(prev_msg, HumanMessage):
            continue
        prev_text = getattr(prev_msg, 'content', '') or ''
        prev_menu = _match_menu_from_query(prev_text)
        if prev_menu:
            menu = prev_menu
            archive("preprocessor.history_carry", {"menu": menu, "from": prev_text[:80]})
            break
        prev_ings = _match_ingredients_from_query(prev_text)
        if prev_ings:
            ingredients = prev_ings
            archive("preprocessor.history_carry", {"ingredients": prev_ings, "from": prev_text[:80]})
            break

    return menu, ingredients


import re as _re

_SIMILAR_KEYWORDS = ["비슷한", "유사한", "닮은", "비슷하게", "같은 종류", "유사", "비슷"]

# "대체 메뉴/음식/요리/레시피" 또는 "대신 먹을/할/만들" = 다른 메뉴 추천 (유사 레시피)
_SIMILAR_PHRASE_PAT = _re.compile(
    r'대체\s*(?:메뉴|음식|요리|레시피|것|거)|대신\s*(?:먹|할|만들|있는)|다른\s*(?:메뉴|음식|요리|레시피)'
)

# "X 대신 Y" / "X 대체" / "X 빼고 뭐" 같은 패턴에서 X(대체할 재료) 추출
# 단, "대체 메뉴" 같은 경우는 제외 (재료가 아니라 메뉴 추천 요청)
_SUBSTITUTE_PAT = _re.compile(
    r'([가-힣]{2,10})\s*(?:대신|대체|말고|없으면|빼고)',
)


def _detect_is_similar(query: str) -> bool:
    if not query:
        return False
    if any(kw in query for kw in _SIMILAR_KEYWORDS):
        return True
    # "대체 메뉴", "대신 먹을", "다른 요리" 같은 표현도 유사 추천 의도로
    if _SIMILAR_PHRASE_PAT.search(query):
        return True
    return False


def _detect_substitute_for(query: str, ingredients=None) -> str | None:
    """'돼지고기 대신 뭐 써?' → '돼지고기' 추출.
    1순위: 정규식 패턴 매칭, 2순위: 알려진 재료 + 대체 키워드 동시 등장.
    단, '대체 메뉴/음식/요리' 같은 메뉴 추천 요청에서는 None 반환 (시나리오 9 진입 차단).
    """
    if not query:
        return None
    # "대체 메뉴"는 substitute_for가 아니라 is_similar로 처리되도록
    if _SIMILAR_PHRASE_PAT.search(query):
        return None
    m = _SUBSTITUTE_PAT.search(query)
    if m:
        return m.group(1).strip()
    # fallback: ingredients 중 하나 + 대체 키워드 동시 등장
    if ingredients:
        has_kw = any(kw in query for kw in ["대신", "대체", "빼고", "없으면"])
        if has_kw:
            return ingredients[0]
    return None


def _build_result(*, is_valid, menu, ingredients, intent, query,
                  exclude=None, conditions=None,
                  is_alternative=False, is_popular=False,
                  rewritten=None) -> dict:
    """표준 출력 형식으로 결과 조립."""
    # 신규: 유사 추천 / 1:1 대체재 의도 감지 (recipe_search.py 시나리오 8/9용)
    is_similar = _detect_is_similar(query)
    substitute_for = _detect_substitute_for(query, ingredients)
    pricing_policy = _detect_pricing_policy(query)

    entities = {
        'menu': menu,
        'ingredient': ingredients if ingredients else None,
        'intent': intent,
        'exclude': exclude,
        'is_alternative': is_alternative,
        'conditions': conditions,
        'is_popular': is_popular,
        # 신규 키
        'is_similar': is_similar,
        'reference_menu': menu if is_similar else None,
        'substitute_for': substitute_for,
        'target_margin_rate': pricing_policy["target_margin_rate"],
        'target_cost_ratio': pricing_policy["target_cost_ratio"],
        'pricing_source': pricing_policy["pricing_source"],
        'pricing_text': pricing_policy["pricing_text"],
    }

    # rewritten_query: LLM 자연어 재해석 우선, 없으면 기계적 조합
    if rewritten:
        rewritten_query = rewritten
    elif menu and intent:
        rewritten_query = f'{menu} {intent}'
    elif ingredients and intent:
        rewritten_query = f'{", ".join(ingredients)} {intent}'
    elif intent:
        rewritten_query = f'{intent} 조회'
    else:
        rewritten_query = query

    return {
        'is_valid': is_valid,
        'entities': entities,
        'rewritten_query': rewritten_query,
    }


def preprocessor_node(state: dict) -> dict:
    """
    멀티쿼리 리트리버 기반 전처리 노드.

    Phase 1: LLM 멀티쿼리 (1순위) — 3가지 재해석 + 통합 엔티티/의도 추출
    Phase 2: Grounding — LLM 출력을 알려진 DB 엔티티에 정규화
    Phase 3: 히스토리 carry-over — LLM이 엔티티 못 찾았을 때 이전 대화에서 상속

    ※ 명백히 무관한 질문만 빠르게 거부 (is_valid=False)
    """
    query = state['messages'][-1].content
    archive("preprocessor.input", {"query": query})

    # ═══ 사전 필터: 명백히 무관한 질문은 LLM 호출 없이 즉시 거부 ═══
    if _is_clearly_invalid(query):
        archive("preprocessor.output", {"phase": "pre_invalid", "query": query})
        return {
            'is_valid': False,
            'entities': {'menu': None, 'ingredient': None, 'intent': 'general',
                         'exclude': None, 'is_alternative': False,
                         'conditions': None, 'is_popular': False,
                         'is_similar': False, 'reference_menu': None,
                         'substitute_for': None,
                         'target_margin_rate': None, 'target_cost_ratio': None,
                         'pricing_source': None, 'pricing_text': None},
            'rewritten_query': '식당 운영/메뉴/원가 관련 질문이 아닙니다',
        }

    # ═══ Phase 1: LLM 멀티쿼리 리트리버 (1순위, ~1.5초) ═══
    # 사용자 질문을 3가지로 재해석하여 정확한 의도 파악
    llm_result = _llm_multiquery_extract(query)

    if not llm_result['is_valid']:
        archive("preprocessor.output", {"phase": "1_invalid", "query": query})
        return {
            'is_valid': False,
            'entities': {'menu': None, 'ingredient': None, 'intent': 'general',
                         'exclude': None, 'is_alternative': False,
                         'conditions': None, 'is_popular': False,
                         'is_similar': False, 'reference_menu': None,
                         'substitute_for': None,
                         'target_margin_rate': None, 'target_cost_ratio': None,
                         'pricing_source': None, 'pricing_text': None},
            'rewritten_query': '식당 운영/메뉴/원가 관련 질문이 아닙니다',
        }

    # ═══ Phase 2: Grounding (0ms) ═══
    # LLM 출력을 알려진 DB 엔티티에 맞춰 정규화
    menu = _ground_menu(llm_result['menu'])
    ingredients = _ground_ingredients(llm_result['ingredients'])
    intent = llm_result['intent']
    exclude = llm_result.get('exclude')
    conditions = llm_result.get('conditions')
    is_alternative = (intent == 'alternative')
    is_popular = any(kw in query for kw in ['인기', 'top', 'best', '추천순', '랭킹'])

    # analytics: 메뉴/재료 추출 없이 원문 그대로 Genie freeform 경로로 전달
    # price_search_node는 ingredients=[] + menu=None 이면 _ask_genie(user_query) freeform path를 탄다
    if intent == 'analytics':
        menu = None
        ingredients = None
    else:
        # 문자열 매칭으로 LLM이 놓친 엔티티 보강 (보조 역할)
        if not menu:
            menu = _match_menu_from_query(query)
        if not ingredients:
            fast_ings = _match_ingredients_from_query(query)
            if fast_ings:
                ingredients = fast_ings
        if not intent or intent == 'general':
            kw_intent = _keyword_intent(query)
            if kw_intent:
                intent = kw_intent

    # ═══ Phase 3: 히스토리 carry-over (0ms, 최후 수단) ═══
    # LLM도 메뉴/재료를 못 찾았을 때만 이전 HumanMessage에서 상속.
    # 단, '대체/대신' 질문은 재료(돼지고기)는 있어도 메뉴 맥락(제육볶음)을 이어받아야
    # 대체재가 메뉴 기준으로 나온다. ("제육볶음 원가" → "돼지고기 대체재료")
    _is_sub_query = bool(_SUBSTITUTE_PAT.search(query)) and not _SIMILAR_PHRASE_PAT.search(query)
    if (not menu and not ingredients) or (not menu and _is_sub_query):
        hist_menu, hist_ings = _carry_from_history(state['messages'])
        if hist_menu:
            menu = hist_menu
        if hist_ings and not ingredients:
            ingredients = hist_ings

    # ═══ Fallback 감지: 모든 단계에서 엔티티 추출 실패 ═══
    if not menu and not ingredients and intent == 'general':
        archive("preprocessor.output", {
            "phase": "fallback", "query": query,
            "reason": "no_entities_extracted",
        })
        return _build_result(
            is_valid=True, menu=None, ingredients=None,
            intent='fallback', query=query, exclude=None,
            conditions=None, is_alternative=False, is_popular=False,
            rewritten="질문을 정확히 이해하지 못했습니다. 메뉴명이나 재료명을 포함해서 다시 질문해주세요.",
        )

    # ═══ 결과 반환 ═══
    archive("preprocessor.output", {
        "phase": "1_llm",
        "menu": menu, "ingredients": ingredients, "intent": intent,
        "exclude": exclude, "rewritten": llm_result.get('rewritten'),
        "pricing": _detect_pricing_policy(query),
        "interpretations": llm_result.get('interpretations', []),
    })

    return _build_result(
        is_valid=True, menu=menu, ingredients=ingredients,
        intent=intent, query=query, exclude=exclude,
        conditions=conditions, is_alternative=is_alternative,
        is_popular=is_popular,
        rewritten=llm_result.get('rewritten'),
    )

