이 
import numpy as np
from databricks_langchain import DatabricksEmbeddings

from backend.debug_log import archive

# ── Qwen3 한국어 임베딩 모델 (lazy init) ──
_emb_ko = None
_indices = None

VALID_PATTERNS = [
    '메뉴 원가 알려줘', '재료 가격', '1인분 비용',
    '레시피 재료', '식재료 시세', '도매가 조회',
    '대체재 추천', '원가 분석', '메뉴 가격 책정',
    '재료 용량', '식당 운영 비용', '마진율 계산',
    '음식 만드는 법', '조리법 알려줘', '식재료 어디서 사',
    '나물 가격', '재료 비교', '메뉴 추천',
    '원가 절감', '대량 구매 가격', '인건비 포함 원가',
]

INVALID_PATTERNS = [
    '날씨 어때', '주식 시세', '비트코인 가격',
    '뉴스 알려줘', '영화 추천', '여행지 추천',
    '수학 문제 풀어줘', '코드 작성해줘', '번역해줘',
    '노래 가사', '게임 추천', '택시 불러줘',
]

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

# 줄임말/오타 → 정규 메뉴명 매핑
MENU_ALIASES = {
    '김찌': '김치찌개', '된찌': '된장찌개', '순찌': '순두부찌개',
    '부찌': '부대찌개', '김치찌게': '김치찌개', '된장찌게': '된장찌개',
    '돈까쓰': '돈까스', '떡볶히': '떡볶이', '제육': '제육볶음',
    '비빔': '비빔밥', '삼겹': '불고기', '순대국': '순두부찌개',
}

INTENT_PATTERNS = {
    '원가': ['원가 알려줘', '얼마야', '가격 얼마', '비용 알려줘', '단가 얼마'],
    '재료': ['재료 알려줘', '뭐 필요해', '재료 뭐뭐', '레시피 알려줘'],
    '대체재': ['대신 쓸 수 있는', '대체할', '바꿀 수 있는'],
    '가격조회': ['가격 알려줘', '도매가', '시세', '얼마인지'],
}

intent_texts = []
intent_labels = []
for label, patterns in INTENT_PATTERNS.items():
    intent_texts.extend(patterns)
    intent_labels.extend([label] * len(patterns))

VALID_THRESHOLD_KO = 0.38
ENTITY_THRESHOLD_KO = 0.75  # 직접 언급 없을 때 임베딩 기반 threshold (높게 유지)

# ── 제외/대체재 의도 키워드 ──
_EXCLUDE_KEYWORDS = ['제외', '빼고', '없는', '알레르기', '못 먹', '빼줘', '없이']
_ALTERNATIVE_KEYWORDS = ['대신', '대체', '바꿀', '대용', '다른 걸로']

# ── 조건 매핑 (기존 agent.py에서 포팅) ──
KIND_MAP = {
    '국/탕': '국/탕', '탕': '국/탕',
    '반찬': '메인반찬', '밑반찬': '밑반찬', '디저트': '디저트',
    '면': '면/만두', '볶음': '볶음', '구이': '구이',
}
# 주의: '찌개', '국' 같은 너무 일반적인 단어는 메뉴명에 포함되므로 조건 매핑에서 제외
# ('김치찌개'는 메뉴명이지 조건이 아님)

METHOD_MAP = {
    '볶음': '볶기', '끓이': '끓이기', '굽': '굽기',
    '찜': '찌기', '튀김': '튀기기',
}


def _get_emb():
    """임베딩 모델 lazy init"""
    global _emb_ko
    if _emb_ko is None:
        _emb_ko = DatabricksEmbeddings(endpoint='databricks-qwen3-embedding-0-6b')
    return _emb_ko


def _get_indices() -> dict:
    """
    인덱스를 lazy하게 빌드하고 캐시.
    [최적화] 모든 텍스트를 하나로 합쳐 1회 embed_documents 호출 후 슬라이싱.
    """
    global _indices
    if _indices is None:
        emb = _get_emb()
        all_texts = VALID_PATTERNS + INVALID_PATTERNS + MENU_NAMES + INGREDIENT_NAMES + intent_texts
        all_vecs = np.array(emb.embed_documents(all_texts))

        i = 0
        valid_vecs = all_vecs[i:i + len(VALID_PATTERNS)]; i += len(VALID_PATTERNS)
        invalid_vecs = all_vecs[i:i + len(INVALID_PATTERNS)]; i += len(INVALID_PATTERNS)
        menu_vecs = all_vecs[i:i + len(MENU_NAMES)]; i += len(MENU_NAMES)
        ingredient_vecs = all_vecs[i:i + len(INGREDIENT_NAMES)]; i += len(INGREDIENT_NAMES)
        intent_vecs = all_vecs[i:i + len(intent_texts)]

        _indices = {
            'valid': valid_vecs,
            'invalid': invalid_vecs,
            'menu': menu_vecs,
            'ingredient': ingredient_vecs,
            'intent': intent_vecs,
        }
    return _indices


def cosine_similarity(query_vec: np.ndarray, index: np.ndarray) -> np.ndarray:
    """쿼리 벡터와 인덱스 간 코사인 유사도 계산"""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    index_norm = index / (np.linalg.norm(index, axis=1, keepdims=True) + 1e-10)
    return index_norm @ query_norm


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
    found = []
    for ing in INGREDIENT_NAMES:
        if ing in query:
            found.append(ing)
    return found


def _detect_exclude_intent(query: str) -> dict:
    """
    제외/대체재 의도 감지.
    반환: {'exclude': '재료명' or None, 'is_alternative': bool}
    """
    exclude = None
    is_alternative = False

    # 대체재 의도
    for kw in _ALTERNATIVE_KEYWORDS:
        if kw in query:
            is_alternative = True
            break

    # 제외 의도: "X 빼고", "X 제외" 패턴에서 X 추출
    for kw in _EXCLUDE_KEYWORDS:
        if kw in query:
            # 키워드 앞의 재료를 찾기
            for ing in INGREDIENT_NAMES:
                if ing in query:
                    exclude = ing
                    break
            break

    return {'exclude': exclude, 'is_alternative': is_alternative}


def _detect_conditions(query: str, menu: str | None) -> dict:
    """
    조건 기반 추천 의도 감지 (난이도, 인분, 종류, 조리법).
    주의: 메뉴명이 이미 감지된 경우 메뉴명에 포함된 조건 키워드는 무시.
    """
    conditions = {}

    # 메뉴명을 제거한 순수 조건 부분만 분석
    query_for_cond = query
    if menu:
        query_for_cond = query.replace(menu, '')

    # 종류 (메뉴명에 포함된 '찌개', '국' 등은 조건으로 간주하지 않음)
    for k, v in KIND_MAP.items():
        if k in query_for_cond:
            conditions['kind'] = v
            break

    # 난이도
    if '초급' in query or '쉬운' in query or '간단' in query:
        conditions['difficulty'] = '초급'
    elif '중급' in query:
        conditions['difficulty'] = '중급'
    elif '고급' in query or '어려운' in query:
        conditions['difficulty'] = '고급'

    # 인분
    for s in ['1인분', '2인분', '3인분', '4인분', '5인분', '6인분이상']:
        if s in query:
            conditions['servings'] = s
            break

    # 조리법
    for k, v in METHOD_MAP.items():
        if k in query_for_cond:
            conditions['cooking_method'] = v
            break

    return conditions


def _detect_popular_intent(query: str) -> bool:
    """인기 레시피 요청 감지"""
    keywords = ['인기', 'top', 'best', '많이', '추천순', '베스트', '랭킹']
    return any(kw in query for kw in keywords)


def preprocessor_node(state: dict) -> dict:
    """
    전처리 노드 (LLM 0회, Qwen3 임베딩).
    1단계: 문자열 매칭 (빠름, 정확)
    2단계: 임베딩 유사도 (느림, fuzzy)
    3단계: 제외/조건/인기 의도 감지 (신규)
    """
    indices = _get_indices()
    emb = _get_emb()

    query = state['messages'][-1].content
    archive("preprocessor.input", {"query": query})
    query_vec = np.array(emb.embed_query(query))

    # 1. is_valid 판별
    valid_scores = cosine_similarity(query_vec, indices['valid'])
    invalid_scores = cosine_similarity(query_vec, indices['invalid'])
    max_valid = float(valid_scores.max())
    max_invalid = float(invalid_scores.max())
    is_valid = max_valid >= VALID_THRESHOLD_KO and max_valid > max_invalid

    # 2. entity 추출: 문자열 매칭 우선 → 임베딩 보조
    menu = None
    ingredients = []  # ★ 복수 재료 지원

    if is_valid:
        # 2a. 문자열 매칭 (줄임말/오타 포함)
        menu = _match_menu_from_query(query)
        ingredients = _match_ingredients_from_query(query)

        # 2b. 문자열 매칭 실패 시 임베딩 유사도로 보조 (높은 threshold)
        if not menu:
            menu_scores = cosine_similarity(query_vec, indices['menu'])
            best_idx = int(menu_scores.argmax())
            if float(menu_scores[best_idx]) >= ENTITY_THRESHOLD_KO:
                menu = MENU_NAMES[best_idx]

        if not ingredients:
            ing_scores = cosine_similarity(query_vec, indices['ingredient'])
            best_idx = int(ing_scores.argmax())
            if float(ing_scores[best_idx]) >= ENTITY_THRESHOLD_KO:
                ingredients = [INGREDIENT_NAMES[best_idx]]

    # 2c. ★ 다중 턴 컨텍스트: 현재 메시지에서 못 잡았으면 이전 히스토리에서 carry-over
    if is_valid and not menu and not ingredients and len(state['messages']) > 1:
        for prev_msg in reversed(state['messages'][:-1]):
            prev_text = getattr(prev_msg, 'content', '') or ''
            prev_menu = _match_menu_from_query(prev_text)
            if prev_menu:
                menu = prev_menu
                archive("preprocessor.context_carry", {"source": "history", "menu": menu, "from_msg_preview": prev_text[:100]})
                break
            prev_ings = _match_ingredients_from_query(prev_text)
            if prev_ings:
                ingredients = prev_ings
                archive("preprocessor.context_carry", {"source": "history", "ingredients": prev_ings, "from_msg_preview": prev_text[:100]})
                break

    # 3. intent 분류 + rewritten_query
    if is_valid:
        intent_scores = cosine_similarity(query_vec, indices['intent'])
        intent = intent_labels[int(intent_scores.argmax())]
        if menu:
            rewritten_query = f'{menu} {intent}'
        elif ingredients:
            rewritten_query = f'{", ".join(ingredients)} {intent}'
        else:
            rewritten_query = f'{intent} 조회'
    else:
        rewritten_query = '식당 운영/메뉴/원가 관련 질문이 아닙니다'

    # 4. ★ 신규: 제외/대체재 의도, 조건, 인기 감지
    exclude_info = _detect_exclude_intent(query)
    conditions = _detect_conditions(query, menu)  # 메뉴명 전달하여 오감지 방지
    is_popular = _detect_popular_intent(query)

    # 메뉴명이 있으면 conditions는 보조적 역할만 (메뉴 검색이 우선)
    # 메뉴명 없이 조건만 있을 때만 conditions를 주 라우팅 신호로 사용
    # 메뉴명이 있어도 conditions를 유지 (recipe_search에서 필터링에 활용)
    # 예: "초보용 김치찌개" → menu="김치찌개", conditions={"difficulty": "초급"}
    pass

    # entities 구성
    entities = {
        'menu': menu,
        'ingredient': ingredients if ingredients else None,  # list or None
        'exclude': exclude_info['exclude'],
        'is_alternative': exclude_info['is_alternative'],
        'conditions': conditions if conditions else None,
        'is_popular': is_popular,
    }

    archive("preprocessor.output", {
        "query": query,
        "is_valid": is_valid,
        "max_valid_score": max_valid,
        "max_invalid_score": max_invalid,
        "entities": entities,
        "rewritten_query": rewritten_query,
    })
    return {
        'is_valid': is_valid,
        'entities': entities,
        'rewritten_query': rewritten_query
    }
