import numpy as np
from langchain_databricks import DatabricksEmbeddings

# ── Qwen3 한국어 임베딩 모델 ──
emb_ko = DatabricksEmbeddings(endpoint='databricks-qwen3-embedding-0-6b')

# ── 1. 의도 분류 인덱스 (is_valid 판단용) ──
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

# ── 2. 메뉴명 인덱스 (entity: menu) ──
MENU_NAMES = [
    '김치찌개', '된장찌개', '불고기', '순두부찌개', '비빔밥',
    '제육볶음', '김치전', '떡볶이', '잡채', '오므라이스',
    '김밥', '부대찌개', '미역국', '소불고기덮밥', '치즈돈까스',
    '파전', '치킨카레', '삼계탕', '갈비탕', '냉면',
    '콩나물국', '마라탕', '부침개', '돈까스', '어묵국',
]

# ── 3. 재료명 인덱스 (entity: ingredient) ──
INGREDIENT_NAMES = [
    '돼지고기', '소고기', '닭고기', '두부', '김치',
    '양파', '대파', '감자', '당근', '애호박',
    '버섯', '고추', '마늘', '계란', '우유',
    '밀가루', '설탕', '간장', '된장', '고춧가루',
    '참기름', '식용유', '소금', '후추', '새우',
]

# ── 4. 의도 키워드 인덱스 (rewritten_query 조합용) ──
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

def build_index_ko(texts: list[str]) -> np.ndarray:
    """텍스트 리스트를 임베딩하여 numpy 배열로 반환"""
    vecs = emb_ko.embed_documents(texts)
    return np.array(vecs)

# 전역 인덱스 생성
valid_index_ko = build_index_ko(VALID_PATTERNS)
invalid_index_ko = build_index_ko(INVALID_PATTERNS)
menu_index_ko = build_index_ko(MENU_NAMES)
ingredient_index_ko = build_index_ko(INGREDIENT_NAMES)
intent_index_ko = build_index_ko(intent_texts)

VALID_THRESHOLD_KO = 0.38
ENTITY_THRESHOLD_KO = 0.68

def cosine_similarity(query_vec: np.ndarray, index: np.ndarray) -> np.ndarray:
    """쿼리 벡터와 인덱스 간 코사인 유사도 계산"""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    index_norm = index / (np.linalg.norm(index, axis=1, keepdims=True) + 1e-10)
    return index_norm @ query_norm

def preprocessor_node(state: dict) -> dict:
    """
    v4-ko 전처리 노드 (LLM 0회, Qwen3 임베딩).
    한국어 줄임말/오타/구어체를 임베딩 유사도로 처리.
    """
    query = state['messages'][-1].content
    query_vec = np.array(emb_ko.embed_query(query))

    # 1. is_valid: 절대 threshold(0.38) + 상대 비교 (valid > invalid)
    valid_scores = cosine_similarity(query_vec, valid_index_ko)
    invalid_scores = cosine_similarity(query_vec, invalid_index_ko)
    max_valid = float(valid_scores.max())
    max_invalid = float(invalid_scores.max())
    is_valid = max_valid >= VALID_THRESHOLD_KO and max_valid > max_invalid

    # 2. entities
    menu = None
    ingredient = None

    if is_valid:
        # 메뉴 매칭
        menu_scores = cosine_similarity(query_vec, menu_index_ko)
        best_menu_idx = int(menu_scores.argmax())
        if float(menu_scores[best_menu_idx]) >= ENTITY_THRESHOLD_KO:
            menu = MENU_NAMES[best_menu_idx]

        # 재료 매칭
        ing_scores = cosine_similarity(query_vec, ingredient_index_ko)
        best_ing_idx = int(ing_scores.argmax())
        if float(ing_scores[best_ing_idx]) >= ENTITY_THRESHOLD_KO:
            ingredient = INGREDIENT_NAMES[best_ing_idx]

    # 3. rewritten_query
    if is_valid:
        intent_scores = cosine_similarity(query_vec, intent_index_ko)
        intent = intent_labels[int(intent_scores.argmax())]
        if menu:
            rewritten_query = f'{menu} {intent}'
        elif ingredient:
            rewritten_query = f'{ingredient} {intent}'
        else:
            rewritten_query = f'{intent} 조회'
    else:
        rewritten_query = '식당 운영/메뉴/원가 관련 질문이 아닙니다'

    return {
        'is_valid': is_valid,
        'entities': {'menu': menu, 'ingredient': ingredient},
        'rewritten_query': rewritten_query
    }
