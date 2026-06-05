import math
import re

from backend.debug_log import archive
from backend.price_bounds import MAX_PRICE_PER_KG
from backend.db import (
    suggest_substitute_ingredient,
    get_menu_main_ingredient,
    suggest_menu_protein_alternatives,
    is_allowed_substitute_for_menu,
)





_MAX_PRICE_PER_KG = MAX_PRICE_PER_KG
_NON_COST_INGREDIENTS = (
    "물", "찬물", "더운물", "뜨거운물", "차가운물", "따뜻한물", "미지근한물",
    "끓는물", "끓인물", "얼음물", "생수", "정수물", "물리터", "미온수", "온수", "냉수",
)
# 주의: '전분물·녹말물·찹쌀풀'은 전분 원가가 있어 제외 대상 아님(가격 계산 유지)
_NON_COST_KEYWORDS = ("국물", "육수", "다시물", "다시마물", "쌀뜨물", "우린물", "다시팩물")


def _is_non_cost_ingredient(name: str) -> bool:
    """원가에서 제외할 재료인지 판정."""
    if name in _NON_COST_INGREDIENTS:
        return True
    return any(kw in name for kw in _NON_COST_KEYWORDS)




_UNIT_TABLE = {
    "큰술": 15.0, "테이블스푼": 15.0, "ts": 15.0, "스푼": 15.0, "T": 15.0,
    "작은술": 5.0, "티스푼": 5.0, "tsp": 5.0, "t": 5.0,
    "컵": 200.0, "cup": 200.0,
    "kg": 1000.0,
    "g": 1.0,
    "ml": 1.0,
    "l": 1000.0, "L": 1000.0,
    "꼬집": 1.0,
    "줌": 5.0,
    "근":  600.0,
    "뿌리": 100.0,
    "포기": 2500.0,
    "단": 250.0,
    "cc": 1.0,      # 1cc = 1ml
    "근": 600.0,    # 1근 = 600g (육류 기준)
}

# 개당 무게 추정 (재료별)
_PER_PIECE_GRAMS = {
    "양파": 200.0,
    "대파": 100.0,
    "쪽파": 30.0,
    "감자": 150.0,
    "당근": 200.0,
    "애호박": 250.0,
    "호박": 250.0,
    "오이": 200.0,
    "토마토": 200.0,
    "고추": 7.0,
    "청양고추": 7.0,
    "풋고추": 10.0,
    "홍고추": 10.0,
    "청고추": 10.0,
    "마늘": 5.0,
    "두부": 300.0,  # 1모 기준
    "계란": 60.0,
    "달걀": 60.0,
    "참치": 150.0,  # 1캔 기준
    "다시마": 5.0,  # 1조각 기준
    "황태머리": 50.0,
    "꽃게": 200.0,  # 1마리 기준
    "게": 200.0,
    "새우": 15.0,  # 1마리 기준
    "홍합": 20.0,
    "바지락": 5.0,
    "청량고추": 7.0,  # 청양고추 변형 표기
    "버섯": 20.0,
    "표고버섯": 20.0,
    "느타리버섯": 15.0,
    "팽이버섯": 100.0,  # 1봉 기준
    "배추": 2500.0,  # 1포기 기준
    "무": 1000.0,  # 1개 기준
    "무우": 1000.0, # 무 변형 표기
    "파": 100.0,
    "생강": 20.0,
    "레몬": 100.0,
    "사과": 250.0,
    "어묵": 40.0,  # 1장 기준
    # 뼈·고기 덩이류 (줄/대/짝 단위로 자주 표기됨)
    "등뼈": 350.0, "돼지등뼈": 350.0, "목등뼈": 350.0, "사골": 500.0,
    "갈비": 250.0, "등갈비": 200.0, "돼지갈비": 250.0, "소갈비": 300.0,
    "닭": 1000.0, "닭다리": 100.0, "닭봉": 50.0, "오리": 1500.0,
}

_QUALITATIVE_GRAMS = {
    "약간": 1.0, "소량": 1.0, "조금": 1.0,
    "적당량": 3.0, "취향껏": 3.0,
}

_DEFAULT_SEASONING_HINTS = (
    "고춧가루", "후춧가루", "후추", "설탕", "소금", "간장", "고추장", "된장", "쌈장",
    "참기름", "들기름", "식용유", "포도씨유", "올리브유", "고추기름",
    "마늘", "다진마늘", "생강", "다진생강", "다시다", "미원", "맛술", "미림", "청주",
    "식초", "물엿", "올리고당", "조청", "매실액", "액젓", "젓갈", "굴소스", "두반장",
    "깨", "통깨", "참깨", "전분", "녹말",
)
_DEFAULT_GRAMS_SEASONING = 10.0   # 양념류 기본 ~1작은술 수준
_DEFAULT_GRAMS_OTHER = 80.0       # 그 외(주/부재료) 기본
_DEFAULT_GRAMS_PIECE = 100.0      # 개수단위인데 품목별 1개 무게를 모를 때 1개당 추정

# 분수 패턴: "1/2", "1/3", "2/3" 등
_FRACTION_RE = re.compile(r"^(\d+)\s*/\s*(\d+)$")
# 수량 + 단위: "1큰술", "1.5kg", "200g", "1/2모", "5개" 등
_QUANTITY_RE = re.compile(
    r"^\s*(?P<num>\d+(?:\.\d+)?|\d+\s*/\s*\d+)\s*(?P<unit>[가-힣A-Za-z]+)?\s*$"
)


def _parse_number(s: str) -> float | None:
    """'1', '1.5', '1/2' → float."""
    s = s.strip()
    m = _FRACTION_RE.match(s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        return num / den if den else None
    try:
        return float(s)
    except ValueError:
        return None


def _quantity_to_grams(quantity: str, ingredient_name: str) -> tuple[float | None, str]:
    """사용량 텍스트 → 그램. (grams, reason) 튜플 반환.

    reason은 디버깅용으로 어떤 경로로 환산했는지 표시.
    환산 실패 시 (None, 사유).
    """
    if not quantity or not quantity.strip():
        if any(h in ingredient_name for h in _DEFAULT_SEASONING_HINTS):
            return (_DEFAULT_GRAMS_SEASONING, "default_seasoning")
        return (_DEFAULT_GRAMS_OTHER, "default_other")
    q = quantity.strip()

    # 정성 표현
    for kw, g in _QUALITATIVE_GRAMS.items():
        if kw in q:
            return (g, f"qualitative:{kw}")

    # 숫자 + 단위 분리
    m = _QUANTITY_RE.match(q)
    if not m:
        # 단위 없이 숫자만 들어오면 '개'로 추정
        num = _parse_number(q)
        if num is not None:
            ppg = _PER_PIECE_GRAMS.get(ingredient_name)
            if ppg:
                return (num * ppg, f"bare_number_piece:{ppg}g/개")
            # 품목 무게를 몰라도 0으로 버리지 말고 기본 개당무게로 추정
            return (num * _DEFAULT_GRAMS_PIECE, f"bare_number_default:{int(_DEFAULT_GRAMS_PIECE)}g")
        return (None, f"unparsed:{q}")

    num = _parse_number(m.group("num"))
    if num is None:
        return (None, f"bad_number:{m.group('num')}")
    unit = (m.group("unit") or "").strip()

    # 공통 부피/중량 단위
    if unit in _UNIT_TABLE:
        return (num * _UNIT_TABLE[unit], f"unit:{unit}")

    # 개수 단위(개/모/대/장/포기/쪽/캔/조각/줄 등)
    if unit in ("개", "알", "모", "대", "장", "포기", "쪽", "캔", "조각", "마리", "송이", "통", "줄", "봉", "팩", "토막", "뿌리", "짝", "덩이", "덩어리"):
        ppg = _PER_PIECE_GRAMS.get(ingredient_name)
        if ppg:
            return (num * ppg, f"piece:{ppg}g/{unit}")
        # 품목별 1개 무게를 몰라도 원가 미확인으로 버리지 말고 기본값으로 추정
        return (num * _DEFAULT_GRAMS_PIECE, f"piece_default:{unit}:{int(_DEFAULT_GRAMS_PIECE)}g")

    # 단위 없음
    if not unit:
        ppg = _PER_PIECE_GRAMS.get(ingredient_name)
        if ppg:
            return (num * ppg, f"no_unit_piece:{ppg}g")
        return (num * _DEFAULT_GRAMS_PIECE, f"no_unit_default:{int(_DEFAULT_GRAMS_PIECE)}g")

    # 알 수 없는 단위 — 그래도 개당 추정으로 원가는 낸다(0 방지)
    return (num * _DEFAULT_GRAMS_PIECE, f"unknown_unit_default:{unit}:{int(_DEFAULT_GRAMS_PIECE)}g")



_PRICE_LINE_PATTERNS = [
    # "재료명: ... ₩12,345/kg" 또는 "재료명: ... 12,345원/kg"
    re.compile(r"(?P<name>[가-힣]+)[^\n]*?(?:₩|약\s*₩)?\s*(?P<price>[\d,]+)\s*원?\s*/\s*kg"),
    # "재료명 ... 12,345원/100g" → /kg 환산
    re.compile(r"(?P<name>[가-힣]+)[^\n]*?(?P<price>[\d,]+)\s*원\s*/\s*100\s*g"),
]


_GENIE_PRICE_PATTERN = re.compile(
    r"(?P<name>[가-힣]+)\s*\(\s*(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>kg|g|개|알|마리|포기|장|손|쪽|단|모|봉|팩|통)\s*\)"
    r"[^\n]*?(?P<price>[\d,]+)\s*원"
)
# 괄호 단위 → 그램 (개수 단위는 _PER_PIECE_GRAMS로 별도 처리하므로 여기선 무게 단위만)
_GENIE_UNIT_TO_GRAMS = {"kg": 1000.0, "g": 1.0}


_GENIE_PRICE_MULTILINE = re.compile(
    r"(?P<name>[가-힣]+)\s*\(\s*(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>kg|g|개|알|마리|포기|장|손|쪽|단|모|봉|팩|통)\s*\)"
    r".{0,80}?(?P<price>[\d,]{3,})\s*원",
    re.DOTALL,
)


_GENIE_PRICE_DECL = re.compile(
    r"(?:평균\s*가격|평균가|도매\s*평균|평균\s*도매가)[은는]?\s*(?P<body>.+?)(?:입니다|이다|이며|\.|\n)",
    re.DOTALL,
)
_GENIE_PRICE_DECL_PAIR = re.compile(r"(?P<name>[가-힣]+)\s+(?P<price>\d[\d,]{2,})\s*원")


def _genie_price_to_per_kg(qty: float, unit: str, price: int, name: str) -> int | None:
    """Genie 괄호 단위(예: '1kg', '100g', '6마리')와 가격을 kg당 단가로 환산."""
    if unit in _GENIE_UNIT_TO_GRAMS:
        grams = qty * _GENIE_UNIT_TO_GRAMS[unit]
    else:
        ppg = _PER_PIECE_GRAMS.get(name)
        if not ppg:
            return None
        grams = qty * ppg
    if grams <= 0:
        return None
    return int(price * 1000 / grams)


def _build_price_map(price_info: dict) -> dict[str, dict]:
    """price_info에서 {재료명: {price_per_kg, source, confidence}} 추출.

    우선순위: structured_prices(있으면 그대로) > text 정규식 파싱.
    """
    if not isinstance(price_info, dict):
        return {}

    price_map: dict[str, dict] = {}

    # 1순위: structured_prices
    structured = price_info.get("structured_prices") or {}
    for name, info in structured.items():
        ppk = info.get("price_per_kg") if isinstance(info, dict) else None
        if not ppk:
            continue
        ppk = int(ppk)
        # sanity 상한 초과 → 네이버가 엉뚱한 상품을 잡은 것으로 보고 버림
        if ppk > _MAX_PRICE_PER_KG:
            archive("cost_calculator.price_rejected", {
                "ingredient": name, "price_per_kg": ppk,
                "reason": "exceeds_max_per_kg", "cap": _MAX_PRICE_PER_KG,
            })
            continue
        note = info.get("note") or ""
        note_lower = note.lower()
        src_hint = (info.get("source") or "").lower()
        if "b2b" in note_lower or "ingredient_recipe" in note:
            source = "recipe_b2b"
        elif "direct_sql" in note_lower or "kamis" in note_lower or "kamis" in src_hint:
            source = "kamis_direct_sql"
        elif src_hint:
            source = src_hint
        else:
            source = "naver_llm"  # 출처 단서가 없으면 종전 기본값(네이버 LLM 정제)
        price_map[name] = {
            "price_per_kg": ppk,
            "source": source,
            "confidence": info.get("confidence", "medium"),
            "note": info.get("note"),
        }

    # 2순위: text/table에서 KAMIS 등 정규식 추출 (structured에 없는 재료만)
    raw_text = ""
    if price_info.get("text"):
        raw_text += price_info["text"] + "\n"
        # text에서 한 줄 단위로 처리하면 패턴이 더 잘 잡힘
    if price_info.get("table"):
        raw_text += price_info["table"] + "\n"

    # 단순 "재료명: ... NNN원/kg" 같은 라인 스캔
    for line in raw_text.split("\n"):
        mg = _GENIE_PRICE_PATTERN.search(line)
        if mg:
            name = mg.group("name")
            if name not in price_map:
                try:
                    raw_price = int(mg.group("price").replace(",", ""))
                    ppk = _genie_price_to_per_kg(
                        float(mg.group("qty")), mg.group("unit"), raw_price, name)
                    if ppk and 100 <= ppk <= _MAX_PRICE_PER_KG:
                        price_map[name] = {
                            "price_per_kg": ppk,
                            "source": "kamis_genie",
                            "confidence": "high",
                        }
                        continue
                except ValueError:
                    pass

        # /100g → /kg 환산
        m100 = _PRICE_LINE_PATTERNS[1].search(line)
        if m100:
            name = m100.group("name")
            if name in price_map:
                continue
            try:
                ppk = int(m100.group("price").replace(",", "")) * 10
                if 100 <= ppk <= _MAX_PRICE_PER_KG:
                    price_map[name] = {
                        "price_per_kg": ppk,
                        "source": "text_parse_100g",
                        "confidence": "low",
                    }
            except ValueError:
                pass
            continue

        m = _PRICE_LINE_PATTERNS[0].search(line)
        if m:
            name = m.group("name")
            if name in price_map:
                continue
            try:
                ppk = int(m.group("price").replace(",", ""))
                if 100 <= ppk <= _MAX_PRICE_PER_KG:
                    price_map[name] = {
                        "price_per_kg": ppk,
                        "source": "text_parse_kg",
                        "confidence": "medium",
                    }
            except ValueError:
                pass

    for mg in _GENIE_PRICE_MULTILINE.finditer(raw_text):
        name = mg.group("name")
        if name in price_map:
            continue
        try:
            raw_price = int(mg.group("price").replace(",", ""))
            ppk = _genie_price_to_per_kg(
                float(mg.group("qty")), mg.group("unit"), raw_price, name)
            if ppk and 100 <= ppk <= _MAX_PRICE_PER_KG:
                price_map[name] = {
                    "price_per_kg": ppk,
                    "source": "kamis_genie_multiline",
                    "confidence": "high",
                }
        except ValueError:
            pass

    for decl in _GENIE_PRICE_DECL.finditer(raw_text):
        body = decl.group("body")
        # 단위 환산 계수: 같은 문장(또는 선언 직전 문맥)에 '100g'이 명시되면 /100g,
        # 그 외(기본값)는 '1kg 단위' 도매가로 보고 그대로 원/kg.
        per_100g = "100g" in body or "100 g" in body
        for mg in _GENIE_PRICE_DECL_PAIR.finditer(body):
            name = mg.group("name")
            if name in price_map:
                continue
            try:
                raw_price = int(mg.group("price").replace(",", ""))
            except ValueError:
                continue
            ppk = raw_price * 10 if per_100g else raw_price
            if 100 <= ppk <= _MAX_PRICE_PER_KG:
                price_map[name] = {
                    "price_per_kg": ppk,
                    "source": "kamis_genie_decl",
                    "confidence": "high",
                }

    return price_map


def _alias_db_name(ingredient_name: str) -> str | None:
    """입력 재료명을 alias 테이블의 DB 재료명으로 변환 (예: '다진마늘' → '깐마늘').

    catalog 모듈을 지연 import하여 순환참조를 피한다. 실패 시 None.
    """
    try:
        from backend.catalog import resolve_ingredient
        r = resolve_ingredient(ingredient_name)
        return r.db_name
    except Exception:
        return None


# ── 재료명 정규화 (매칭률 향상) ──
# 레시피 재료명에 붙은 조리상태/수량/의태어/오타를 떼어내 표준명으로.
# 예: '후추톡'->'후추', '참기름큰술'->'참기름', '다진대파'->'대파', '식용류'->'식용유'
# ※ 표시용 이름은 안 바꾸고, 가격 매칭 시도용으로만 사용.
_COOK_PREFIXES = ("다진", "삶은", "송송썬", "채썬", "갈은", "녹인", "불린", "익은",
                  "찐", "볶은", "데친", "으깬", "잘게썬", "곱게간", "어슷썬", "얇게썬", "송송 썬")
_QTY_SUFFIXES = ("큰술", "작은술", "컵", "톡", "방울", "솔", "듬뿍", "씩", "번", "리터", "줌", "꼬집")
_SIZE_SUFFIXES = ("작은것", "작은거", "큰것", "큰거", "흰부분", "흰대", "초록부분",
                  "흰부분만", "조금", "약간")
_NAME_SYNONYM = {
    "식용류": "식용유", "후춧가루": "후추", "후추가루": "후추",
    "슈거파우더": "설탕", "원당": "설탕", "녹말가루": "전분", "녹말": "전분",
    "대파흰부분": "대파", "대파흰대": "대파", "대파초록부분": "대파", "대파흰대부분": "대파",
}


def _normalize_ingredient_name(name: str) -> str:
    """가격 매칭용 표준명. 조리상태·수량·크기 수식 제거 + 동의어/오타 통일."""
    n = (name or "").strip()
    if n in _NAME_SYNONYM:
        return _NAME_SYNONYM[n]
    for p in _COOK_PREFIXES:
        if n.startswith(p) and len(n) > len(p) + 1:
            n = n[len(p):]
            break
    for suf in _QTY_SUFFIXES + _SIZE_SUFFIXES:
        if n.endswith(suf) and len(n) > len(suf) + 1:
            n = n[:-len(suf)]
            break
    n = n.strip()
    return _NAME_SYNONYM.get(n, n)


# ── 핵심 staple 큐레이트 단가 (원/kg, 도매 기준) ──
# KAMIS엔 없고 B2B(ingredient_recipe)는 std_name 파싱이 깨져(김치→'김치찜' 7,900/kg)
# 과대·누락이 잦은 주재료는 신뢰 가능한 고정가를 DB보다 우선 적용한다.
# (배추김치 도매 평균 ~4,000원/kg 기준. 새 staple 관측되면 여기에 추가.)
_STAPLE_PPK = {"김치": 4000, "묵은지": 4500, "깍두기": 4500, "총각김치": 5000}


def _staple_price(name: str) -> dict | None:
    """김치류 등 큐레이트 staple이면 고정 단가 dict 반환, 아니면 None."""
    n = (name or "").strip()
    if not n:
        return None
    ppk = None
    if n == "묵은지" or n.endswith("묵은지"):
        ppk = _STAPLE_PPK["묵은지"]
    elif n in _STAPLE_PPK:
        ppk = _STAPLE_PPK[n]
    elif n.endswith("김치"):              # 신김치·배추김치·포기김치·김장김치 등
        ppk = _STAPLE_PPK["김치"]
    if ppk is None:
        return None
    return {"price_per_kg": ppk, "source": "staple_curated", "confidence": "medium"}


def _lookup_price(ingredient_name: str, price_map: dict) -> dict | None:
    """재료명 → 가격 dict. (staple 큐레이트 우선) 정확 → alias → 정규화 → 부분 매칭 순서로 시도."""
    # 핵심 staple(김치류)은 DB가 부정확/누락이라 큐레이트 단가를 DB보다 우선 적용
    staple = _staple_price(ingredient_name)
    if staple:
        return staple
    if ingredient_name in price_map:
        return price_map[ingredient_name]

    # alias 매칭: Genie는 DB명(예: '깐마늘')으로 가격을 주는데 레시피는 '다진마늘'로
    # 들어오는 케이스. alias 테이블로 입력명 → db_name 변환 후 그 이름으로 재조회.
    db_name = _alias_db_name(ingredient_name)
    if db_name and db_name in price_map:
        return price_map[db_name]

    # 정규화 매칭: '후추톡'->'후추', '참기름큰술'->'참기름' 등 수식 제거 후 재조회
    norm = _normalize_ingredient_name(ingredient_name)
    if norm != ingredient_name:
        if norm in price_map:
            return price_map[norm]
        ndb = _alias_db_name(norm)
        if ndb and ndb in price_map:
            return price_map[ndb]

    # 부분 매칭: 토큰 포함 관계로 보조 매칭 (예: '마늘' ↔ '다진마늘').
    for key, info in price_map.items():
        if key in ingredient_name or ingredient_name in key:
            return info
    return None


# ── 참고 판매가 = 예상 원가 ÷ 기준 원가율 ──
# 기준 원가율(BASE_COST_RATE)은 사장님이 입력한 '목표값'이 아니라 서비스가 임시로 쓰는 '기본 기준값'이다.
#   참고 판매가 = 예상 원가 / 기준 원가율  (원가율 0.30 → 판매가 = 원가/0.3), 100원 단위 올림.
#   재료 기준 이익 = 참고 판매가 - 예상 원가  (인건비·임대료·가스비·배달수수료 미포함)
BASE_COST_RATE = 0.30


def _valid_rate(value) -> float | None:
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if rate > 1:
        rate = rate / 100
    if 0 < rate < 0.95:
        return round(rate, 4)
    return None


def _pct_text(rate: float | None) -> str:
    if rate is None:
        return "-"
    pct = round(rate * 100, 1)
    return f"{pct:g}%"


def _pricing_policy_from_state(state: dict) -> dict:
    entities = state.get("entities") if isinstance(state, dict) else {}
    entities = entities if isinstance(entities, dict) else {}

    user_margin = _valid_rate(entities.get("target_margin_rate"))
    user_cost_ratio = _valid_rate(entities.get("target_cost_ratio"))
    pricing_source = entities.get("pricing_source")

    if pricing_source == "user_margin" and user_margin is not None:
        cost_ratio = round(1 - user_margin, 4)
        margin_rate = user_margin
        source = "user_margin"
        label = f"사용자 마진율 {_pct_text(margin_rate)}"
    elif pricing_source == "user_cost_ratio" and user_cost_ratio is not None:
        cost_ratio = user_cost_ratio
        margin_rate = round(1 - cost_ratio, 4)
        source = "user_cost_ratio"
        label = f"사용자 원가율 {_pct_text(cost_ratio)}"
    elif user_margin is not None:
        cost_ratio = round(1 - user_margin, 4)
        margin_rate = user_margin
        source = "user_margin"
        label = f"사용자 마진율 {_pct_text(margin_rate)}"
    elif user_cost_ratio is not None:
        cost_ratio = user_cost_ratio
        margin_rate = round(1 - cost_ratio, 4)
        source = "user_cost_ratio"
        label = f"사용자 원가율 {_pct_text(cost_ratio)}"
    else:
        cost_ratio = BASE_COST_RATE
        margin_rate = round(1 - cost_ratio, 4)
        source = "default_cost_ratio"
        label = f"기본 원가율 {_pct_text(cost_ratio)}"

    servings_num = _parse_servings(recipe.get("servings"))
    return {
        "cost_ratio": cost_ratio,
        "margin_rate": margin_rate,
        "pricing_source": source,
        "pricing_label": label,
        "pricing_text": entities.get("pricing_text"),
    }


def ceil_to_100(value: float) -> int:
    """100원 단위 올림."""
    return int(math.ceil(value / 100) * 100)


def reference_price(estimated_cost, cost_rate: float = BASE_COST_RATE):
    """참고 판매가 = 예상 원가 / 기준 원가율 (100원 올림). 원가 없으면 None."""
    if estimated_cost is None or cost_rate <= 0:
        return None
    return ceil_to_100(estimated_cost / cost_rate)


def material_profit(ref_price, estimated_cost):
    """재료 기준 이익 = 참고 판매가 - 예상 원가."""
    if ref_price is None or estimated_cost is None:
        return None
    return ref_price - estimated_cost


def _extract_industry(state: dict) -> str:
    """메시지의 '[업종: X, 지역: Y]'에서 업종 추출. 없으면 ''."""
    try:
        for msg in reversed(state.get("messages", []) or []):
            content = getattr(msg, "content", "") or ""
            m = re.search(r"\[업종:\s*([^,\]]+)", content)
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    return ""


# 사용자 입력이 없을 때만 BASE_COST_RATE를 쓰고, 입력된 마진율/원가율은 _pricing_policy_from_state에서 해석한다.


def _parse_servings(servings) -> int:
    """'3인분'->3, '2~3인분'->2, '4인분 이상'->4, 숫자 없으면 1.
    레시피 재료 수량은 이 인분 기준이므로, 1인분 원가는 (전체÷인분수)."""
    if not servings:
        return 1
    m = re.search(r'\d+', str(servings))
    if m:
        n = int(m.group())
        return n if n > 0 else 1
    return 1


def _calc_recipe_cost(recipe: dict, price_map: dict) -> dict:
    """한 레시피의 재료별 원가 계산.

    Returns:
      {
        "menu": "김치찌개",
        "rank": 1,
        "items": [
            {"name": "두부", "quantity": "1/2모", "grams": 150.0,
             "price_per_kg": 10000, "cost": 1500, "source": "naver_llm", "confidence": "high"},
            ...
        ],
        "total_cost": int,
        "unconfirmed_count": int,
      }
    """
    items_out = []
    total = 0
    unconfirmed = 0

    for ing in recipe.get("ingredients", []):
        if not isinstance(ing, dict):
            continue
        name = ing.get("name", "").strip()
        if not name:
            continue
        quantity = (ing.get("quantity") or ing.get("amount") or "").strip()

        # 물·국물류 등 비-원가 재료는 원가 합산에서 제외 (₩0으로 표시)
        if _is_non_cost_ingredient(name):
            items_out.append({
                "name": name,
                "quantity": quantity,
                "grams": None,
                "price_per_kg": None,
                "cost": 0,
                "source": "non_cost",
                "confidence": "n/a",
                "qty_reason": "non_cost_ingredient",
            })
            continue

        price_info = _lookup_price(name, price_map)
        grams, qty_reason = _quantity_to_grams(quantity, name)

        if price_info and grams is not None:
            cost = int(price_info["price_per_kg"] * grams / 1000)
            items_out.append({
                "name": name,
                "quantity": quantity,
                "grams": round(grams, 1),
                "price_per_kg": price_info["price_per_kg"],
                "cost": cost,
                "source": price_info.get("source"),
                "confidence": price_info.get("confidence"),
                "qty_reason": qty_reason,
            })
            total += cost
        else:
            unconfirmed += 1
            items_out.append({
                "name": name,
                "quantity": quantity,
                "grams": grams,
                "price_per_kg": price_info["price_per_kg"] if price_info else None,
                "cost": None,
                "source": price_info.get("source") if price_info else None,
                "confidence": price_info.get("confidence") if price_info else "none",
                "qty_reason": qty_reason,
                "missing_reason": (
                    "no_price" if not price_info
                    else "no_quantity_conversion"
                ),
            })

    servings_num = _parse_servings(recipe.get("servings"))
    return {
        "menu": recipe.get("menu") or recipe.get("name") or "이름없음",
        "rank": recipe.get("_rank"),
        "servings": recipe.get("servings"),
        "servings_num": servings_num,                       # 인분 수 (숫자)
        "difficulty": recipe.get("difficulty"),
        "items": items_out,
        "total_cost": total,                                # 전체(N인분) 재료비
        "per_serving_cost": int(total / servings_num) if servings_num else total,  # 1인분 원가
        "unconfirmed_count": unconfirmed,
    }


def _format_recipe_section(calc: dict) -> str:
    """한 레시피의 계산 결과를 markdown 표로."""
    header = f"## [인기 {calc['rank']}위] {calc['menu']}"
    meta = []
    if calc.get("servings"):
        meta.append(f"분량 {calc['servings']}")
    if calc.get("difficulty"):
        meta.append(f"난이도 {calc['difficulty']}")
    sub = " · ".join(meta)

    lines = [header]
    if sub:
        lines.append(f"_{sub}_")
    lines.append("")
    lines.append("| 재료 | 사용량 | 단가(원/kg) | 원가 |")
    lines.append("|---|---:|---:|---:|")
    for it in calc["items"]:
        ppk = f"{it['price_per_kg']:,}" if it.get("price_per_kg") else "—"
        if it.get("source") == "non_cost":
            cost = "원가 제외"
        elif it["cost"] is not None:
            cost = f"{it['cost']:,}원"
        else:
            cost = "시세/사용량 미확인"
        qty = it.get("quantity") or "-"
        # 수량이 비어 기본값으로 추정한 경우 → 추정한 그램값을 같이 보여줌 (예: '약 10g(추정)')
        if (it.get("qty_reason") or "").startswith("default") and not (it.get("quantity") or "").strip():
            g = it.get("grams")
            qty = f"약 {int(g)}g(추정)" if g else "추정"
        lines.append(f"| {it['name']} | {qty} | {ppk} | {cost} |")

    total = calc["total_cost"]
    sv = calc.get("servings_num") or 1
    per = calc.get("per_serving_cost")
    if per is None:
        per = int(total / sv) if total else 0
    cr = calc.get("cost_ratio") or BASE_COST_RATE
    margin_rate = calc.get("margin_rate")
    if margin_rate is None:
        margin_rate = round(1 - cr, 4)
    pricing_label = calc.get("pricing_label") or f"기본 원가율 {_pct_text(cr)}"
    sell_price = reference_price(per, cr) or 0   # 1인분 기준, 원가율 역산 + 100원 올림
    profit = (sell_price - per) if (sell_price and per) else 0
    lines.append("")
    lines.append(f"**총 원가 ({sv}인분 전체):** {total:,}원" + (
        f"  (시세/사용량 미확인 재료 {calc['unconfirmed_count']}개 제외)"
        if calc["unconfirmed_count"] else ""
    ))
    if total:
        lines.append(f"**1인분 예상 원가:** {per:,}원" + (f"  (전체 {total:,}원 ÷ {sv}인분)" if sv > 1 else ""))
        lines.append(f"**참고 판매가 [{pricing_label} 기준]:** {sell_price:,}원")
        lines.append(f"**기준 원가율 / 예상 마진율:** {_pct_text(cr)} / {_pct_text(margin_rate)}")
        lines.append(f"**재료 기준 이익:** {profit:,}원")
    return "\n".join(lines)




def _strong_family(target_name: str, cand_name: str) -> bool:
    if not target_name or not cand_name or target_name == cand_name:
        return False
    # 한쪽이 다른 쪽을 포함 + 포함되는 이름이 3글자 이상 (의미있는 계열명)
    if target_name in cand_name and len(target_name) >= 3:
        return True
    if cand_name in target_name and len(cand_name) >= 3:
        return True
    # 공통 접두 3글자 이상
    common = 0
    for a, b in zip(target_name, cand_name):
        if a == b:
            common += 1
        else:
            break
    return common >= 3


# 주재료(단백질) 판정용 — 이 재료가 비쌀 때만 대체 제안 대상
_PROTEIN_HINTS = (
    "고기", "살", "갈비", "삼겹", "목살", "안심", "등심", "닭", "돼지", "소", "오리",
    "새우", "오징어", "낙지", "문어", "조개", "꽃게", "전복", "관자", "홍합", "바지락",
    "연어", "참치", "고등어", "갈치", "명태", "코다리", "꽁치", "광어", "대구", "동태", "아귀", "아구",
    "햄", "소시지", "베이컨", "스팸", "두부", "유부", "계란", "달걀", "어묵",
)


def _is_protein(name: str) -> bool:
    return any(h in name for h in _PROTEIN_HINTS)


# B2B std_name이 요리/가공식품으로 깨진 행(소고기→'장터국밥', 참치→'참치액젓')을 후보 가격에서 배제.
# 생재료 단가만 후보로 쓰기 위함 — 틀린 대체 제안을 막는다.
_B2B_DISH_NOISE = (
    "국밥", "액젓", "액기스", "찜", "만두", "볶음밥", "주먹밥", "스프", "수프",
    "사발면", "컵라면", "김밥", "떡볶이", "소스", "양념", "육수", "다시", "조미",
    "젓갈", "장조림", "조림", "전골", "밀키트",
)


def _candidate_ppk(cname: str, price_map: dict) -> int | None:
    """대체 후보의 원/kg 단가를 구한다.

    후보(참치·목살 등)는 보통 '현재 레시피 재료'가 아니라서 price_map에 없다.
    그래서 price_map만 보면 대체재가 거의 안 떴다 → staple/이번 가격맵 → B2B 유통가
    (in-memory recipe_catalog, 빠름) 순으로 후보를 '따로' 가격 조회한다.
    B2B 상품명이 요리/가공식품(국밥·액젓 등)으로 깨진 행은 배제(틀린 제안 방지).
    """
    info = _lookup_price(cname, price_map)   # staple(김치류) + price_map(정확/alias/정규화/부분)
    if isinstance(info, dict) and info.get("price_per_kg"):
        return int(info["price_per_kg"])
    try:
        from backend.catalog import get_recipe_price
        ri = get_recipe_price(cname)
        if ri and ri.price_per_kg and ri.price_per_kg > 0:
            product = ri.product_name or ""
            if not any(kw in product for kw in _B2B_DISH_NOISE):
                return int(ri.price_per_kg)
    except Exception:
        pass
    return None


# 대체 제안 임계값: 싼 재료(참치 한 캔 684원 등)나 미미한 절감엔 제안하지 않는다.
_SUB_MIN_TARGET_COST = 1500   # 이 금액(원) 미만 주재료는 대체 제안 대상 아님
_SUB_MIN_SAVING_WON = 300     # 접시당 절감이 이 미만이면 제안 생략


def _build_substitute_line(calc: dict, price_map: dict) -> str:
    """비싼 '주재료(단백질)'를 같은 메뉴에서 통하는 더 싼 재료로 대체 제안.

    규칙(사용자 요구):
      - 비싼 단백질부터 시도 (돼지고기·소고기·참치 등)
      - 후보 = '이 메뉴의 다른 레시피들이 실제로 쓰는 단백질' (데이터로 '어울림' 판정)
        예) 김치찌개 돼지고기(비쌈) → 참치 (참치김치찌개가 실제 있으니 OK)
            제육볶음 돼지앞다리살 → 더 싼 돼지부위/오징어
            미역국 소고기 → 미역국에 안 쓰는 참치는 후보에 없음 (괴식 차단)
      - 그 후보 중 price_map에서 '더 싼' 것을 선택. 없으면 추천 안 함.
    """
    items = [
        it for it in calc.get("items", [])
        if it.get("cost") and it.get("source") != "non_cost" and it.get("price_per_kg")
    ]
    if not items:
        return ""
    items.sort(key=lambda it: it["cost"], reverse=True)
    menu = calc.get("menu") or ""

    for target in items[:6]:  # 비싼 순 최대 6개
        tname = target["name"]
        tppk = target["price_per_kg"]
        tcost = target.get("cost") or 0
        if not _is_protein(tname):      # 단백질(주재료)만 대체 대상
            continue
        # 충분히 비싼 주재료에만 제안. 참치 한 캔(684원)처럼 싼 재료는 바꿔도 의미 없음.
        if tcost < _SUB_MIN_TARGET_COST:
            continue
        try:
            cands = suggest_menu_protein_alternatives(menu, tname, limit=10)
        except Exception:
            cands = []
        # 후보 중 '더 싼' 것 선택 (가장 많이 절감되는 순)
        best = None
        for c in cands:
            cname = c.get("name", "")
            if not cname or cname == tname:
                continue
            if not is_allowed_substitute_for_menu(menu, tname, cname):
                continue
            # 후보도 '진짜 단백질명'이어야 한다. (Neo4j lv1 분류 노이즈로 김치국물·국물류가
            #  단백질로 새서 돼지고기 대체재로 뜨는 것 차단 — 물/국물류도 제외)
            if not _is_protein(cname) or _is_non_cost_ingredient(cname):
                continue
            # 후보 가격은 price_map에 없을 수 있어 따로 조회(B2B 등) — 이게 핵심 수정
            cppk = _candidate_ppk(cname, price_map)
            if not cppk or cppk >= tppk:
                continue  # 후보가 더 싸야 의미 있음
            if best is None or cppk < best[1]:
                best = (cname, cppk)
        if not best:
            continue
        cname, cppk = best
        saving_pct = round((tppk - cppk) / tppk * 100)
        # 접시당 절감액 (target 사용 그램 기준)
        grams = target.get("grams")
        saving_won = int((tppk - cppk) * grams / 1000) if grams else None
        # 절감이 미미하면(접시당 기준) 제안 생략 — 다음 비싼 재료로
        if saving_won is not None and saving_won < _SUB_MIN_SAVING_WON:
            continue
        archive("cost_calculator.substitute", {
            "menu": menu, "target": tname, "target_ppk": tppk,
            "candidate": cname, "candidate_ppk": cppk,
            "saving_pct": saving_pct, "saving_won": saving_won,
        })
        # ★ 카드 UI용 구조화 저장
        calc["substitute"] = {
            "target": tname, "candidates": [cname], "saving_pct": saving_pct,
            "target_ppk": tppk, "candidate_ppk": cppk, "saving_won": saving_won,
        }
        won_txt = f" · 약 -{saving_won:,}원/접시 절감" if saving_won else ""
        return (
            f"\n\n💡 **대체 제안:** **{tname}**(단가 {tppk:,}원/kg)이 비싼 편이에요 "
            f"→ **{cname}**(단가 {cppk:,}원/kg)(으)로 바꾸면 약 **{saving_pct}%** 저렴{won_txt}. "
            f"('{menu}'에 두루 쓰이는 재료예요)"
        )
    return ""  # 더 싼 후보 없음 → 추천 생략


def cost_calculator_node(state: dict) -> dict:
    """레시피 재료 + 가격 데이터를 코드로 매칭하여 원가를 계산합니다."""
    recipe_info = state.get("recipe_info", {})
    price_info = state.get("price_info", {})

    archive("cost_calculator.input", {
        "has_recipe": bool(recipe_info),
        "has_price": bool(price_info),
        "price_source": price_info.get("estimation_source") if isinstance(price_info, dict) else None,
        "num_structured_prices": len(price_info.get("structured_prices") or {}) if isinstance(price_info, dict) else 0,
    })

    if not recipe_info or not price_info:
        archive("cost_calculator.output", {"reason": "insufficient_data"})
        return {"cost_info": {"error": "레시피 또는 가격 정보 부족"}}

    # 가격 맵 구축
    price_map = _build_price_map(price_info)
    archive("cost_calculator.price_map", {
        "num_prices": len(price_map),
        "ingredients": list(price_map.keys()),
        "sources": {
            src: [k for k, v in price_map.items() if v.get("source") == src]
            for src in ("naver_llm", "text_parse_kg", "text_parse_100g")
        },
    })

    # 레시피별 계산 (rank 부여)
    recipes = recipe_info.get("data") if isinstance(recipe_info, dict) else recipe_info
    if not isinstance(recipes, list):
        recipes = [recipes] if recipes else []

    # 사용자 마진율/원가율 입력이 있으면 우선하고, 없으면 서비스 기본 원가율을 쓴다.
    pricing_policy = _pricing_policy_from_state(state)
    cost_ratio = pricing_policy["cost_ratio"]
    margin_rate = pricing_policy["margin_rate"]

    calc_results = []
    for idx, recipe in enumerate(recipes, start=1):
        if not isinstance(recipe, dict):
            continue
        recipe = {**recipe, "_rank": idx}
        c = _calc_recipe_cost(recipe, price_map)
        c["cost_ratio"] = cost_ratio            # 마크다운·카드 판매가 계산 기준
        c["margin_rate"] = margin_rate
        c["pricing_source"] = pricing_policy["pricing_source"]
        c["pricing_label"] = pricing_policy["pricing_label"]
        c["pricing_text"] = pricing_policy.get("pricing_text")
        # 검증 로그: 레시피명/인분수/전체원가/1인분원가/카드표시원가/참고판매가/재료기준이익/제외재료
        _per = c.get("per_serving_cost")
        _ref = reference_price(_per, cost_ratio)
        _excluded = [it.get("name") for it in (c.get("items") or [])
                     if it.get("name") and (it.get("cost") is None or it.get("source") == "non_cost")]
        archive("cost_calculator.card_check", {
            "menu": c.get("menu"),
            "servings_num": c.get("servings_num"),
            "full_cost_total": c.get("total_cost"),       # 전체 재료 원가 합계(= 상세표 합계)
            "per_serving_cost": _per,                     # 1인분 예상 원가
            "card_estimated_cost": _per,                  # 카드에 표시되는 원가(1인분)
            "reference_price": _ref,                      # 참고 판매가
            "material_profit": material_profit(_ref, _per),
            "cost_ratio": cost_ratio,
            "margin_rate": margin_rate,
            "pricing_source": pricing_policy["pricing_source"],
            "excluded_ingredients": _excluded,            # 원가 계산 제외 재료
        })
        calc_results.append(c)

    archive("cost_calculator.calc", {
        "num_recipes": len(calc_results),
        "totals": [c["total_cost"] for c in calc_results],
        "unconfirmed_counts": [c["unconfirmed_count"] for c in calc_results],
        "pricing": pricing_policy,
    })

    # 마크다운 조합 (+ 비싼 보조재료 대체 제안 라인 부착)
    sections = []
    for c in calc_results:
        sec = _format_recipe_section(c)
        sub_line = _build_substitute_line(c, price_map)
        if sub_line:
            sec += sub_line
        sections.append(sec)

    # 인기 순위별 비교 요약
    if len(calc_results) > 1:
        non_zero = [(c["rank"], c["menu"], c["total_cost"])
                    for c in calc_results if c["total_cost"] > 0]
        if non_zero:
            cheapest = min(non_zero, key=lambda x: x[2])
            summary = (
                f"\n---\n**요약:** 인기 {cheapest[0]}위 '{cheapest[1]}'이(가) "
                f"총 원가 {cheapest[2]:,}원으로 가장 저렴합니다."
            )
            sections.append(summary)

    result = "\n\n".join(sections) if sections else "원가 계산 결과 없음"

    archive("cost_calculator.output", {
        "success": True,
        "result_preview": result[:400],
        "num_sections": len(sections),
    })

    return {"cost_info": {
        "analysis": result,
        "calc_results": calc_results,  # report_generator/디버깅용 raw
    }}
