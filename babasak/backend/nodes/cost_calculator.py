import re

from backend.debug_log import archive
from backend.price_bounds import MAX_PRICE_PER_KG
from backend.db import suggest_substitute_ingredient, get_menu_main_ingredient





_MAX_PRICE_PER_KG = MAX_PRICE_PER_KG
_NON_COST_INGREDIENTS = ("물",)
_NON_COST_KEYWORDS = ("국물",)


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
        return (None, f"unparsed:{q}")

    num = _parse_number(m.group("num"))
    if num is None:
        return (None, f"bad_number:{m.group('num')}")
    unit = (m.group("unit") or "").strip()

    # 공통 부피/중량 단위
    if unit in _UNIT_TABLE:
        return (num * _UNIT_TABLE[unit], f"unit:{unit}")

    # 개수 단위(개/모/대/장/포기/쪽/캔/조각 등)
    if unit in ("개", "알", "모", "대", "장", "포기", "쪽", "캔", "조각", "마리", "송이", "통", "줄", "봉", "팩", "토막", "뿌리"):
        ppg = _PER_PIECE_GRAMS.get(ingredient_name)
        if ppg:
            return (num * ppg, f"piece:{ppg}g/{unit}")
        return (None, f"piece_unit_no_table:{ingredient_name}/{unit}")

    # 단위 없음
    if not unit:
        ppg = _PER_PIECE_GRAMS.get(ingredient_name)
        if ppg:
            return (num * ppg, f"no_unit_piece:{ppg}g")
        return (None, "no_unit_no_table")

    return (None, f"unknown_unit:{unit}")



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


def _lookup_price(ingredient_name: str, price_map: dict) -> dict | None:
    """재료명 → 가격 dict. 정확 매칭 → alias db_name 매칭 → 부분 매칭 순서로 시도."""
    if ingredient_name in price_map:
        return price_map[ingredient_name]

    # alias 매칭: Genie는 DB명(예: '깐마늘')으로 가격을 주는데 레시피는 '다진마늘'로
    # 들어오는 케이스. alias 테이블로 입력명 → db_name 변환 후 그 이름으로 재조회.
    db_name = _alias_db_name(ingredient_name)
    if db_name and db_name in price_map:
        return price_map[db_name]

    # 부분 매칭: 토큰 포함 관계로 보조 매칭 (예: '마늘' ↔ '다진마늘').
    for key, info in price_map.items():
        if key in ingredient_name or ingredient_name in key:
            return info
    return None


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

    return {
        "menu": recipe.get("menu") or recipe.get("name") or "이름없음",
        "rank": recipe.get("_rank"),
        "servings": recipe.get("servings"),
        "difficulty": recipe.get("difficulty"),
        "items": items_out,
        "total_cost": total,
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
    margin_price = int(total / 0.7) if total else 0  # 마진 30%
    lines.append("")
    lines.append(f"**총 원가:** {total:,}원" + (
        f"  (시세/사용량 미확인 재료 {calc['unconfirmed_count']}개 제외)"
        if calc["unconfirmed_count"] else ""
    ))
    if total:
        lines.append(f"**1인분 원가:** {total:,}원")
        lines.append(f"**권장 판매가 (마진 30%):** {margin_price:,}원")
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


def _build_substitute_line(calc: dict, price_map: dict) -> str:
    """비싼 재료를 '같은 계열 + 더 싼' 재료로 1개만 제안하는 라인 생성.

    규칙(사용자 요구 반영):
      - 비싼 재료부터 순서대로 시도 (핵심재료 포함 — 돼지고기앞다리살→돼지고기목살 OK)
      - 후보는 ① 강한 계열 매칭(_strong_family) ② price_map에서 실제로 더 싼 것만
      - 조건 만족하는 게 없으면 추천 안 함('' 반환) — 엉뚱한 대체(미역→다시마) 금지
    """
    items = [
        it for it in calc.get("items", [])
        if it.get("cost") and it.get("source") != "non_cost" and it.get("price_per_kg")
    ]
    if not items:
        return ""
    items.sort(key=lambda it: it["cost"], reverse=True)
    menu = calc.get("menu") or ""

    for target in items[:6]:  # 비싼 순 최대 6개까지만 시도(성능)
        tname = target["name"]
        tppk = target["price_per_kg"]
        try:
            cands = suggest_substitute_ingredient(menu, tname, limit=8)
        except Exception:
            cands = []
        for c in cands:
            cname = c.get("name", "")
            if not _strong_family(tname, cname):
                continue
            cinfo = price_map.get(cname) if isinstance(price_map, dict) else None
            cppk = cinfo.get("price_per_kg") if isinstance(cinfo, dict) else None
            if not cppk or cppk >= tppk:
                continue  # 후보가 더 싸야 의미 있음
            saving_pct = round((tppk - cppk) / tppk * 100)
            archive("cost_calculator.substitute", {
                "menu": menu, "target": tname, "target_ppk": tppk,
                "candidate": cname, "candidate_ppk": cppk, "saving_pct": saving_pct,
            })
            # ★ 카드 UI용 구조화 저장 (calc_results에 실려 report_generator→프론트로 전달)
            calc["substitute"] = {
                "target": tname, "candidates": [cname], "saving_pct": saving_pct,
                "target_ppk": tppk, "candidate_ppk": cppk,
            }
            return (
                f"\n\n💡 **대체 제안:** **{tname}**(단가 {tppk:,}원/kg)이 비싼 편이에요 "
                f"→ 같은 계열 **{cname}**(단가 {cppk:,}원/kg)(으)로 바꾸면 "
                f"약 **{saving_pct}%** 저렴해요."
            )
    return ""  # 적절한 '같은 계열 + 더 싼' 대체재 없음 → 추천 생략


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

    calc_results = []
    for idx, recipe in enumerate(recipes, start=1):
        if not isinstance(recipe, dict):
            continue
        recipe = {**recipe, "_rank": idx}
        calc_results.append(_calc_recipe_cost(recipe, price_map))

    archive("cost_calculator.calc", {
        "num_recipes": len(calc_results),
        "totals": [c["total_cost"] for c in calc_results],
        "unconfirmed_counts": [c["unconfirmed_count"] for c in calc_results],
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
