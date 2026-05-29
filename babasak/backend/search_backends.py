import json
import os
import re
import requests
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from backend.debug_log import archive
from backend.catalog import resolve_ingredient

_NAVER_RATE_LOCK = threading.Lock()
_NAVER_LAST_REQUEST_AT = 0.0
_NAVER_DEFAULT_MAX_WORKERS = 2
_NAVER_DEFAULT_MIN_INTERVAL = 0.25
_NAVER_RETRY_DELAYS = (0.7, 1.5)

# 합성 재료명(예: '돼지고기앞다리살')에서 부위명을 따로 떼어내 상품명과
# 순서·띄어쓰기 무관하게 매칭하기 위한 부위 토큰 목록. 긴 것부터 검사.
_MEAT_PART_TOKENS = (
    "앞다리살", "뒷다리살", "항정살", "갈매기살", "가브리살", "등심덧살",
    "삼겹살", "오겹살", "목살", "등심", "안심", "갈비", "사태", "전지", "후지",
    "다리살", "닭가슴살", "닭다리살", "우삼겹", "차돌박이", "양지", "차돌",
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _wait_for_naver_slot() -> None:
    global _NAVER_LAST_REQUEST_AT

    min_interval = max(0.0, _env_float("NAVER_SEARCH_MIN_INTERVAL", _NAVER_DEFAULT_MIN_INTERVAL))
    with _NAVER_RATE_LOCK:
        now = time.monotonic()
        wait_seconds = _NAVER_LAST_REQUEST_AT + min_interval - now
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _NAVER_LAST_REQUEST_AT = time.monotonic()


def _parse_ingredients(text: str) -> list[str]:
    items = []
    for line in text.replace("，", ",").replace("\n", ",").split(","):
        line = line.strip()
        if not line:
            continue
        # "재료명: 수량" 형식이면 재료명만 추출
        name = line.split(":")[0].strip()
        if name:
            items.append(name)
    return items


# ────────────────────────────────────────────────────────────
# 단위 정규화: 네이버 상품가 → kg당 단가 환산
# ────────────────────────────────────────────────────────────

_TITLE_TAG_RE = re.compile(r"<[^>]+>")
_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|g|ml|l)\b", re.IGNORECASE)
_PACK_RE = re.compile(r"(?:^|[\s,])(\d{1,2})\s*(?:개|팩|봉|입|박스|세트)|x\s*(\d{1,2})\b", re.IGNORECASE)


def _clean_title(title: str) -> str:
    """네이버 응답 title에서 <b> 같은 태그 제거."""
    return _TITLE_TAG_RE.sub("", title or "")


def _extract_total_grams(title: str) -> int | None:
    """제목에서 무게 + 묶음 수를 파싱하여 총 g(또는 ml)을 반환.

    예: "고춧가루 2.5kg 8개" → 2.5 * 1000 * 8 = 20000
        "오뚜기 참기름 320ml" → 320
        "두부 500g x12" → 500 * 12 = 6000
    파싱 실패 시 None.
    """
    clean = _clean_title(title).lower()

    weight_match = _WEIGHT_RE.search(clean)
    if not weight_match:
        return None
    val = float(weight_match.group(1))
    unit = weight_match.group(2).lower()
    multiplier = {"kg": 1000, "g": 1, "ml": 1, "l": 1000}.get(unit, 0)
    if multiplier == 0:
        return None
    grams = val * multiplier

    pack_match = _PACK_RE.search(clean)
    if pack_match:
        pack = int(pack_match.group(1) or pack_match.group(2))
        if 1 < pack <= 50:  # 합리적 범위만 적용
            grams *= pack

    if grams < 30:  # 너무 작은 단위 (잘못된 매칭) 제외
        return None
    return int(grams)


def _per_kg_prices(items: list[dict]) -> list[int]:
    """네이버 items에서 단위 환산 가능한 것만 골라 kg당 단가 리스트 반환."""
    out = []
    for it in items:
        try:
            lprice = int(it.get("lprice", 0))
            if lprice <= 0:
                continue
            grams = _extract_total_grams(it.get("title", ""))
            if grams:
                per_kg = int(lprice * 1000 / grams)
                if 100 <= per_kg <= 500_000:  # 합리적 kg당 단가만
                    out.append(per_kg)
        except (TypeError, ValueError):
            continue
    return out


# ────────────────────────────────────────────────────────────
# LLM 정제: 네이버 raw 결과 5개 → 적합 상품 선별 + kg당 단가 산정
# ────────────────────────────────────────────────────────────
# 재료당 1회 호출(작고 빠름). 호출 부담을 줄이려고 lazy init + 짧은 출력.
_refine_llm = None


def _get_refine_llm():
    """LLM 정제용 lazy init. 결정적 답을 위해 temperature=0."""
    global _refine_llm
    if _refine_llm is None:
        from databricks_langchain import ChatDatabricks
        _refine_llm = ChatDatabricks(
            endpoint="databricks-gpt-5-4-mini",
            temperature=0,
            max_tokens=400,
        )
    return _refine_llm


_REFINE_SYSTEM = """당신은 식재료 도매가 분석가입니다.
네이버 검색 결과 중 사용자가 찾는 식재료에 정확히 일치하는 상품만 골라서 kg당 도매가를 산정합니다.

[작업 절차]
1. 각 상품 title을 보고 사용자가 찾는 재료와 같은 종류인지 판단 (예: '두부' 검색에 '두부젤리'/세제/꼬치 등은 제외).
2. 적합한 상품들의 title에서 총 무게(g)를 추정. title에 무게 명시가 없으면 통념으로 추정 (예: 두부 1모≈300g, 양파 1개≈200g, 참치 1캔≈150g, 청양고추 1개≈7g).
3. 각 상품의 kg당 단가 계산 = lprice × 1000 / 총 무게(g).
4. 적합 상품들의 kg당 단가 중 **최저가**를 최종 가격으로 산정 (도매가 기준 가장 저렴한 옵션을 보여주기 위함). 단위 환산이 모호한 경우는 confidence를 낮춤.
5. 한 상품도 적합하지 않으면 price_per_kg=null, confidence="none".

[출력 형식 — 반드시 순수 JSON만, 다른 텍스트 금지]
{
  "ingredient": "<재료명>",
  "price_per_kg": <정수 또는 null>,
  "unit_hint": "<예: 1모≈300g, 1개≈200g, kg 단위 도매 등 — 환산 근거>",
  "confidence": "high" | "medium" | "low" | "none",
  "selected_indices": [<적합으로 판정한 상품 인덱스 0~4>],
  "rejected_reasons": {"<인덱스>": "<제외 사유>"},
  "note": "<한 줄 요약>"
}

[confidence 기준]
- high: 적합 상품 3개 이상 + kg당 단가 분산 작음
- medium: 적합 상품 1~2개 또는 단위 추정 보강 필요
- low: 추정에 의존이 큼
- none: 적합 상품 없음
"""


def _refine_with_llm(item: str, items_raw: list[dict]) -> dict:
    """네이버 raw 5개를 LLM에 던져 dict로 정제.

    실패하거나 JSON 파싱 안 되면 {"confidence": "none", ...} 반환하여
    호출부에서 폴백 처리.
    """
    if not items_raw:
        return {"ingredient": item, "price_per_kg": None, "confidence": "none",
                "note": "no_items"}

    # LLM 입력용 후보 목록 직렬화 (인덱스 + title + 가격만)
    candidates = []
    for i, it in enumerate(items_raw[:5]):
        candidates.append({
            "index": i,
            "title": _clean_title(it.get("title", ""))[:120],
            "lprice": int(it.get("lprice", 0)) if it.get("lprice") else 0,
        })

    user_prompt = (
        f"사용자가 찾는 재료: {item}\n\n"
        f"네이버 검색 결과 (상위 {len(candidates)}개):\n"
        + "\n".join(f"  [{c['index']}] {c['title']}  (lprice={c['lprice']}원)" for c in candidates)
        + "\n\n위 5개 중 적합한 상품만 골라서 kg당 단가를 산정하고 JSON으로만 답하세요."
    )

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        response = _get_refine_llm().invoke([
            SystemMessage(content=_REFINE_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        raw = (response.content or "").strip()
        # 일부 LLM이 ```json 코드펜스 붙이는 경우 제거
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        # 최소 필드 보강
        parsed.setdefault("ingredient", item)
        parsed.setdefault("price_per_kg", None)
        parsed.setdefault("confidence", "none")
        # price_per_kg 정수 캐스팅 (LLM이 가끔 문자열로 줌)
        ppk = parsed.get("price_per_kg")
        if isinstance(ppk, str):
            digits = re.sub(r"[^\d]", "", ppk)
            parsed["price_per_kg"] = int(digits) if digits else None
        elif isinstance(ppk, float):
            parsed["price_per_kg"] = int(ppk)
        archive("naver_search.llm_refine", {
            "item": item,
            "num_candidates": len(candidates),
            "result": parsed,
        })
        return parsed
    except Exception as e:
        archive("naver_search.llm_refine_error", {"item": item, "error": str(e)})
        return {"ingredient": item, "price_per_kg": None,
                "confidence": "none", "note": f"refine_error: {e}"}


# ────────────────────────────────────────────────────────────
# 버전 1 : DuckDuckGo  ← 기본값 / 롤백용  (SEARCH_BACKEND=duckduckgo)
# ────────────────────────────────────────────────────────────

def duckduckgo_search(ingredients_text: str) -> str:
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun().run(f"{ingredients_text} 도매 시세 가격 kg")
    except Exception as e:
        return f"웹 검색 실패: {e}"


# ────────────────────────────────────────────────────────────
# 버전 2 : 네이버 쇼핑 API  (SEARCH_BACKEND=naver)
# ────────────────────────────────────────────────────────────

def naver_search_structured(ingredients_text: str) -> dict:
    """네이버 검색 + 재료별 LLM 정제 → 구조화된 결과 dict.

    Returns:
      {
        "items": {
            "<재료명>": {
                "price_per_kg": int|None,
                "confidence": "high|medium|low|none",
                "unit_hint": str|None,
                "note": str|None,
                "raw_titles": [str, ...]   # 디버깅용
            },
            ...
        },
        "text": "<재료1: ...\n재료2: ...>"  # 호환용 텍스트 출력
      }
    """
    client_id = os.getenv("NAVER_CLIENT_ID", "")
    client_secret = os.getenv("NAVER_CLIENT_SECRET", "")
    archive("naver_search.input", {
        "ingredients_text": ingredients_text,
        "has_client_id": bool(client_id),
        "has_client_secret": bool(client_secret),
    })
    if not client_id or not client_secret:
        msg = f"[오류] 네이버 API 키가 설정되지 않았습니다. (NAVER_CLIENT_ID={'있음' if client_id else '없음'}, NAVER_CLIENT_SECRET={'있음' if client_secret else '없음'})"
        archive("naver_search.output", {"reason": "missing_credentials", "result": msg})
        return {"items": {}, "text": msg}

    ingredients = _parse_ingredients(ingredients_text)
    archive("naver_search.parsed", {"ingredients": ingredients})
    if not ingredients:
        msg = "검색할 재료가 없습니다."
        archive("naver_search.output", {"reason": "no_ingredients", "result": msg})
        return {"items": {}, "text": msg}

    def _call_naver_api(item: str, query: str, display: int, attempt_label: str) -> tuple[int, list[dict]]:
        """네이버 쇼핑 API 1회 호출. (status_code, items_raw) 반환.

        429 retry 내장. 호출 실패 시 빈 리스트 반환.
        """
        _wait_for_naver_slot()
        r = requests.get(
            "https://openapi.naver.com/v1/search/shop.json",
            params={"query": query, "display": display},
            headers={
                "X-Naver-Client-Id": client_id,
                "X-Naver-Client-Secret": client_secret,
            },
            timeout=5,
        )
        archive("naver_search.api_call", {
            "item": item, "attempt": attempt_label,
            "query": query, "display": display,
            "status": r.status_code, "body_preview": r.text[:200],
        })
        for retry_idx, delay in enumerate(_NAVER_RETRY_DELAYS, start=1):
            if r.status_code != 429:
                break
            archive("naver_search.rate_limited", {
                "item": item, "attempt": attempt_label,
                "retry": retry_idx, "retry_after_seconds": delay,
            })
            time.sleep(delay)
            _wait_for_naver_slot()
            r = requests.get(
                "https://openapi.naver.com/v1/search/shop.json",
                params={"query": query, "display": display},
                headers={
                    "X-Naver-Client-Id": client_id,
                    "X-Naver-Client-Secret": client_secret,
                },
                timeout=5,
            )
        if r.status_code != 200:
            return (r.status_code, [])
        return (r.status_code, r.json().get("items", []))

    def _filter_by_token_and_price(item: str, items_raw: list[dict], attempt_label: str) -> list[dict]:
        """title 핵심 토큰 매칭 + 가격 범위 필터. archive에 reject 사유 기록.

        매칭 전략 (느슨한 순서로 OR):
          1) 검색 토큰이 상품명에 그대로 포함
          2) 공백 제거 후 포함 ("돼지고기 앞다리살" → "돼지고기앞다리살")
          3) 검색 토큰을 2글자+ 조각으로 나눠 핵심 조각이 포함
        '돼지고기앞다리살'처럼 합성 재료명이 띄어쓰기/순서 차이로 전부 탈락하던 문제 완화.
        """
        resolved = resolve_ingredient(item)
        key_name = resolved.db_name or item
        key_token = key_name.split()[0] if key_name else item
        key_token_nospace = key_token.replace(" ", "")
        # 합성 재료명에서 핵심 조각 추출. 공백으로 나뉜 조각 + 고기 부위명.
        # 예: "돼지고기앞다리살" → 공백 없으나 부위명 '앞다리살'을 따로 잡아 순서 무관 매칭.
        sub_tokens = [t for t in re.split(r"\s+", key_token) if len(t) >= 2]
        for cut in _MEAT_PART_TOKENS:
            if cut in key_token_nospace:
                sub_tokens.append(cut)

        def _matches(title_nospace: str) -> bool:
            if key_token and key_token in title_nospace:
                return True
            if key_token_nospace and key_token_nospace in title_nospace:
                return True
            # 핵심 조각(부위명 등)이 상품명에 있으면 통과 — 띄어쓰기/순서 차이 흡수
            for tok in sub_tokens:
                if len(tok) >= 3 and tok in title_nospace:
                    return True
            return False

        kept = []
        rejected_titles = []
        for it in items_raw:
            try:
                lprice = int(it.get("lprice", 0))
            except (TypeError, ValueError):
                lprice = 0
            if lprice < 500 or lprice > 500_000:
                rejected_titles.append((_clean_title(it.get("title", ""))[:60], "price_out_of_range"))
                continue
            title_clean = _clean_title(it.get("title", ""))
            title_nospace = title_clean.replace(" ", "")
            if not _matches(title_nospace):
                rejected_titles.append((title_clean[:60], "token_mismatch"))
                continue
            kept.append(it)
        archive("naver_search.title_validation", {
            "item": item, "attempt": attempt_label,
            "key_token": key_token,
            "kept": len(kept),
            "rejected": len(items_raw) - len(kept),
            "rejected_sample": rejected_titles[:3],
        })
        return kept

    # ── 재시도 단계 정의 ───────────────────────────────────────
    # 1차: "{재료} 식자재 도매" + display=5 → 업소 묶음 위주
    # 2차: "{재료}" 단독 + display=15 → 일반 상품 포함 더 넓은 후보
    # LLM이 1차에서 "전부 부적합" 판정하면 2차로 자동 진행.
    _ATTEMPTS = [
        {"label": "narrow", "query_suffix": " 식자재 도매", "display": 5},
        {"label": "wider",  "query_suffix": "",             "display": 15},
    ]

    def _search_one(item: str) -> tuple[str, dict, str]:
        """한 재료 처리. (item, structured_dict, text_line) 반환.

        재시도 전략: _ATTEMPTS를 순서대로 돌면서 LLM이 confidence!=none 줄 때까지.
        모든 attempt 실패하면 마지막 attempt의 kept를 가지고 정규식/절대가 폴백.
        """
        try:
            last_status = None
            last_kept: list[dict] = []
            last_refined: dict = {}
            last_attempt_label = None

            for attempt in _ATTEMPTS:
                label = attempt["label"]
                query = f"{item}{attempt['query_suffix']}"
                last_attempt_label = label

                status, items_raw = _call_naver_api(item, query, attempt["display"], label)
                last_status = status

                if status == 429:
                    archive("naver_search.rate_limit_final", {"item": item, "attempt": label})
                    line = f"{item}: 네이버 API 한도 초과 (잠시 후 재시도 필요)"
                    return (item, {"price_per_kg": None, "confidence": "none",
                                   "note": f"rate_limit_attempt_{label}"}, line)
                if status != 200:
                    archive("naver_search.http_error", {"item": item, "attempt": label, "status": status})
                    # 한 단계 실패해도 다음 attempt 시도
                    continue

                kept = _filter_by_token_and_price(item, items_raw, label)
                per_kg_regex = _per_kg_prices(kept)
                kept_titles = [_clean_title(it.get("title", ""))[:80] for it in kept]
                prices = [int(it["lprice"]) for it in kept if it.get("lprice")]

                archive("naver_search.parsed_prices", {
                    "item": item, "attempt": label,
                    "num_items_raw": len(items_raw),
                    "num_items_kept": len(kept),
                    "prices": prices,
                    "per_kg_prices_regex": per_kg_regex,
                    "titles": kept_titles[:3],
                })

                # title 검증 통과 0건이면 다음 attempt로
                if not kept:
                    last_kept = []
                    continue

                # LLM 정제
                refined = _refine_with_llm(item, kept)
                refined["raw_titles"] = kept_titles
                refined["attempt"] = label
                last_kept = kept
                last_refined = refined

                ppk = refined.get("price_per_kg")
                conf = refined.get("confidence", "none")
                if ppk and conf != "none":
                    # 교차 검증: 정규식으로 환산한 kg당 최저가와 비교.
                    # LLM이 최저가 선택을 자주 틀리므로(예: 멸치 51,415가 실제
                    # 최저인데 92,800으로 산정), 정규식 최저가보다 1.4배 넘게
                    # 비싸면 정규식 값으로 보정한다.
                    if per_kg_regex:
                        regex_min = min(per_kg_regex)
                        if ppk > regex_min * 1.4:
                            archive("naver_search.llm_price_corrected", {
                                "item": item, "attempt": label,
                                "llm_price": ppk, "regex_min": regex_min,
                            })
                            refined["price_per_kg"] = regex_min
                            refined["note"] = (
                                f"llm={ppk} → 정규식 최저가로 보정 "
                                f"({refined.get('note', '')})"
                            )
                            ppk = regex_min
                    # 성공 — 즉시 반환
                    line = (f"{item}: 약 ₩{ppk:,}/kg "
                            f"(LLM 정제 {label}, conf={conf}"
                            f"{', ' + refined['unit_hint'] if refined.get('unit_hint') else ''})")
                    return (item, refined, line)

                # LLM이 none 판정 — 다음 attempt로 (있으면)
                archive("naver_search.llm_refine_none", {
                    "item": item, "attempt": label,
                    "next_attempt": "yes" if label != _ATTEMPTS[-1]["label"] else "no",
                })

            # ─── 모든 attempt 실패 — 마지막 attempt의 kept로 폴백 ──────
            if last_kept:
                per_kg_regex = _per_kg_prices(last_kept)
                prices = [int(it["lprice"]) for it in last_kept if it.get("lprice")]
                refined = last_refined or {"price_per_kg": None, "confidence": "none",
                                            "raw_titles": [_clean_title(it.get("title", ""))[:80] for it in last_kept]}

                # 정규식 환산 폴백 (최저가)
                if per_kg_regex:
                    min_pkg = min(per_kg_regex)
                    refined["note"] = "llm_none_regex_fallback"
                    refined["price_per_kg"] = min_pkg
                    refined["confidence"] = "low"
                    line = (f"{item}: 약 ₩{min_pkg:,}/kg "
                            f"(정규식 환산 최저가 폴백, {len(per_kg_regex)}개 상품)")
                    return (item, refined, line)

                # 절대가 최저가 폴백
                sorted_prices = sorted(prices) if prices else []
                if sorted_prices:
                    min_abs = sorted_prices[0]
                    refined.setdefault("note", "absolute_price_only")
                    line = f"{item}: 약 ₩{min_abs:,} (상품가 최저, 단위 미상)"
                    return (item, refined, line)

            # 진짜 아무것도 못 찾음
            archive("naver_search.all_attempts_exhausted", {
                "item": item, "last_status": last_status, "last_attempt": last_attempt_label,
            })
            line = f"{item}: 검색 결과 없음"
            return (item, {"price_per_kg": None, "confidence": "none",
                           "raw_titles": [], "note": "all_attempts_exhausted"}, line)

        except Exception as e:
            archive("naver_search.error", {"item": item, "error": str(e)})
            line = f"{item}: 검색 실패 ({e})"
            return (item, {"price_per_kg": None, "confidence": "none",
                           "note": f"error: {e}"}, line)

    max_workers = max(1, _env_int("NAVER_SEARCH_MAX_WORKERS", _NAVER_DEFAULT_MAX_WORKERS))
    with ThreadPoolExecutor(max_workers=min(len(ingredients), max_workers)) as ex:
        results = list(ex.map(_search_one, ingredients))

    items_dict: dict[str, dict] = {}
    text_lines: list[str] = []
    for item_name, structured, text_line in results:
        items_dict[item_name] = structured
        text_lines.append(text_line)

    final_text = (
        "\n".join(text_lines)
        + "\n\n※ 네이버 쇼핑 소매가 기준이며, KAMIS DB에 없는 재료에 한해 참고용으로 제공됩니다."
    )
    archive("naver_search.output", {
        "result_preview": final_text[:500],
        "num_items": len(items_dict),
        "by_confidence": {
            c: [name for name, d in items_dict.items() if d.get("confidence") == c]
            for c in ("high", "medium", "low", "none")
        },
    })
    return {"items": items_dict, "text": final_text}


def naver_search(ingredients_text: str) -> str:
    """호환용 래퍼: 기존 호출처가 문자열을 기대하므로 text만 반환.

    구조화된 dict가 필요하면 naver_search_structured()를 직접 호출.
    """
    return naver_search_structured(ingredients_text)["text"]


# ────────────────────────────────────────────────────────────
# 디스패처  ←  app.yml의 SEARCH_BACKEND 값으로 버전 전환
# ────────────────────────────────────────────────────────────

def search(ingredients_text: str) -> str:
    backend = os.getenv("SEARCH_BACKEND", "naver")
    if backend == "naver":
        return naver_search(ingredients_text)
    return duckduckgo_search(ingredients_text)  # 롤백용
