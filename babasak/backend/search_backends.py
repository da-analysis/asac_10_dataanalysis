import os
import re
import requests
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from backend.debug_log import archive

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_NAVER_RATE_LOCK = threading.Lock()
_NAVER_LAST_REQUEST_AT = 0.0
_NAVER_DEFAULT_MAX_WORKERS = 2
_NAVER_DEFAULT_MIN_INTERVAL = 0.25
_NAVER_RETRY_DELAYS = (0.7, 1.5)


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


def _median(values: list[int]) -> int:
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) // 2


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

def naver_search(ingredients_text: str) -> str:
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
        return msg

    ingredients = _parse_ingredients(ingredients_text)
    archive("naver_search.parsed", {"ingredients": ingredients})
    if not ingredients:
        msg = "검색할 재료가 없습니다."
        archive("naver_search.output", {"reason": "no_ingredients", "result": msg})
        return msg

    def _search_one(item: str) -> str:
        try:
            _wait_for_naver_slot()
            r = requests.get(
                "https://openapi.naver.com/v1/search/shop.json",
                params={"query": f"{item} 식자재 도매", "display": 5},
                headers={
                    "X-Naver-Client-Id": client_id,
                    "X-Naver-Client-Secret": client_secret,
                },
                timeout=5,
            )
            archive("naver_search.api_call", {
                "item": item,
                "status": r.status_code,
                "body_preview": r.text[:300],
            })
            for attempt, delay in enumerate(_NAVER_RETRY_DELAYS, start=1):
                if r.status_code != 429:
                    break
                archive("naver_search.rate_limited", {
                    "item": item,
                    "attempt": attempt,
                    "retry_after_seconds": delay,
                })
                time.sleep(delay)
                _wait_for_naver_slot()
                r = requests.get(
                    "https://openapi.naver.com/v1/search/shop.json",
                    params={"query": f"{item} 식자재 도매", "display": 5},
                    headers={
                        "X-Naver-Client-Id": client_id,
                        "X-Naver-Client-Secret": client_secret,
                    },
                    timeout=5,
                )
                archive("naver_search.api_call", {
                    "item": item,
                    "status": r.status_code,
                    "attempt": attempt + 1,
                    "body_preview": r.text[:300],
                })
            # 재시도 후에도 429면 "한도 초과"로 명시 (Medium 1)
            if r.status_code == 429:
                archive("naver_search.rate_limit_final", {"item": item})
                return f"{item}: 네이버 API 한도 초과 (잠시 후 재시도 필요)"
            if r.status_code != 200:
                archive("naver_search.http_error", {"item": item, "status": r.status_code})
                return f"{item}: 네이버 API 오류 (status {r.status_code})"

            items_raw = r.json().get("items", [])
            prices = [int(it["lprice"]) for it in items_raw if it.get("lprice")]
            per_kg = _per_kg_prices(items_raw)  # 단위 환산 가능한 것만
            archive("naver_search.parsed_prices", {
                "item": item,
                "num_items": len(items_raw),
                "prices": prices,
                "per_kg_prices": per_kg,
                "titles": [_clean_title(it.get("title", ""))[:60] for it in items_raw[:3]],
            })
            if not prices:
                return f"{item}: 검색 결과 없음"

            # ── 1순위: per-kg 환산 (단위 정보 있으면) ──
            if per_kg:
                median_pkg = _median(per_kg)
                return f"{item}: 약 ₩{median_pkg:,}/kg (네이버 쇼핑 환산, {len(per_kg)}개 상품 기준)"

            # ── 2순위: 단위 환산 불가 → 절대 가격 중앙값 (참고용) ──
            sorted_prices = sorted(prices)
            return f"{item}: 약 ₩{_median(sorted_prices):,} (네이버 쇼핑 상품가 중앙값, 단위 미상·범위 ₩{sorted_prices[0]:,}~₩{sorted_prices[-1]:,})"
        except Exception as e:
            archive("naver_search.error", {"item": item, "error": str(e)})
            return f"{item}: 검색 실패 ({e})"

    max_workers = max(1, _env_int("NAVER_SEARCH_MAX_WORKERS", _NAVER_DEFAULT_MAX_WORKERS))
    with ThreadPoolExecutor(max_workers=min(len(ingredients), max_workers)) as ex:
        results = list(ex.map(_search_one, ingredients))

    final = (
        "\n".join(results)
        + "\n\n※ 네이버 쇼핑 소매가 기준이며, KAMIS DB에 없는 재료에 한해 참고용으로 제공됩니다."
    )
    archive("naver_search.output", {"result_preview": final[:500]})
    return final


# ────────────────────────────────────────────────────────────
# 버전 3 : CJ프레시웨이 + 에이스식자재몰 크롤링  (SEARCH_BACKEND=crawl)
# ────────────────────────────────────────────────────────────

def _extract_min_price(text: str) -> int | None:
    matches = re.findall(r'[\d,]+', text)
    values = []
    for m in matches:
        val = int(m.replace(",", ""))
        if 500 <= val <= 500_000:
            values.append(val)
    return min(values) if values else None


def _crawl_cj(item: str) -> str | None:
    try:
        from bs4 import BeautifulSoup
        r = requests.get(
            "https://www.cjfls.co.kr/search",
            params={"searchKeyword": item},
            headers=_HEADERS,
            timeout=7,
        )
        soup = BeautifulSoup(r.text, "html.parser")
        for img in soup.find_all("img", src=lambda s: s and "price" in s.lower()):
            price = _extract_min_price(img.parent.get_text(" ", strip=True))
            if price:
                return f"₩{price:,} (CJ프레시웨이)"
    except Exception:
        pass
    return None


def _crawl_ace(item: str) -> str | None:
    try:
        from bs4 import BeautifulSoup
        r = requests.get(
            "https://www.acemall.asia/goods/goods_search.php",
            params={"keyword": item},
            headers=_HEADERS,
            timeout=7,
        )
        soup = BeautifulSoup(r.text, "html.parser")
        # <strong> 태그에 가격이 있음
        for tag in soup.find_all("strong"):
            text = tag.get_text(strip=True).replace(",", "")
            if text.isdigit():
                val = int(text)
                if 500 <= val <= 500_000:
                    return f"₩{val:,} (에이스식자재몰)"
    except Exception:
        pass
    return None


def crawl_search(ingredients_text: str) -> str:
    ingredients = _parse_ingredients(ingredients_text)
    if not ingredients:
        return "검색할 재료가 없습니다."

    def _crawl_one(item: str) -> str:
        price = _crawl_cj(item) or _crawl_ace(item)
        return f"{item}: {price}" if price else f"{item}: 검색 결과 없음"

    with ThreadPoolExecutor(max_workers=min(len(ingredients), 5)) as ex:
        results = list(ex.map(_crawl_one, ingredients))

    return (
        "\n".join(results)
        + "\n\n※ 위 가격은 식자재몰 기준이며, KAMIS DB에 없는 재료에 한해 참고용으로 제공됩니다."
    )


# ────────────────────────────────────────────────────────────
# 디스패처  ←  app.yml의 SEARCH_BACKEND 값으로 버전 전환
# ────────────────────────────────────────────────────────────

def search(ingredients_text: str) -> str:
    backend = os.getenv("SEARCH_BACKEND", "duckduckgo")
    if backend == "naver":
        return naver_search(ingredients_text)
    if backend == "crawl":
        return crawl_search(ingredients_text)
    return duckduckgo_search(ingredients_text)  # 기본값 (롤백용)
