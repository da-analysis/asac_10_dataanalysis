import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


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
    print(f"[naver_search] called. id={'있음' if client_id else '없음'}, secret={'있음' if client_secret else '없음'}, text={ingredients_text!r}")
    if not client_id or not client_secret:
        return f"[오류] 네이버 API 키가 설정되지 않았습니다. (NAVER_CLIENT_ID={'있음' if client_id else '없음'}, NAVER_CLIENT_SECRET={'있음' if client_secret else '없음'})"

    ingredients = _parse_ingredients(ingredients_text)
    print(f"[naver_search] parsed ingredients: {ingredients}")
    if not ingredients:
        return "검색할 재료가 없습니다."

    def _search_one(item: str) -> str:
        try:
            r = requests.get(
                "https://openapi.naver.com/v1/search/shop.json",
                params={"query": f"{item} 식자재 도매", "display": 5},
                headers={
                    "X-Naver-Client-Id": client_id,
                    "X-Naver-Client-Secret": client_secret,
                },
                timeout=5,
            )
            print(f"[naver_search] {item}: status={r.status_code}, body={r.text[:200]}")
            items = r.json().get("items", [])
            prices = [int(it["lprice"]) for it in items if it.get("lprice")]
            if not prices:
                return f"{item}: 검색 결과 없음"
            avg = sum(prices) // len(prices)
            return f"{item}: 약 ₩{avg:,} (네이버 쇼핑 소매가 기준)"
        except Exception as e:
            return f"{item}: 검색 실패 ({e})"

    with ThreadPoolExecutor(max_workers=min(len(ingredients), 5)) as ex:
        results = list(ex.map(_search_one, ingredients))

    return (
        "\n".join(results)
        + "\n\n※ 네이버 쇼핑 소매가 기준이며, KAMIS DB에 없는 재료에 한해 참고용으로 제공됩니다."
    )


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
