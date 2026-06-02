"""
price_search.py: 기간 파싱을 동적으로 개선.

문제: "세달", "6개월", "1년" 등 _PERIOD_MAP에 없는 기간은 기본값 7일로 폴백.
해결: 정규식으로 "N개월", "N달", "N일", "N주", "N년" 등을 동적 파싱.
      기본값도 30일로 변경 (추이를 물어보는 사용자에겐 7일보다 30일이 적절).
"""

file_path = "/Workspace/Users/rimmyeb@gmail.com/asac_10_dataanalysis/babasak/backend/nodes/price_search.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

old_block = '''_PERIOD_MAP = {
    "일주일": 7, "1주일": 7, "7일": 7, "한주": 7,
    "2주": 14, "이주일": 14,
    "한달": 30, "1달": 30, "한 달": 30, "30일": 30, "1개월": 30,
    "두달": 60, "2달": 60, "2개월": 60,
    "3개월": 90, "석달": 90,
}


def _detect_trend_request(user_query: str) -> bool:
    """사용자 질문이 시계열 추이 요청인지 감지."""
    return bool(_TREND_KEYWORDS.search(user_query))


def _extract_trend_days(user_query: str) -> int:
    """추이 기간 추출. 기본 7일."""
    for keyword, days in _PERIOD_MAP.items():
        if keyword in user_query:
            return days
    return 7'''

new_block = '''_PERIOD_MAP = {
    "일주일": 7, "1주일": 7, "7일": 7, "한주": 7,
    "2주": 14, "이주일": 14,
    "한달": 30, "1달": 30, "한 달": 30, "30일": 30, "1개월": 30,
    "두달": 60, "2달": 60, "2개월": 60,
    "세달": 90, "석달": 90, "3달": 90, "3개월": 90,
    "넉달": 120, "4달": 120, "4개월": 120,
    "5개월": 150, "5달": 150,
    "반년": 180, "6개월": 180, "6달": 180,
    "1년": 365, "일년": 365, "12개월": 365,
}
# 동적 기간 파싱 정규식: "N개월", "N달", "N일", "N주", "N년"
_PERIOD_DYNAMIC_RE = re.compile(r"(\\d+)\\s*(개월|달|일|주|주일|년)")
# 한글 숫자 → 정수 매핑
_KOR_NUM = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
_PERIOD_KOR_RE = re.compile(r"(" + "|".join(_KOR_NUM.keys()) + r")\\s*(개월|달|일|주|주일|년)")


def _detect_trend_request(user_query: str) -> bool:
    """사용자 질문이 시계열 추이 요청인지 감지."""
    return bool(_TREND_KEYWORDS.search(user_query))


def _extract_trend_days(user_query: str) -> int:
    """추이 기간 추출. 기간 제한 없이 자유롭게 지원. 기본 30일."""
    # 1단계: 고정 키워드 매칭 (정확한 표현 우선)
    for keyword, days in _PERIOD_MAP.items():
        if keyword in user_query:
            return days
    # 2단계: 동적 숫자 파싱 ("5개월", "14일", "2주" 등)
    m = _PERIOD_DYNAMIC_RE.search(user_query)
    if m:
        num = int(m.group(1))
        unit = m.group(2)
        if unit in ("개월", "달"):
            return num * 30
        elif unit == "일":
            return num
        elif unit in ("주", "주일"):
            return num * 7
        elif unit == "년":
            return num * 365
    # 3단계: 한글 숫자 파싱 ("세 달", "다섯 개월" 등)
    km = _PERIOD_KOR_RE.search(user_query)
    if km:
        num = _KOR_NUM[km.group(1)]
        unit = km.group(2)
        if unit in ("개월", "달"):
            return num * 30
        elif unit == "일":
            return num
        elif unit in ("주", "주일"):
            return num * 7
        elif unit == "년":
            return num * 365
    # 기본값: 30일 (추이를 물어보는 사용자에겐 7일보다 30일이 더 적절)
    return 30'''

if old_block in content:
    content = content.replace(old_block, new_block)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("OK: 동적 기간 파싱으로 업그레이드 완료")
    print("  - 세달/넉달/반년/1년 등 추가")
    print("  - 'N개월', 'N달', 'N일', 'N주', 'N년' 동적 파싱")
    print("  - 한글 숫자 (세/네/다섯...) 파싱")
    print("  - 기본값 7일 → 30일로 변경")
else:
    print("FAIL: 매칭 실패")
    idx = content.find("_PERIOD_MAP")
    print(f"  _PERIOD_MAP 위치: {idx}")
