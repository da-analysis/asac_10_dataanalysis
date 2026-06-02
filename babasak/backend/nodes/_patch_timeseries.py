import os

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
# 교체할 구 코드 (SQL 부분만 정확히 매칭)
old_sql_block = '''    sql = f"""
SELECT `날짜`, `재료명`, `가격`, `단위`
FROM silver.ingredient.ingredient
WHERE `재료명` = '{safe_name}'
  AND `단위` = '{safe_unit}'
  AND `날짜` >= DATE_SUB(CURRENT_DATE(), {days})
  AND `가격` IS NOT NULL
ORDER BY `날짜`
""".strip()'''

new_sql_block = '''    sql = f"""
SELECT `날짜`, ROUND(AVG(`가격`)) AS `평균가격`
FROM silver.ingredient.ingredient
WHERE `재료명` = '{safe_name}'
  AND `단위` = '{safe_unit}'
  AND `날짜` >= DATE_SUB(CURRENT_DATE(), {days})
  AND `가격` IS NOT NULL
GROUP BY `날짜`
ORDER BY `날짜`
""".strip()'''

# 교체할 DataFrame 생성 + 텍스트 요약 부분
old_df_block = '''        df = pd.DataFrame(rows, columns=["날짜", "재료명", "가격", "단위"])
        df["가격"] = pd.to_numeric(df["가격"], errors="coerce")
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
        df = df.dropna(subset=["날짜", "가격"]).sort_values("날짜")

        text_lines = [f"[{ingredient_name} 최근 {days}일 가격 추이]"]
        for _, row in df.iterrows():
            text_lines.append(f"  {row['날짜'].strftime('%Y-%m-%d')}: ₩{int(row['가격']):,}/{db_unit}")

        return {"dataframe": df, "text": "\\n".join(text_lines), "sql": sql, "error": None}'''

new_df_block = '''        df = pd.DataFrame(rows, columns=["날짜", "평균가격"])
        df["평균가격"] = pd.to_numeric(df["평균가격"], errors="coerce")
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
        df = df.dropna(subset=["날짜", "평균가격"]).sort_values("날짜")

        # 가격 텍스트 요약 제거 — 차트만 표시 (가독성 향상)
        return {"dataframe": df, "text": "", "sql": sql, "error": None}'''

# 패치 적용
changed = False

if old_sql_block in content:
    content = content.replace(old_sql_block, new_sql_block)
    print("✅ SQL 블록 교체 완료 (GROUP BY + AVG)")
    changed = True
else:
    print("❌ SQL 블록 매칭 실패")

if old_df_block in content:
    content = content.replace(old_df_block, new_df_block)
    print("✅ DataFrame + 텍스트 요약 교체 완료")
    changed = True
else:
    print("❌ DataFrame 블록 매칭 실패")

# docstring도 업데이트
old_docstring = '    """시계열 가격 데이터 직접 SQL 조회 → DataFrame 반환.'
new_docstring = '    """시계열 가격 데이터 직접 SQL 조회 → 날짜별 평균가격 DataFrame 반환.'

if old_docstring in content:
    content = content.replace(old_docstring, new_docstring)
    print("✅ docstring 업데이트 완료")
    changed = True

if changed:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n✅ 파일 저장 완료: {file_path}")
else:
    print("\n⚠️ 변경 사항 없음 — 파일이 이미 수정되었거나 구조가 다릅니다.")
    # 디버깅
    idx = content.find("SELECT `날짜`, `재료명`, `가격`, `단위`")
    print(f"  'SELECT 날짜, 재료명' 위치: {idx}")
    idx2 = content.find("text_lines = [f\"[{ingredient_name}")
    print(f"  'text_lines' 위치: {idx2}")
