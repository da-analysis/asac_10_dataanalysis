"""
price_search.py의 _timeseries_sql_query 함수를 패치하는 스크립트.
실행 후 삭제해도 됩니다.
"""
import os

file_path = "/Workspace/Users/rimmyeb@gmail.com/asac_10_dataanalysis/babasak/backend/nodes/price_search.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

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
