# Databricks notebook source
# MAGIC %md
# MAGIC # 일별 가격 데이터 병합 및 기초통계
# MAGIC
# MAGIC **데이터 소스**: `/Volumes/data_api/agrofood_perday/test`
# MAGIC
# MAGIC **기간**: 2016-01-04 ~ 2026-04-13 (약 2,903개 CSV)
# MAGIC
# MAGIC | 단계 | 작업 |
# MAGIC |------|------|
# MAGIC | 1 | 데이터 병합 |
# MAGIC | 2 | 데이터 정제 |
# MAGIC | 3 | 기초통계 |
# MAGIC | 4 | 시각화 |
# MAGIC | 5 | 결과 저장 |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1단계: 데이터 병합
# MAGIC Spark로 전체 CSV를 한 번에 읽어 병합합니다.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *

vol_path = "/Volumes/bronze_api/agrofood_perday/test"

# Spark로 전체 CSV 한 번에 읽기 (가장 빠른 방법)
df_spark = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("encoding", "UTF-8")
    .csv(f"{vol_path}/*.csv")
)

total_rows = df_spark.count()
total_cols = len(df_spark.columns)

print(f"전체 행 수: {total_rows:,}")
print(f"전체 컬럼 수: {total_cols}")
print(f"\n컬럼 목록:")
for col in df_spark.columns:
    print(f"  - {col}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2단계: 데이터 정제

# COMMAND ----------

display(df_spark.select("item_nm").distinct())

# COMMAND ----------

# 스키마 확인
df_spark.printSchema()

# COMMAND ----------

# 각 컬럼에 값이 몇 종류씩 있는지
display(df_spark.agg(
    F.countDistinct("item_nm").alias("품목수"),
    F.countDistinct("ctgry_nm").alias("카테고리수"),
    F.countDistinct("mrkt_nm").alias("시장수"),
    F.countDistinct("se_nm").alias("구분수"),
    F.countDistinct("sgg_nm").alias("지역수"),
    F.countDistinct("grd_nm").alias("등급수"),
    F.countDistinct("vrty_nm").alias("품종수"),
    F.countDistinct("exmn_ymd").alias("조사기간수")
))

# COMMAND ----------

# 가격 컬럼 숫자 변환 및 날짜 변환
df_clean = (df_spark
    .withColumn("exmn_dd_prc", F.col("exmn_dd_prc").cast("double"))
    .withColumn("exmn_dd_cnvs_prc", F.col("exmn_dd_cnvs_prc").cast("double"))
    .withColumn("exmn_date", F.expr("try_to_date(exmn_ymd, 'yyyyMMdd')"))
    .withColumn("year", F.year("exmn_date"))
    .withColumn("month", F.month("exmn_date"))
    .withColumn("day_of_week", F.dayofweek("exmn_date"))
)

print(f"정제 후 행 수: {df_clean.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##1_1 결측치 확인

# COMMAND ----------

# 결측치 확인
from pyspark.sql import functions as F

print("=== 결측치 현황 ===")
null_counts = df_clean.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c) 
    for c in df_clean.columns
])
display(null_counts)

# COMMAND ----------

df_null = df_clean.filter(F.col("mrkt_cd").isNull())

display(df_null.select("ctgry_cd", "ctgry_nm","item_nm", "se_nm","sgg_nm", "exmn_ymd", "mrkt_cd", "mrkt_nm", "orgnl_reg_dt"))

# COMMAND ----------

display(df_null.select("se_nm").distinct())

# COMMAND ----------

display(df_null.select("exmn_ymd").distinct().orderBy(F.desc("exmn_ymd")))

# COMMAND ----------

# MAGIC %md
# MAGIC 결측치 상황-> 소매로 거래된 축산물 품목에서 결측치 확인

# COMMAND ----------

# MAGIC %md
# MAGIC ##1_2 데이터없는 품목 파악

# COMMAND ----------

comp_path = "/Workspace/Users/biod1614@gmail.com/ref_sheet_품목코드.csv"

df_comp = spark.createDataFrame(
    pd.read_csv(comp_path, encoding="UTF-8")
)
display(df_comp)


# COMMAND ----------

# DBTITLE 1,품목명 비교: 참조시트 vs 실제데이터
# 품목명 비교: df_comp(참조시트) vs df_clean(실제 데이터)
comp_items = df_comp.select(F.col("품목명").alias("item_nm")).distinct()
clean_items = df_clean.select("item_nm").distinct()

# 참조시트에만 있는 품목 (실제 데이터에 없음)
only_in_comp = comp_items.subtract(clean_items).orderBy("item_nm")

print(f"참조시트 품목 수: {comp_items.count()}")
print(f"실제 데이터 품목 수: {clean_items.count()}")
print(f"\n--- 참조시트에만 있는 품목 ({only_in_comp.count()}개) ---")
display(only_in_comp)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

display(df_clean.limit(10))

# COMMAND ----------

# 중복 확인
total = df_clean.count()
distinct = df_clean.distinct().count()
print(f"전체 행: {total:,}")
print(f"고유 행: {distinct:,}")
print(f"중복 행: {total - distinct:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3단계: 기초통계

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-1. 전체 요약 통계

# COMMAND ----------

display(df_clean.select("exmn_dd_cnvs_prc","exmn_dd_prc").describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-2. 품목별 통계

# COMMAND ----------

# 카테고리별 데이터 건수 + 품목 수
display(df_raw.groupBy("ctgry_nm").agg(
    F.count("*").alias("데이터수"),
    F.countDistinct("item_nm").alias("품목수")
).orderBy(F.desc("데이터수")))

# COMMAND ----------

# 품목별로 데이터가 몇 건씩 있는지
display(df_spark.groupBy("item_nm").count().orderBy(F.desc("count")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-3. 연도별 통계

# COMMAND ----------

year_stats = (df_clean
    .groupBy("year")
    .agg(
        F.count("*").alias("데이터수"),
        F.round(F.avg("exmn_dd_prc"), 0).alias("평균가격"),
        F.countDistinct("item_nm").alias("품목수"),
        F.countDistinct("exmn_date").alias("조사일수")
    )
    .orderBy("year")
)

display(year_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-4. 카테고리(부류)별 통계

# COMMAND ----------

ctgry_stats = (df_clean
    .groupBy("ctgry_cd", "ctgry_nm")
    .agg(
        F.count("*").alias("데이터수"),
        F.round(F.avg("exmn_dd_prc"), 0).alias("평균가격"),
        F.countDistinct("item_nm").alias("품목수"),
        F.round(F.stddev("exmn_dd_prc"), 0).alias("가격표준편차")
    )
    .orderBy(F.desc("데이터수"))
)

display(ctgry_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-5. 지역별 통계

# COMMAND ----------

region_stats = (df_clean
    .groupBy("sgg_cd", "sgg_nm")
    .agg(
        F.count("*").alias("데이터수"),
        F.round(F.avg("exmn_dd_prc"), 0).alias("평균가격"),
        F.countDistinct("item_nm").alias("품목수")
    )
    .orderBy(F.desc("데이터수"))
)

display(region_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-6. 월별 가격 추이 (연도x월)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 시각화 시 필요한 한글폰트 모듈 다운로드

# COMMAND ----------

# MAGIC %pip install koreanize-matplotlib

# COMMAND ----------

import koreanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# COMMAND ----------

#plt.rc('font', family='NanumGothic')  # 한글 폰트 설정 (윈도우)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 품목별 평균가격 Top 15
item_pd = item_stats.limit(15).toPandas()

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(item_pd['item_nm'], item_pd['평균가격'], color='steelblue')
ax.set_xlabel('평균가격 (원)')
ax.set_title('평균가격 상위 15개 품목')
ax.invert_yaxis()

for bar, val in zip(bars, item_pd['평균가격']):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
            f'{val:,.0f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()

# COMMAND ----------

# 연도별 평균가격 추이
year_pd = year_stats.toPandas()

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.bar(year_pd['year'], year_pd['데이터수'], color='lightblue', alpha=0.7, label='Data Count')
ax1.set_ylabel('Data Count')
ax1.set_xlabel('Year')

ax2 = ax1.twinx()
ax2.plot(year_pd['year'], year_pd['평균가격'], color='red', marker='o', linewidth=2, label='Avg Price')
ax2.set_ylabel('Average Price (Won)')

ax1.set_title('Yearly Price Trend & Data Volume')
fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.88))
plt.tight_layout()
plt.show()

# COMMAND ----------

# 카테고리별 평균가격
ctgry_pd = ctgry_stats.toPandas()

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(ctgry_pd['ctgry_nm'], ctgry_pd['평균가격'], color='coral')
ax.set_xlabel('Average Price (Won)')
ax.set_title('Average Price by Category')
ax.invert_yaxis()
plt.tight_layout()
plt.show()
