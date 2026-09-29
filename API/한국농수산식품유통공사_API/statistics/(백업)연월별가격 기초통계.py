# Databricks notebook source
# MAGIC %md
# MAGIC # 📊 연월별 가격정보 기초통계 분석
# MAGIC
# MAGIC **데이터 소스**: `/Volumes/data_api/agrofood_peryearmonth/volumn`
# MAGIC
# MAGIC | 단계 | 내용 |
# MAGIC |------|------|
# MAGIC | 1 | 데이터 로드 & 구조 탐색 |
# MAGIC | 2 | 데이터 정제 |
# MAGIC | 3 | 기초통계 (전체/품목별/카테고리별/연도별/지역별) |
# MAGIC | 4 | 시계열 분석 (추세/계절성/변동성) |
# MAGIC | 5 | 시각화 |
# MAGIC | 6 | 결과 저장 |

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ 설정

# COMMAND ----------

dbutils.widgets.text("volume_path", "/Volumes/bronze_api/agrofood_peryearmonth/volumn", "분석 데이터 볼륨 경로")
vol_path = dbutils.widgets.get("volume_path")
print(f"분석 대상 경로: {vol_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1단계: 데이터 로드 & 구조 탐색

# COMMAND ----------

import os

# 파일 목록 확인
all_files = [f for f in os.listdir(vol_path) if f.endswith('.csv')]
all_files.sort()
print(f"CSV 파일 수: {len(all_files)}")
print(f"처음 5개: {all_files[:5]}")
print(f"마지막 5개: {all_files[-5:]}")

# COMMAND ----------

# Spark로 전체 CSV 로드
from pyspark.sql import functions as F
from pyspark.sql.window import Window

df_raw = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("encoding", "UTF-8")
    .csv(f"{vol_path}/*.csv")
)

print(f"전체 행 수: {df_raw.count():,}")
print(f"컬럼 수: {len(df_raw.columns)}")
print(f"\n컬럼 목록:")
for c in df_raw.columns:
    print(f"  - {c}")

# COMMAND ----------

# 스키마 및 샘플 확인
df_raw.printSchema()

# COMMAND ----------

# 각 컬럼에 값이 몇 종류씩 있는지
display(df_raw.agg(
    F.countDistinct("item_nm").alias("품목수"),
    F.countDistinct("ctgry_nm").alias("카테고리수"),
    F.countDistinct("se_nm").alias("구분수"),
    F.countDistinct("sgg_nm").alias("지역수"),
    F.countDistinct("grd_nm").alias("등급수"),
    F.countDistinct("vrty_nm").alias("품종수"),
    F.countDistinct("exmn_ym").alias("조사기간수")
))


# COMMAND ----------

# 카테고리별 데이터 건수 + 품목 수
display(df_raw.groupBy("ctgry_nm").agg(
    F.count("*").alias("데이터수"),
    F.countDistinct("item_nm").alias("품목수")
).orderBy(F.desc("데이터수")))


# COMMAND ----------

# 품목별로 데이터가 몇 건씩 있는지
display(df_raw.groupBy("item_nm").count().orderBy(F.desc("count")))


# COMMAND ----------

display(df_raw.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ##1_1 결측치 확인

# COMMAND ----------

# 결측치 확인

null_counts = df_raw.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c) 
    for c in df_raw.columns
])
display(null_counts)

# COMMAND ----------

# orgnl_reg_dt 원본등록일시, pmm_cfcntrng 월별진폭계수, pmm_cfcntvrtn 월별변동계수, pyy_cfcntrng 연별진폭계수, unit 단위, unit_sz 단위크기

df_null = df_raw.filter(F.col('orgnl_reg_dt').isNull())
display(df_null)

# COMMAND ----------

# MAGIC %md
# MAGIC ##1_2 데이터 없는 품목 파악

# COMMAND ----------

comp_path = "/Workspace/Users/biod1614@gmail.com/ref_sheet_품목코드.csv"

df_comp = spark.createDataFrame(
    pd.read_csv(comp_path, encoding="UTF-8")
)
display(df_comp)


# COMMAND ----------

# 품목명 비교: df_comp(참조시트) vs df_clean(실제 데이터)
comp_items = df_comp.select(F.col("품목명").alias("item_nm")).distinct()
clean_items = df_raw.select("item_nm").distinct()

# 참조시트에만 있는 품목 (실제 데이터에 없음)
only_in_comp = comp_items.subtract(clean_items).orderBy("item_nm")

print(f"참조시트 품목 수: {comp_items.count()}")
print(f"실제 데이터 품목 수: {clean_items.count()}")
print(f"\n--- 참조시트에만 있는 품목 ({only_in_comp.count()}개) ---")
display(only_in_comp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2단계: 데이터 정제

# COMMAND ----------

# 가격 컬럼명
PRICE_COL = "pmm_avgprc"             # 당월 평균가격
PRICE_CONV_COL = "pyy_avgprc"        # 전년 평균가격
DATE_COL = "exmn_ym"             # 날짜 컬럼 (YYYYMM 또는 YYYYMMDD)
ITEM_NAME_COL = "item_nm"         # 품목명
ITEM_CODE_COL = "item_cd"         # 품목코드
CTGRY_NAME_COL = "ctgry_nm"       # 카테고리명
CTGRY_CODE_COL = "ctgry_cd"       # 카테고리코드
REGION_NAME_COL = "sgg_nm"        # 지역명
GRADE_NAME_COL = "grd_nm"         # 등급명
UNIT_COL = "unit"                 # 단위

print(" 컬럼 매핑 설정 완료.")

# COMMAND ----------

# 타입 변환 및 파생 컬럼 생성
df = (df_raw
    .withColumn("price", F.col(PRICE_COL).cast("double"))
    .withColumn("price_conv", F.col(PRICE_CONV_COL).cast("double"))
    .filter(F.col("price").isNotNull() & (F.col("price") > 0))
)

# 날짜 처리 (YYYYMM 형식인 경우)
date_sample = df.select(DATE_COL).first()[0]
date_str = str(date_sample)

if len(date_str) == 6:  # YYYYMM
    df = (df
        .withColumn("year", F.substring(F.col(DATE_COL).cast("string"), 1, 4).cast("int"))
        .withColumn("month", F.substring(F.col(DATE_COL).cast("string"), 5, 2).cast("int"))
        .withColumn("date_ym", F.concat_ws("-",
            F.substring(F.col(DATE_COL).cast("string"), 1, 4),
            F.substring(F.col(DATE_COL).cast("string"), 5, 2)))
    )
    print(f"날짜 형식: YYYYMM (예: {date_str})")
elif len(date_str) == 8:  # YYYYMMDD
    df = (df
        .withColumn("date", F.to_date(F.col(DATE_COL).cast("string"), "yyyyMMdd"))
        .withColumn("year", F.year("date"))
        .withColumn("month", F.month("date"))
        .withColumn("date_ym", F.date_format("date", "yyyy-MM"))
    )
    print(f"날짜 형식: YYYYMMDD (예: {date_str})")
else:
    print(f"⚠️ 알 수 없는 날짜 형식: {date_str} (길이: {len(date_str)})")

print(f"정제 후 행 수: {df.count():,}")

# COMMAND ----------

# 결측치 현황
print("=== 결측치 현황 ===")
null_counts = df.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c)
    for c in [ITEM_NAME_COL, CTGRY_NAME_COL, "price", "price_conv", "year", "month"]
])
display(null_counts)

# COMMAND ----------

# 중복 확인
total = df.count()
distinct = df.distinct().count()
print(f"전체: {total:,} / 고유: {distinct:,} / 중복: {total - distinct:,}")

# COMMAND ----------

display(df.limit(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3단계: 기초통계

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-1. 전체 요약

# COMMAND ----------

display(df.select("price", "price_conv").describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-2. 품목별 통계

# COMMAND ----------

trend_raw = (df
    .select(DATE_COL, ITEM_NAME_COL, REGION_NAME_COL, PRICE_COL, PRICE_CONV_COL)
    .orderBy(ITEM_NAME_COL, DATE_COL)
)
display(trend_raw)

# COMMAND ----------

# 11월 전국 가리비 평균값
month_item_stats = (df
    .groupBy(DATE_COL, REGION_NAME_COL, ITEM_NAME_COL, GRADE_NAME_COL)
    .agg(
        F.count("*").alias("데이터수"),
        F.max("price").alias("가격최대값"),
        F.min("price").alias("가격최소값")
    )
    .orderBy(DATE_COL, REGION_NAME_COL, ITEM_NAME_COL)
)
display(month_item_stats)
print(month_item_stats.count())

# COMMAND ----------

item_stats = (df
    .groupBy(ITEM_NAME_COL, DATE_COL, REGION_NAME_COL)
    .agg(
        F.count("*").alias("데이터수"),
        F.first("price").alias("price"),
        F.round(F.median("price"), 0).alias("중앙값가격"),
        F.max("price").alias("최고가격"),
        F.min("price").alias("최저가격"),
        F.round(F.stddev("price"), 0).alias("가격표준편차"),
        F.countDistinct("date_ym").alias("데이터수(월별)")
    )
    #.withColumn("cv", F.round(F.col("std_price") / F.col("avg_price") * 100, 2))
    #.withColumn("price_range", F.col("max_price") - F.col("min_price"))
    .orderBy(ITEM_NAME_COL, DATE_COL)
)

print("=== 품목별 기초통계 (평균가격 상위) ===")
display(item_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-3. 카테고리(부류)별 통계

# COMMAND ----------

ctgry_stats = (df
    .groupBy(CTGRY_CODE_COL, CTGRY_NAME_COL,ITEM_NAME_COL,GRADE_NAME_COL)
    .agg(
        F.count("*").alias("데이터수"),
        F.round(F.max("price")).alias("최대가격"),
        F.round(F.min("price")).alias("최소가격"),
    )
    .orderBy(F.desc("데이터수"))
)

display(ctgry_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-4. 연도별 통계

# COMMAND ----------

# 품목 × 연월별로 나누면 exmn_dd_prc가 이미 1개씩이므로
# 별도 집계 없이 원본 월평균 가격을 그대로 볼 수 있음
year_item_stats_2 = (df
    .select("year", "month", ITEM_NAME_COL, CTGRY_NAME_COL, GRADE_NAME_COL, REGION_NAME_COL, PRICE_COL, PRICE_CONV_COL, )
    .orderBy("year", "month", "item_nm")
)

display(year_item_stats_2)


# COMMAND ----------

year_stats = (df
    .groupBy("year")
    .agg(
        F.count("*").alias("데이터수"),
        
    )
    .orderBy("year")
)

display(year_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-5. 지역별 통계

# COMMAND ----------

region_stats = (df
    .groupBy(REGION_NAME_COL)
    .agg(
        F.count("*").alias("데이터수"),
        
    )
    .orderBy(F.desc("데이터수"))
)

display(region_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ##시각화 시 한글폰트작업에 필요한 모듈

# COMMAND ----------

# MAGIC %pip install koreanize-matplotlib

# COMMAND ----------

import koreanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
