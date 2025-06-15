# Databricks notebook source
# MAGIC %md
# MAGIC 1. 데이터로딩, 폐업일자 필터링
# MAGIC 2. 필요한 컬럼 추출
# MAGIC 3. 시도/시군구/법정동 작업 \
# MAGIC 3-1) 위도/경도변환 \
# MAGIC 3-2) shp 파일로 시도 시군구 법정동 채우기 \
# MAGIC 3-3) API 테이블이랑 조인해서 값 채우고, null값 제거
# MAGIC 4. 사업장명 작업
# MAGIC 5. 개방서비스명 필터링 \
# MAGIC ㄴ 폐업유무 건수 합이 100이하 이거나 폐업(1) == 0인경우 제거
# MAGIC 6. 운영기간 범주화
# MAGIC 7. 모델 돌리기 \
# MAGIC 7-1) 계층적 샘플링 \
# MAGIC ㄴ 개방서비스명, 폐업유무 기준으로 (8:2) \
# MAGIC 7-2) 원핫인코딩 적용 \
# MAGIC 7-3) ColumnTransformer \
# MAGIC 7-4) XGBoost 파이프라인 \
# MAGIC 7-5) 모델 학습

# COMMAND ----------

# MAGIC %pip install xgboost pyproj geopandas optuna scikit-optimize imbalanced-learn   

# COMMAND ----------

import mlflow

mlflow.autolog(
log_input_examples=False,
log_model_signatures=True,
log_models=True,
disable=False,
exclusive=False,
disable_for_unsupported_versions=True,
silent=False
)

# COMMAND ----------

from pyspark.sql.functions import col, to_date, regexp_extract, year, struct
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.functions import pandas_udf

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

######################################################
# Step 1: 데이터 로딩, 폐업일자 필터링
######################################################

base_df = spark.table("silver.mois.mois_table_with_id")

print("기본 테이블 후 행 수 :", base_df.count())

# 폐업일자 → 날짜형
df_str_to_date = base_df.withColumn(
    "폐업일자",
    to_date(col("폐업일자"), "yyyy-MM-dd")
)

df_clean_date = (
    df_str_to_date
    .withColumn(
        "데이터갱신일자",
        regexp_extract(col("데이터갱신일자"), r"(^\d{4}-\d{2}-\d{2})", 0)
    )
    .withColumn(
        "최종수정시점",
        regexp_extract(col("최종수정시점"), r"(^\d{4}-\d{2}-\d{2})", 0)
    )
)

df_clean_date = df_clean_date.dropna(subset=["인허가일자"])


# 폐업일자: null이거나 2020년 이후인 데이터 필터
closed_filter_df = df_clean_date.filter(
    (col("폐업일자").isNull()) | (year(col("폐업일자")) >= 2020)
)



print("폐업일자 필터링 후 행 수 :", closed_filter_df.count())

# 폐업유무 라벨링: 폐업일자가 있으면 1, 없으면 0
closed_df = closed_filter_df.withColumn(
    "폐업유무",
    col("폐업일자").isNotNull().cast("integer")
)

print("폐업유무 라벨링 후 행 수 :", closed_df.count())

######################################################
# Step 2: 필요한 컬럼 추출
######################################################

# 필요한 컬럼만 선택
selected_columns = [
    "unique_id",
    "폐업일자",
    "개방서비스명",
    "업태구분명",
    "영업상태명",
    "영업상태구분코드",
    "상세영업상태명",
    "상세영업상태코드",
    "소재지면적",
    "위생업태명",
    "사업장명",
    "인허가일자",
    "최종수정시점",
    "데이터갱신일자",
    "데이터갱신구분",
    "좌표정보x_epsg5174_",
    "좌표정보y_epsg5174_"]

df_selected = closed_df.select(*selected_columns, "폐업유무")

print("컬럼 추출 후 행 수 :", df_selected.count())

######################################################
# Step 3: 위도/경도 변환 (EPSG:5174 → EPSG:4326)
######################################################
from pyproj import Transformer
from pyspark.sql.types import StructType, StructField, DoubleType

transformer = Transformer.from_crs("EPSG:5174", "EPSG:4326", always_xy=True)

schema = StructType([StructField("경도", DoubleType()), StructField("위도", DoubleType())])

@pandas_udf(schema)
def transform_coords(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    lon_arr, lat_arr = transformer.transform(x.values, y.values)
    return pd.DataFrame({"경도": lon_arr, "위도": lat_arr})

df_transformed = (
    df_selected
    .withColumn("coord", transform_coords("좌표정보x_epsg5174_", "좌표정보y_epsg5174_"))
    .selectExpr("*", "coord.`경도` as `경도`", "coord.`위도` as `위도`")
    .drop("coord")
)

print("위도/경도 변환 후 행 수 :", df_transformed.count())

######################################################
# Step 4: shp 파일로 시도 시군구 법정동 채우기
######################################################

# 1. 좌표만 뽑아서 distinct
coord_df = df_transformed.select("경도", "위도").dropna().distinct()
coords_pd = coord_df.toPandas()
coords_pd["geometry"] = coords_pd.apply(lambda row: Point(row["경도"], row["위도"]), axis=1)
gdf = gpd.GeoDataFrame(coords_pd, geometry="geometry", crs="EPSG:4326")

# 2. SHP 파일 로드 & 라벨링
sido_gdf = gpd.read_file("/Volumes/bronze/file_mois/shp/시도/ctp_rvn.shp", encoding="cp949").to_crs(epsg=4326)[["CTP_KOR_NM", "geometry"]]
sgg_gdf = gpd.read_file("/Volumes/bronze/file_mois/shp/시군구/sig.shp", encoding="cp949").to_crs(epsg=4326)[["SIG_KOR_NM", "geometry"]]
emd_gdf = gpd.read_file("/Volumes/bronze/file_mois/shp/읍면동/emd.shp", encoding="cp949").to_crs(epsg=4326)[["EMD_KOR_NM", "geometry"]]

# 시도
gdf = gpd.sjoin(gdf, sido_gdf, how="left", predicate="within")
gdf.rename(columns={"CTP_KOR_NM": "시도"}, inplace=True)
gdf.drop(columns=["index_right"], inplace=True)

# 시군구
gdf = gpd.sjoin(gdf, sgg_gdf, how="left", predicate="within")
gdf.rename(columns={"SIG_KOR_NM": "시군구"}, inplace=True)
gdf.drop(columns=["index_right"], inplace=True)

# 읍면동
gdf = gpd.sjoin(gdf, emd_gdf, how="left", predicate="within")
gdf.rename(columns={"EMD_KOR_NM": "법정동"}, inplace=True)
gdf.drop(columns=["index_right"], inplace=True)

labeled_coords = gdf[["경도", "위도", "시도", "시군구", "법정동"]] \
    .drop_duplicates(subset=["경도", "위도"])

# 3. Spark로 다시 변환 후 Join
labeled_coords_spark = spark.createDataFrame(labeled_coords)

df_with_shp = (
    df_transformed
    .join(labeled_coords_spark, on=["경도", "위도"], how="left")
)

# 샘플 출력
df_with_shp.filter(col("좌표정보x_epsg5174_").isNotNull()).limit(5).display()

print("시도 시군구 법정동 채운 후 행 수 :", df_with_shp.count())

######################################################
# Step 5: API 테이블이랑 조인해서 값 채우고, null값 제거
######################################################

from pyspark.sql.functions import when, col

# 1. API 데이터 로딩
df_api = spark.table("silver.mois.mois_address_info_v2")

# API 테이블에서 주소 정보가 있는 경우만 필터링
df_api_filtered = (
    df_api
    .filter(col("siNm").isNotNull() & col("sggNm").isNotNull() & col("emdNm").isNotNull())
    .select(
        col("uid").alias("unique_id"),
        col("siNm").alias("시도_api"),
        col("sggNm").alias("시군구_api"),
        col("emdNm").alias("법정동_api")
    )
)

print("API 테이블에서 주소 정보가 있는 경우만 필터링한 후 행 수 :", df_api_filtered.count())

# 그런 다음 조인
df_joined = df_with_shp.join(df_api_filtered, on="unique_id", how="left")

print("API 테이블 조인한 후 행 수 :", df_joined.count())

# 컬럼 정리
from pyspark.sql.functions import when, col

df_enriched = (
    df_joined
    .withColumn("시도", when(col("시도").isNull(), col("시도_api")).otherwise(col("시도")))
    .withColumn("시군구", when(col("시군구").isNull(), col("시군구_api")).otherwise(col("시군구")))
    .withColumn("법정동", when(col("법정동").isNull(), col("법정동_api")).otherwise(col("법정동")))
    .drop("시도_api", "시군구_api", "법정동_api")
)

print("컬럼 정리한 후 행 수 :", df_enriched.count())

# 시도, 시군구, 법정동 null값 제거
df_clean_addr = df_enriched.dropna(subset=["시도", "시군구", "법정동"])

print("시도, 시군구, 법정동 null값 제거 후 행 수 :", df_clean_addr.count())

######################################################
# Step 6: 사업장명 이상값 제거 및 전처리
######################################################

# 사업장명 이상값 제거 및 날짜 포맷 정리
invalid_values = ['무', '없음', '-', '.']
df_na_removed = df_clean_addr.dropna(subset=["사업장명"])
df_clean = df_na_removed.filter(~F.col("사업장명").isin(invalid_values))

print("사업장명 이상값 제거 후 행 수 :", df_clean.count())

######################################################
"""
1차 전처리
정제 결과가 없는(isnull()) 경우만 골라서 별도로 처리
그중 괄호 안 영어가 있는 경우 복구
지점, 지점명, 센터 등 제외
사업장명+지점명 붙어있는 경우 보류 처리
"""
######################################################

import re
import pandas as pd

# 사업장명 1차 전처리 함수 
def clean_bizname_final(name):
    if name is None or not isinstance(name, str):
        return None

    name = re.sub(r"\s+", " ", name).strip()
    name = name.replace('(주)', '').replace('( 주)', '').replace('(주', '').replace('주)', '').replace('( 주', '')
    name = name.replace('주식회사', '')
    name = name.replace('-한시적', '').replace(' -한시적', '').replace('(한시적)', '')

    eng_in_parens = re.findall(r"\(([^)]*[a-z][^)]*)\)", name)

    name = re.sub(r"\([^)]*\)", "", name).strip()
    name = re.sub(r"\s+\S{1,20}(지점|점|센터|분점)$", "", name)

    if re.search(r'[가-힣a-z0-9]{2,}(지점|점|센터|분점)$', name):
        return "보류"

    if re.fullmatch(r"[^\w\s]", name):
        return None

    name = name.strip()
    if name == "":
        return None

    if name == "주" and not eng_in_parens:
        return None

    if eng_in_parens:
        return f"{name} {' '.join(eng_in_parens)}"

    return name

# 괄호 안에 영어 있는 경우 전처리 함수 (괄호만 제거 후 소문자로 변경)
def recover_english_in_parentheses(name):
    if isinstance(name, str):
        match = re.search(r"\(([^)]*[a-zA-Z][^)]*)\)", name)
        if match:
            # 괄호 안에 영어 있는 경우
            clean_name = re.sub(r"[()]", "", name).strip()
            return clean_name.lower()
    return None 

# 1차 전처리 적용 함수 
def process_business_names(df_clean):
    df_bizname_pd = df_clean.select("unique_id", "사업장명").toPandas()

    df_bizname_pd["정제된사업장명"] = df_bizname_pd["사업장명"].apply(clean_bizname_final)
    df_bizname_pd["정제결과"] = df_bizname_pd["정제된사업장명"]

    valid_all_df = df_bizname_pd[df_bizname_pd["정제결과"].isnull()][["unique_id", "사업장명", "정제된사업장명", "정제결과"]].copy()

    valid_all_df["정제된사업장명"] = valid_all_df["사업장명"].apply(recover_english_in_parentheses)
    valid_all_df["정제결과"] = valid_all_df["정제된사업장명"]

    valid_all_df = valid_all_df[
        (valid_all_df["정제된사업장명"].notnull()) & 
        (valid_all_df["정제된사업장명"].str.lower() != "null")
    ].copy()

    cleaned_business_name_df = df_bizname_pd.copy()
    cleaned_business_name_df.set_index("unique_id", inplace=True)
    valid_all_df.set_index("unique_id", inplace=True)
    cleaned_business_name_df.update(valid_all_df[["정제된사업장명", "정제결과"]])
    cleaned_business_name_df.reset_index(inplace=True)

    return cleaned_business_name_df

######################################################
"""
2차 전처리
보류로 남긴 것 2차 전처리
1차 전처리 완료된 것 중에서 사전 생성하여 사업장명 추출 후 적용
"""
######################################################

# 보류로 처리한 데이터 전처리하는데 필요한 함수들 

# 관련 텍스트만 제거하는 함수
def clean_parentheses_ju(name):
    if name is None or not isinstance(name, str):
        return None
    name = re.sub(r"\s*\( ?주\)?\s*", "", name)
    name = name.replace('주식회사', '')
    return name.strip()

# 브랜드 매칭 함수
def extract_from_dict_better(name, brand_list):
    if not isinstance(name, str):
        return None

    for brand in brand_list:
        if len(brand) <= 2:
            if re.search(r'[a-z]', brand):
                if re.search(rf'\b{re.escape(brand)}\b', name):
                    return brand
            else:
                if brand in name:
                    return brand
        else:
            if brand in name:
                return brand
    return None

# 최종 전처리 함수
def process_final_business_names(df_clean, brand_top_n=500, min_brand_freq=500, words_to_remove=None):
    if words_to_remove is None:
        words_to_remove = [
            "중앙", "잡화", "우리", "우정", "제일",
            "그린", "대성", "드림", "타임", "스타",
            "마루", "명가", "정성", "코코", "현대"
        ]

    df_bizname_cleaned = df_clean.copy()

    df_hold = df_bizname_cleaned[df_bizname_cleaned["정제결과"] == "보류"].copy()
    df_vocab_source = df_bizname_cleaned[df_bizname_cleaned["정제결과"] != "보류"].copy()

    # 단어사전 만들기
    brand_dict = (
        df_vocab_source[df_vocab_source["정제결과"].str.len() >= 2]["정제결과"]
        .value_counts()
        .head(brand_top_n)
        .index
        .tolist()
    )
    brand_dict_sorted = sorted(brand_dict, key=lambda x: -len(x))  # 긴 것부터 매칭

    # 보류 데이터 전처리
    df_hold["사업장명_cleaned"] = df_hold["사업장명"].apply(clean_parentheses_ju)

    # 대표상호 추출
    df_hold["대표상호"] = df_hold["사업장명_cleaned"].apply(lambda x: extract_from_dict_better(x, brand_dict_sorted))

    # 등장횟수 기준 필터
    brand_counts = df_hold["대표상호"].value_counts()
    valid_brands_final = brand_counts[brand_counts >= min_brand_freq].index.tolist()
    df_hold_filtered = df_hold[df_hold["대표상호"].isin(valid_brands_final)].copy()

    # 의미 없는 단어 제거
    df_hold_filtered.loc[
        df_hold_filtered["대표상호"].isin(words_to_remove),
        "대표상호"
    ] = None

    # None 제거
    df_hold_valid = df_hold_filtered[df_hold_filtered["대표상호"].notnull()].copy()

    # 정제된사업장명, 정제결과 갱신
    df_hold_valid["정제된사업장명"] = df_hold_valid["대표상호"]
    df_hold_valid["정제결과"] = df_hold_valid["대표상호"]

    # 필요한 컬럼만 유지
    df_hold_valid = df_hold_valid[["unique_id", "사업장명", "정제된사업장명", "정제결과"]].copy()

    # 보류아닌 것과 합치기
    final_business_name_df = pd.concat([df_vocab_source, df_hold_valid], ignore_index=True)

    return final_business_name_df

######################################################
# 임베딩을 위한 전처리
######################################################

import re

# 숫자만 있는 경우 처리 함수
def is_only_number_except_year(name):
    if not isinstance(name, str):
        return False
    if name.isdigit():
        return len(name) == 4
    return True

# 특수문자 정리 함수
def clean_special_tokens(name):
    if not isinstance(name, str):
        return name
    name = re.sub(r"[()]", "", name)
    name = name.replace(",", " ").replace("/", " ")
    name = re.sub(r"[^가-힣a-z0-9\s\-&\.·]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

# 임베딩을 위한 전처리 함수 
def preprocess_for_embedding(df_clean):
    # 브랜드 통일 매핑
    brand_standard_mapping = {
        "씨유": "CU",
        "씨유 CU": "CU",
        "CU": "CU",
        "지에스25": "GS25",
        "지에스25 GS": "GS25",
        "지에스25 GS25": "GS25",
        "GS25": "GS25",
        "이디야": "이디야커피",
        "이디야커피": "이디야커피",
    }
    df_clean["정제된사업장명"] = df_clean["정제된사업장명"].replace(brand_standard_mapping)
    df_clean["정제결과"] = df_clean["정제결과"].replace(brand_standard_mapping)

    # 소문자 변환
    df_clean["정제된사업장명"] = df_clean["정제된사업장명"].str.lower()
    df_clean["정제결과"] = df_clean["정제결과"].str.lower()

    # Null 제거
    df_clean = df_clean[df_clean["정제된사업장명"].notnull()].copy()

    # 숫자만 존재하는 경우 처리
    df_clean = df_clean[df_clean["정제된사업장명"].apply(is_only_number_except_year)].copy()

    # 특수문자 정리
    df_clean["정제된사업장명"] = df_clean["정제된사업장명"].apply(clean_special_tokens)
    df_clean["정제결과"] = df_clean["정제결과"].apply(clean_special_tokens)

    # 1글자 이하 제거
    df_clean = df_clean[df_clean["정제된사업장명"].apply(lambda x: len(x.strip()) > 1)].copy()

    return df_clean

######################################################
# 최종 전처리 완료된 df 생성
######################################################

# 1차 기본 전처리
cleaned_business_name_df = process_business_names(df_clean)

# 2차 전처리 
final_business_name_df = process_final_business_names(cleaned_business_name_df)

# 임베딩용 최종 전처리 적용
final_business_name_df = preprocess_for_embedding(final_business_name_df)

print("사업장명 전처리 후 행 수:", len(cleaned_business_name_df)) 
print("최종 전처리 완료된 데이터프레임 수:", len(final_business_name_df)) 
print("임베딩용 전처리 완료 후 데이터 수:", len(final_business_name_df))
display(final_business_name_df) 

######################################################
# 테이블에 적용
######################################################
from pyspark.sql.types import StructType, StructField, StringType, LongType

# 1. 스키마 정의
schema = StructType([
    StructField("unique_id", LongType(), True),        # int64 -> LongType
    StructField("사업장명", StringType(), True),        # object -> StringType
    StructField("정제된사업장명", StringType(), True),
    StructField("정제결과", StringType(), True),
])

# 2. pandas DataFrame → Spark DataFrame 변환
final_business_name = spark.createDataFrame(final_business_name_df, schema=schema)

# 3. 정제된사업장명 컬럼만 df_clean에 left join으로 추가 (기존 row 보존)
df_final = df_clean.join(
    final_business_name.select("unique_id", "정제된사업장명"),
    on="unique_id",
    how="left"
)

df_final = df_final.filter(col("정제된사업장명").isNotNull())

######################################################
# Step 7: 개방서비스명 필터링
######################################################
from pyspark.sql.functions import col, sum as _sum, count, when
from pyspark.sql.functions import trim

agg_df = df_final.groupBy("개방서비스명").agg(
    count("*").alias("전체건수"),
    _sum("폐업유무").alias("폐업건수")
)

print("개방서비스명 그룹화한 후 행 수 :", agg_df.count())

# 조건에 해당하는 개방서비스명 필터링 (제외 대상)
to_remove = agg_df.filter((col("전체건수") <= 100) | (col("폐업건수") == 0)) \
                  .select("개방서비스명")

print("조건에 해당하는 개방서비스명 제외 대상 :", to_remove.count())

# 원래 df에서 제외
result_df = df_final.join(to_remove, on="개방서비스명", how="left_anti")

print("원래 df에서 제외한 후 행 수 :", result_df.count())

######################################################
# Step 8: 운영기간 범주화
######################################################
from pyspark.sql.functions import datediff, lit, round, when

# 기준일 정의
cutoff_date = lit("2025-03-31")

period_df = result_df.withColumn(
    '종료일자',
    when(col('폐업일자').isNotNull(), col('폐업일자')).otherwise(cutoff_date)
)

# 운영기간(연) 계산
period_df = period_df.withColumn('운영기간', round(datediff(col('종료일자'), col('인허가일자')) / 365.25, 2))

# 운영기간 범주 생성
period_df = period_df.withColumn(
    "운영기간_범주",
    when(col("운영기간") < 1, "운영기간_1년미만")
    .when((col("운영기간") >= 1) & (col("운영기간") < 5), "운영기간_5년미만")
    .when((col("운영기간") >= 5) & (col("운영기간") < 10), "운영기간_10년미만")
    .when((col("운영기간") >= 10) & (col("운영기간") < 15), "운영기간_15년미만")
    .otherwise("운영기간_15년이상")
)

# 필요없는 컬럼 제외
period_df_result = period_df.drop("종료일자", "경도", "위도", "사업장명")

period_df_result.select("인허가일자", "폐업일자", "운영기간", "운영기간_범주").show(10, truncate=False)

print("운영기간 범주화한 후 행 수 :", period_df_result.count())

# COMMAND ----------

# ## 작업 완료한 것 델타 테이블에 저장

# period_df_result.write \
#     .format("delta") \
#     .mode("overwrite") \
#     .saveAsTable("gold.ml_result_mois.pipeline_df")

# COMMAND ----------

("gold.ml_result_mois.pipeline_df")
period_df_result.limit(5).display()

# COMMAND ----------

######################################################
# Step 9: 모델 돌리기
######################################################

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer

######################################################
# 1. 계층적 샘플링
######################################################

from sklearn.model_selection import train_test_split

df = spark.table("gold.ml_result_mois.pipeline_df")

# 모델에 사용할 컬럼들 Pandas로 변환
model_df = df.select(
    "개방서비스명", "운영기간_범주", "시도", "시군구", "폐업유무"
).toPandas()

# 1. stratify_key 생성: 개방서비스명 + 폐업유무
model_df['stratify_key'] = model_df['개방서비스명'].astype(str) + "_" + model_df['폐업유무'].astype(str)

# 2. stratify_key별 개수 확인 및 2개 이상만 필터링
valid_keys = model_df['stratify_key'].value_counts()
valid_keys = valid_keys[valid_keys >= 2].index.tolist()
model_df_filtered = model_df[model_df['stratify_key'].isin(valid_keys)].copy()

# 3. X / y / stratify 나누기
X = model_df_filtered.drop(["폐업유무", "stratify_key"], axis=1)
y = model_df_filtered["폐업유무"]
stratify_col = model_df_filtered["stratify_key"]

# 4. Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=stratify_col, random_state=42
)
######################################################
# 원핫인코딩 적용
######################################################

categorical_features = ['개방서비스명', '운영기간_범주', '시도', '시군구']

# 원핫인코딩
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ]
)

# XGBoost 파이프라인
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6
    ))
])

## 모델 학습
clf.fit(X_train, y_train)

# COMMAND ----------

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# 예측값
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # XGBClassifier 확률 출력

# 정확도
print("Accuracy:", accuracy_score(y_test, y_pred))

# ROC AUC
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# 분류 리포트
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 혼동 행렬
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
