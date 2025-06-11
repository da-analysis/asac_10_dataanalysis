# Databricks notebook source
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

# MAGIC %md
# MAGIC ## 전처리

# COMMAND ----------

# MAGIC %md
# MAGIC ### 기본 전처리

# COMMAND ----------

df = spark.table("bronze.file_smartpolicing.`100대_생활밀접업종_개_폐업수_률`")
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC 변수명 정리  
# MAGIC
# MAGIC STRD_YR_CD	기준년코드	VARCHAR	4	Y	N  
# MAGIC STRD_QTR_CD	기준분기코드	VARCHAR	1	Y	N  
# MAGIC ADSTRD_CD	행정동코드	VARCHAR	10	Y	N  
# MAGIC ADSTRD_CD_NM	행정동코드명	VARCHAR	255	N	Y  
# MAGIC SRVIC_INDUTY_CD	서비스업종코드	VARCHAR	8	Y	N  
# MAGIC SRVIC_INDUTY_CD_NM	서비스업종명	VARCHAR	100	N	Y  
# MAGIC STR_CNT	점포수	NUMBER	20	N	Y  
# MAGIC SMLRT_INDUTY_STR_CNT	유사업종점포수	NUMBER	10	N	Y  
# MAGIC OPBIZ_RATE	개업율	NUMBER	10	N	Y  
# MAGIC OPBIZ_STR_CNT	개업점포수	NUMBER	10	N	Y  
# MAGIC CUS_RT	폐업률	NUMBER	10	N	Y  
# MAGIC CUS_STR_CNT	폐업점포수	NUMBER	10	N	Y  
# MAGIC FRCHS_STR_CNT	프랜차이즈점포수	NUMBER	10	N	Y  

# COMMAND ----------

from pyspark.sql.functions import col, round, when, lag
from pyspark.sql.window import Window

# CAL_CUS_RT 계산 (폐업률) 
df_with_rate = df.withColumn(
    "CAL_CUS_RT", #폐업 점포수(CUS_STR_CNT) / 전체 점포수(STR_CNT)
    when(col("STR_CNT") > 0, round((col("CUS_STR_CNT") / col("STR_CNT")) * 100, 2))
)

# 연도 + 분기 숫자형 컬럼 생성
df_with_rate = df_with_rate.withColumn(
    "YEAR_QTR",
    col("STRD_YR_CD") * 10 + col("STRD_QTR_CD")
)

# 이전 연도분기 계산
df_with_rate = df_with_rate.withColumn(
    "PREV_YEAR_QTR",
    when(col("STRD_QTR_CD") > 1,
         col("STRD_YR_CD") * 10 + (col("STRD_QTR_CD") - 1)
    ).otherwise(
         (col("STRD_YR_CD") - 1) * 10 + 4
    )
)

# 업종 + 행정동 단위로 정렬
window_spec = Window.partitionBy("SRVIC_INDUTY_CD_NM", "ADSTRD_CD_NM").orderBy("YEAR_QTR")

# 이전 분기 점포 수, 연도분기 적용
df_with_rate = df_with_rate.withColumn("PREV_STR_CNT_TEMP", lag("STR_CNT").over(window_spec))
df_with_rate = df_with_rate.withColumn("PREV_YEAR_QTR_LAG", lag("YEAR_QTR").over(window_spec))

# 이전 점포 수(PREV_STR_CNT) 계산
df_with_rate = df_with_rate.withColumn(
    "PREV_STR_CNT",
    when(col("PREV_YEAR_QTR_LAG") == col("PREV_YEAR_QTR"), col("PREV_STR_CNT_TEMP"))
)

df_model= df_with_rate.select(
    "STRD_YR_CD", "STRD_QTR_CD", "YEAR_QTR", "PREV_YEAR_QTR",
    "ADSTRD_CD_NM", "SRVIC_INDUTY_CD_NM",
    "STR_CNT", "PREV_STR_CNT", "CUS_STR_CNT", "CUS_RT", "CAL_CUS_RT"
)

display(df_model)


# COMMAND ----------

# 이전분기 점포 수 (PREV_STR_CNT) 첫 분기(2014년 1월)가 아닌 NULL값 확인
df_model.filter(
    col("PREV_STR_CNT").isNull() & (col("YEAR_QTR") != 20141)
).display()

# COMMAND ----------

# YEAR_QTR가 20141이 아닌 null 값은 0으로(전년 행정동&업종 없음)
from pyspark.sql.functions import when, col

df_model = df_model.withColumn(
    "PREV_STR_CNT",
    when((col("PREV_STR_CNT").isNull()) & (col("YEAR_QTR") != 20141), 0)
    .otherwise(col("PREV_STR_CNT"))
)


# COMMAND ----------

df_model.filter(col("YEAR_QTR") == 20141).count()

# COMMAND ----------

df_model.filter(
    (col("YEAR_QTR") == 20141) & col("PREV_STR_CNT").isNull() # 첫분기라 불러올 이전 데이터가 없어서 null값
).count()

# COMMAND ----------

# MAGIC %md
# MAGIC 차이 알아보기 위해 "CAL_CUS_RT", "CUS_RT" 쌍체 t-검정

# COMMAND ----------

# 둘의 차이 알아보기 위해 "CAL_CUS_RT", "CUS_RT" 쌍체 t-검정
df_model = df_model.withColumn(
    "CUS_RT",
    when(col("CUS_RT").isNull(), 0).otherwise(col("CUS_RT"))
)

df_pd = df_model.select("CAL_CUS_RT", "CUS_RT").dropna().toPandas()

# t-검정
from scipy.stats import ttest_rel

t_stat, p_value = ttest_rel(df_pd["CAL_CUS_RT"], df_pd["CUS_RT"])

print(f" 쌍체 t-검정 결과\nT-statistic: {t_stat:.3f}\nP-value: {p_value:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC 폐업률(CUS_RT)이 전체 사업체 수에 대한 해당 분기 폐업 사업체 수의 비율(%) 이라고 하였지만, 계산해보니 차이가 존재함을 발견. 계산한 CAL_CUS_RT을 사용할 것

# COMMAND ----------

from pyspark.sql.functions import dense_rank
from pyspark.sql.window import Window

# 업종 인코딩
w_upjong = Window.orderBy("SRVIC_INDUTY_CD_NM")
df_model = df_model.withColumn("업종_index", dense_rank().over(w_upjong) - 1)

# 행정동 인코딩
w_region = Window.orderBy("ADSTRD_CD_NM")
df_model = df_model.withColumn("행정동_index", dense_rank().over(w_region) - 1)


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### (추가)분포도 확인

# COMMAND ----------

!wget -O /tmp/NanumGothic.ttf https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 폰트 경로 지정 및 객체 생성
font_path = "/tmp/NanumGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)

# 폰트 이름 적용 (자동 인식)
plt.rcParams['font.family'] = font_prop.get_name()

# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(6, 4))
plt.title("한글 폰트 적용 테스트", fontproperties=font_prop)
plt.xlabel("X축", fontproperties=font_prop)
plt.ylabel("Y축", fontproperties=font_prop)
plt.plot([1, 2, 3], [3, 2, 1])
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import seaborn as sns

# 1. 한글 폰트 로드 및 설정
font_path = "/tmp/NanumGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)
font_name = font_prop.get_name()

mpl.rcParams['font.family'] = font_name
mpl.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# 2. 시각화할 컬럼 목록
columns_to_plot = [
    "업종_index", "행정동_index", "PREV_STR_CNT", "YEAR_QTR", "CAL_CUS_RT"
]

n_cols = 3
n_rows = (len(columns_to_plot) + n_cols - 1) // n_cols

plt.figure(figsize=(n_cols * 5, n_rows * 4))

for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(df_model.select(col).toPandas().dropna(), kde=True, bins=30)
    plt.title(f"{col} 분포", fontproperties=font_prop)
    plt.xlabel(col, fontproperties=font_prop)
    plt.ylabel("Count", fontproperties=font_prop)

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC - 업종_index, 행정동_index는 범주형이기에 그대로
# MAGIC - PREV_STR_CNT, CAL_CUS_RT는 로그 변환 필요
# MAGIC - YEAR_QTRsms는 연도/분기 분리

# COMMAND ----------

# MAGIC %sql
# MAGIC select STR_CNT, count(*) cnts
# MAGIC from bronze.file_smartpolicing.`100대_생활밀접업종_개_폐업수_률`
# MAGIC group by STR_CNT
# MAGIC order by STR_CNT asc

# COMMAND ----------

df_model.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### STR_CNT >= 20

# COMMAND ----------

from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ✅ 한글 폰트 설정
font_path = "/tmp/NanumGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

# ✅ 1. 점포 수 20 이상 + 폐업률 존재 필터링 → Spark DataFrame
filtered_20_df = df_model.filter(
    (col("STR_CNT") >= 20) & col("CAL_CUS_RT").isNotNull()
)

# ✅ 2. Pandas 변환 (→ df_model_20)
df_model_20 = filtered_20_df.toPandas().reset_index(drop=True)

# ✅ 3. 시각화
rates = df_model_20["CAL_CUS_RT"]
plt.figure(figsize=(8, 5))
plt.hist(rates, bins=30)
plt.title("점포수 20 이상 필터링 후 폐업률 분포", fontproperties=font_prop)
plt.xlabel("폐업률 (%)", fontproperties=font_prop)
plt.ylabel("빈도", fontproperties=font_prop)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - 여전히 왼쪽으로 치우침

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

# 업종 인코딩
df_model_20["업종_index"] = LabelEncoder().fit_transform(df_model_20["SRVIC_INDUTY_CD_NM"])

# 행정동 인코딩
df_model_20["행정동_index"] = LabelEncoder().fit_transform(df_model_20["ADSTRD_CD_NM"])


# COMMAND ----------

display(df_model)
display(df_model_20)

# COMMAND ----------

print(f"전체 데이터 개수:", df_model_20.count())

# COMMAND ----------

import numpy as np
df_model = df_model.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 시행착오

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ✅ 한글 폰트 설정
font_path = "/tmp/NanumGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)

for min_cnt in [5, 10, 20, 30]:
    filtered = df_model[(df_model["STR_CNT"] >= min_cnt) & (df_model["CAL_CUS_RT"].notnull())]
    
    rates = filtered["CAL_CUS_RT"]

    plt.figure(figsize=(6, 4))
    plt.hist(rates, bins=30)
    plt.title(f"점포수 ≥ {min_cnt} 필터링 후 폐업률 분포", fontproperties=font_prop)
    plt.xlabel("폐업률 (%)", fontproperties=font_prop)
    plt.ylabel("빈도", fontproperties=font_prop)
    plt.tight_layout()
    plt.show()



# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ✅ 한글 폰트 설정
font_path = "/tmp/NanumGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)

# 필터 기준 목록
thresholds = [5, 10, 20, 30]
result_counts = []

# ✅ Pandas 스타일 필터 + 카운트
for th in thresholds:
    count = df_model[(df_model["STR_CNT"] >= th) & (df_model["CAL_CUS_RT"].notnull())].shape[0]
    result_counts.append((f"STR_CNT ≥ {th}", count))

# ✅ DataFrame으로 변환
df_counts = pd.DataFrame(result_counts, columns=["기준", "남은 데이터 수"])

# ✅ 시각화
plt.figure(figsize=(8, 5))
plt.bar(df_counts["기준"], df_counts["남은 데이터 수"], color="orange")
plt.title("STR_CNT 필터 기준별 남은 데이터 수", fontproperties=font_prop)
plt.xlabel("필터 기준", fontproperties=font_prop)
plt.ylabel("남은 데이터 수", fontproperties=font_prop)
plt.xticks(rotation=45, fontproperties=font_prop)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# COMMAND ----------

# 추가 df_model_20 변수 변환
import numpy as np

# log1p 변환 적용
df_model_20["LOG_PREV_STR_CNT"] = np.log1p(df_model_20["PREV_STR_CNT"])

# CAL_CUS_RT 변수 변환
import numpy as np

df_model_20["LOG_CAL_CUS_RT"] = np.log1p(df_model_20["CAL_CUS_RT"])

# COMMAND ----------

# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# import seaborn as sns

# # 한글 폰트 설정
# font_path = "/tmp/NanumGothic.ttf"
# font_prop = fm.FontProperties(fname=font_path)

# plt.figure(figsize=(12, 5))

# # 원본
# plt.subplot(1, 2, 1)
# sns.histplot(df_model_20["CAL_CUS_RT"], bins=50, kde=True)
# plt.title("변환 전 CAL_CUS_RT", fontproperties=font_prop)

# # log 변환 후
# plt.subplot(1, 2, 2)
# sns.histplot(df_model_20["LOG_CAL_CUS_RT"], bins=50, kde=True)
# plt.title("log1p 변환 후 CAL_CUS_RT", fontproperties=font_prop)

# plt.tight_layout()
# plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### 변수변환 정리

# COMMAND ----------

# from sklearn.preprocessing import PowerTransformer
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm

# # ✅ 한글 폰트 설정
# font_path = "/tmp/NanumGothic.ttf"
# font_prop = fm.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = font_prop.get_name()
# plt.rcParams['axes.unicode_minus'] = False

# # 1. PySpark -> Pandas 변환
# df_filtered_pd = filtered_20_df.select("CAL_CUS_RT").dropna().toPandas()

# # 2. Yeo-Johnson 변환
# yeo_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
# cal_cus_rt_yeojohnson = yeo_transformer.fit_transform(df_filtered_pd[["CAL_CUS_RT"]])

# # 변환된 값 DataFrame에 추가
# df_filtered_pd["CAL_CUS_RT_YEO"] = cal_cus_rt_yeojohnson

# # 3. 시각화 (Before vs After)
# plt.figure(figsize=(12, 5))

# # 변환 전
# plt.subplot(1, 2, 1)
# plt.hist(df_filtered_pd["CAL_CUS_RT"], bins=30)
# plt.title("변환 전 CAL_CUS_RT", fontproperties=font_prop)
# plt.xlabel("CAL_CUS_RT", fontproperties=font_prop)
# plt.ylabel("Count", fontproperties=font_prop)

# # 변환 후
# plt.subplot(1, 2, 2)
# plt.hist(df_filtered_pd["CAL_CUS_RT_YEO"], bins=30)
# plt.title("Yeo-Johnson 변환 후 CAL_CUS_RT", fontproperties=font_prop)
# plt.xlabel("CAL_CUS_RT_YEO", fontproperties=font_prop)
# plt.ylabel("Count", fontproperties=font_prop)

# plt.tight_layout()
# plt.show()


# COMMAND ----------

# from scipy.stats import boxcox
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# import numpy as np
# import pandas as pd

# # ✅ 한글 폰트 설정
# font_path = "/tmp/NanumGothic.ttf"
# font_prop = fm.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = font_prop.get_name()
# plt.rcParams['axes.unicode_minus'] = False

# # 1. PySpark → Pandas 변환 (이미 되어있다고 가정)
# df_filtered_pd = filtered_20_df.select("CAL_CUS_RT").dropna().toPandas()

# # 2. Box-Cox 변환
# # -> 0 이하 값 있으면 shift 필요
# if (df_filtered_pd["CAL_CUS_RT"] <= 0).any():
#     min_val = df_filtered_pd["CAL_CUS_RT"].min()
#     shift = abs(min_val) + 1e-6  # 약간의 여유를 줌
#     cal_cus_rt_shifted = df_filtered_pd["CAL_CUS_RT"] + shift
#     cal_cus_rt_boxcox, fitted_lambda = boxcox(cal_cus_rt_shifted)
# else:
#     cal_cus_rt_boxcox, fitted_lambda = boxcox(df_filtered_pd["CAL_CUS_RT"])

# # 변환된 값 추가
# df_filtered_pd["CAL_CUS_RT_BOXCOX"] = cal_cus_rt_boxcox

# print(f"Box-Cox 변환 λ(lambda) 값: {fitted_lambda:.4f}")

# # 3. 시각화 (Before vs After)
# plt.figure(figsize=(12, 5))

# # 변환 전
# plt.subplot(1, 2, 1)
# plt.hist(df_filtered_pd["CAL_CUS_RT"], bins=30)
# plt.title("변환 전 CAL_CUS_RT", fontproperties=font_prop)
# plt.xlabel("CAL_CUS_RT", fontproperties=font_prop)
# plt.ylabel("Count", fontproperties=font_prop)

# # 변환 후
# plt.subplot(1, 2, 2)
# plt.hist(df_filtered_pd["CAL_CUS_RT_BOXCOX"], bins=30)
# plt.title("Box-Cox 변환 후 CAL_CUS_RT", fontproperties=font_prop)
# plt.xlabel("CAL_CUS_RT_BOXCOX", fontproperties=font_prop)
# plt.ylabel("Count", fontproperties=font_prop)

# plt.tight_layout()
# plt.show()


# COMMAND ----------

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm

# # ✅ 한글 폰트 설정
# font_path = "/tmp/NanumGothic.ttf"
# font_prop = fm.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = font_prop.get_name()
# plt.rcParams['axes.unicode_minus'] = False

# # 1. PySpark → Pandas 변환 (이미 되어있다고 가정)
# df_filtered_pd = filtered_20_df.select("CAL_CUS_RT").dropna().toPandas()

# # 2. sqrt 변환
# # (만약 음수 값이 있으면 shift)
# if (df_filtered_pd["CAL_CUS_RT"] < 0).any():
#     min_val = df_filtered_pd["CAL_CUS_RT"].min()
#     shift = abs(min_val) + 1e-6  # 살짝 더해줌
#     cal_cus_rt_shifted = df_filtered_pd["CAL_CUS_RT"] + shift
#     cal_cus_rt_sqrt = np.sqrt(cal_cus_rt_shifted)
# else:
#     cal_cus_rt_sqrt = np.sqrt(df_filtered_pd["CAL_CUS_RT"])

# # 변환된 값 추가
# df_filtered_pd["CAL_CUS_RT_SQRT"] = cal_cus_rt_sqrt

# # 3. 시각화
# plt.figure(figsize=(12, 5))

# # 변환 전
# plt.subplot(1, 2, 1)
# plt.hist(df_filtered_pd["CAL_CUS_RT"], bins=30)
# plt.title("변환 전 CAL_CUS_RT", fontproperties=font_prop)
# plt.xlabel("CAL_CUS_RT", fontproperties=font_prop)
# plt.ylabel("Count", fontproperties=font_prop)

# # sqrt 변환 후
# plt.subplot(1, 2, 2)
# plt.hist(df_filtered_pd["CAL_CUS_RT_SQRT"], bins=30)
# plt.title("Sqrt 변환 후 CAL_CUS_RT", fontproperties=font_prop)
# plt.xlabel("CAL_CUS_RT_SQRT", fontproperties=font_prop)
# plt.ylabel("Count", fontproperties=font_prop)

# plt.tight_layout()
# plt.show()


# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import PowerTransformer

# ✅ 한글 폰트 설정
font_path = "/tmp/NanumGothic.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ✅ 1. 데이터 준비 (filtered_df 사용)
df_filtered_pd = filtered_20_df.select("CAL_CUS_RT").dropna().toPandas()

# ✅ 2. 0값을 최솟값의 절반으로 대체
min_positive = df_filtered_pd[df_filtered_pd["CAL_CUS_RT"] > 0]["CAL_CUS_RT"].min()
adjusted_value = min_positive / 2


print(f"최솟값: {min_positive:.6f}, 대체할 값: {adjusted_value:.6f}")

df_filtered_pd["CAL_CUS_RT_ADJ"] = df_filtered_pd["CAL_CUS_RT"].apply(lambda x: adjusted_value if x == 0 else x)




# ✅ 4. 변환 전후 시각화
plt.figure(figsize=(14, 6))

# 변환 전
plt.subplot(1, 3, 1)
plt.hist(df_filtered_pd["CAL_CUS_RT"], bins=30)
plt.title("변환 전 CAL_CUS_RT", fontproperties=font_prop)
plt.xlabel("CAL_CUS_RT", fontproperties=font_prop)
plt.ylabel("Count", fontproperties=font_prop)

# 0 대체 후
plt.subplot(1, 3, 2)
plt.hist(df_filtered_pd["CAL_CUS_RT_ADJ"], bins=30)
plt.title("0값 조정 후 CAL_CUS_RT", fontproperties=font_prop)
plt.xlabel("CAL_CUS_RT_ADJ", fontproperties=font_prop)
plt.ylabel("Count", fontproperties=font_prop)


plt.tight_layout()
plt.show()

# COMMAND ----------

# 연, 분기 rename
df_model = df_model.rename(columns={
    "STRD_YR_CD": "YEAR",
    "STRD_QTR_CD": "QUARTER"
})

df_model_20 = df_model_20.rename(columns={
    "STRD_YR_CD": "YEAR",
    "STRD_QTR_CD": "QUARTER"
})

# COMMAND ----------

# 분기의 주기성 반영
import numpy as np
df_model_20["QUARTER_SIN"] = np.sin(2 * np.pi * df_model["QUARTER"] / 4)
df_model_20["QUARTER_COS"] = np.cos(2 * np.pi * df_model["QUARTER"] / 4)


# COMMAND ----------

# MAGIC %md
# MAGIC #### X,y 정의

# COMMAND ----------



# ✅ 2. X, y 정의

X = df_model_20[[
    "업종_index", "행정동_index",
    "YEAR"
]]


y = df_filtered_pd["CAL_CUS_RT_ADJ"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# y 분포도 확인
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.hist(y, bins=30)
plt.title("최종 Y (CAL_CUS_RT_ADJ) 분포", fontproperties=font_prop)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# COMMAND ----------

X_train = X[df_model_20["YEAR_QTR"] < 20221]
X_test = X[df_model_20["YEAR_QTR"] >= 20221]

y_train = y[df_model_20["YEAR_QTR"] < 20221]
y_test = y[df_model_20["YEAR_QTR"] >= 20221]

total_count = len(X)
train_count = len(X_train)
test_count = len(X_test)

# 비율 계산
train_ratio = train_count / total_count * 100
test_ratio = test_count / total_count * 100

print(f"Train 데이터: {train_count}개 ({train_ratio:.2f}%)")
print(f"Test 데이터: {test_count}개 ({test_ratio:.2f}%)")

# COMMAND ----------

print("X_train 결측치 여부:", X_train.isnull().sum().sum())
print("y_train 결측치 여부:", y_train.isnull().sum())
print("X_train 무한값 존재 여부:", np.isinf(X_train).sum().sum())
print("y_train 무한값 존재 여부:", np.isinf(y_train).sum())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 구단위로 묶기

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from bronze.etc.code_kikcd_h
# MAGIC where deleted_date is null
# MAGIC and h_dong_name = '둔촌제1동'

# COMMAND ----------

code_df = spark.sql("""
    SELECT h_dong_code, h_dong_name, sigungu
    FROM bronze.etc.code_kikcd_h
    WHERE deleted_date IS NULL
      AND (sido = '서울특별시' OR sido = '서울시')
""").toPandas().drop_duplicates()

# COMMAND ----------

df_model_20.head()

# COMMAND ----------

# 행정동 join 위해 이름 수정
code_df["clean_dong"] = (
    code_df["h_dong_name"]
    .str.replace(r"제\d+동", "동", regex=True)    # 예: 면목제1동 → 면목동
    .str.replace(r"제동", "동", regex=True)       # 예: 창제동 → 창동
    .str.replace(r"[\d\W_]+", "", regex=True)     # 숫자 및 특수기호 제거
)
code_df["clean_dong"] = code_df["clean_dong"].replace("홍동", "홍제동")
code_df

# COMMAND ----------

# 행정동 join 위해 이름 수정
df_model_20["clean_dong"] = (
    df_model_20["ADSTRD_CD_NM"]
    .str.replace(r"제\d+", "", regex=True)     # '제숫자' 제거
    .str.replace(r"[\d\W_]+", "", regex=True)  # 나머지 숫자/기호 제거
)
df_model_20["clean_dong"] = df_model_20["clean_dong"].replace("홍동", "홍제동")
df_model_20["clean_dong"]

# COMMAND ----------

df_model_20 = df_model_20.merge(
    code_df[["clean_dong", "sigungu"]].drop_duplicates(),
    how="left",
    on="clean_dong"
)

print("sigungu 누락 행:", df_model_20["sigungu"].isnull().sum())

# COMMAND ----------

# ✅ 시군구 인덱스 생성 (sigungu → 시군구_index)
df_model_20 = df_model_20.copy()  # 원본 보호

# unique 값 기준으로 정렬 후 정수 인코딩
sigungu_mapping = {
    name: idx for idx, name in enumerate(sorted(df_model_20["sigungu"].dropna().unique()))
}

# 매핑 적용
df_model_20["시군구_index"] = df_model_20["sigungu"].map(sigungu_mapping)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 최종 x, y 생성

# COMMAND ----------

df_model_20.head()

# COMMAND ----------

df_model_20["YEAR"].drop_duplicates().sort_values().reset_index(drop=True)


# COMMAND ----------

#  인덱스 정렬 (병합 전 반드시 필요)
df_model_20 = df_model_20.reset_index(drop=True)
#  0을 최솟값 절반으로 대체하여 CAL_CUS_RT_ADJ 생성
min_positive = df_model_20[df_model_20["CAL_CUS_RT"] > 0]["CAL_CUS_RT"].min()
adjusted_value = min_positive / 2

# CAL_CUS_RT_ADJ 컬럼 생성
df_model_20["CAL_CUS_RT_ADJ"] = df_model_20["CAL_CUS_RT"].apply(
    lambda x: adjusted_value if x == 0 else x
)



#  X, y 정의
feature_cols = [
    "업종_index", "행정동_index",
    "시군구_index", "YEAR"
]

X = df_model_20[feature_cols]
y = y = df_model_20["CAL_CUS_RT_ADJ"]

print("X shape:", X.shape)
print("y shape:", y.shape)

#  y 분포 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.hist(y, bins=30)
plt.title("최종 Y (CAL_CUS_RT_ADJ) 분포", fontproperties=font_prop)
plt.xlabel("CAL_CUS_RT_ADJ", fontproperties=font_prop)
plt.ylabel("Count", fontproperties=font_prop)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# COMMAND ----------

print("최솟값:", y.min())
print("최댓값:", y.max())

# COMMAND ----------

print(y.describe())

# COMMAND ----------

df_model_20 = df_model_20.reset_index(drop=True)
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

#  train/test 분할
X_train = X[df_model_20["YEAR_QTR"] < 20221].copy()
X_test = X[df_model_20["YEAR_QTR"] >= 20221].copy()
y_train = y[df_model_20["YEAR_QTR"] < 20221].copy()
y_test = y[df_model_20["YEAR_QTR"] >= 20221].copy()

print(type(X_train))
print(type(y_train))
print(X_train.shape)
print(y_train.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 분류기

# COMMAND ----------

# MAGIC %md
# MAGIC - best_params
# MAGIC
# MAGIC --- 
# MAGIC
# MAGIC     'n_estimators': 278,
# MAGIC     'max_depth': 9,
# MAGIC     'learning_rate': 0.11222351457605487,
# MAGIC     'subsample': 0.8995971063266778,
# MAGIC     'colsample_bytree': 0.8911899012969402
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC - threshold
# MAGIC ---
# MAGIC 0.390

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 히스토그램상 대부분의 낮은 값들이 0.5 미만에 몰려 있고, 폐업률 0.5는 상징적으로 "절반 수준"이라는 비교적 타당한 기준

# COMMAND ----------

# MAGIC %md
# MAGIC ### XGBoost

# COMMAND ----------

# MAGIC  %pip install optuna

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

import xgboost as xgb
print("XGBoost 버전:", xgb.__version__)

# COMMAND ----------

import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

# 폐업률이 0.5 이하이면 생존(0), 초과하면 폐업(1)
y_train_bin = (y_train > 0.5).astype(int)
y_test_bin = (y_test > 0.5).astype(int)



# COMMAND ----------

from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import pandas as pd

# 클래스 비율 계산
neg = (y_train <= 0.5).sum()
pos = (y_train > 0.5).sum()
scale_pos_weight = neg / pos  

# 이진 타겟
y_train_bin = (y_train > 0.5).astype(int)
y_test_bin = (y_test > 0.5).astype(int)

# XGBoost 분류기 정의
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 모델 학습
xgb_model.fit(X_train, y_train_bin)

# 클래스 예측
y_pred_bin = xgb_model.predict(X_test)

#  예측 확률 추가 
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# 결과 요약
print("Confusion Matrix:\n", confusion_matrix(y_test_bin, y_pred_bin))
print("\nClassification Report:\n", classification_report(y_test_bin, y_pred_bin))

# 결과 데이터프레임으로 정리
df_results = pd.DataFrame({
    '실제값': y_test_bin,
    '예측값': y_pred_bin,
    '예측확률': y_pred_proba
})

print(df_results.head())


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### OPTUNA 파라미터 조정

# COMMAND ----------

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import numpy as np

# 타겟 정의 (폐업률 > 0.5 → 1 / 이하 → 0)
y_train_bin = (y_train > 0.5).astype(int)
y_test_bin = (y_test > 0.5).astype(int)

# 클래스 비율 계산
neg = (y_train_bin == 0).sum()
pos = (y_train_bin == 1).sum()
scale_pos_weight = neg / pos

# Optuna objective 함수
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0)
    }

    model = XGBClassifier(**params)
    preds = model.fit(X_train, y_train_bin).predict(X_test)
    return f1_score(y_test_bin, preds)  # 또는 recall, precision 등으로 바꿀 수 있음

# 최적화 수행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 최적 하이퍼파라미터 출력
print("Best trial:")
print(study.best_trial)

# 최적 파라미터로 모델 학습
best_model = XGBClassifier(**study.best_params, objective="binary:logistic", eval_metric="logloss", use_label_encoder=False, random_state=42)
best_model.fit(X_train, y_train_bin)

# 평가
y_pred = best_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test_bin, y_pred))
print("\nClassification Report:\n", classification_report(y_test_bin, y_pred))


# COMMAND ----------

best_model = XGBClassifier(**study.best_params,
                           objective="binary:logistic",
                           eval_metric="logloss",
                           use_label_encoder=False,
                           random_state=42)

best_model.fit(
    X_train, y_train_bin,
    eval_set=[(X_test, y_test_bin)],
    early_stopping_rounds=10,
    verbose=False
)


# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## mlflow

# COMMAND ----------

# import mlflow
# import mlflow.xgboost
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# # ✅ MLflow 자동 로깅 활성화
# mlflow.xgboost.autolog(
#     log_input_examples=False,
#     log_model_signatures=True,
#     log_models=True,
#     disable=False,
#     exclusive=False,
#     disable_for_unsupported_versions=True,
#     silent=False
# )

# with mlflow.start_run(run_name="폐업률 분류기 모델"):
#     # 이미 훈련된 best_model 사용
#     y_pred = best_model.predict(X_test)
#     y_proba = best_model.predict_proba(X_test)[:, 1]

#     # 주요 성능지표 기록
#     auc = roc_auc_score(y_test_bin, y_proba)
#     mlflow.log_metric("AUC", auc)

#     # confusion matrix나 classification report 로그도 원하면 아래 추가
#     print("Confusion Matrix:\n", confusion_matrix(y_test_bin, y_pred))
#     print("Classification Report:\n", classification_report(y_test_bin, y_pred))

#     # 모델 직접 저장하려면 (선택)
#     mlflow.xgboost.log_model(best_model, artifact_path="model")


# COMMAND ----------

# MAGIC %md
# MAGIC ### 파이프라인 구축

# COMMAND ----------

X_pipe = df_model_20[[
    "SRVIC_INDUTY_CD_NM", "ADSTRD_CD_NM", "sigungu",
    "YEAR"
]]

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# 1. string 상태의 raw feature만 추출
X_pipe = df_model_20[[
    "SRVIC_INDUTY_CD_NM", "ADSTRD_CD_NM", "sigungu", "YEAR"
]].reset_index(drop=True)

# 2. train 데이터만 필터링
X_pipe_train = X_pipe[df_model_20["YEAR_QTR"] < 20221].reset_index(drop=True)
y_train_bin = (y[df_model_20["YEAR_QTR"] < 20221] > 0.5).astype(int).reset_index(drop=True)

# 행 수 검증
assert len(X_pipe_train) == len(y_train_bin), "X와 y의 길이가 다릅니다."

# 3. 전처리기 구성
categorical_features = ["SRVIC_INDUTY_CD_NM", "ADSTRD_CD_NM", "sigungu"]
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# 4. 최적화된 하이퍼파라미터 기반 XGBClassifier 정의
clf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        scale_pos_weight=scale_pos_weight,  # 클래스 불균형 비율 유지

        # ✅ Optuna 최적 하이퍼파라미터 반영
        n_estimators=425,
        max_depth=9,
        learning_rate=0.23954201152772756,
        colsample_bytree=0.879387332682185,
        min_child_weight=9,
        gamma=0.6718888547295224,
        reg_alpha=0.0,
        reg_lambda=0.0
    ))
])

# 5. 학습 수행
clf_pipeline.fit(X_pipe_train, y_train_bin)

# 6. 테스트 데이터 정의
X_pipe_test = X_pipe[df_model_20["YEAR_QTR"] >= 20221].reset_index(drop=True)
y_test_bin = (y[df_model_20["YEAR_QTR"] >= 20221] > 0.5).astype(int).reset_index(drop=True)

# 7. 예측 수행
y_pred = clf_pipeline.predict(X_pipe_test)
y_pred_proba = clf_pipeline.predict_proba(X_pipe_test)[:, 1]

# 8. 결과 정리
df_results = X_pipe_test.copy()
df_results['실제값'] = y_test_bin
df_results['예측값'] = y_pred
df_results['예측확률'] = y_pred_proba


# COMMAND ----------

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("Confusion Matrix:\n", confusion_matrix(y_test_bin, y_pred))
print("\nClassification Report:\n", classification_report(y_test_bin, y_pred))
print(f"AUC: {roc_auc_score(y_test_bin, y_pred_proba):.4f}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### 예측값 모델결과 테이블 생성

# COMMAND ----------

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 1. 테스트 데이터 준비
X_pipe_test = X_pipe[df_model_20["YEAR_QTR"] >= 20221].reset_index(drop=True)
y_test_bin = (y[df_model_20["YEAR_QTR"] >= 20221] > 0.5).astype(int).reset_index(drop=True)

# 2. 확률 예측 (1일 확률)
y_proba = clf_pipeline.predict_proba(X_pipe_test)[:, 1]

# 3. ROC 곡선 좌표 계산
fpr, tpr, thresholds = roc_curve(y_test_bin, y_proba)

# 4. AUC 계산
auc_score = roc_auc_score(y_test_bin, y_proba)
print(f"AUC: {auc_score:.4f}")

# 5. 시각화
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# COMMAND ----------

# 테스트 데이터 구성
X_pipe_test = X_pipe[df_model_20["YEAR_QTR"] >= 20221].reset_index(drop=True)
y_test_bin = (y[df_model_20["YEAR_QTR"] >= 20221] > 0.5).astype(int).reset_index(drop=True)


# COMMAND ----------

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# 예측 (클래스)
y_pred = clf_pipeline.predict(X_pipe_test)

# 예측 확률 (클래스 1일 확률)
y_pred_proba = clf_pipeline.predict_proba(X_pipe_test)[:, 1]

# 결과 출력
print("Confusion Matrix:")
print(confusion_matrix(y_test_bin, y_pred))

print("\nClassification Report:")
print(classification_report(y_test_bin, y_pred))

print(f"\nAccuracy: {accuracy_score(y_test_bin, y_pred):.4f}")

# 예측 결과 요약 테이블
df_results = pd.DataFrame({
    "실제값": y_test_bin,
    "예측값": y_pred,
    "예측확률": y_pred_proba
})

print("\n예측 결과 샘플:")
print(df_results.head())


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


