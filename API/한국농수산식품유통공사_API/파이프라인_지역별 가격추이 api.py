# Databricks notebook source
# Databricks notebook source
import requests
import pandas as pd
import math
import time
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
yesterday_kst = (datetime.now(KST) - timedelta(days=1)).strftime("%Y%m%d")

dbutils.widgets.text("target_date", yesterday_kst, "수집 대상 날짜 (YYYYMMDD)")
target_date = dbutils.widgets.get("target_date")


service_key = dbutils.secrets.get(scope="agrofood-api-prod", key="publicdata-service-key_rkj")

API_URL = "https://apis.data.go.kr/B552845/perRegion/price"
OUTPUT_DIR = "/Volumes/bronze_api/agrofood_perregion/volumn"
MAX_RETRIES = 4

SGG_CODES = {
    "1101": "서울",
    "2100": "부산", "2200": "대구", "2300": "인천",
    "2401": "광주", "2501": "대전", "2601": "울산", "2701": "세종",
    "3100": "경기", "3111": "수원", "3112": "성남", "3113": "의정부",
    "3138": "고양", "3145": "용인",
    "3201": "강원", "3211": "춘천", "3214": "강릉",
    "3300": "충북",
    "3411": "천안",
    "3500": "전북", "3511": "전주", "3512": "군산",
    "3600": "전남", "3611": "목포", "3613": "순천",
    "3700": "경북", "3711": "포항", "3714": "안동",
    "3800": "경남", "3811": "마산", "3814": "창원", "3818": "김해",
    "3911": "제주",
    "9998": "온라인",
}
def fetch_with_retry(url, params, max_retries=MAX_RETRIES):
    """요청 실패 시 최대 4회 지수 백오프 재시도"""
    delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"  재시도 {attempt}/{max_retries - 1} (대기 {delay}s): {e}")
            time.sleep(delay)
            delay *= 2



print(f"수집 대상 날짜: {target_date}")
print("=" * 50)

daily_items = []

for sgg_cd, sgg_nm in SGG_CODES.items():
    params = {
        "serviceKey": service_key,
        "returnType": "JSON",
        "pageNo": 1,
        "numOfRows": 1000,
        "cond[exmn_ymd::EQ]": target_date,
        "cond[sgg_cd::EQ]": sgg_cd,
    }

    try:
        data = fetch_with_retry(API_URL, params)

        total_count = int(data.get("response", {}).get("body", {}).get("totalCount", 0))
        num_of_rows = int(data.get("response", {}).get("body", {}).get("numOfRows", 1000))
        pages = math.ceil(total_count / num_of_rows) if total_count > 0 else 0

        if total_count == 0:
            print(f"  [{sgg_cd} {sgg_nm}] 데이터 없음")
            time.sleep(0.2)
            continue

        print(f"  [{sgg_cd} {sgg_nm}] 총 {total_count}건 / {pages}페이지")

        for page in range(1, pages + 1):
            params["pageNo"] = page
            data = fetch_with_retry(API_URL, params)
            items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])

            if isinstance(items, dict):
                items = [items]
            if not items:
                continue

            collect_time = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
            for item in items:
                item["collect_time"] = collect_time
            daily_items.extend(items)
            print(f"    [{page}/{pages}페이지] {len(items)}건 수집 / 누적: {len(daily_items)}건")
            time.sleep(0.2)

    except Exception as e:
        print(f"  [{sgg_cd} {sgg_nm}] 에러: {e} - 다음 지역으로")
        time.sleep(0.5)



if daily_items:
    df = pd.DataFrame(daily_items)
    file_path = f"{OUTPUT_DIR}/지역별가격정보_{target_date}.csv"
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {file_path} ({len(daily_items)}건)")
else:
    print(f"{target_date}: 수집된 데이터 없음")


dbutils.widgets.text("result_count", str(len(daily_items)), "수집 건수")
print(f"총 수집 건수: {len(daily_items)}")
