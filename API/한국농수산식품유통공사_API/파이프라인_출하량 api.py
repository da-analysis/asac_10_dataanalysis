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


API_URL = "https://apis.data.go.kr/B552845/shipmentSequel/info"
OUTPUT_DIR = "/Volumes/bronze_api/agrofood_shipmentsequel/volumn"
MAX_RETRIES = 4


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

params = {
    "serviceKey": service_key,
    "returnType": "JSON",
    "pageNo": 1,
    "numOfRows": 1000,
    "cond[spmt_ymd::EQ]": target_date,
}

try:
    data = fetch_with_retry(API_URL, params)

    total_count = int(data.get("response", {}).get("body", {}).get("totalCount", 0))
    num_of_rows = int(data.get("response", {}).get("body", {}).get("numOfRows", 1000))
    pages = math.ceil(total_count / num_of_rows) if total_count > 0 else 0

    if total_count == 0:
        print("데이터 없음")
    else:
        print(f"총 {total_count}건 / {pages}페이지")

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
    print(f"에러 발생: {e}")
    raise

if daily_items:
    df = pd.DataFrame(daily_items)
    file_path = f"{OUTPUT_DIR}/출하량추이정보_{target_date}.csv"
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {file_path} ({len(daily_items)}건)")
else:
    print(f"{target_date}: 수집된 데이터 없음")

dbutils.widgets.text("result_count", str(len(daily_items)), "수집 건수")
print(f"총 수집 건수: {len(daily_items)}")
