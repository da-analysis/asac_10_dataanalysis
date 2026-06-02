import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

INGREDIENT_ICONS = {
    "감자": "🥔", "양파": "🧅", "대파": "🥬", "파": "🥬",
    "배추": "🥬", "상추": "🥬", "시금치": "🥬", "깻잎": "🥬",
    "당근": "🥕", "오이": "🥒", "토마토": "🍅", "고추": "🌶️", "마늘": "🧄",
    "돼지": "🥩", "닭": "🍗", "소": "🥩",
    "쌀": "🌾", "우유": "🥛", "달걀": "🥚", "계란": "🥚",
    "고등어": "🐟", "명태": "🐟", "오징어": "🦑", "새우": "🍤",
    "수박": "🍉", "사과": "🍎",
}

# 기준일·직전일을 '전 품목 공통 1일'로 고정한 뒤, 두 날짜 데이터가 모두 있는
# 품목의 변동률을 계산해 상승 상위 6 + 하락 하위 6 = 총 12개를 뽑는다.
#   기준일  = MAX(날짜)                (전 품목 통틀어 최신 1일)
#   직전일  = MAX(날짜 < 기준일)        (전체 공통 직전 영업일 1일)
# 같은 재료에 단위가 여러 개면(예: 양파 15kg vs 1.5kg) 단위 문자열의 숫자가
# 가장 큰 단위 하나만 사용한다(대용량 단위 우선).
# DB에 새 날짜가 쌓이면 기준일이 자동으로 바뀌므로 매일 구성이 달라진다.
_SQL = """
WITH daily AS (
    SELECT `재료명`, `단위`, `날짜`, ROUND(AVG(`가격`)) AS `가격`
    FROM silver.ingredient.ingredient
    GROUP BY `재료명`, `단위`, `날짜`
),
unit_pick AS (  -- 재료별로 단위 숫자가 가장 큰 단위 1개만 선택
    SELECT `재료명`, `단위`
    FROM (
        SELECT `재료명`, `단위`,
            ROW_NUMBER() OVER (
                PARTITION BY `재료명`
                ORDER BY CAST(NULLIF(regexp_extract(`단위`, '([0-9.]+)', 1), '') AS DOUBLE) DESC NULLS LAST, `단위`
            ) AS urn
        FROM (SELECT DISTINCT `재료명`, `단위` FROM daily)
    )
    WHERE urn = 1
),
ref AS (
    SELECT
        MAX(`날짜`) AS `기준일`,
        MAX(CASE WHEN `날짜` < (SELECT MAX(`날짜`) FROM daily) THEN `날짜` END) AS `직전일`
    FROM daily
),
cur AS (
    SELECT d.`재료명`, d.`단위`, d.`날짜` AS `기준일`, d.`가격`
    FROM daily d
    JOIN ref r ON d.`날짜` = r.`기준일`
    JOIN unit_pick u ON d.`재료명` = u.`재료명` AND d.`단위` = u.`단위`
),
prev AS (
    SELECT d.`재료명`, d.`단위`, d.`가격` AS `이전가격`
    FROM daily d
    JOIN ref r ON d.`날짜` = r.`직전일`
    JOIN unit_pick u ON d.`재료명` = u.`재료명` AND d.`단위` = u.`단위`
),
scored AS (
    SELECT
        cur.`재료명`,
        cur.`단위`,
        cur.`기준일` AS `날짜`,
        CAST(cur.`가격` AS INT) AS `가격`,
        CAST(prev.`이전가격` AS INT) AS `이전가격`,
        ROUND((cur.`가격` - prev.`이전가격`) / prev.`이전가격` * 100, 1) AS `변동률`
    FROM cur JOIN prev
        ON cur.`재료명` = prev.`재료명` AND cur.`단위` = prev.`단위`
    WHERE cur.`가격` IS NOT NULL
      AND prev.`이전가격` IS NOT NULL AND prev.`이전가격` > 0
      AND ABS((cur.`가격` - prev.`이전가격`) / prev.`이전가격` * 100) <= 100   -- 데이터 오류성 이상치 컷
)
(SELECT `재료명`, `단위`, `날짜`, `가격`, `이전가격`, `변동률`, 'up'   AS trend
 FROM scored WHERE `변동률` > 0 ORDER BY `변동률` DESC LIMIT 6)
UNION ALL
(SELECT `재료명`, `단위`, `날짜`, `가격`, `이전가격`, `변동률`, 'down' AS trend
 FROM scored WHERE `변동률` < 0 ORDER BY `변동률` ASC  LIMIT 6)
"""

last_error: str | None = None


def _get_warehouse_id(w: WorkspaceClient) -> str | None:
    wh_id = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if wh_id:
        return wh_id
    for wh in w.warehouses.list():
        if wh.state and wh.state.value in ("RUNNING", "STARTING"):
            return wh.id
    warehouses = list(w.warehouses.list())
    return warehouses[0].id if warehouses else None


def get_ingredient_prices_live() -> list[dict] | None:
    global last_error
    last_error = None
    try:
        host = os.environ.get("DATABRICKS_HOST")
        token = os.environ.get("DATABRICKS_TOKEN")
        w = WorkspaceClient(host=host, token=token) if host and token else WorkspaceClient()
        warehouse_id = _get_warehouse_id(w)
        if not warehouse_id:
            last_error = "warehouse를 찾을 수 없음 (DATABRICKS_WAREHOUSE_ID 미설정)"
            return None

        resp = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=_SQL,
            wait_timeout="50s",
        )

        if not resp.status or resp.status.state != StatementState.SUCCEEDED:
            err_msg = ""
            if resp.status and resp.status.error:
                err_msg = f": {resp.status.error.message}"
            last_error = f"SQL 실행 실패{err_msg}"
            return None

        rows = []
        for row in (resp.result.data_array or []):
            name = str(row[0] or "")
            unit = str(row[1] or "")
            date = str(row[2] or "")
            price = int(float(row[3])) if row[3] is not None else 0
            prev_price = int(float(row[4])) if row[4] is not None else None
            change = float(row[5]) if row[5] is not None else 0.0
            trend = str(row[6]) if len(row) > 6 and row[6] is not None else ("up" if change > 0 else "down")
            icon = next((v for k, v in INGREDIENT_ICONS.items() if k in name), "🥬")
            rows.append(
                {
                    "name": f"{name} ({unit})",
                    "icon": icon,
                    "price": price,
                    "prev_price": prev_price,
                    "change": change,
                    "date": date,
                    "trend": trend,
                }
            )
        return rows or None
    except Exception as e:
        last_error = str(e)
        return None
