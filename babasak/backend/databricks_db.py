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

_SQL = """
WITH daily AS (
    SELECT `재료명`, `단위`, `날짜`, ROUND(AVG(`가격`)) AS `가격`
    FROM silver.ingredient.ingredient
    WHERE (
        (`재료명` = '계란'   AND `단위` = '30개') OR
        (`재료명` = '깐마늘' AND `단위` = '20kg') OR
        (`재료명` = '당근'   AND `단위` = '10kg') OR
        (`재료명` = '대파'   AND `단위` = '1kg') OR
        (`재료명` = '양파'   AND `단위` = '15kg') OR
        (`재료명` = '풋고추' AND `단위` = '4kg')
    )
    GROUP BY `재료명`, `단위`, `날짜`
),
with_prev AS (
    SELECT
        `재료명`,
        `단위`,
        `날짜`,
        `가격`,
        LAG(`가격`) OVER (PARTITION BY `재료명`, `단위` ORDER BY `날짜`) AS `이전가격`
    FROM daily
),
latest AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY `재료명`, `단위` ORDER BY `날짜` DESC) AS rn
    FROM with_prev
)
SELECT
    `재료명`,
    `단위`,
    `날짜`,
    CAST(`가격` AS INT) AS `가격`,
    CAST(`이전가격` AS INT) AS `이전가격`,
    CASE
        WHEN `이전가격` IS NULL OR `이전가격` = 0 THEN 0
        ELSE ROUND((`가격` - `이전가격`) / `이전가격` * 100, 1)
    END AS `변동률`
FROM latest
WHERE rn = 1 AND `가격` IS NOT NULL
ORDER BY `재료명`
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
            icon = next((v for k, v in INGREDIENT_ICONS.items() if k in name), "🥬")
            rows.append(
                {
                    "name": f"{name} ({unit})",
                    "icon": icon,
                    "price": price,
                    "prev_price": prev_price,
                    "change": change,
                    "date": date,
                }
            )
        return rows or None
    except Exception as e:
        last_error = str(e)
        return None
