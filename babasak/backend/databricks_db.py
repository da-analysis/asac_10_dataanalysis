import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

_DEFAULT_ICON = "🛒"  # 매핑에 없는 품목용 중립 아이콘

# 실제 silver.ingredient 카탈로그 전 품목 ↔ 이모지 매핑.
# 매칭은 _pick_icon()이 (1)정확 일치 → (2)긴 키워드부터 부분 일치 순으로 처리하므로
# "고추"가 "고춧가루"를, "오징어"가 "마른오징어"를 가로채지 않는다.
INGREDIENT_ICONS = {
    # 과일류
    "감귤": "🍊", "거봉": "🍇", "건블루베리": "🫐", "건포도": "🍇", "단감": "🍊",
    "딸기": "🍓", "레몬": "🍋", "망고": "🥭", "멜론": "🍈", "바나나": "🍌",
    "배": "🍐", "복숭아": "🍑", "블루베리": "🫐", "사과": "🍎", "샤인머스켓": "🍇",
    "수박": "🍉", "아보카도": "🥑", "오렌지": "🍊", "자몽": "🍊", "참다래": "🥝",
    "참외": "🥭", "체리": "🍒", "파인애플": "🍍", "포도": "🍇",
    # 수산물
    "가리비": "🐚", "갈치": "🐟", "건다시마": "🪸", "고등어": "🐟", "고등어필렛": "🐟",
    "굴": "🐚", "김": "🪸", "꽁치": "🐟", "꽃게": "🦀", "대하": "🐟",
    "마른멸치": "🐟", "마른미역": "🪸", "마른오징어": "🦑", "멸치액젓": "🐟", "명태": "🐟",
    "물오징어": "🦑", "바지락": "🐚", "부세": "🐟", "북어": "🐟", "삼치": "🐟",
    "새우": "🦐", "새우젓": "🦐", "수입조기": "🐟", "연어필렛": "🐟", "오징어": "🦑",
    "오징어젓": "🦑", "오징어젓갈": "🦑", "전복": "🐚", "전어": "🐟", "조기": "🐟",
    "참조기": "🐟", "천일염": "🧂", "홍합": "🐚",
    # 쌀/잡곡
    "고구마": "🍠", "귀리": "🌾", "기장": "🌾", "녹두": "🫛", "메밀": "🌾",
    "보리": "🌾", "수수": "🌾", "쌀": "🌾", "율무": "🌾", "찹쌀": "🌾",
    "콩": "🫛", "팥": "🫘", "현미": "🌾", "혼합곡": "🌾",
    # 채소류
    "가지": "🍆", "감자": "🥔", "갓": "🥦", "건고추": "🌶️", "고춧가루": "🌶️",
    "깐마늘": "🧄", "깻잎": "🥬", "단호박": "🎃", "당근": "🥕", "대파": "🥦",
    "마늘": "🧄", "무": "🥦", "미나리": "🥬", "방울토마토": "🍅", "배추": "🥬",
    "부추": "🥦", "붉은고추": "🌶️", "브로콜리": "🥦", "상추": "🥬", "생강": "🫚",
    "시금치": "🥬", "알배기배추": "🥬", "애호박": "🥒", "양배추": "🥬", "양상추": "🥬",
    "양파": "🧅", "얼갈이배추": "🥬", "연근": "🥦", "열무": "🥦", "오이": "🥒",
    "우엉": "🥦", "적상추": "🥬", "절임배추": "🥬", "쥬키니": "🥒", "쪽파": "🥦",
    "청경채": "🥬", "케일": "🥬", "콩나물": "🌱", "토마토": "🍅", "파프리카": "🫑",
    "풋고추": "🌶️", "피마늘": "🧄", "피망": "🫑",
    # 축산물
    "계란": "🥚", "닭고기": "🐔", "돼지고기": "🐷", "소고기": "🐮", "우유": "🥛",
    "훈제오리": "🦆",
    # 특용작물
    "느타리버섯": "🍄", "들깨": "🥜", "땅콩": "🥜", "새송이버섯": "🍄", "아몬드": "🥜",
    "양송이버섯": "🍄", "참깨": "🥜", "팽이버섯": "🍄", "표고버섯": "🍄", "호두": "🥜",
}

# 부분 일치용 키워드를 길이 내림차순으로 미리 정렬(긴 이름이 짧은 이름을 가로채지 않도록).
_ICON_KEYS_BY_LEN = sorted(INGREDIENT_ICONS, key=len, reverse=True)


def _pick_icon(name: str) -> str:
    """재료명에 맞는 이모지 선택: 정확 일치 우선, 없으면 긴 키워드부터 부분 일치."""
    if name in INGREDIENT_ICONS:
        return INGREDIENT_ICONS[name]
    for k in _ICON_KEYS_BY_LEN:
        if k in name:
            return INGREDIENT_ICONS[k]
    return _DEFAULT_ICON

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
            icon = _pick_icon(name)
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
