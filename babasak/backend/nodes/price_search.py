import os
import time
import re
import pandas as pd
import mlflow
from concurrent.futures import ThreadPoolExecutor
from mlflow.entities import SpanType
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

from backend.debug_log import archive
from backend.catalog import resolve_many, ResolveResult, get_recipe_prices_for_items

from backend.nodes.chart_utils import generate_chart_html
# 개수 단위(개/포기/마리 등) → kg 환산용 1개당 그램 추정표. cost_calculator와 동일 기준을
#써야 원가 계산이 일관되므로 그쪽 표를 그대로 재사용한다(단일 출처).
from backend.nodes.cost_calculator import _PER_PIECE_GRAMS

_COUNT_UNIT_RE = re.compile(r"^\d+(?:\.\d+)?\s*(?:개|포기|마리|통|단|모|장|알|봉|손|쪽)$")
_WEIGHT_UNIT_RE_PS = re.compile(r"^(\d+(?:\.\d+)?)(kg|g|ml|l)$", re.IGNORECASE)


def _is_count_unit_str(unit: str) -> bool:
    return bool(_COUNT_UNIT_RE.match((unit or "").replace(" ", "")))


def _weight_unit_to_grams(unit: str) -> float:
    m = _WEIGHT_UNIT_RE_PS.match((unit or "").replace(" ", "").lower())
    if not m:
        return 0.0
    return float(m.group(1)) * (1000 if m.group(2) in ("kg", "l") else 1)


def _best_weight_row(rows_by_db: dict, db_name: str):
    """같은 품목의 무게단위 행 중 중량이 가장 큰(=소포장 과대 회피) 것 반환. (unit, avg) | None."""
    best = None
    for (dn, du), avg in rows_by_db.items():
        if dn != db_name or not avg or avg <= 0:
            continue
        g = _weight_unit_to_grams(du)
        if g > 0 and (best is None or g > best[2]):
            best = (du, avg, g)
    return (best[0], best[1]) if best else None


_TREND_KEYWORDS = re.compile(r"(추이|추세|그래프|트렌드|변동|변화|시세|동향|trend|graph|일별|주별|월별)")
_PERIOD_MAP = {
    "일주일": 7, "1주일": 7, "7일": 7, "한주": 7,
    "2주": 14, "이주일": 14,
    "한달": 30, "1달": 30, "한 달": 30, "30일": 30, "1개월": 30,
    "두달": 60, "2달": 60, "2개월": 60,
    "세달": 90, "석달": 90, "3달": 90, "3개월": 90,
    "넉달": 120, "4달": 120, "4개월": 120,
    "5개월": 150, "5달": 150,
    "반년": 180, "6개월": 180, "6달": 180,
    "1년": 365, "일년": 365, "12개월": 365,
}
# 동적 기간 파싱 정규식: "N개월", "N달", "N일", "N주", "N년"
_PERIOD_DYNAMIC_RE = re.compile(r"(\d+)\s*(개월|달|일|주|주일|년)")
_KOR_NUM = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
_PERIOD_KOR_RE = re.compile(r"(" + "|".join(_KOR_NUM.keys()) + r")\s*(개월|달|일|주|주일|년)")


def _detect_trend_request(user_query: str) -> bool:
    """사용자 질문이 시계열 추이 요청인지 감지."""
    return bool(_TREND_KEYWORDS.search(user_query))


def _extract_trend_days(user_query: str) -> int:
    """추이 기간 추출. 기간 제한 없이 자유롭게 지원. 기본 30일."""
    # 1단계: 고정 키워드 매칭 (정확한 표현 우선)
    for keyword, days in _PERIOD_MAP.items():
        if keyword in user_query:
            return days
    # 2단계: 동적 숫자 파싱 ("5개월", "14일", "2주" 등)
    m = _PERIOD_DYNAMIC_RE.search(user_query)
    if m:
        num = int(m.group(1))
        unit = m.group(2)
        if unit in ("개월", "달"):
            return num * 30
        elif unit == "일":
            return num
        elif unit in ("주", "주일"):
            return num * 7
        elif unit == "년":
            return num * 365
    # 3단계: 한글 숫자 파싱 ("세 달", "다섯 개월" 등)
    km = _PERIOD_KOR_RE.search(user_query)
    if km:
        num = _KOR_NUM[km.group(1)]
        unit = km.group(2)
        if unit in ("개월", "달"):
            return num * 30
        elif unit == "일":
            return num
        elif unit in ("주", "주일"):
            return num * 7
        elif unit == "년":
            return num * 365
    # 기본값: 30일 (추이를 물어보는 사용자에겐 7일보다 30일이 더 적절)
    return 30


def _timeseries_sql_query(ingredient_name: str, db_name: str, db_unit: str, days: int) -> dict:
    """시계열 가격 데이터 직접 SQL 조회 -> 날짜별 평균가격 DataFrame 반환.

    날짜당 여러 지역/출처의 가격이 있으므로 GROUP BY + AVG로 집계하여
    Genie Space와 동일한 깔끔한 라인 차트를 생성.
    """
    safe_name = (db_name or "").replace("'", "''")
    safe_unit = (db_unit or "").replace("'", "''")
    sql = f"""
SELECT `날짜`, ROUND(AVG(`가격`)) AS `평균가격`
FROM silver.ingredient.ingredient
WHERE `재료명` = '{safe_name}'
  AND `단위` = '{safe_unit}'
  AND `날짜` >= DATE_SUB(CURRENT_DATE(), {days})
  AND `가격` IS NOT NULL
GROUP BY `날짜`
ORDER BY `날짜`
""".strip()

    try:
        host = os.environ.get("DATABRICKS_HOST")
        token = os.environ.get("DATABRICKS_TOKEN")
        w = WorkspaceClient(host=host, token=token) if host and token else WorkspaceClient()
        warehouse_id = _get_warehouse_id(w)
        if not warehouse_id:
            return {"dataframe": None, "text": "", "sql": sql, "error": "warehouse_not_found"}

        resp = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=sql,
            wait_timeout="50s",
        )
        if not resp.status or resp.status.state != StatementState.SUCCEEDED:
            err_msg = resp.status.error.message if resp.status and resp.status.error else "unknown"
            return {"dataframe": None, "text": "", "sql": sql, "error": f"sql_failed: {err_msg}"}

        rows = resp.result.data_array or []
        if not rows:
            return {"dataframe": None, "text": f"{ingredient_name}: 최근 {days}일 데이터 없음", "sql": sql, "error": None}

        df = pd.DataFrame(rows, columns=["날짜", "평균가격"])
        df["평균가격"] = pd.to_numeric(df["평균가격"], errors="coerce")
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
        df = df.dropna(subset=["날짜", "평균가격"]).sort_values("날짜")

        # 가격 텍스트 요약 제거 - 차트만 표시
        return {"dataframe": df, "text": "", "sql": sql, "error": None}
    except Exception as e:
        return {"dataframe": None, "text": "", "sql": sql, "error": f"exception: {str(e)}"}


GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "01f148e5845f1f68843892ceb53abd32")

# Genie 한 번에 조회할 재료 수 제한 (작게 잡을수록 SQL 생성 안정성 ↑)
_GENIE_BATCH_SIZE = 7
# Genie 동시 호출 워커 수 (Databricks rate limit 고려)
_GENIE_MAX_WORKERS = 5
# 배치가 FAILED로 실패했을 때 항목별 단건 재시도 최대 동시 워커 수
_GENIE_RETRY_MAX_WORKERS = 2

# WorkspaceClient 싱글턴 — 매 호출마다 재생성하지 않고 재사용
_ws_client: WorkspaceClient | None = None


def _get_client() -> WorkspaceClient:
    """WorkspaceClient 싱글턴 반환. 인증 핸드셰이크를 1회만 수행."""
    global _ws_client
    if _ws_client is None:
        _ws_client = WorkspaceClient()
    return _ws_client


# negation 패턴: Genie가 "X는 조회되지 않았습니다" 식으로 말할 때 잡기 위함
_NEGATION_PATTERNS = [
    re.compile(r"조회되지\s*않"),
    re.compile(r"확인되지\s*않"),
    re.compile(r"데이터가\s*없"),
    re.compile(r"제공할\s*수\s*없"),
    re.compile(r"찾을\s*수\s*없"),
    re.compile(r"존재하지\s*않"),
    re.compile(r"기록이\s*없"),
]

# 가격 표기 패턴 — "1,200원", "₩1,200", "1.2kg당 3,000원" 등
_PRICE_PATTERN = re.compile(r"(?:₩|\d[\d,]{2,}\s*원)|(?:\d+\s*(?:kg|g|ml|L|개)\s*당?\s*\d[\d,]+)")

# --- 빠른 응답과 일관성을 위한 인메모리 캐시 (용량 제한 및 TTL 적용) ---
_batch_cache = {}
_MAX_CACHE_SIZE = 100       # 최대 100개의 배치 묶음만 기억
_CACHE_TTL_SECONDS = 3600   # 1시간(3600초)이 지나면 만료

# direct_sql 1차 조회 결과 캐시 (같은 (재료명,단위) 조합 반복 조회 시 warehouse 재호출 회피)
_direct_sql_cache: dict[tuple, tuple[float, dict]] = {}


def _get_warehouse_id(w: WorkspaceClient) -> str | None:
    """direct_sql 폴백용 warehouse 조회. databricks_db.py / catalog.py와 동일 패턴."""
    wh_id = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if wh_id:
        return wh_id
    for wh in w.warehouses.list():
        if wh.state and wh.state.value in ("RUNNING", "STARTING"):
            return wh.id
    warehouses = list(w.warehouses.list())
    return warehouses[0].id if warehouses else None


# ─── 지역(시도) 매핑 ─────────────────────────────────────────────
# 프로필 '[업종: X, 지역: Y]'의 지역은 자유 입력("서울 강남구")이라 silver.ingredient.ingredient의
# `시도`(서울/부산/경기...) 값으로 매핑해 그 지역 도매가로 원가를 낸다. 지역 데이터가 없는
# 재료는 전국 평균으로 폴백(COALESCE). B2B/네이버 가격엔 지역이 없어 그대로 둔다.
_SIDO_LIST = ("서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
              "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주")
_SIDO_ALIASES = {"충청북도": "충북", "충청남도": "충남", "전라북도": "전북",
                 "전라남도": "전남", "경상북도": "경북", "경상남도": "경남"}


def _sido_from_region(region: str) -> str | None:
    """자유 입력 지역('서울 강남구', '경기도 수원') → silver `시도` 값. 못 찾으면 None."""
    t = (region or "").strip()
    if not t:
        return None
    for long_name, short in _SIDO_ALIASES.items():
        if t.startswith(long_name):
            return short
    for s in _SIDO_LIST:
        if t.startswith(s) or s in t:
            return s
    return None


def _extract_region(state: dict) -> str:
    """메시지의 '[업종: X, 지역: Y]'에서 지역 추출. 없으면 ''."""
    try:
        for msg in reversed(state.get("messages", []) or []):
            content = getattr(msg, "content", "") or ""
            m = re.search(r"지역:\s*([^,\]\n]+)", content)
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    return ""


def _direct_sql_query(targets: list[tuple[str, str, str]], region: str | None = None) -> dict:
    """카탈로그 (재료명, 단위) 조합으로 statement_execution을 직접 호출.

    Genie 우회 — LLM SQL 생성을 건너뛰고 결정적인 WHERE 절로 최근 30일 평균 도매가 조회.
    재질의도 실패한 matched 항목의 마지막 폴백으로 사용한다.

    Returns:
        {
          "text": "재료명: ₩가격/kg ..." 형식 문자열(report_generator/사람용),
          "found": [회복된 input_name],
          "prices": {input_name: {"price_per_kg": int, "unit_hint": str}}  # 무게단위 환산분만
          "error": Optional[str], "sql": str,
        }
        결과가 없거나 실패하면 found=[]/prices={}로 반환.
    """
    if not targets:
        return {"text": "", "found": [], "prices": {}, "error": None}

    # (db_name, db_unit) → 행 데이터(avg_price, db_unit). 캐시는 이 db단위 행만 저장하고,
    # input_name 입히기(text/found/prices 구성)는 캐시 hit/miss 공통으로 마지막에 1회만
    # 수행한다. → 같은 db재료를 다른 input_name으로 조회해도(예: '마늘'/'다진마늘'→'깐마늘')
    # 캐시가 input_name에 오염되지 않는다.
    sido = _sido_from_region(region)   # 지역 설정 시 그 시도 도매가, 없으면 None(전국)
    cache_key = tuple(sorted((t[1], t[2]) for t in targets)) + (sido or "",)
    cached = _direct_sql_cache.get(cache_key)
    if cached and (time.time() - cached[0] < _CACHE_TTL_SECONDS):
        rows_by_db, sql = cached[1], cached[2]
        return _direct_sql_assemble(targets, rows_by_db, sql, cached_hit=True, sido=sido)

    # WHERE 절 동적 생성: (재료명='X' AND 단위='Y') OR ...
    where_clauses = []
    for _input_name, db_name, db_unit in targets:
        # SQL 인젝션 방지: 작은따옴표 escape
        safe_name = (db_name or "").replace("'", "''")
        safe_unit = (db_unit or "").replace("'", "''")
        where_clauses.append(f"(`재료명` = '{safe_name}' AND `단위` = '{safe_unit}')")
        # 개수단위 타깃은 같은 품목의 무게단위 행도 함께 조회(무게 우선 환산용)
        if _is_count_unit_str(db_unit):
            where_clauses.append(f"(`재료명` = '{safe_name}')")

    # 지역(시도) 설정 시: 그 지역 평균을 우선, 그 지역에 데이터 없으면 전국 평균으로 폴백(COALESCE).
    if sido:
        safe_sido = sido.replace("'", "''")
        price_expr = ("ROUND(COALESCE("
                      f"AVG(CASE WHEN `시도` = '{safe_sido}' THEN `가격` END), AVG(`가격`)))")
    else:
        price_expr = "ROUND(AVG(`가격`))"
    sql = f"""
SELECT `재료명`, `단위`, {price_expr} AS `평균가격`, COUNT(*) AS `행수`
FROM silver.ingredient.ingredient
WHERE ({" OR ".join(where_clauses)})
  AND `날짜` >= DATE_SUB(CURRENT_DATE(), 30)
  AND `가격` IS NOT NULL
GROUP BY `재료명`, `단위`
""".strip()

    try:
        # WorkspaceClient 싱글턴 재사용 (매 호출 인증 핸드셰이크 회피).
        w = _get_client()
        warehouse_id = _get_warehouse_id(w)
        if not warehouse_id:
            return {"text": "", "found": [], "prices": {}, "error": "warehouse_not_found"}

        resp = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=sql,
            wait_timeout="50s",
        )
        if not resp.status or resp.status.state != StatementState.SUCCEEDED:
            err_msg = resp.status.error.message if resp.status and resp.status.error else "unknown"
            return {"text": "", "found": [], "prices": {}, "error": f"sql_failed: {err_msg}"}

        # db단위 행 데이터만 추출 (input_name 무관).
        rows_by_db: dict[tuple[str, str], int] = {}
        for row in (resp.result.data_array or []):
            db_name = str(row[0] or "").strip()
            db_unit = str(row[1] or "").strip()
            try:
                avg_price = int(float(row[2])) if row[2] is not None else None
            except (TypeError, ValueError):
                avg_price = None
            if avg_price is None or avg_price <= 0:
                continue
            rows_by_db[(db_name, db_unit)] = avg_price

        # 캐시 적재 (용량 초과 시 가장 오래된 항목 제거)
        if len(_direct_sql_cache) >= _MAX_CACHE_SIZE:
            del _direct_sql_cache[next(iter(_direct_sql_cache))]
        _direct_sql_cache[cache_key] = (time.time(), rows_by_db, sql)

        return _direct_sql_assemble(targets, rows_by_db, sql, cached_hit=False, sido=sido)
    except Exception as e:
        return {"text": "", "found": [], "prices": {}, "error": f"exception: {str(e)}"}


def _direct_sql_assemble(
    targets: list[tuple[str, str, str]],
    rows_by_db: dict[tuple[str, str], int],
    sql: str,
    cached_hit: bool,
    sido: str | None = None,
) -> dict:
    """db단위 행 데이터(rows_by_db)에 호출자 targets의 input_name을 입혀 결과 구성.

    무게 단위(kg/g)는 원/kg로 환산해 prices(구조화)에 직접 담아 cost_calculator가
    텍스트 파싱 없이 1순위로 쓰게 한다. 그 외 단위(개/마리 등)는 환산 불가라 text/found
    에만 넣고 prices에는 안 넣는다(cost_calculator가 사용량 기준으로 환산).
    """
    lines: list[str] = []
    found_inputs: list[str] = []
    prices: dict[str, dict] = {}
    _note = "KAMIS direct_sql 평균 도매가" + (f" ({sido} 기준)" if sido else "")
    for input_name, db_name, db_unit in targets:
        avg_price = rows_by_db.get((db_name, db_unit))
        if avg_price is None or avg_price <= 0:
            continue
        unit_lower = db_unit.lower().replace(" ", "")
        kg_match = re.match(r"^(\d+(?:\.\d+)?)kg$", unit_lower)
        g_match = re.match(r"^(\d+(?:\.\d+)?)g$", unit_lower)
        if kg_match:
            kg_val = float(kg_match.group(1))
            price_per_kg = int(avg_price / kg_val) if kg_val > 0 else avg_price
            prices[input_name] = {
                "price_per_kg": price_per_kg,
                "confidence": "high",
                "unit_hint": f"{db_unit} (KAMIS direct_sql)",
                "note": _note,
            }
            lines.append(
                f"{input_name}: 약 ₩{price_per_kg:,}/kg "
                f"(KAMIS direct_sql, {db_name}/{db_unit} 평균)"
            )
        elif g_match:
            g_val = float(g_match.group(1))
            price_per_kg = int(avg_price * 1000 / g_val) if g_val > 0 else avg_price
            prices[input_name] = {
                "price_per_kg": price_per_kg,
                "confidence": "high",
                "unit_hint": f"{db_unit} → kg 환산 (KAMIS direct_sql)",
                "note": _note,
            }
            lines.append(
                f"{input_name}: 약 ₩{price_per_kg:,}/kg "
                f"(KAMIS direct_sql, {db_name}/{db_unit} → kg 환산)"
            )
        else:
            # 무게가 아닌 단위(개/포기/마리 등).
            # 1순위: 같은 품목에 무게단위 도매가가 있으면 그걸로 환산한다. 개수→그램 추정은
            # 마리/개당 무게 가정이 부정확해 단가가 크게 튄다(예: 새우 10마리÷15g → 33,913/kg
            # vs 2kg 기준 13,870/kg). 무게단위 행이 있으면 그게 더 신뢰도 높다.
            weight_row = _best_weight_row(rows_by_db, db_name)
            # 2순위: 무게단위가 없으면 1개당 그램 추정치로 환산(무우 같은 '개' 단위 누락 방지).
            piece_match = re.match(r"^(\d+(?:\.\d+)?)\s*(?:개|포기|마리|통|단|모|장|알|봉|손|쪽)$", unit_lower)
            grams_per_piece = _PER_PIECE_GRAMS.get(db_name) or _PER_PIECE_GRAMS.get(input_name)
            if weight_row:
                w_unit, w_avg = weight_row
                w_grams = _weight_unit_to_grams(w_unit)
                price_per_kg = int(w_avg * 1000 / w_grams) if w_grams > 0 else w_avg
                prices[input_name] = {
                    "price_per_kg": price_per_kg,
                    "confidence": "high",
                    "unit_hint": f"{w_unit} 무게단위 우선 ({db_unit} 대신, KAMIS direct_sql)",
                    "note": _note,
                }
                lines.append(
                    f"{input_name}: 약 ₩{price_per_kg:,}/kg "
                    f"(KAMIS direct_sql, {db_name}/{w_unit} 무게단위 우선)"
                )
            elif piece_match and grams_per_piece:
                qty_val = float(piece_match.group(1))
                total_grams = qty_val * grams_per_piece
                price_per_kg = int(avg_price * 1000 / total_grams) if total_grams > 0 else avg_price
                prices[input_name] = {
                    "price_per_kg": price_per_kg,
                    "confidence": "high",
                    "unit_hint": f"{db_unit} → kg 환산 (1{db_unit.rstrip('0123456789')}≈{grams_per_piece:g}g, KAMIS direct_sql)",
                    "note": _note,
                }
                lines.append(
                    f"{input_name}: 약 ₩{price_per_kg:,}/kg "
                    f"(KAMIS direct_sql, {db_name}/{db_unit} → kg 환산)"
                )
            else:
                # 환산 계수 미상 — 종전대로 text에만 (cost_calculator가 사용량 기준 처리)
                lines.append(
                    f"{input_name}: 약 ₩{avg_price:,}/{db_unit} "
                    f"(KAMIS direct_sql, {db_name}/{db_unit} 평균)"
                )
        found_inputs.append(input_name)

    return {
        "text": "\n".join(lines),
        "found": found_inputs,
        "prices": prices,
        "error": None,
        "sql": sql,
        "cached_hit": cached_hit,
    }


def _ask_genie(question: str, conversation_id: str = None) -> dict:
    w = _get_client()
    result = {"text": None, "sql": None, "dataframe": None, "conversation_id": None}

    if conversation_id is None:
        response = w.genie.start_conversation_and_wait(
            space_id=GENIE_SPACE_ID, content=question)
        result["conversation_id"] = response.conversation_id
    else:
        response = w.genie.create_message_and_wait(
            space_id=GENIE_SPACE_ID, conversation_id=conversation_id, content=question)
        result["conversation_id"] = conversation_id

    if response.attachments:
        text_parts = []
        for att in response.attachments:
            if att.text and att.text.content:
                text_parts.append(att.text.content)
            if att.query and att.query.query:
                result["sql"] = att.query.query
            if att.attachment_id:
                try:
                    qr = w.genie.get_message_query_result(
                        space_id=GENIE_SPACE_ID,
                        conversation_id=result["conversation_id"],
                        message_id=response.id,
                        attachment_id=att.attachment_id)
                    if qr.columns and qr.data_array:
                        result["dataframe"] = pd.DataFrame(
                            qr.data_array, columns=[c.name for c in qr.columns])
                except Exception:
                    pass
        if text_parts:
            result["text"] = "\n".join(text_parts)

    return result


# Genie 구조적 응답 파싱 패턴
_FOUND_LIST_RE = re.compile(
    r"조회(?:된|되는)?\s*재료(?:는|가|로)?\s*[(\[]?(.+?)(?:입니다|이며|이다|입\b|이고|\.)",
    re.DOTALL,
)
_NOT_FOUND_LIST_RE = re.compile(
    r"나머지\s*재료\s*[(\[]?(.+?)(?:[)\]]|는\s*DB|는\s*'없음'|는\s*\"없음\"|는\s*데이터|입니다)",
    re.DOTALL,
)


def _has_price_evidence(ing: str, genie_text: str, other_names: list | None = None) -> bool:
    """genie_text 안에서 ing '자신'의 가격 숫자가 부정어 없이 있으면 True.

    이전 버전은 이름 주변 윈도우(-30~+80)에 '아무' 가격 숫자만 있으면 True였다.
    그 결과 "조회된 재료는 …양파, 청양고추입니다. … 고춧가루 49,500원, …"처럼
    재료를 콤마로 나열하면, 양파 본인 가격이 없어도 뒤따라오는 '남의 가격'을
    자기 것으로 오인해 True가 되어 unavailable 폴백을 못 받는 결함이 있었다.
    (cf. project_onion_judgment_fix / project_genie_format_parsing)

    수정: 이름 '바로 뒤(+35자)' 또는 '바로 앞(-25자)' 좁은 구간만 보고,
    이름과 그 가격 사이에 콤마/줄바꿈/다른 재료명 같은 '경계'가 끼면 남의 가격으로
    간주하여 제외한다. 이로써 판정이 cost_calculator의 실제 가격 파싱과 일치하게 된다.
    """
    if ing not in genie_text:
        return False
    others = [n for n in (other_names or []) if n and n != ing]
    for match in re.finditer(re.escape(ing), genie_text):
        # 뒤쪽 구간(가장 흔한 "재료명 NN,NNN원" 순서). 콤마 나열에서 다음 재료
        # 가격이 안 새어들도록 +80 → +35로 좁힘.
        fwd = genie_text[match.end():match.end() + 35]
        # 앞쪽 구간(역순 표기 대비). 이름과 가격 사이 경계는 가격 '뒤쪽'으로 검사.
        bwd = genie_text[max(0, match.start() - 25):match.start()]
        for window, is_forward in ((fwd, True), (bwd, False)):
            if any(p.search(window) for p in _NEGATION_PATTERNS):
                continue
            pm = _PRICE_PATTERN.search(window)
            if not pm:
                continue
            # 이름 ↔ 가격 사이 구간(seg)에 경계가 있으면 그 가격은 남의 것.
            seg = window[:pm.start()] if is_forward else window[pm.end():]
            if "," in seg or "\n" in seg:
                continue
            if any(o in seg for o in others):
                continue
            return True
    return False


def _detect_unavailable(ingredients: list, genie_text: str) -> list:
    """요청한 재료 vs Genie 응답을 비교하여 누락된 재료 반환.

    전략:
    1) Genie가 "조회된 재료는 X, Y, Z" 식으로 명시한 경우 → 그 외는 모두 unavailable
    2) "나머지 재료(A, B, C)는 ... 없" 패턴이 있으면 그것도 활용
    3) 명시적 패턴 없으면 기존 윈도우 기반 polling으로 폴백
    """
    if not genie_text or not ingredients:
        return list(ingredients)

    # ── Step 1: "조회된 재료는 X, Y, Z" 명시적 추출 ──
    explicit_found = set()
    for fm in _FOUND_LIST_RE.finditer(genie_text):
        found_text = re.sub(r"\*+", "", fm.group(1))  # markdown ** 제거
        for ing in ingredients:
            if ing in found_text:
                explicit_found.add(ing)

    # ── Step 2: "나머지 재료(A, B, C)는 ... 없" 명시적 추출 ──
    explicit_not_found = set()
    for nm in _NOT_FOUND_LIST_RE.finditer(genie_text):
        nf_text = nm.group(1)
        for ing in ingredients:
            if ing in nf_text:
                explicit_not_found.add(ing)

    # ── Step 3: 명시적 found 리스트가 있으면, 단 가격 숫자가 실제로 있는지 재확인 ──
    # Genie가 "조회된 재료는 …양파…입니다"라고 이름만 말하고 가격 줄을 안 주는 경우가 있다.
    # 이름이 found 목록에 있어도 응답 본문에 그 재료의 가격 숫자가 없으면 unavailable로 내려
    # 기존 폴백(카탈로그 재질의 → direct_sql)이 가격을 채우게 한다.
    if explicit_found:
        unavailable = []
        for ing in ingredients:
            if ing not in explicit_found:
                unavailable.append(ing)
                continue
            if not _has_price_evidence(ing, genie_text, ingredients):
                unavailable.append(ing)
        return unavailable

    # ── Step 4: 명시적 not_found 리스트만 있으면 그것만 unavailable ──
    if explicit_not_found:
        return [ing for ing in ingredients if ing in explicit_not_found]

    # ── Step 5: 폴백 — 기존 윈도우 기반 polling ──
    return [ing for ing in ingredients if not _has_price_evidence(ing, genie_text, ingredients)]


def _build_catalog_query(targets: list[tuple[str, str, str]]) -> str:
    """(input_name, db_name, db_unit) 튜플 목록으로 정확명 쿼리 생성."""
    lines = []
    for input_name, db_name, db_unit in targets:
        lines.append(f"- 재료명='{db_name}', 단위='{db_unit}'  (사용자 입력: '{input_name}')")
    return (
        "아래는 silver.ingredient.ingredient 테이블에 확실히 존재하는 (재료명, 단위) 조합입니다. "
        "각 항목에 대해 정확히 그 재료명/단위로 WHERE 절을 작성하여 최근 도매가를 조회해줘. "
        "다른 단위로 대체하거나 LIKE 검색하지 말고 명시된 조건만 사용해줘. "
        "반드시 WHERE 재료명 IN ('A', 'B', 'C') 형식으로 단 1개의 SQL 쿼리를 작성해서 모든 재료를 한 번에 조회해줘. "
        "재료별로 개별 쿼리를 실행하지 마. "
        "응답은 반드시 다음 형식을 포함해줘:\n"
        "  '조회된 재료는 X, Y, Z입니다.'\n"
        "  '나머지 재료(A, B, C)는 DB에 데이터가 없어 없음으로 분류됩니다.'\n"
        "그 다음 각 재료별 가격을 자세히 알려줘.\n"
        + "\n".join(lines)
    )


def _build_passthrough_query(batch_items: list[str]) -> str:
    """alias 미등록 재료를 위한 자유 텍스트 쿼리."""
    items_str = "', '".join(batch_items)
    return (
        "다음 재료들에 대해서만 최근 도매가를 조회해줘. "
        "반드시 실제 DB에 있는 데이터만 보고해줘. DB에 없는 재료는 '없음'으로 표시해줘. 추정값 사용 금지. "
        f"반드시 WHERE 재료명 IN ('{items_str}') 형식으로 단 1개의 SQL 쿼리를 작성해서 모든 재료를 한 번에 조회해줘. "
        "재료별로 개별 쿼리를 실행하지 마. "
        "응답은 반드시 다음 형식을 포함해줘:\n"
        "  '조회된 재료는 X, Y, Z입니다.'\n"
        "  '나머지 재료(A, B, C)는 DB에 데이터가 없어 없음으로 분류됩니다.'\n"
        "그 다음 각 재료별 가격을 자세히 알려줘. "
        f"재료 목록: {', '.join(batch_items)}"
    )


def price_search_node(state: dict) -> dict:
    """Genie Space로 도매가 조회. 조회 후 누락 재료를 unavailable 필드로 반환."""
    entities = state.get("entities", {})
    recipe_info = state.get("recipe_info", {})
    loop_count = state.get("loop_count", 0)
    region = _extract_region(state)   # 지역 설정 시 그 시도(서울/부산..) 도매가로 원가 산정

    ingredients = []
    if recipe_info and recipe_info.get("data"):
        for recipe in recipe_info["data"]:
            for ing in recipe.get("ingredients", []):
                name = ing.get("name", "") if isinstance(ing, dict) else str(ing)
                if name:
                    ingredients.append(name)
    elif entities.get("ingredient"):
        ingredients = [entities["ingredient"]] if isinstance(entities["ingredient"], str) else entities["ingredient"]
    elif entities.get("menu"):
        ingredients = [entities["menu"]] if isinstance(entities["menu"], str) else entities["menu"]

    # ═══ [시계열 추이 요청 감지] ═══════════════════════════════════════════════
    # "계란 가격 추이 보여줘", "양파 시세 알려줘" 등 시계열 요청은
    # Genie 배치 대신 직접 SQL로 일별 평균가격을 조회 -> 차트 생성.
    # 차트를 비활성화하려면 chart_utils.py의 ENABLE_CHART = False로 변경.
    # ═══════════════════════════════════════════════════════════════════════════
    user_query = state["messages"][-1].content if state.get("messages") else ""
    if ingredients and _detect_trend_request(user_query):
        from backend.catalog import resolve_ingredient
        trend_days = _extract_trend_days(user_query)
        _TREND_MAX_INGREDIENTS = 5
        trend_targets = ingredients[:_TREND_MAX_INGREDIENTS]
        trend_dropped = ingredients[_TREND_MAX_INGREDIENTS:]
        frames = []      # 차트용 (재료명 태그된 df)
        summaries = []   # 재료별 요약 한 줄
        names = []       # 차트에 실린 재료명
        sqls = []
        no_data = []     # 매칭 실패 또는 데이터 없음
        for target_name in trend_targets:
            resolved = resolve_ingredient(target_name)
            if not (resolved.status == "matched" and resolved.db_name and resolved.db_unit):
                no_data.append(target_name)
                continue
            ts_result = _timeseries_sql_query(
                target_name, resolved.db_name, resolved.db_unit, trend_days
            )
            if ts_result["sql"]:
                sqls.append(ts_result["sql"])
            df = ts_result["dataframe"]
            if df is not None and not df.empty:
                tagged = df.copy()
                tagged["재료명"] = target_name
                frames.append(tagged)
                names.append(target_name)
                summaries.append(
                    f"{target_name}: 평균 \u20a9{int(df['평균가격'].mean()):,}, "
                    f"최고 \u20a9{int(df['평균가격'].max()):,}, 최저 \u20a9{int(df['평균가격'].min()):,}"
                )
            else:
                no_data.append(target_name)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            price_data = {"source": "timeseries_direct", "data": names}
            if sqls:
                price_data["sql"] = "\n\n".join(sqls)
            chart_html = generate_chart_html(combined, user_query=user_query)
            start_date = combined["날짜"].min().strftime("%Y-%m-%d")
            end_date = combined["날짜"].max().strftime("%Y-%m-%d")
            note = ""
            if trend_dropped:
                note += f" (재료가 많아 {len(trend_dropped)}개 제외: {', '.join(trend_dropped)})"
            if no_data:
                note += f" [데이터 없음: {', '.join(no_data)}]"
            if chart_html:
                price_data["chart_html"] = chart_html
                price_data["text"] = (
                    f"최근 {trend_days}일 가격 추이를 조회했습니다 "
                    f"(기간: {start_date} ~ {end_date}). " + " / ".join(summaries) + note
                    + " 인터랙티브 차트가 아래에 표시됩니다. "
                    f"[주의: 날짜별 가격 목록, 텍스트 그래프, ASCII 차트를 절대 작성하지 마세요. "
                    f"위 요약만 전달하세요.]"
                )
            else:
                price_data["table"] = combined.to_string(index=False, max_rows=7)
            archive("price_search.timeseries_direct", {
                "ingredients": names,
                "dropped": trend_dropped,
                "no_data": no_data,
                "days": trend_days,
                "has_chart": bool(price_data.get("chart_html")),
                "num_rows": len(combined),
            })
            return {"price_info": price_data}

    if not ingredients:
        user_query = state["messages"][-1].content if state.get("messages") else ""
        if user_query:
            try:
                with mlflow.start_span(name="genie_freeform_query", span_type=SpanType.RETRIEVER) as span:
                    span.set_inputs({"question": user_query})
                    result = _ask_genie(user_query)
                    span.set_outputs({
                        "text": result.get("text"),
                        "has_sql": bool(result.get("sql")),
                        "has_dataframe": result.get("dataframe") is not None,
                    })
                if result["text"] or result["dataframe"] is not None:
                    price_data = {"source": "genie", "data": [user_query]}
                    if result["text"]:
                        price_data["text"] = result["text"]
                    if result["sql"]:
                        price_data["sql"] = result["sql"]
                    if result["dataframe"] is not None:
                        price_data["table"] = result["dataframe"].to_string(index=False, max_rows=15)

                    # ═══ [차트 생성] ═══════════════════════════════════════
                    # Genie API는 차트를 반환하지 않으므로 DataFrame에서 직접 생성.
                    # 차트를 비활성화하려면 chart_utils.py의 ENABLE_CHART = False로 변경.
                    # ═══════════════════════════════════════════════════════════════
                    chart_html = generate_chart_html(result["dataframe"], user_query=user_query)
                    if chart_html:
                        price_data["chart_html"] = chart_html
           
                    return {"price_info": price_data}   

            except Exception:
                pass
        return {"price_info": {"source": "genie", "data": [], "note": "조회할 재료 없음"}}

    ingredients = list(dict.fromkeys(ingredients))

    # 재료별 trace_id — 모든 archive에 동일 형식으로 박아 grep 추적용.
    trace_ids = {ing: f"{loop_count}:{ing}" for ing in ingredients}

    # ─── 카탈로그 기반 그룹 분리 (4그룹) ────────────────────────
    # resolve_many()는 각 재료를 5가지 status로 분류한다:
    #   matched         → 정확명 쿼리 그룹 (catalog_targets)
    #   recipe_matched  → B2B 유통가 직접 사용 (recipe_direct) ★ NEW
    #   unmapped        → alias 미등록, 자유 쿼리로 시도 (passthrough)
    #   ambiguous       → 자동 매칭 금지, Genie 건너뛰고 곧장 unavailable (skip)
    #   not_in_catalog  → alias 매핑은 있는데 카탈로그엔 없음 (skip)
    resolved: list[ResolveResult] = resolve_many(ingredients)
    resolved_by_input: dict[str, ResolveResult] = {r.input_name: r for r in resolved}

    catalog_targets: list[tuple[str, str, str]] = []  # (input, db_name, db_unit)
    recipe_direct: list[str] = []  # recipe_matched → B2B 가격 직접 사용, Genie 건너뜀
    passthrough: list[str] = []
    skip_unavailable: list[str] = []
    for r in resolved:
        if r.status == "matched" and r.db_name and r.db_unit:
            catalog_targets.append((r.input_name, r.db_name, r.db_unit))
        elif r.status == "recipe_matched":
            recipe_direct.append(r.input_name)
        elif r.status == "unmapped":
            passthrough.append(r.input_name)
        else:
            # ambiguous / not_in_catalog → Genie를 거치지 않고 곧장 unavailable로
            skip_unavailable.append(r.input_name)

    archive("price_search.input", {
        "ingredients": ingredients,
        "num_ingredients": len(ingredients),
        "region": region,                          # 사용자 설정 지역(자유 입력)
        "sido": _sido_from_region(region),         # 매핑된 시도(가격 필터 적용값) — None이면 전국
        "batch_size": _GENIE_BATCH_SIZE,
        "trace_ids": trace_ids,
        "groups": {
            "catalog_targets": [t[0] for t in catalog_targets],
            "recipe_direct": recipe_direct,
            "passthrough": passthrough,
            "skip_unavailable": skip_unavailable,
        },
        "resolved": [
            {"input": r.input_name, "status": r.status,
             "db_name": r.db_name, "db_unit": r.db_unit, "reason": r.reason}
            for r in resolved
        ],
    })

    try:
        # ─── direct_sql 1차 조회 (catalog 재료) ──────────────
        # 방침 변경(2026-06-01): KAMIS 정확명(matched)으로 매칭된 재료는 결정적인
        # statement_execution(direct_sql)을 '먼저' 돌린다. LLM이 SQL을 생성·실행하는
        # Genie(배치당 ~40초)를 건너뛰어 속도가 크게 빨라지고, 자연어→정규식 파싱의
        # 비결정성도 회피한다. direct_sql이 못 잡은 catalog 재료만 Genie로 폴백.
        # passthrough(alias 미등록)는 정확명이 없어 direct_sql 불가 → 기존대로 Genie.
        direct_first_texts: list[str] = []
        direct_first_sqls: list[str] = []
        direct_first_recovered: set[str] = set()
        # direct_sql이 환산한 원/kg 구조화 가격 — structured_prices에 직접 투입해
        # cost_calculator가 텍스트 파싱 없이 1순위로 쓰게 한다(B 전환의 핵심 이득).
        direct_first_prices: dict[str, dict] = {}
        remaining_catalog: list[tuple[str, str, str]] = list(catalog_targets)
        if catalog_targets:
            archive("price_search.direct_first_attempt", {
                "items": [t[0] for t in catalog_targets],
                "trace_ids": [trace_ids.get(t[0], t[0]) for t in catalog_targets],
            })
            try:
                with mlflow.start_span(
                    name=f"direct_sql_first_[{','.join(t[0] for t in catalog_targets)[:60]}]",
                    span_type=SpanType.RETRIEVER,
                ) as span:
                    span.set_inputs({"targets": catalog_targets, "region": region})
                    df_result = _direct_sql_query(catalog_targets, region=region)
                    span.set_outputs({
                        "found": df_result.get("found"),
                        "error": df_result.get("error"),
                        "text_preview": (df_result.get("text") or "")[:200],
                    })
                direct_first_recovered = set(df_result.get("found") or [])
                direct_first_prices = df_result.get("prices") or {}
                if df_result.get("text"):
                    direct_first_texts.append(df_result["text"])
                if df_result.get("sql"):
                    direct_first_sqls.append(df_result["sql"])
                remaining_catalog = [
                    t for t in catalog_targets if t[0] not in direct_first_recovered
                ]
                archive("price_search.direct_first_result", {
                    "recovered": list(direct_first_recovered),
                    "num_structured": len(direct_first_prices),
                    "still_missing": [t[0] for t in remaining_catalog],
                    "error": df_result.get("error"),
                    "cached_hit": df_result.get("cached_hit"),
                })
            except Exception as df_exc:
                archive("price_search.direct_first_error", {
                    "error": str(df_exc),
                    "items": [t[0] for t in catalog_targets],
                })
                remaining_catalog = list(catalog_targets)

        # ─── 배치 계획 (direct_sql 실패 catalog + passthrough만 Genie로) ───
        batches: list[tuple[str, list]] = []
        for i in range(0, len(remaining_catalog), _GENIE_BATCH_SIZE):
            batches.append(("catalog", remaining_catalog[i:i + _GENIE_BATCH_SIZE]))
        for i in range(0, len(passthrough), _GENIE_BATCH_SIZE):
            batches.append(("passthrough", passthrough[i:i + _GENIE_BATCH_SIZE]))

        archive("price_search.batch_plan", {
            "num_batches": len(batches),
            "num_direct_first_recovered": len(direct_first_recovered),
            "num_recipe_direct": len(recipe_direct),
            "num_skip": len(skip_unavailable),
            "max_workers": _GENIE_MAX_WORKERS,
            "plan": [
                {"mode": mode, "size": len(items),
                 "items": [it[0] if mode == "catalog" else it for it in items]}
                for mode, items in batches
            ],
        })

        def _process_batch(idx_batch: tuple) -> tuple:
            """배치 1개를 Genie에 질의. (idx, mode, result|None, input_names, err|None)."""
            idx, (mode, items) = idx_batch
            if mode == "catalog":
                query = _build_catalog_query(items)
                input_names = [t[0] for t in items]
            else:
                query = _build_passthrough_query(items)
                input_names = items

            cache_key = tuple(sorted(input_names))
            if cache_key in _batch_cache:
                cached_time, cached_result = _batch_cache[cache_key]
                if time.time() - cached_time < _CACHE_TTL_SECONDS:
                    return (idx, mode, cached_result, input_names, None)
                else:
                    del _batch_cache[cache_key]

            archive("price_search.genie_query", {
                "batch_index": idx,
                "mode": mode,
                "batch_items": input_names,
                "trace_ids": [trace_ids.get(n, n) for n in input_names],
                "query": query,
            })
            try:
                with mlflow.start_span(
                    name=f"genie_batch_{idx}_{mode}_[{','.join(input_names)[:60]}]",
                    span_type=SpanType.RETRIEVER,
                ) as span:
                    span.set_inputs({"batch_index": idx, "mode": mode,
                                      "batch_items": input_names, "question": query})
                    result = _ask_genie(query)
                    span.set_outputs({
                        "text": result.get("text"),
                        "has_sql": bool(result.get("sql")),
                        "has_dataframe": result.get("dataframe") is not None,
                    })

                if result.get("text") or result.get("dataframe") is not None:
                    if len(_batch_cache) >= _MAX_CACHE_SIZE:
                        oldest_key = next(iter(_batch_cache))
                        del _batch_cache[oldest_key]
                    _batch_cache[cache_key] = (time.time(), result)

                return (idx, mode, result, input_names, None)
            except Exception as batch_exc:
                return (idx, mode, None, input_names, str(batch_exc))

        # ── 병렬 호출 ──
        # direct_sql 1차에서 이미 회복한 결과를 출발점으로 둔다(텍스트/SQL 합류).
        all_texts: list[str] = list(direct_first_texts)
        all_sqls: list[str] = list(direct_first_sqls)
        all_tables: list[str] = []
        failed_batch_items: list[str] = []

        if batches:
            with ThreadPoolExecutor(max_workers=min(len(batches), _GENIE_MAX_WORKERS)) as ex:
                batch_results = list(ex.map(_process_batch, enumerate(batches)))
            batch_results.sort(key=lambda r: r[0])

            for idx, mode, result, batch_items, err in batch_results:
                if err:
                    archive("price_search.batch_failed", {
                        "batch_index": idx,
                        "mode": mode,
                        "batch_items": batch_items,
                        "trace_ids": [trace_ids.get(n, n) for n in batch_items],
                        "error": err,
                    })
                    failed_batch_items.extend(batch_items)
                    continue

                archive("price_search.genie_response", {
                    "batch_index": idx,
                    "mode": mode,
                    "batch_items": batch_items,
                    "trace_ids": [trace_ids.get(n, n) for n in batch_items],
                    "has_text": bool(result.get("text")),
                    "has_sql": bool(result.get("sql")),
                    "has_dataframe": result.get("dataframe") is not None,
                    "text_preview": (result.get("text") or "")[:300],
                })

                if result.get("text"):
                    all_texts.append(result["text"])
                if result.get("sql"):
                    all_sqls.append(result["sql"])
                if result.get("dataframe") is not None:
                    all_tables.append(result["dataframe"].to_string(index=False, max_rows=15))

        # ─── 실패 배치 단건 재시도 ────────────────────────────
        retry_recovered: list[str] = []
        catalog_input_map = {t[0]: t for t in catalog_targets}
        retry_targets = [i for i in failed_batch_items if i in catalog_input_map]
        skip_retry_passthrough = [i for i in failed_batch_items if i not in catalog_input_map]
        if skip_retry_passthrough:
            archive("price_search.retry_skipped_passthrough", {
                "items": skip_retry_passthrough,
                "reason": "passthrough_goes_to_naver",
            })
        if retry_targets:

            def _retry_single(item: str) -> tuple:
                """단건 재시도(catalog 전용). (item, result|None, err|None)."""
                query = _build_catalog_query([catalog_input_map[item]])
                item_mode = "catalog"
                archive("price_search.retry_single_attempt", {
                    "item": item,
                    "mode": item_mode,
                    "trace_id": trace_ids.get(item, item),
                })
                try:
                    with mlflow.start_span(
                        name=f"genie_retry_{item_mode}_[{item}]",
                        span_type=SpanType.RETRIEVER,
                    ) as span:
                        span.set_inputs({"item": item, "mode": item_mode, "question": query})
                        r = _ask_genie(query)
                        span.set_outputs({
                            "text": r.get("text"),
                            "has_sql": bool(r.get("sql")),
                            "has_dataframe": r.get("dataframe") is not None,
                        })
                    return (item, r, None)
                except Exception as retry_exc:
                    return (item, None, str(retry_exc))

            with ThreadPoolExecutor(
                max_workers=min(len(retry_targets), _GENIE_RETRY_MAX_WORKERS)
            ) as ex:
                retry_results = list(ex.map(_retry_single, retry_targets))

            still_failed_after_retry: list[str] = []
            for item, r, err in retry_results:
                if err or not r:
                    archive("price_search.retry_single_failed", {
                        "item": item,
                        "trace_id": trace_ids.get(item, item),
                        "error": err or "no_result",
                    })
                    still_failed_after_retry.append(item)
                    continue
                archive("price_search.retry_single_success", {
                    "item": item,
                    "trace_id": trace_ids.get(item, item),
                    "has_text": bool(r.get("text")),
                    "has_sql": bool(r.get("sql")),
                    "has_dataframe": r.get("dataframe") is not None,
                    "text_preview": (r.get("text") or "")[:200],
                })
                retry_recovered.append(item)
                if r.get("text"):
                    all_texts.append(r["text"])
                if r.get("sql"):
                    all_sqls.append(r["sql"])
                if r.get("dataframe") is not None:
                    all_tables.append(r["dataframe"].to_string(index=False, max_rows=15))

            failed_batch_items = still_failed_after_retry + skip_retry_passthrough
        else:
            failed_batch_items = skip_retry_passthrough

        price_data = {"source": "genie", "data": ingredients}
        if all_texts:
            price_data["text"] = "\n\n".join(all_texts)
        if all_sqls:
            price_data["sql"] = "\n---\n".join(all_sqls)
        if all_tables:
            price_data["table"] = "\n---\n".join(all_tables)

        # ─── direct_sql 1차 구조화 가격 투입 ──────────────────
        # direct_sql이 환산한 원/kg를 structured_prices에 직접 넣어 cost_calculator가
        # 텍스트 파싱 없이 1순위로 쓰게 한다(B 전환의 핵심 — 정규식 비결정성 우회).
        if direct_first_prices:
            structured_prices = price_data.get("structured_prices", {})
            structured_prices.update(direct_first_prices)
            price_data["structured_prices"] = structured_prices

        # ─── recipe_matched 재료 B2B 가격 즉시 투입 ────────────
        # Genie를 거치지 않고 ingredient_recipe의 유통가를 structured_prices에 직접 삽입.
        # cost_calculator가 structured_prices를 1순위로 참조하므로 이 재료들은 확정 가격.
        if recipe_direct:
            structured_prices = price_data.get("structured_prices", {})
            recipe_text_lines = []
            recipe_priced_count = 0  # recipe_direct로 실제 structured에 넣은 개수(전용 카운트)
            for ing_name in recipe_direct:
                r = resolved_by_input.get(ing_name)
                if r and r.recipe_info:
                    ri_info = r.recipe_info
                    price = ri_info.get("price", 0)
                    unit = ri_info.get("unit", "")
                    product_name = ri_info.get("product_name", "")
                    unit_numeric = ri_info.get("unit_numeric")
                    unit_text = (ri_info.get("unit_text") or "").lower().strip()
                    price_per_kg = None
                    if unit_numeric and unit_numeric > 0:
                        if unit_text in ("g", "그램"):
                            price_per_kg = int(price * 1000 / unit_numeric)
                        elif unit_text in ("kg", "킬로그램"):
                            price_per_kg = int(price / unit_numeric)
                        elif unit_text in ("ml", "밀리리터"):
                            price_per_kg = int(price * 1000 / unit_numeric)
                        elif unit_text in ("l", "리터"):
                            price_per_kg = int(price / unit_numeric)
                    # price_per_kg를 환산했고(=None 아님), 아직 다른 출처(direct_sql 등)가
                    # 그 재료를 안 채웠을 때만 structured에 넣는다.
                    #  - None 주입 금지: cost_calculator가 어차피 거르고, 의미만 흐림.
                    #  - 덮어쓰기 금지: direct_sql(KAMIS 확정가)을 B2B로 날리지 않게.
                    if price_per_kg and ing_name not in structured_prices:
                        structured_prices[ing_name] = {
                            "price_per_kg": price_per_kg,
                            "confidence": "high",
                            "unit_hint": f"{unit} (B2B 유통가, 상품: {product_name})",
                            "note": "ingredient_recipe B2B 가격 직접 사용",
                        }
                        recipe_priced_count += 1
                        recipe_text_lines.append(
                            f"{ing_name}: 약 ₩{price_per_kg:,}/kg (B2B 유통가, {product_name} {unit})"
                        )
                    else:
                        # 환산 불가(개/봉 등)거나 이미 다른 출처가 채운 경우: 텍스트로만 참고 제공.
                        recipe_text_lines.append(
                            f"{ing_name}: ₩{price:,}/{unit} (B2B 유통가, {product_name})"
                        )
            if structured_prices:
                price_data["structured_prices"] = structured_prices
            if recipe_text_lines:
                existing_text = price_data.get("text", "")
                price_data["text"] = (
                    existing_text + "\n\n[ingredient_recipe B2B 유통가]\n"
                    + "\n".join(recipe_text_lines)
                )
            archive("price_search.recipe_direct_applied", {
                "items": recipe_direct,
                "num_priced": recipe_priced_count,
                "num_structured_total": len(structured_prices),
            })

        # ─── unavailable 판정 ─────────────────────────────────
        queried_input_names = [t[0] for t in catalog_targets] + passthrough
        judge_targets: list[str] = []
        for name in queried_input_names:
            judge_targets.append(name)
            r = resolved_by_input.get(name)
            if r and r.db_name and r.db_name not in judge_targets:
                judge_targets.append(r.db_name)

        genie_response_text = (price_data.get("text") or "") + "\n" + (price_data.get("table") or "")
        missing_set = set(_detect_unavailable(judge_targets, genie_response_text))

        unavailable: list[str] = []
        for name in queried_input_names:
            # direct_sql 1차에서 확정된 재료는 구조화 가격이 이미 있으므로 정규식
            # 판정을 거치지 않고 무조건 available 처리한다(B 전환의 판정 일치).
            if name in direct_first_recovered:
                continue
            r = resolved_by_input.get(name)
            in_response_by_input = name not in missing_set
            in_response_by_db = bool(r and r.db_name and r.db_name not in missing_set)
            if not (in_response_by_input or in_response_by_db):
                unavailable.append(name)

        # 실패 배치는 무조건 unavailable
        for item in failed_batch_items:
            if item not in unavailable:
                unavailable.append(item)

        # ─── 카탈로그 재질의 ──────────────────────────────────
        catalog_input_set = {t[0] for t in catalog_targets}
        recoverable = [ing for ing in unavailable if ing in catalog_input_set]
        requery_recovered: list[str] = []
        if recoverable:
            requery_targets: list[tuple[str, str, str]] = [
                (ing, resolved_by_input[ing].db_name, resolved_by_input[ing].db_unit)
                for ing in recoverable
            ]
            requery_query = _build_catalog_query(requery_targets)
            archive("price_search.catalog_requery_attempt", {
                "recoverable": recoverable,
                "trace_ids": [trace_ids.get(ing, ing) for ing in recoverable],
                "targets": [
                    {"input": ing, "db_name": db_name, "db_unit": db_unit}
                    for ing, db_name, db_unit in requery_targets
                ],
            })
            try:
                with mlflow.start_span(
                    name=f"genie_catalog_requery_[{','.join(recoverable)[:60]}]",
                    span_type=SpanType.RETRIEVER,
                ) as span:
                    span.set_inputs({"recoverable": recoverable, "question": requery_query})
                    requery_result = _ask_genie(requery_query)
                    span.set_outputs({
                        "text": requery_result.get("text"),
                        "has_sql": bool(requery_result.get("sql")),
                        "has_dataframe": requery_result.get("dataframe") is not None,
                    })

                requery_text = requery_result.get("text") or ""
                if requery_result.get("dataframe") is not None:
                    requery_text += "\n" + requery_result["dataframe"].to_string(index=False, max_rows=15)

                requery_judge_targets: list[str] = []
                for ing in recoverable:
                    requery_judge_targets.append(ing)
                    db_name = resolved_by_input[ing].db_name
                    if db_name and db_name not in requery_judge_targets:
                        requery_judge_targets.append(db_name)
                requery_missing_set = set(_detect_unavailable(requery_judge_targets, requery_text))

                found_now: list[str] = []
                for ing in recoverable:
                    r = resolved_by_input[ing]
                    in_resp_by_input = ing not in requery_missing_set
                    in_resp_by_db = bool(r.db_name and r.db_name not in requery_missing_set)
                    if in_resp_by_input or in_resp_by_db:
                        found_now.append(ing)

                archive("price_search.catalog_requery_result", {
                    "recoverable": recoverable,
                    "found_now": found_now,
                    "still_missing": [ing for ing in recoverable if ing not in found_now],
                    "text_preview": requery_text[:300],
                })

                if found_now:
                    requery_recovered = list(found_now)
                    unavailable = [ing for ing in unavailable if ing not in found_now]
                    if requery_result.get("text"):
                        existing_text = price_data.get("text", "")
                        price_data["text"] = (
                            existing_text + "\n\n[카탈로그 재질의 추가 조회]\n" + requery_result["text"]
                        )
                    if requery_result.get("sql"):
                        existing_sql = price_data.get("sql", "")
                        price_data["sql"] = (
                            existing_sql + "\n---\n" + requery_result["sql"] if existing_sql
                            else requery_result["sql"]
                        )
                    if requery_result.get("dataframe") is not None:
                        existing_table = price_data.get("table", "")
                        requery_table = requery_result["dataframe"].to_string(index=False, max_rows=15)
                        price_data["table"] = (
                            existing_table + "\n---\n" + requery_table if existing_table
                            else requery_table
                        )
            except Exception as requery_exc:
                archive("price_search.catalog_requery_error", {
                    "error": str(requery_exc),
                    "recoverable": recoverable,
                })

        # ─── direct_sql_fallback (안전망) ─────────────────────
        # B 전환(2026-06-01) 후 catalog 재료는 이미 함수 앞단에서 direct_sql 1차를
        # 거쳤다. 정상 흐름에선 여기 도달하는 matched 재료가 거의 없다(있어도 1차에서
        # 못 잡은 것이라 재시도 의미 적음). 다만 1차 direct_sql이 예외로 통째 실패한
        # 경우(direct_first_error)엔 catalog가 Genie로 갔다가 여기로 떨어지므로,
        # 그때의 마지막 재시도 안전망으로 남겨둔다.
        direct_sql_recovered: list[str] = []
        unrecoverable_matched = [ing for ing in unavailable if ing in catalog_input_set]
        if unrecoverable_matched:
            direct_targets: list[tuple[str, str, str]] = [
                (ing, resolved_by_input[ing].db_name, resolved_by_input[ing].db_unit)
                for ing in unrecoverable_matched
            ]
            archive("price_search.direct_sql_attempt", {
                "items": unrecoverable_matched,
                "trace_ids": [trace_ids.get(ing, ing) for ing in unrecoverable_matched],
                "targets": [
                    {"input": ing, "db_name": db_name, "db_unit": db_unit}
                    for ing, db_name, db_unit in direct_targets
                ],
            })
            try:
                with mlflow.start_span(
                    name=f"direct_sql_[{','.join(unrecoverable_matched)[:60]}]",
                    span_type=SpanType.RETRIEVER,
                ) as span:
                    span.set_inputs({"items": unrecoverable_matched, "targets": direct_targets})
                    direct_result = _direct_sql_query(direct_targets)
                    span.set_outputs({
                        "found": direct_result.get("found"),
                        "error": direct_result.get("error"),
                        "text_preview": (direct_result.get("text") or "")[:200],
                    })

                direct_found = direct_result.get("found") or []
                archive("price_search.direct_sql_result", {
                    "items": unrecoverable_matched,
                    "found": direct_found,
                    "still_missing": [ing for ing in unrecoverable_matched if ing not in direct_found],
                    "error": direct_result.get("error"),
                    "text_preview": (direct_result.get("text") or "")[:300],
                })

                if direct_found:
                    direct_sql_recovered = list(direct_found)
                    unavailable = [ing for ing in unavailable if ing not in direct_found]
                    if direct_result.get("text"):
                        existing_text = price_data.get("text", "")
                        price_data["text"] = (
                            existing_text + "\n\n[KAMIS direct_sql 폴백 조회]\n"
                            + direct_result["text"]
                        )
                    if direct_result.get("sql"):
                        existing_sql = price_data.get("sql", "")
                        price_data["sql"] = (
                            existing_sql + "\n---\n" + direct_result["sql"] if existing_sql
                            else direct_result["sql"]
                        )
            except Exception as direct_exc:
                archive("price_search.direct_sql_error", {
                    "error": str(direct_exc),
                    "items": unrecoverable_matched,
                })

        # ─── recipe B2B 최종 폴백 ─────────────────────────────
        # direct_sql에서도 회복 안 된 항목 + passthrough 실패분에 대해
        # ingredient_recipe에서 B2B 유통가를 마지막으로 시도.
        # 이렇게 하면 Naver로 넘어가는 재료를 최소화.
        recipe_fallback_recovered: list[str] = []
        recipe_fallback_candidates = [
            ing for ing in unavailable if ing not in skip_unavailable
        ]
        if recipe_fallback_candidates:
            recipe_fallback_result = get_recipe_prices_for_items(recipe_fallback_candidates)
            if recipe_fallback_result:
                recipe_fallback_recovered = list(recipe_fallback_result.keys())
                unavailable = [ing for ing in unavailable if ing not in recipe_fallback_recovered]
                # structured_prices에 추가
                structured_prices = price_data.get("structured_prices", {})
                recipe_fb_lines = []
                for ing_name, info in recipe_fallback_result.items():
                    ppk = info.get("price_per_kg")
                    structured_prices[ing_name] = {
                        "price_per_kg": ppk,
                        "confidence": "medium",
                        "unit_hint": f"{info.get('unit', '')} (B2B 유통가 폴백, {info.get('product_name', '')})",
                        "note": "Genie/direct_sql 실패 후 recipe B2B 폴백",
                    }
                    if ppk:
                        recipe_fb_lines.append(
                            f"{ing_name}: 약 ₩{ppk:,}/kg (B2B 유통가 폴백, {info.get('product_name', '')} {info.get('unit', '')})"
                        )
                    else:
                        recipe_fb_lines.append(
                            f"{ing_name}: ₩{info.get('price', 0):,}/{info.get('unit', '')} (B2B 유통가 폴백)"
                        )
                if structured_prices:
                    price_data["structured_prices"] = structured_prices
                if recipe_fb_lines:
                    existing_text = price_data.get("text", "")
                    price_data["text"] = (
                        existing_text + "\n\n[recipe B2B 유통가 폴백]\n"
                        + "\n".join(recipe_fb_lines)
                    )
                archive("price_search.recipe_fallback_applied", {
                    "candidates": recipe_fallback_candidates,
                    "recovered": recipe_fallback_recovered,
                    "still_unavailable": [ing for ing in recipe_fallback_candidates if ing not in recipe_fallback_recovered],
                })

        # skip 그룹 합류 (Genie를 거치지 않은 ambiguous/not_in_catalog)
        for item in skip_unavailable:
            if item not in unavailable:
                unavailable.append(item)

        if unavailable:
            price_data["unavailable"] = unavailable
            price_data["unavailable_resolved"] = {
                ing: {
                    "status": resolved_by_input[ing].status,
                    "db_name": resolved_by_input[ing].db_name,
                    "db_unit": resolved_by_input[ing].db_unit,
                }
                for ing in unavailable if ing in resolved_by_input
            }

        # missing_price_search가 같은 trace_id를 재사용하도록 전달
        price_data["trace_ids"] = trace_ids

        archive("price_search.output", {
            "requested": ingredients,
            "unavailable": unavailable,
            "num_batches_used": len(batches),
            "num_recipe_direct": len(recipe_direct),
            "num_skipped_to_unavailable": len(skip_unavailable),
            "num_failed_batches": len(failed_batch_items),
            "num_retry_recovered": len(retry_recovered),
            "retry_recovered": retry_recovered,
            "num_requery_recovered": len(requery_recovered),
            "requery_recovered": requery_recovered,
            "num_direct_sql_recovered": len(direct_sql_recovered),
            "direct_sql_recovered": direct_sql_recovered,
            "num_recipe_fallback_recovered": len(recipe_fallback_recovered),
            "recipe_fallback_recovered": recipe_fallback_recovered,
            "has_text": bool(price_data.get("text")),
            "has_table": bool(price_data.get("table")),
            "text_preview": (price_data.get("text") or "")[:300],
            "unavailable_by_status": {
                status: [ing for ing in unavailable
                         if ing in resolved_by_input and resolved_by_input[ing].status == status]
                for status in ("matched", "recipe_matched", "ambiguous", "not_in_catalog", "unmapped")
            },
        })
        return {"price_info": price_data}

    except Exception as e:
        archive("price_search.error", {"error": str(e), "ingredients": ingredients})
        return {"price_info": {"source": "error", "data": [], "error": str(e)}, "error_log": [f"price_search: {str(e)}"]}
