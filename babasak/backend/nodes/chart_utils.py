from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from backend.debug_log import archive


ENABLE_CHART: bool = True


# ── plotly는 선택적 의존성 (없으면 차트 비활성화) ──
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


# ── 날짜/시간 컬럼 감지 패턴 ──
_DATE_COL_PATTERNS = re.compile(
    r"(날짜|date|일자|일시|기간|month|year|week|시간|timestamp)",
    re.IGNORECASE,
)

# ── 숫자(가격) 컬럼 감지 패턴 ──
_PRICE_COL_PATTERNS = re.compile(
    r"(가격|price|평균|금액|원|단가|도매|소매|avg|mean|sum|total|cost)",
    re.IGNORECASE,
)

# ── 카테고리(그룹) 컬럼 감지 패턴 ──
_CATEGORY_COL_PATTERNS = re.compile(
    r"(재료|품목|이름|name|종류|분류|카테고리|category|item|ingredient|단위)",
    re.IGNORECASE,
)


def _detect_date_col(df: pd.DataFrame) -> Optional[str]:
    """DataFrame에서 날짜/시간 컬럼을 자동 감지"""
    # 1) 컬럼명 패턴 매칭
    for col in df.columns:
        if _DATE_COL_PATTERNS.search(col):
            return col
    # 2) dtype이 datetime인 컬럼
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    # 3) 문자열인데 날짜 형식(YYYY-MM-DD)인 컬럼
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(5)
            if len(sample) > 0 and sample.str.match(r"^\d{4}[-/]\d{2}[-/]\d{2}").all():
                return col
    return None


def _detect_price_col(df: pd.DataFrame) -> Optional[str]:
    """DataFrame에서 가격/숫자 컬럼을 자동 감지."""
    # 1) 컬럼명 패턴 + 숫자형
    for col in df.columns:
        if _PRICE_COL_PATTERNS.search(col):
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
            # 문자열이지만 숫자로 변환 가능한 경우
            try:
                pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="raise")
                return col
            except (ValueError, TypeError, AttributeError):
                pass
    # 2) 패턴에 안 맞아도 숫자 컬럼이 있으면 첫 번째 반환 (count 계열 제외)
    skip_cols = {"행수", "count", "cnt", "건수", "횟수"}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col.lower() not in skip_cols:
            return col
    return None


def _detect_category_col(df: pd.DataFrame) -> Optional[str]:
    """DataFrame에서 카테고리(재료명 등) 컬럼을 자동 감지."""
    for col in df.columns:
        if _CATEGORY_COL_PATTERNS.search(col):
            return col
    return None


def _ensure_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """컬럼을 숫자형으로 변환 (쉼표 제거 등)."""
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ""), errors="coerce"
        )
    return df


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """컬럼을 datetime형으로 변환."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


_PLOTLY_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToInclude": [
        "zoom2d",          # 드래그로 영역 확대
        "zoomIn2d",        # + 버튼 확대
        "zoomOut2d",       # - 버튼 축소
        "pan2d",           # 드래그로 팬 (이동)
        "resetScale2d",    # 원래 범위로 초기화 (더블클릭과 동일)
        "autoScale2d",     # 오토스케일
    ],
    "modeBarButtonsToRemove": [
        "select2d", "lasso2d", "toImage",
    ],
    "displaylogo": False,
    "scrollZoom": True,     # 마우스 휠로 확대/축소
}


def generate_chart_html(
    df: pd.DataFrame,
    user_query: str = "",
) -> Optional[str]:
    """DataFrame을 분석하여 Genie Space 스타일의 차트 HTML을 생성.

    ★ ENABLE_CHART = False이면 항상 None을 반환합니다.

    차트 스타일:
      - Genie Space와 동일: 흰 배경, 파란 라인, 마커 없음
      - 확대/축소/팬/초기화 모두 지원 (툴바 + 마우스 휠)
      - X축: 날짜, Y축: 가격(원)

    Args:
        df: Genie API에서 반환된 DataFrame
        user_query: 사용자 원본 질문 (차트 제목에 활용)

    Returns:
        차트 HTML 문자열 (plotly, CDN 포함) 또는 None
    """
    # ═══ 차트 기능 비활성화 체크 ═══
    if not ENABLE_CHART:
        return None

    if not _HAS_PLOTLY:
        archive("chart_utils.plotly_not_installed", {
            "note": "pip install plotly 필요. 차트 기능 비활성화됨.",
        })
        return None

    if df is None or df.empty or len(df) < 2:
        return None

    try:
        date_col = _detect_date_col(df)
        price_col = _detect_price_col(df)
        category_col = _detect_category_col(df)

        # ── Case 1: 시계열 라인 차트 (날짜 + 가격) ── Genie 스타일
        if date_col and price_col:
            df = _ensure_datetime(df, date_col)
            df = _ensure_numeric(df, price_col)
            df = df.dropna(subset=[date_col, price_col])
            df = df.sort_values(date_col)

            if df.empty:
                return None

            fig = go.Figure()

            # 여러 재료가 있으면 각각 별도 trace
            if category_col and df[category_col].nunique() > 1:
                for name, group in df.groupby(category_col):
                    fig.add_trace(go.Scatter(
                        x=group[date_col],
                        y=group[price_col],
                        mode="lines",
                        name=str(name),
                        line=dict(width=2),
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=df[date_col],
                    y=df[price_col],
                    mode="lines",
                    name=user_query.split()[0] if user_query else "가격",
                    line=dict(color="#1f77b4", width=2),  # Genie 기본 파란색
                ))

            # Genie Space 동일 레이아웃: 흰 배경, 간결한 그리드
            fig.update_layout(
                title=dict(
                    text=user_query or "가격 추이",
                    font=dict(size=14, color="#333333"),
                    x=0.01, xanchor="left",
                ),
                xaxis=dict(
                    title="날짜",
                    showgrid=True,
                    gridcolor="#f0f0f0",
                    linecolor="#cccccc",
                    tickformat="%b %d, %Y",
                ),
                yaxis=dict(
                    title="평균 가격 (원)",
                    showgrid=True,
                    gridcolor="#f0f0f0",
                    linecolor="#cccccc",
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                hovermode="x unified",
                height=400,
                margin=dict(l=60, r=30, t=40, b=60),
                showlegend=(category_col is not None and df[category_col].nunique() > 1) if category_col else False,
            )

            chart_html = fig.to_html(
                include_plotlyjs="cdn",
                full_html=False,
                config=_PLOTLY_CONFIG,
            )
            archive("chart_utils.line_chart_generated", {
                "date_col": date_col,
                "price_col": price_col,
                "category_col": category_col,
                "num_rows": len(df),
                "query": user_query[:100],
            })
            return chart_html

        # ── Case 2: 막대 차트 (카테고리 + 가격, 날짜 없음) ──
        if category_col and price_col:
            df = _ensure_numeric(df, price_col)
            df = df.dropna(subset=[price_col])

            if df.empty:
                return None

            if len(df) > 15:
                df = df.nlargest(15, price_col)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df[category_col],
                y=df[price_col],
                marker_color="#1f77b4",
                text=df[price_col].apply(lambda v: f"{int(v):,}"),
                textposition="outside",
            ))
            fig.update_layout(
                title=dict(
                    text=user_query or "가격 비교",
                    font=dict(size=14, color="#333333"),
                    x=0.01, xanchor="left",
                ),
                xaxis=dict(
                    title="재료",
                    tickangle=-45,
                    linecolor="#cccccc",
                ),
                yaxis=dict(
                    title="가격 (원)",
                    showgrid=True,
                    gridcolor="#f0f0f0",
                    linecolor="#cccccc",
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=400,
                margin=dict(l=60, r=30, t=40, b=80),
                showlegend=False,
            )

            chart_html = fig.to_html(
                include_plotlyjs="cdn",
                full_html=False,
                config=_PLOTLY_CONFIG,
            )
            archive("chart_utils.bar_chart_generated", {
                "category_col": category_col,
                "price_col": price_col,
                "num_items": len(df),
                "query": user_query[:100],
            })
            return chart_html

        # ── Case 3: 차트 생성 불가 ──
        archive("chart_utils.no_suitable_columns", {
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "query": user_query[:100],
        })
        return None

    except Exception as e:
        archive("chart_utils.generation_error", {"error": str(e), "query": user_query[:100]})
        return None
