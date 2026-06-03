import base64
from pathlib import Path

import streamlit as st
import backend.databricks_db as _db
from backend.databricks_db import get_ingredient_prices_live

# 'AI 챗봇' 기능카드 아이콘으로 쓸 새 로고. logo_new.png 우선, 없으면 logo.png, 둘 다 없으면 💬.
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def _chat_icon_html() -> str:
    for fname in ("logo_new.png", "logo.png"):
        p = _ASSETS_DIR / fname
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            return (f'<img src="data:image/png;base64,{b64}" alt="챗봇" '
                    f'style="width:60%;height:60%;object-fit:contain;" />')
    return "💬"


# hero 배너 우측 재료 비주얼. assets/hero.png 한 장이 있으면 이미지로, 없으면 이모지 폴백.
def _hero_visual_html() -> str:
    p = _ASSETS_DIR / "hero.png"
    if p.exists():
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'<img class="hero-ing" src="data:image/png;base64,{b64}" alt="재료" />'
    return '<span class="hero-ing-emoji">🥬🥔🧅🥩</span>'

FALLBACK_INGREDIENTS = [
    # 상승 6
    {"name": "풋고추 (4kg)", "icon": "🌶️", "price": 19800, "change": 8.2, "trend": "up"},
    {"name": "깐마늘 (20kg)", "icon": "🧄", "price": 85000, "change": 5.4, "trend": "up"},
    {"name": "상추 (4kg)", "icon": "🥬", "price": 24000, "change": 4.1, "trend": "up"},
    {"name": "오이 (10kg)", "icon": "🥒", "price": 28000, "change": 3.3, "trend": "up"},
    {"name": "시금치 (4kg)", "icon": "🥬", "price": 16000, "change": 2.6, "trend": "up"},
    {"name": "토마토 (10kg)", "icon": "🍅", "price": 32000, "change": 1.5, "trend": "up"},
    # 하락 6
    {"name": "감자 (20kg)", "icon": "🥔", "price": 26000, "change": -14.3, "trend": "down"},
    {"name": "양파 (15kg)", "icon": "🧅", "price": 17200, "change": -4.2, "trend": "down"},
    {"name": "당근 (10kg)", "icon": "🥕", "price": 14400, "change": -3.8, "trend": "down"},
    {"name": "대파 (1kg)", "icon": "🥬", "price": 2000, "change": -2.9, "trend": "down"},
    {"name": "배추 (10kg)", "icon": "🥬", "price": 9800, "change": -2.1, "trend": "down"},
    {"name": "계란 (30개)", "icon": "🥚", "price": 7000, "change": -1.4, "trend": "down"},
]


# 전 품목 스캔 쿼리라 무거우므로 1시간 캐싱 (시세 DB는 하루 단위 갱신).
@st.cache_data(ttl=3600, show_spinner=False)
def _load_ingredients() -> tuple[list[dict], str | None]:
    live = get_ingredient_prices_live()
    if live:
        return live, None
    return FALLBACK_INGREDIENTS, _db.last_error or "알 수 없는 오류"


def _render_price_cards(items: list[dict]) -> None:
    """시세 카드를 6열씩 줄바꿈하여 렌더링."""
    if not items:
        return
    n_cols = min(6, max(1, len(items)))
    for row_start in range(0, len(items), n_cols):
        cols = st.columns(n_cols)
        for col, ing in zip(cols, items[row_start : row_start + n_cols]):
            change = ing.get("change", 0) or 0
            if change > 0:
                direction, cls = "▲", "price-up"
            elif change < 0:
                direction, cls = "▼", "price-down"
            else:
                direction, cls = "─", "price-neutral"
            date = ing.get("date", "")
            date_html = (
                f'<div style="font-size:12px; color:#94a3b8; margin-top:4px;">{date} 기준</div>' if date else ""
            )
            with col:
                st.markdown(
                    f"""
                <div class="price-card">
                    <div class="price-icon">{ing.get("icon", "")}</div>
                    <div class="price-name">{ing.get("name", "")}</div>
                    <div class="price-value">₩{ing.get("price", 0):,}</div>
                    <div class="{cls}">{direction} {abs(change)}%</div>
                    {date_html}
                </div>
                """,
                    unsafe_allow_html=True,
                )


def _render_feature_shortcuts() -> None:
    """주요 기능 바로가기 섹션 (챗봇 / 가격 추이 카드 + 버튼)."""
    st.markdown('<div class="section-title">주요 기능 바로가기</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
        <div class="feature-card">
            <div class="feature-icon icon-green">{_chat_icon_html()}</div>
            <div class="feature-title">AI 챗봇</div>
            <div class="feature-desc">메뉴 고민, 재료 궁금증을 AI에게 바로 물어보세요.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("챗봇 열기", key="home_chatbot"):
            st.session_state.page = "chatbot"
            st.rerun()

    with c2:
        st.markdown(
            """
        <div class="feature-card">
            <div class="feature-icon icon-blue">📊</div>
            <div class="feature-title">가격 추이 알아보기</div>
            <div class="feature-desc">주요 식재료 가격 변화를 한눈에 확인하세요.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("가격 추이 보기", key="home_dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()


# 추천 메뉴 TOP 3 (하드코딩). 예상 원가는 표기하지 않는다.
RECOMMENDED_MENUS = [
    {"rank": 1, "name": "제육볶음"},
    {"rank": 2, "name": "김치찌개"},
    {"rank": 3, "name": "우거지 감자탕"},
]


# '전반적 물가 지표: 연-반기별 평균가격 추이' 위젯 하나만 임베드.
# fullscreenWidget=<page>~<widget> 으로 특정 차트만 전체화면 표시한다.
# (대시보드 전체가 아니라 위젯 단위. 위젯 ID는 published 링크의 fullscreenWidget 값)
_DASHBOARD_EMBED_URL = (
    "https://dbc-b2983598-29d8.cloud.databricks.com/embed/dashboardsv3/"
    "01f1477ee2361f28a2da2e770290bfd7"
    "?o=7474645180304154"
    "&fullscreenWidget=c4ad9d27~53f16a93"
)


def _render_price_trend_chart() -> None:
    """전반적 물가 지표(연-반기별 평균가격 추이) — Databricks 대시보드 임베드."""
    import streamlit.components.v1 as components

    st.markdown('<div class="section-title">전반적 물가 지표</div>', unsafe_allow_html=True)
    st.caption("연-반기별 평균가격 추이")
    components.iframe(_DASHBOARD_EMBED_URL, height=300, scrolling=True)
    st.caption("정확한 가격과 추이를 확인하려면 메뉴에서 '가격 추이 알아보기'를 눌러주세요!")


def _render_recommended_menus() -> None:
    """바바삭 추천 메뉴 TOP 3 (메뉴명만, 원가 미표기)."""
    st.markdown('<div class="section-title">바바삭 추천 메뉴 TOP 3</div>', unsafe_allow_html=True)
    for m in RECOMMENDED_MENUS:
        st.markdown(
            f"""
        <div class="menu-card">
            <span class="menu-rank">{m['rank']}</span>
            <span class="menu-name">{m['name']}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )


def _render_price_summary() -> None:
    """최근 식재료 시세 요약 섹션 (오르는 중 / 내리는 중 카드)."""
    st.markdown(
        """
    <div class="price-wrap">
        <div class="price-title">최근 식재료 시세 요약</div>
    """,
        unsafe_allow_html=True,
    )

    ingredients, error = _load_ingredients()
    if error:
        st.warning(f"실시간 시세 데이터를 가져오지 못했습니다. ({error})")

    if ingredients:
        ups = [i for i in ingredients if i.get("trend") == "up"]
        downs = [i for i in ingredients if i.get("trend") == "down"]
        # trend 키가 없는 옛 데이터 호환: change 부호로 분류
        if not ups and not downs:
            ups = [i for i in ingredients if (i.get("change") or 0) > 0]
            downs = [i for i in ingredients if (i.get("change") or 0) < 0]

        if ups:
            st.markdown(
                '<div style="font-weight:800;color:#dc2626;margin:6px 0 10px;">📈 오르는 중</div>',
                unsafe_allow_html=True,
            )
            _render_price_cards(ups)
        if downs:
            st.markdown(
                '<div style="font-weight:800;color:#2563eb;margin:18px 0 10px;">📉 내리는 중</div>',
                unsafe_allow_html=True,
            )
            _render_price_cards(downs)

    st.markdown("</div>", unsafe_allow_html=True)


def render():
    st.markdown('<div class="hello">안녕하세요, 사장님!</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hello-sub">오늘도 바바삭이 스마트한 메뉴 결정을 도와드릴게요.</div>',
        unsafe_allow_html=True,
    )

    # 하나의 hero 박스 안에 [텍스트 | 시세 동향 차트 | 추천 메뉴]를 가로로 배치 (목업 레이아웃).
    # st.line_chart 위젯은 HTML 문자열에 못 넣으므로, 컬럼들을 st.container(border=True)로
    # 감싸 Streamlit이 만드는 테두리 래퍼(stVerticalBlockBorderWrapper)에 hero 박스 배경을
    # CSS로 입혀 '한 박스'처럼 보이게 한다. (:has() 방식보다 DOM 구조에 안정적)
    with st.container(border=True):
        st.markdown('<span class="hero-marker"></span>', unsafe_allow_html=True)
        c_text, c_chart, c_menu = st.columns([1.1, 2, 1], gap="medium")
        with c_text:
            st.markdown(
                """
            <div class="hero-text">
                <div class="hero-title">
                    실시간 재료 시세를 분석해<br>
                    <span>최적의 메뉴를 추천</span>해드려요!
                </div>
                <div class="hero-desc">
                    재료 가격의 추이를 바탕으로 마진을 높이는<br>
                    스마트한 메뉴 관리의 시작
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with c_chart:
            _render_price_trend_chart()
        with c_menu:
            _render_recommended_menus()

    _render_price_summary()
    _render_feature_shortcuts()
    # 프로젝트 소개는 사이드바의 'ℹ️ 프로젝트 소개' 메뉴(views/about.py)로 분리됨
