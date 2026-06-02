import streamlit as st
import backend.databricks_db as _db
from backend.databricks_db import get_ingredient_prices_live

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


def render():
    st.markdown('<div class="hello">안녕하세요, 사장님!</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hello-sub">오늘도 바바삭이 스마트한 메뉴 결정을 도와드릴게요.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="hero">
        <div>
            <div class="hero-title">
                실시간 재료 시세를 분석해<br>
                <span>최적의 메뉴를 추천</span>해드려요!
            </div>
            <div class="hero-desc">
                재료 가격의 추이를 바탕으로 마진을 높이는<br>
                스마트한 메뉴 관리의 시작
            </div>
        </div>
        <div class="hero-visual">🥬🥔🧅🥩</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">주요 기능 바로가기</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
        <div class="feature-card">
            <div class="feature-icon icon-green">💬</div>
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

    st.markdown(
        """
    <div class="price-wrap">
        <div class="price-title">오늘의 식재료 시세 요약</div>
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

    st.markdown('<div class="section-title" style="margin-top:40px;">프로젝트 소개</div>', unsafe_allow_html=True)
    st.markdown(
        """
    <div class="about-card">
        <div class="about-badge">✦ 바바삭</div>
        <div class="about-title">
            소상공인을 위한<br>
            <span>물가 연동형 메뉴 추천 AI</span>
        </div>
        <div class="about-desc">
            식재료 가격 변동 정보와 레시피 데이터를 결합하여<br>
            수익성 높은 메뉴와 대체 재료를 추천합니다.
        </div>
        <div class="about-features">
            <div class="about-feature-item">
                <div class="about-feature-icon">📈</div>
                <div class="about-feature-text">실시간 시세 분석</div>
            </div>
            <div class="about-feature-item">
                <div class="about-feature-icon">🍽️</div>
                <div class="about-feature-text">메뉴 최적화 추천</div>
            </div>
            <div class="about-feature-item">
                <div class="about-feature-icon">🤖</div>
                <div class="about-feature-text">AI 챗봇 상담</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
