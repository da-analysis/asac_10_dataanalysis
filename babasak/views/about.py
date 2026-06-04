import streamlit as st


def render():
    """프로젝트 소개 페이지. (기존 홈 하단의 about-card 블록을 분리)"""
    st.markdown('<div class="section-title">프로젝트 소개</div>', unsafe_allow_html=True)
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
