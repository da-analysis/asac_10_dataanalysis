import base64
from pathlib import Path

import streamlit as st
from views import home, chatbot, dashboard


def _img_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


_ASSETS = Path(__file__).parent / "assets"


st.set_page_config(page_title="바바삭", page_icon="🍽️", layout="wide", initial_sidebar_state="collapsed")

# databricks workspace import-dir app /Workspace/Users/jyj000818@gmail.com/databricks_apps_jyjtest --profile DEFAULT --overwrite
# databricks apps deploy jyjtest --source-code-path /Workspace/Users/jyj000818@gmail.com/databricks_apps_jyjtest --profile DEFAULT


if st.query_params.get("goto"):
    st.session_state.page = st.query_params["goto"]
    st.query_params.clear()

if "page" not in st.session_state:
    st.session_state.page = "home"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_industry" not in st.session_state:
    st.session_state.user_industry = ""
if "user_region" not in st.session_state:
    st.session_state.user_region = ""
if "profile_saved" not in st.session_state:
    st.session_state.profile_saved = False


def go_page(page_name: str):
    st.session_state.page = page_name
    st.rerun()


st.markdown(
    """
<style>
[data-testid="stSidebarNav"] {
    display: none;
}

[data-testid="collapsedControl"] button svg,
[data-testid="stSidebarCollapseButton"] svg {
    display: none !important;
}

[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"] {
    position: relative !important;
    overflow: visible !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

[data-testid="collapsedControl"] button::before,
[data-testid="stSidebarCollapseButton"]::before {
    content: '' !important;
    display: block !important;
    width: 18px !important;
    height: 2px !important;
    background-color: #374151 !important;
    box-shadow: 0 6px 0 #374151, 0 12px 0 #374151 !important;
    margin-top: -6px !important;
    flex-shrink: 0 !important;
}

.stApp {
    background: #f8fafc;
}

.block-container {
    padding-top: 4rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 4rem;
}

.stButton > button {
    width: 100%;
    height: auto;
    min-height: 48px;
    padding: 10px 16px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    background: #ffffff;
    color: #111827;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #ef4444;
    color: #ef4444;
    background: #fff5f5;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ef4444, #f97316);
    color: white;
    border: none;
}

.logo-box {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}

.logo-icon {
    font-size: 120px;
    width: 100%;
    max-width: 200px;
    height: auto;
    object-fit: contain;
    vertical-align: middle;
}

.logo-text {
    font-size: 28px;
    font-weight: 900;
    color: #ef4444;
}

.logo-sub {
    font-size: 13px;
    color: #64748b;
    margin-left: 42px;
    margin-bottom: 28px;
}

.hello {
    font-size: 34px;
    font-weight: 900;
    color: #0f172a;
    margin-bottom: 8px;
}

.hello-sub {
    font-size: 17px;
    color: #64748b;
    margin-bottom: 24px;
}

.hero {
    background: linear-gradient(120deg, #fff7ed 0%, #fff5f5 45%, #eef2ff 100%);
    border: 1px solid #fecaca;
    border-radius: 22px;
    padding: 42px 48px;
    margin-bottom: 34px;
    min-height: 230px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
}

.hero-title {
    font-size: 32px;
    font-weight: 900;
    color: #0f172a;
    line-height: 1.35;
    margin-bottom: 20px;
}

.hero-title span {
    color: #ef4444;
}

.hero-desc {
    font-size: 17px;
    color: #334155;
    line-height: 1.75;
}

.hero-visual {
    font-size: 92px;
    text-align: right;
    opacity: 0.95;
}

.section-title {
    font-size: 24px;
    font-weight: 900;
    color: #0f172a;
    margin-bottom: 16px;
}

.feature-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 28px 24px;
    text-align: center;
    min-height: 230px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    margin-bottom: 10px;
}

.feature-icon {
    width: 72px;
    height: 72px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 34px;
    margin-bottom: 18px;
}

.icon-green {
    background: #dcfce7;
}

.icon-blue {
    background: #dbeafe;
}

.price-wrap {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 22px;
    margin-top: 18px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
}

.price-title {
    font-size: 18px;
    font-weight: 900;
    color: #111827;
    margin-bottom: 16px;
}

.price-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 18px;
    text-align: center;
}

.price-icon {
    font-size: 34px;
    margin-bottom: 8px;
}

.price-name {
    font-size: 15px;
    color: #334155;
    margin-bottom: 6px;
}

.price-value {
    font-size: 20px;
    font-weight: 900;
    color: #111827;
}

.price-up {
    font-size: 13px;
    color: #ef4444;
    font-weight: 700;
}

.price-down {
    font-size: 13px;
    color: #2563eb;
    font-weight: 700;
}

.price-neutral {
    font-size: 13px;
    color: #94a3b8;
    font-weight: 700;
}

.profile-prompt {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 16px;
    padding: 18px 24px;
    margin-top: 20px;
    margin-bottom: 4px;
}

.profile-prompt-title {
    font-size: 15px;
    font-weight: 700;
    color: #92400e;
    margin-bottom: 4px;
}

.profile-prompt-desc {
    font-size: 13px;
    color: #78350f;
}

.profile-saved {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 12px;
    padding: 12px 18px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 20px;
    margin-bottom: 4px;
}

.profile-saved-text {
    font-size: 14px;
    color: #166534;
    font-weight: 700;
}

#fab-chatbot, #fab-chatbot:visited {
    text-decoration: none;
}

#fab-chatbot {
    position: fixed;
    bottom: 32px;
    right: 32px;
    width: 62px;
    height: 62px;
    border-radius: 50%;
    background: linear-gradient(135deg, #ef4444, #f97316);
    color: white;
    font-size: 28px;
    border: none;
    box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
    z-index: 9999;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

#fab-chatbot:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 28px rgba(239, 68, 68, 0.5);
}

.about-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 22px;
    padding: 44px 52px;
    margin-top: 8px;
    box-shadow: 0 8px 24px rgba(15,23,42,0.04);
}

.about-badge {
    display: inline-flex;
    align-items: center;
    background: #fff5f5;
    border: 1px solid #fecaca;
    color: #ef4444;
    font-size: 12px;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 18px;
    letter-spacing: 0.6px;
}

.about-title {
    font-size: 26px;
    font-weight: 900;
    color: #0f172a;
    line-height: 1.45;
    margin-bottom: 14px;
}

.about-title span {
    color: #ef4444;
}

.about-desc {
    font-size: 15px;
    color: #64748b;
    line-height: 1.85;
    margin-bottom: 32px;
}

.about-features {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}

.about-feature-item {
    display: flex;
    align-items: center;
    gap: 9px;
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 11px 18px;
    transition: background 0.2s;
}

.about-feature-item:hover {
    background: #f1f5f9;
}

.about-feature-icon {
    font-size: 18px;
}

.about-feature-text {
    font-size: 13px;
    color: #334155;
    font-weight: 600;
    white-space: nowrap;
}
</style>
<a id="fab-chatbot" href="?goto=chatbot" title="챗봇 바로가기" target="_self">💬</a>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    _logo_path = _ASSETS / "logo.png"
    _logo_icon = (
        f'<img class="logo-icon" src="data:image/png;base64,{_img_b64(str(_logo_path))}" />'
        if _logo_path.exists()
        else '<div class="logo-icon">🍽️</div>'
    )
    st.markdown(
        f"""
    <a href="?goto=home" target="_self" style="text-decoration:none;">
        <div class="logo-box">
            {_logo_icon}
        </div>
    </a>
    <div class="logo-sub">소상공인을 지원하는 비서</div>
    """,
        unsafe_allow_html=True,
    )

    if st.button("🏠 홈"):
        go_page("home")
    if st.button("💬 챗봇에게 물어보기"):
        go_page("chatbot")
    if st.button("📊 가격 추이 알아보기"):
        go_page("dashboard")


def margin_page():
    st.title("💰 원가 · 마진 계산")

    col1, col2, col3 = st.columns(3)
    with col1:
        cost = st.number_input("1인분 원가", min_value=0, value=2600, step=100)
    with col2:
        price = st.number_input("판매가", min_value=0, value=8000, step=500)
    with col3:
        servings = st.number_input("예상 판매 수량", min_value=1, value=30, step=1)

    if price > 0:
        margin_rate = (price - cost) / price * 100
        profit = (price - cost) * servings
        r1, r2 = st.columns(2)
        r1.metric("예상 마진율", f"{margin_rate:.1f}%")
        r2.metric("예상 총이익", f"₩{profit:,.0f}")


if st.session_state.page == "home":
    home.render()
elif st.session_state.page == "chatbot":
    chatbot.render()
elif st.session_state.page == "dashboard":
    dashboard.render()
elif st.session_state.page == "margin":
    margin_page()
