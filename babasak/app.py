import base64
from pathlib import Path

import streamlit as st
from views import home, chatbot, dashboard, about


def _img_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


_ASSETS = Path(__file__).parent / "assets"

# 챗봇 말풍선 대신 쓸 새 로고. logo_new.png 우선, 없으면 logo.png 폴백.
# data URI(없으면 None)를 만들어 플로팅 버튼·홈 카드에서 재사용한다.
_NEW_LOGO_PATH = _ASSETS / "logo_new.png"
if not _NEW_LOGO_PATH.exists():
    _NEW_LOGO_PATH = _ASSETS / "logo.png"
_NEW_LOGO_URI = (
    f"data:image/png;base64,{_img_b64(str(_NEW_LOGO_PATH))}"
    if _NEW_LOGO_PATH.exists()
    else None
)


st.set_page_config(page_title="바바삭", page_icon="🍽️", layout="wide", initial_sidebar_state="expanded")

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

/* 사이드바 토글 버튼(접기/펴기).
   Streamlit 기본 화살표 아이콘(svg)을 그대로 살리고, 버튼 박스만 깔끔히 정돈한다.
   버튼 자체의 testid는 stBaseButton-headerNoPadding 이며, 펼침/접힘 상태는
   각각 stSidebarCollapseButton / collapsedControl 컨테이너 안에 들어 있다. */
[data-testid="stSidebarCollapseButton"] [data-testid="stBaseButton-headerNoPadding"],
[data-testid="collapsedControl"] [data-testid="stBaseButton-headerNoPadding"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    padding: 0 !important;
    border-radius: 10px !important;
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08) !important;
    color: #374151 !important;
    cursor: pointer !important;
    transition: background 0.15s ease, border-color 0.15s ease !important;
}

[data-testid="stSidebarCollapseButton"] [data-testid="stBaseButton-headerNoPadding"]:hover,
[data-testid="collapsedControl"] [data-testid="stBaseButton-headerNoPadding"]:hover {
    background: #fff5f5 !important;
    border-color: #ef4444 !important;
    color: #ef4444 !important;
}

/* 화살표 아이콘 크기·정렬 */
[data-testid="stSidebarCollapseButton"] [data-testid="stBaseButton-headerNoPadding"] svg,
[data-testid="collapsedControl"] [data-testid="stBaseButton-headerNoPadding"] svg {
    width: 22px !important;
    height: 22px !important;
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
    /* 옅은 회색 배경 */
    background: #f1f5f9;
    border-right: 1px solid #e2e8f0;
    /* 좁은 화면에서도 자동으로 접히지 않도록 최소 폭 보장 */
    min-width: 244px !important;
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

/* 사이드바 메뉴 버튼만 초록 테마 (본문 버튼은 위 전역 규칙 그대로 유지) */
[data-testid="stSidebar"] .stButton > button {
    border: 1px solid #bbf7d0;
    background: #ffffff;
    color: #15803d;
}

[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #16a34a;
    color: #15803d;
    background: #f0fdf4;
}

.logo-box {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-bottom: 4px;
}

/* 사이드바 로고: Streamlit 기본 이미지 스타일을 이기도록 크기를 강제 지정 */
[data-testid="stSidebar"] img.logo-icon,
.logo-icon {
    font-size: 120px;
    width: 150px !important;
    max-width: 150px !important;
    height: 150px !important;
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
    text-align: center;
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

/* hero-marker를 포함한 st.container(border=True)를 hero 박스로 스타일링.
   Streamlit 1.51의 테두리 컨테이너는 stVerticalBlockBorderWrapper 래퍼를 만든다.
   그중 .hero-marker를 품은 것만 골라 hero 그라데이션 박스로 꾸민다. */
.hero-marker {
    display: none;
}

[data-testid="stVerticalBlockBorderWrapper"]:has(.hero-marker) {
    background: linear-gradient(120deg, #fff7ed 0%, #fff5f5 45%, #eef2ff 100%);
    border: 1px solid #fecaca !important;
    border-radius: 22px;
    padding: 28px 32px;
    margin-bottom: 34px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
}

/* 박스 안 컬럼들을 세로 중앙 정렬 */
[data-testid="stVerticalBlockBorderWrapper"]:has(.hero-marker)
    [data-testid="stHorizontalBlock"] {
    align-items: center;
}

.hero-text {
    padding-right: 8px;
    text-align: left;   /* 왼쪽 정렬 */
}

/* hero 텍스트 칸만 세로 중앙 정렬에서 빼서 위쪽(상단)에 붙인다.
   컬럼 컨테이너는 align-items:center 라 차트·메뉴는 중앙 유지하되,
   .hero-text를 가진 컬럼만 align-self:flex-start 로 위로 올린다. */
[data-testid="stVerticalBlockBorderWrapper"]:has(.hero-marker)
    [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:has(.hero-text) {
    align-self: flex-start;
}

.hero-text .hero-title {
    font-size: 18px;
    margin-bottom: 14px;
}

.hero-text .hero-desc {
    font-size: 12px;
    line-height: 1.65;
}

/* 박스 안 차트/메뉴 영역의 섹션 제목은 조금 작게 */
[data-testid="stVerticalBlockBorderWrapper"]:has(.hero-marker)
    .section-title {
    font-size: 13px;
    margin-bottom: 10px;
    white-space: nowrap;   /* '바바삭 추천 메뉴 TOP 3'가 세로로 깨지지 않게 */
}

/* AI 브리핑: 차트 아래에 두는 시세 요약 박스 */
.ai-briefing {
    margin-top: 12px;
    background: #ffffff;
    border: 1px solid #fde4d3;
    border-radius: 12px;
    padding: 12px 14px;
    box-shadow: 0 4px 12px rgba(15,23,42,.04);
}
.ai-briefing-title {
    font-size: 13px;
    font-weight: 800;
    color: #ea580c;
    margin-bottom: 6px;
}
.ai-briefing-body {
    font-size: 13px;
    line-height: 1.6;
    color: #334155;
}
.ai-briefing-body strong {
    color: #0f172a;
    font-weight: 800;
}
/* 브리핑 안 상승(빨강)/하락(파랑) 퍼센트 강조 */
.ai-briefing-body .brief-up {
    color: #dc2626;
    font-weight: 800;
}
.ai-briefing-body .brief-down {
    color: #2563eb;
    font-weight: 800;
}


/* 좁은 좌측 컬럼용 hero 변형: 세로 배치 + 폰트/패딩 축소, 우측 패널과 높이 맞춤 */
.hero-compact {
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    padding: 28px 28px;
    height: 100%;
    margin-bottom: 0;
}

.hero-compact .hero-title {
    font-size: 24px;
    margin-bottom: 14px;
}

.hero-compact .hero-desc {
    font-size: 14px;
    line-height: 1.6;
}

.hero-compact .hero-visual {
    margin-top: 18px;
    justify-content: flex-start;
}

.hero-compact .hero-ing {
    height: 110px;
}

.hero-compact .hero-ing-emoji {
    font-size: 56px;
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
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 10px;
}

/* hero 재료 PNG 이미지 (한 장) */
.hero-ing {
    height: 160px;
    width: auto;
    max-width: 100%;
    object-fit: contain;
}

/* png 없을 때 이모지 폴백 */
.hero-ing-emoji {
    font-size: 84px;
    line-height: 1;
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

.menu-card {
    display: flex;
    align-items: center;
    gap: 12px;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
}

.menu-rank {
    flex: 0 0 auto;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: #fb923c;
    color: white;
    font-weight: 800;
    font-size: 15px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

.menu-name {
    font-size: 13px;
    font-weight: 700;
    color: #0f172a;
    white-space: nowrap;   /* '우거지 감자탕' 등이 두 줄로 깨지지 않게 한 줄 유지 */
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
    /* 주변 박스(흰 배경·테두리·그림자) 제거 — 배경에 그대로 얹히는 형태 */
    background: transparent;
    border: none;
    padding: 0;
    margin-top: 18px;
    margin-bottom: 18px;
    box-shadow: none;
}

.price-title {
    font-size: 24px;
    font-weight: 900;
    color: #111827;
    margin-top: 12px;
    margin-bottom: 16px;
    padding-left: 8px;
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
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: #ffffff;
    color: white;
    font-size: 28px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.18);
    z-index: 9999;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

/* 새 로고 이미지를 동그란 버튼에 꽉 차게 */
#fab-chatbot img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    padding: 7px;
    box-sizing: border-box;
}

#fab-chatbot:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 28px rgba(15, 23, 42, 0.28);
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

/* ── 프로젝트 소개 + 팀 소개를 한 카드에 담는 래퍼 ──────────── */
.about-wrap {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 22px;
    padding: 44px 52px;
    margin-top: 8px;
    box-shadow: 0 8px 24px rgba(15,23,42,0.04);
}
/* 카드 안 맨 위 '프로젝트 소개' 제목 */
.about-wrap-title {
    font-size: 24px;
    font-weight: 900;
    color: #0f172a;
    margin-bottom: 24px;
}
/* 두 섹션을 나누는 구분선 (한 카드 안에서 '나뉜 느낌'을 준다) */
.about-divider {
    height: 1px;
    background: #eef1f6;
    margin: 36px 0;
}

/* ── 팀/멘토 소개 ───────────────────────────────────────────── */
/* 래퍼(.about-wrap) 안에 들어가므로 자체 박스 스타일은 제거하고 내용만. */
.team-section {
    padding: 0;
    margin: 0;
}
.team-section-title {
    font-size: 22px;
    font-weight: 900;
    color: #0f172a;
    margin-bottom: 6px;
}
.team-section-desc {
    font-size: 14px;
    color: #64748b;
    line-height: 1.7;
    margin-bottom: 24px;
}
/* 멘토 카드: 가로형(사진 왼쪽 + 텍스트 오른쪽)으로 팀원과 구분해 강조.
   전체 폭을 채우지 않고 내용 폭(최대 480px)으로 줄여 섹션 가운데에 둔다.
   inline-flex + margin:auto 로 카드 자체를 중앙 정렬한다. */
.mentor-card {
    display: flex;
    align-items: center;
    gap: 22px;
    background: #fbfbfd;
    border: 1px solid #ece9f5;
    border-radius: 18px;
    padding: 22px 30px;
    /* 내용 폭만큼만 차지하게 한 뒤 좌우 auto 마진으로 섹션 가운데 정렬 */
    width: fit-content;
    max-width: 480px;
    margin: 0 auto 26px;
}
.mentor-photo {
    flex: 0 0 auto;
    width: 96px;
    height: 96px;
    border-radius: 50%;
    object-fit: cover;
    background: #ede9fe;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 44px;
}
.mentor-info-name { font-size: 19px; font-weight: 900; color: #0f172a; }
.mentor-info-role {
    display: inline-block;
    margin: 8px 0 10px;
    background: #ede9fe; color: #6d28d9;
    font-size: 12px; font-weight: 800;
    padding: 4px 12px; border-radius: 100px;
}
.mentor-info-desc { font-size: 13px; color: #475569; line-height: 1.7; }

/* 팀원 카드 그리드 (한 줄 4개, 좁으면 자동 줄바꿈) */
.team-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
}
.member-card {
    flex: 1 1 calc(25% - 14px);
    min-width: 210px;
    box-sizing: border-box;
    background: #ffffff;
    border: 1px solid #eef1f6;
    border-radius: 18px;
    padding: 26px 22px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(15,23,42,0.03);
}
.member-photo {
    width: 104px;
    height: 104px;
    border-radius: 50%;
    object-fit: cover;
    margin: 0 auto 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 48px;
}
.member-name { font-size: 17px; font-weight: 900; color: #0f172a; margin-bottom: 14px; }
/* 배지 + 불릿을 한 줄에 두고 세로 중앙 정렬 (배지가 불릿 묶음 중앙에 오게) */
.member-detail {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
}
.member-role {
    flex: 0 0 auto;
    display: inline-block;
    font-size: 11px; font-weight: 800;
    padding: 4px 12px; border-radius: 100px;
    text-align: center;
    line-height: 1.35;
}
/* 배지 둘째 줄 보조 텍스트((조기취업) 등) — 폰트만 작게 해서 배지 높이를
   '팀원' 한 줄 배지(정유정 등)와 비슷하게 맞춘다. */
.member-role .role-sub {
    font-size: 8px;
    font-weight: 700;
    line-height: 1;
    opacity: 0.9;
}
/* 역할 배지 색상 팔레트 (카드 순서대로 초록/파랑/주황/보라) */
.member-role.c0 { background:#dcfce7; color:#15803d; }
.member-role.c1 { background:#dbeafe; color:#1d4ed8; }
.member-role.c2 { background:#ffedd5; color:#c2410c; }
.member-role.c3 { background:#ede9fe; color:#6d28d9; }
/* 사진 배경 톤도 배지와 맞춰 은은하게 */
.member-photo.c0 { background:#dcfce7; }
.member-photo.c1 { background:#dbeafe; }
.member-photo.c2 { background:#ffedd5; }
.member-photo.c3 { background:#ede9fe; }
.member-bullets {
    list-style: none;
    padding: 0;
    margin: 0;
    text-align: left;
}
.member-bullets li {
    font-size: 11px;
    color: #475569;
    line-height: 1.7;
    padding-left: 12px;
    position: relative;
}
.member-bullets li::before {
    content: "•";
    position: absolute;
    left: 0;
    color: #94a3b8;
}
</style>
""",
    unsafe_allow_html=True,
)

# 우하단 플로팅 챗봇 버튼 — 새 로고 이미지(없으면 💬 이모지 폴백)
_fab_inner = (
    f'<img src="{_NEW_LOGO_URI}" alt="챗봇" />' if _NEW_LOGO_URI else "💬"
)
st.markdown(
    f'<a id="fab-chatbot" href="?goto=chatbot" title="챗봇 바로가기" target="_self">{_fab_inner}</a>',
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
    """,
        unsafe_allow_html=True,
    )

    if st.button("홈"):
        go_page("home")
    if st.button("챗봇에게 물어보기"):
        go_page("chatbot")
    if st.button("가격 추이 알아보기"):
        go_page("dashboard")
    if st.button("프로젝트 소개"):
        go_page("about")


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
elif st.session_state.page == "about":
    about.render()
elif st.session_state.page == "margin":
    margin_page()
