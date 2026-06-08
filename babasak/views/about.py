import base64
import html
from pathlib import Path

import streamlit as st

# 프로필 사진(PNG)은 assets 폴더에 둔다. 파일이 없으면 이모지로 폴백한다.
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def _esc(v) -> str:
    return html.escape(str(v if v is not None else ""))


def _photo_html(filename: str, css_class: str, fallback_emoji: str) -> str:
    """assets/<filename> 이 있으면 <img>로, 없으면 이모지 플레이스홀더로 반환한다.
    css_class 는 .mentor-photo / .member-photo(+색상 c0~c3) 등 컨테이너 클래스."""
    path = _ASSETS_DIR / filename
    if path.exists():
        b64 = base64.b64encode(path.read_bytes()).decode()
        return (f'<img class="{css_class}" '
                f'src="data:image/png;base64,{b64}" alt="" />')
    # 폴백: 같은 크기의 원형 박스 안에 이모지
    return f'<div class="{css_class}">{fallback_emoji}</div>'


# ── 멘토 정보 (여기만 고치면 됨) ─────────────────────────────
_MENTOR = {
    "name": "이상열",
    "role": "멘토",
    "photo": "mentor.png",   # assets/mentor.png 넣으면 자동 표시, 없으면 이모지
}

# ── 팀원 정보 (이름/역할/불릿/사진만 고치면 됨) ──────────────
# 역할 배지·사진 톤 색은 순서대로 초록·파랑·주황·보라(c0~c3)로 자동 배정된다.
_MEMBERS = [
    {
        "name": "노경준",
        "role": "팀장",
        "bullets": ["데이터 수집 및 전처리", "Neo4j 기반 재료 그래프 구축", "RAG 로직 개발", "대체 재료 추천 로직 개선", "통합 테스트 및 품질 개선"],
        "photo": "member_1.png",
    },
    {
        "name": "이예림",
        "role": "팀원",
        "bullets": ["데이터 리서치 및 수집", "데이터 정규화 및 표준화", "파이프라인 및 대시보드 구축", "Genie Space 환경 구축", "챗봇 성능 개선 및 평가"],
        "photo": "member_2.png",
    },
    {
        "name": "정유정",
        "role": "팀원",
        "bullets": ["데이터 수집 및 전처리", "대시보드 구축", "챗봇 네이버 API 검색 구현", "웹앱&챗봇 UI 개발", "챗봇 속도 개선"],
        "photo": "member_3.png",
    },
    {
        "name": "윤정식",
        "role": "팀원",
        "role_sub": "(조기취업)",   # 배지 둘째 줄 — 더 작은 글씨로 표시
        "bullets": ["데이터 수집", "협업 인프라 환경 구축", "Langgraph 아키텍처 설계", "RAG 설계", "답변 품질 검증 환경 구축"],
        "photo": "member_4.png",
    },
]


def _mentor_html() -> str:
    m = _MENTOR
    photo = _photo_html(m["photo"], "mentor-photo", "🧑‍🏫")
    # desc(소개 문구)는 선택 항목 — 없으면 그 줄을 생략한다.
    desc = m.get("desc")
    desc_html = f'<div class="mentor-info-desc">{_esc(desc)}</div>' if desc else ""
    return (
        '<div class="mentor-card">'
        f'{photo}'
        '<div class="mentor-info">'
        f'<div class="mentor-info-name">{_esc(m["name"])}</div>'
        f'<span class="mentor-info-role">✦ {_esc(m["role"])}</span>'
        f'{desc_html}'
        '</div>'
        '</div>'
    )


def _member_html(mb: dict, idx: int) -> str:
    # idx(0~3)로 색상 클래스(c0~c3)를 돌려쓴다. 5명 이상이면 다시 처음 색으로.
    c = f"c{idx % 4}"
    photo = _photo_html(mb["photo"], f"member-photo {c}", "👤")
    bullets = "".join(f"<li>{_esc(b)}</li>" for b in mb.get("bullets", []))
    # 배지: 메인 역할(role) + 선택적 보조 텍스트(role_sub, 더 작은 둘째 줄).
    role_html = _esc(mb["role"])
    if mb.get("role_sub"):
        role_html += f'<br><span class="role-sub">{_esc(mb["role_sub"])}</span>'
    return (
        '<div class="member-card">'
        f'{photo}'
        f'<div class="member-name">{_esc(mb["name"])}</div>'
        '<div class="member-detail">'
        f'<span class="member-role {c}">{role_html}</span>'
        f'<ul class="member-bullets">{bullets}</ul>'
        '</div>'
        '</div>'
    )


def render():
    """프로젝트 소개 페이지: 하나의 카드 안에 [제목 | 서비스 소개 | 팀 소개]를
    구분선으로 나눠 담는다."""
    members_html = "".join(_member_html(mb, i) for i, mb in enumerate(_MEMBERS))

    # 주의: st.markdown에 넘기는 HTML은 줄 앞에 공백 4칸 이상이 있으면 마크다운이
    # '코드 블록'으로 해석해 태그가 그대로 보인다. 따라서 한 줄로 이어붙여 전달한다.
    about_block = (
        '<div class="about-block">'
        '<div class="about-badge">✦ 바바삭</div>'
        '<div class="about-title">소상공인을 위한<br>'
        '<span>물가 연동형 메뉴 추천 AI</span></div>'
        '<div class="about-desc">'
        '식재료 가격 변동 정보와 레시피 데이터를 결합하여<br>'
        '수익성 높은 메뉴와 대체 재료를 추천합니다.</div>'
        '<div class="about-features">'
        '<div class="about-feature-item">'
        '<div class="about-feature-icon">📈</div>'
        '<div class="about-feature-text">실시간 시세 분석</div></div>'
        '<div class="about-feature-item">'
        '<div class="about-feature-icon">🍽️</div>'
        '<div class="about-feature-text">메뉴 최적화 추천</div></div>'
        '<div class="about-feature-item">'
        '<div class="about-feature-icon">🤖</div>'
        '<div class="about-feature-text">AI 챗봇 상담</div></div>'
        '</div>'
        '</div>'
    )

    team_block = (
        '<div class="team-block">'
        '<div class="team-section-title">팀 소개</div>'
        '<div class="team-section-desc">'
        '바바삭은 소상공인의 어려움을 해결하기 위해<br>'
        '데이터 분석, AI 역량을 바탕으로 챗봇 서비스를 개발했습니다.</div>'
        f'{_mentor_html()}'
        f'<div class="team-grid">{members_html}</div>'
        '</div>'
    )

    st.markdown(
        '<div class="about-wrap">'
        '<div class="about-wrap-title">프로젝트 소개</div>'
        f'{about_block}'
        '<div class="about-divider"></div>'
        f'{team_block}'
        '</div>',
        unsafe_allow_html=True,
    )
