import base64
import html
import os
import json
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st
import streamlit.components.v1 as components


def _esc(v) -> str:
    """카드에 넣는 모든 외부 문자열(메뉴명·재료명 등)을 HTML 이스케이프한다.
    LLM/DB에서 온 값에 <, &, " 가 섞여도 레이아웃이 깨지지 않게 한다."""
    return html.escape(str(v if v is not None else ""))


def _md_to_html(text) -> str:
    """챗봇 텍스트 답변(마크다운)을 말풍선 안에 넣을 안전한 HTML로 변환한다.
    외부 markdown 라이브러리는 배포 환경(requirements.txt)에 없으므로 쓰지 않고,
    핵심 마크다운만 정규식으로 직접 처리한다.
    - 먼저 전부 이스케이프(XSS 방지) → 그 다음 마크다운 토큰만 HTML로 복원.
    - 지원: **굵게**, *기울임*, `코드`, 줄바꿈, 줄머리 '- '/'* '/'1. ' 글머리.
    카드 응답이 아닌 일반 텍스트 답변에만 쓰인다."""
    import re

    s = _esc(text)
    # 굵게 → 기울임 → 인라인코드 순서로 변환(굵게의 ** 가 * 변환에 먹히지 않게 먼저).
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s, flags=re.S)
    s = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+?)`",
               r'<code style="background:#f1f5f9;border-radius:4px;'
               r'padding:1px 4px;font-size:.85em">\1</code>', s)
    # 줄머리 글머리표( - , * , 1. )를 불릿(•)으로 치환해 평문 줄로 보여준다.
    lines = []
    for ln in s.split("\n"):
        m = re.match(r"\s*(?:[-*]|\d+\.)\s+(.*)", ln)
        lines.append(f"• {m.group(1)}" if m else ln)
    # 줄바꿈 → <br>
    return "<br>".join(lines)


API_URL = os.getenv("BACKEND_API_URL", "http://localhost:9000")

# 챗봇 헤더 로고. 새 로고(logo_new.png)가 있으면 그것, 없으면 기존 logo.png, 둘 다 없으면 이모지.
# (사이드바 로고는 app.py가 별도로 logo.png 를 사용하므로 여기 변경은 챗봇 헤더에만 적용된다.)
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_LOGO_PATH = _ASSETS_DIR / "logo_new.png"
if not _LOGO_PATH.exists():
    _LOGO_PATH = _ASSETS_DIR / "logo.png"

# 우측 '추천 질문' 패널의 칩 (라벨, 보낼 질문). 클릭 시 바로 전송된다.
_SUGGESTED_QUESTIONS = [
    ("메뉴", "지금 마진 좋은 메뉴 추천해줘"),
    ("원가", "감자탕 1인분 원가 계산해줘"),
    ("대체", "대파 대신 쓸 대체 재료는?"),
    ("시세", "최근 1개월 간 감자 가격 시세 알려줘"),
]

# 메뉴 칩만 사용자가 요청한 문구로 교체
_SUGGESTED_QUESTIONS[0] = ("메뉴", "제육볶음 레시피 알려줘")


def _logo_img_html(height: int = 34) -> str:
    if _LOGO_PATH.exists():
        b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
        return (f'<img src="data:image/png;base64,{b64}" '
                f'style="height:{height}px;width:auto;object-fit:contain;vertical-align:middle;" />')
    return f'<span style="font-size:{height}px;line-height:1;">🍽️</span>'


def _unpack(item):
    """
    chat_history 항목을 (role, message, chart_html, card) 구조로 안전하게 정규화합니다.
    기존의 tuple(2구조, 3구조) 및 dict 형식 모두와 호환됩니다.
    """
    if isinstance(item, dict):
        return (
            item.get("role", "assistant"),
            item.get("message", ""),
            item.get("chart_html"),
            item.get("card"),
        )
    elif isinstance(item, (list, tuple)):
        role = item[0]
        message = item[1]
        chart_html = None
        card = None
        if len(item) >= 3:
            # 하단 코드 호환 (role, message, card)
            card = item[2]
        return role, message, chart_html, card
    return "assistant", str(item), None, None


def _render_turn(role: str, message, chart_html, card, turn: int):
    """한 대화 턴의 본문을 현재 열린 컨테이너 안에 그린다.
    히스토리 재생(루프)과 방금 받은 응답을 그 자리에서 그릴 때 같은 로직을 쓰도록
    분리한 함수. (호출부에서 with st.chat_message(role): 안에 둘 것)"""
    if role == "assistant" and card and card.get("recipes"):
        # 레시피 카드 응답은 _render_cards가 와이드 버블로 감싸 렌더한다.
        _render_cards(card, message, turn)
    elif role == "user":
        # 사용자 질문: 오른쪽 브랜드 버블. 순수 텍스트라 이스케이프해도 안전.
        st.markdown(
            f'<div class="cb-row user"><div class="cb-bubble user">'
            f'{_esc(message)}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        # 챗봇 텍스트 답변: 왼쪽 흰 버블. 마크다운(**굵게** 등)을 살려 버블에 통째로 넣는다.
        st.markdown(
            f'<div class="cb-row bot"><div class="cb-bubble bot">'
            f'{_md_to_html(message)}</div></div>',
            unsafe_allow_html=True,
        )
    if chart_html:
        components.html(chart_html, height=420, scrolling=False)


def _queue_message(text: str):
    """사용자 메시지를 히스토리에 넣고 '답변 대기' 플래그를 세운 뒤 즉시 rerun.
    실제 백엔드 호출은 채팅 영역(_chat_fragment)이 그려진 뒤 그 안에서 수행한다.
    입력창과 추천 질문 칩이 공유하는 진입점.

    여기서 st.rerun()이 꼭 필요한 이유: pending(_pending_answer) 처리 지점은
    _chat_fragment 위쪽(채팅 영역)에 있고, 이 함수를 호출하는 입력 위젯은 아래쪽에
    있다. rerun 없이 위젯의 자동 rerun에만 맡기면, user 메시지를 append하기 전에 이미
    위쪽 pending 처리를 지나친 뒤라 한 박자 늦게 반영되는 off-by-one이 생긴다.
    즉시 rerun하면 처음부터 다시 그려져 user 메시지·응답이 같은 사이클에 처리된다.

    이 함수는 항상 _chat_fragment(@st.fragment) 안에서 호출되므로, 이 st.rerun()은
    전체 페이지가 아니라 fragment 범위만 부분 재실행한다 → 헤더·프로필은 그대로 두고
    채팅 영역만 갱신되어 '페이지 리로드' 느낌이 없다.
    (응답 후 2차 rerun도 제거한 상태 → 응답 시점의 재렌더 깜빡임은 없음.)"""
    text = (text or "").strip()
    if not text:
        return
    st.session_state.chat_history.append(
        {"role": "user", "message": text, "chart_html": None, "card": None}
    )
    st.session_state["_pending_answer"] = text
    st.rerun(scope="fragment")


@st.fragment
def _chat_fragment():
    """채팅 본문(좌: 대화/응답, 우: 추천칩·Tip)과 입력창을 묶은 부분 재실행 단위.
    입력·칩 클릭으로 발생하는 rerun이 이 fragment 범위로 한정되어 전체 페이지가
    다시 그려지지 않는다."""
    # ── 본문: 좌(채팅) / 우(추천질문 + Tip) 2단 ──────────────────
    # 우측 추천질문 패널을 넉넉히(2) 두어 칩 라벨이 한 줄에 들어가게 한다.
    col_chat, col_side = st.columns([8, 2], gap="large")

    with col_chat:
        # 채팅 히스토리를 고정 높이 컨테이너로 감싼다 → 페이지 대신 이 박스만 스크롤된다.
        # rerun마다 페이지 높이가 줄었다 늘며 스크롤이 위로 클램핑됐다 돌아오는
        # '올라갔다 내려오는' 흔들림을, 페이지를 아예 안 움직이게 해서 원천 차단한다.
        # (입력창·사이드패널·스크롤 스크립트는 이 박스 밖에 둔다.)
        chat_box = st.container(height=460)
        with chat_box:
            for turn, item in enumerate(st.session_state.chat_history):
                role, message, chart_html, card = _unpack(item)
                with st.chat_message(role):
                    _render_turn(role, message, chart_html, card, turn)

            # 대기 중인 질문이 있으면 채팅 영역 안에서 로딩 스피너만 띄우고 응답을 받는다.
            # 응답은 '그 자리'에서 그리지 않는다: 그 자리 렌더 + 위쪽 히스토리 루프가 같은
            # 답변을 이중으로 그려, fragment 부분 재실행 때 옛 요소가 안 지워지고 흐릿한
            # 잔상(ghosting)으로 남는 문제가 있었다. 대신 히스토리에 append만 하고
            # fragment를 부분 재실행하면, 답변은 오직 위쪽 루프 한 곳에서만 그려진다.
            # scope="fragment"라 전체 페이지 리로드/깜빡임은 없다.
            pending = st.session_state.pop("_pending_answer", None)
            if pending:
                with st.chat_message("assistant"):
                    with st.spinner("챗봇이 답변을 생성 중입니다..."):
                        bot_text, chart_html, bot_card = _fetch_response(pending)
                st.session_state.chat_history.append(
                    {"role": "assistant", "message": bot_text,
                     "chart_html": chart_html, "card": bot_card}
                )
                st.rerun(scope="fragment")

            # 채팅 맨 아래 스크롤 기준점(앵커). rerun 직후 여기로 자동 스크롤한다.
            st.markdown('<div id="chat-bottom-anchor"></div>', unsafe_allow_html=True)

    with col_side:
        _render_side_panel()

    # 입력창 (placeholder 제거)
    user_input = st.chat_input("")
    if user_input:
        _queue_message(user_input)

    # 채팅 갱신 직후 최신 메시지로 자동 스크롤. fragment 안에 둬서 부분 재실행 때도
    # 동작한다. scrollIntoView는 가장 가까운 스크롤 부모(=위 고정 높이 박스)를 스크롤하므로
    # 페이지는 안 움직이고 박스 안에서만 내려간다. 2단 모션(auto 즉시점프 + 150ms smooth)을
    # 빼고 requestAnimationFrame으로 렌더 완료 직후 한 번만 부드럽게 스크롤해 튕김을 없앤다.
    # iframe에서 부모 문서 스크롤을 시도하되 막히면(보안정책) 조용히 무시.
    if st.session_state.chat_history:
        components.html(
            """
            <script>
            (function () {
                try {
                    const doc = window.parent.document;
                    const anchor = doc.getElementById("chat-bottom-anchor");
                    if (anchor) {
                        requestAnimationFrame(function () {
                            anchor.scrollIntoView({behavior: "smooth", block: "end"});
                        });
                    }
                } catch (e) { /* cross-origin 등으로 막히면 무시 */ }
            })();
            </script>
            """,
            height=0,
        )


def render():
    _inject_card_css()

    # ── 상단 헤더: 로고 + 타이틀 + 지역/업종 ─────────────────────
    region = st.session_state.get("user_region", "")
    industry = st.session_state.get("user_industry", "")
    loc_txt = region or "지역 미설정"
    ind_txt = industry or "업종 미설정"
    st.markdown(
        f"""
    <div class="cb-header">
        <div class="cb-header-left">
            <div class="cb-logo">{_logo_img_html(64)}</div>
            <div>
                <div class="cb-title">바바삭 챗봇 <span class="cb-badge">● 응답중</span></div>
                <div class="cb-subtitle">메뉴 추천 · 원가 계산 · 대체 재료 · 시세 분석</div>
            </div>
        </div>
        <div class="cb-header-right">
            <div class="cb-chip-info">📍 지역 · {loc_txt}</div>
            <div class="cb-chip-info">🏷️ 업종 · {ind_txt}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    _render_profile()
    st.divider()

    # ── 본문(채팅+입력+추천칩)을 fragment로 감싼다 ───────────────
    # st.fragment 안에서 일어나는 st.rerun()은 '전체 페이지'가 아니라 이 함수 범위만
    # 부분 재실행한다. 따라서 질문을 보낼 때(_queue_message의 rerun)도 헤더·프로필은
    # 그대로 두고 채팅 영역만 다시 그려져 '페이지 리로드' 느낌이 사라진다.
    # 추천칩(col_side)도 클릭 시 같은 부분 재실행이 되도록 컬럼 전체를 fragment에 넣는다.
    # 채팅·입력·추천칩·스크롤 보정은 모두 _chat_fragment 안에서 처리한다.
    _chat_fragment()


def _render_side_panel():
    """우측 패널: 추천 질문 칩 + 챗봇 답변 활용 Tip."""
    st.markdown('<div class="cb-panel-title">✦ 추천 질문</div>', unsafe_allow_html=True)
    st.markdown('<div class="cb-panel-sub">이런 걸 물어보세요</div>', unsafe_allow_html=True)
    # 칩 버튼만 겨냥하는 CSS(.cb-suggest + div ...)가 잡도록 직전에 마커를 둔다.
    st.markdown('<div class="cb-suggest"></div>', unsafe_allow_html=True)
    for i, (label, question) in enumerate(_SUGGESTED_QUESTIONS):
        if st.button(f"[{label}]  {question}", key=f"suggest_{i}", use_container_width=True):
            _queue_message(question)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown(
        """
    <div class="cb-tip">
        <div class="cb-tip-title">💡 챗봇 답변 활용 Tip</div>
        <div class="cb-tip-item">원가 · 마진율은 추정치예요.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# 카드 렌더링 스타일 및 함수 (하단 코드 로직 그대로 유지)
# ──────────────────────────────────────────────────────────────
def _inject_card_css():
    st.markdown("""
    <style>
      /* 챗봇 화면 전체를 '줌아웃'한 느낌으로 압축해 한 화면에 더 많은 내용이 담기게 한다.
         루트 폰트(rem 기준)를 줄이면 rem 기반 요소들이 일괄로 작아진다. */
      html { font-size: 14px; }              /* 기본 16px → 14px (약 87.5% 축소) */

      /* 본문 폭은 넓게 유지(카드 3개 가로 배치). 상하 여백은 줄여 밀도를 높인다. */
      .block-container {
          max-width: 1800px !important;
          padding-top: 1.2rem !important;
          padding-bottom: 1.2rem !important;
      }
      /* 요소 사이 기본 간격(Streamlit 수직 갭) 약간만 축소. 너무 줄이면 입력 라벨이
         위 박스에 붙어 답답하므로 숨 쉴 공간은 남긴다. */
      div[data-testid="stVerticalBlock"] { gap: .85rem !important; }
      hr { margin: .8rem 0 !important; }     /* st.divider 여백 축소 */

      /* 카드 3개를 한 줄에 가로로 나열하는 그리드 컨테이너 */
      .fc-grid { display:flex; flex-wrap:wrap; gap:14px; margin:6px 0 14px; align-items:flex-start; }
      .fc-card { background:#fff; border:1px solid #dbe3ef; border-radius:16px;
                 padding:18px 20px; box-shadow:0 6px 18px rgba(15,23,42,.06);
                 /* 기본: 한 줄에 3개 (gap 14px 2개분 고려). 화면 좁으면 자동 줄바꿈 */
                 flex:1 1 calc(33.333% - 10px); min-width:200px; box-sizing:border-box; }
      /* 펼쳐진 카드는 한 줄 전체를 차지해 나머지 카드를 아래로 밀어냄 */
      .fc-card:has(> details[open]) { flex-basis:100%; }
      .fc-head { display:flex; justify-content:space-between; align-items:center; }
      .fc-pill { background:#dcfce7; color:#15803d; font-size:.72rem; font-weight:800;
                 padding:3px 11px; border-radius:100px; margin-right:9px; }
      .fc-pill.rank { background:#eef2ff; color:#4f46e5; }
      .fc-menu { font-weight:800; font-size:1.12rem; color:#0f172a; }
      .fc-up { color:#16a34a; font-weight:800; }
      .fc-summary { margin-top:12px; }
      .fc-srow { display:flex; justify-content:space-between; gap:8px; padding:7px 0;
                 border-bottom:1px solid #f1f5f9; font-size:.9rem; }
      .fc-srow .k { color:#64748b; }
      .fc-srow .v { color:#0f172a; font-weight:700; }
      .fc-green { color:#16a34a; font-weight:800; }
      .fc-details { margin-top:6px; }
      .fc-details > summary { cursor:pointer; color:#4f6cf7; font-weight:700; font-size:.88rem;
                 list-style:none; padding:10px 0 4px; text-align:center; user-select:none; }
      .fc-details > summary::-webkit-details-marker { display:none; }
      .fc-details > summary:hover { color:#3b53d6; }
      /* 펼침 상태에 따라 두 라벨 중 하나만 보여준다 (JS 없이 토글) */
      .fc-details > summary .fc-open-label { display:none; }
      .fc-details[open] > summary .fc-shut-label { display:none; }
      .fc-details[open] > summary .fc-open-label { display:inline; }
      .fc-sec { font-weight:800; color:#334155; margin:14px 0 6px; }
      .fc-itable { width:100%; border-collapse:collapse; font-size:.9rem; }
      .fc-itable th { background:#eef2ff; color:#475569; text-align:left; padding:7px 10px; }
      .fc-itable td { border-bottom:1px solid #f1f5f9; padding:7px 10px; }
      /* 재료명 → 네이버 쇼핑 검색 링크 */
      .fc-buy { color:#0f172a; text-decoration:none; border-bottom:1px dashed #93c5fd; }
      .fc-buy:hover { color:#2563eb; border-bottom-color:#2563eb; }
      .fc-sub { background:#fff7ed; border:1px solid #fed7aa; border-radius:12px;
                padding:13px 15px; margin:12px 0; font-size:.9rem; }
      .fc-chip { background:#f0fdf4; border:1px solid #86efac; border-radius:9px;
                 padding:4px 11px; font-weight:800; color:#15803d; display:inline-block; }
      .fc-muted { color:#64748b; font-size:.82rem; }
      .fc-steps { margin:4px 0 0; padding-left:20px; }
      .fc-steps li { margin:5px 0; color:#334155; }
      .fc-nm { background:#faf5ff; border:1px solid #e9d5ff; border-radius:14px;
               padding:14px 16px; margin:8px 0 12px; }
      .fc-nm-tag { background:#ede9fe; color:#7c3aed; font-size:.7rem; font-weight:800;
                   padding:3px 10px; border-radius:100px; }
      .fc-nm-name { font-weight:800; font-size:1.05rem; color:#0f172a; margin-left:6px; }
      .fc-combo { background:#fff; border:1px dashed #cbd5e1; border-radius:10px;
                  padding:9px 12px; text-align:center; color:#475569; margin:9px 0; font-weight:600; }

      /* ── 챗봇 헤더 ── */
      .cb-header { display:flex; justify-content:space-between; align-items:center;
                   gap:16px; flex-wrap:wrap; margin:4px 0 10px; }
      .cb-header-left { display:flex; align-items:center; gap:14px; }
      .cb-logo { display:flex; align-items:center; }
      .cb-title { font-size:1.5rem; font-weight:900; color:#0f172a; }
      .cb-badge { font-size:.72rem; font-weight:800; color:#16a34a;
                  background:#dcfce7; border-radius:100px; padding:2px 10px;
                  vertical-align:middle; margin-left:6px; }
      .cb-subtitle { font-size:.9rem; color:#64748b; margin-top:2px; }
      .cb-header-right { display:flex; gap:10px; flex-wrap:wrap; }
      .cb-chip-info { background:#fff; border:1px solid #e5e7eb; border-radius:12px;
                      padding:8px 14px; font-size:.85rem; font-weight:700; color:#334155;
                      box-shadow:0 2px 8px rgba(15,23,42,.04); }

      /* ── 우측 패널 ── (좁은 폭에 맞춰 제목·부제목·칩 전반 축소) */
      .cb-panel-title { font-size:.9rem; font-weight:900; color:#0f172a; margin-bottom:2px; }
      .cb-panel-sub { font-size:.72rem; color:#94a3b8; margin-bottom:10px; }
      /* 추천 질문 칩(st.button): cb-suggest 래퍼 다음에 오는 Streamlit 버튼만 스타일.
         라벨이 2줄로 접혀도 답답하지 않도록 세로 패딩·줄간격을 넉넉히 준다. */
      .cb-suggest + div [data-testid="stButton"] > button {
          font-size:.76rem; font-weight:700; line-height:1.4;
          padding:10px 12px; min-height:0; text-align:left;
          white-space:nowrap; border-radius:12px; }
      /* 칩 사이 간격 약간 확보 */
      .cb-suggest + div [data-testid="stButton"] { margin-bottom:7px; }
      .cb-tip { background:#fffbeb; border:1px solid #fde68a; border-radius:14px;
                padding:12px 13px; }
      .cb-tip-title { font-size:.8rem; font-weight:800; color:#92400e; margin-bottom:6px; }
      .cb-tip-item { font-size:.74rem; color:#78350f; line-height:1.55; }

      /* ── 말풍선(버블) UI ──────────────────────────────────────
         Streamlit 기본 chat_message의 회색 박스/아바타 배경을 지우고,
         그 안에 우리가 그린 .cb-bubble 말풍선만 보이게 한다.
         - 사용자 질문: 오른쪽 정렬 + 브랜드(빨강→주황) 그라데이션 버블
         - 챗봇 답변: 왼쪽 정렬 + 흰 버블 (꼬리 포함)
         카드 응답(레시피)은 와이드 버블(.cards)로 감싼다. */
      [data-testid="stChatMessage"] {
          background: transparent !important;
          padding: 0 !important;
          box-shadow: none !important;
      }
      /* 기본 아바타 아이콘 숨김 (말풍선만으로 화자 구분).
         Streamlit 버전마다 아바타 컨테이너의 testid/클래스가 달라
         testid 접두 매칭 + 이미지/아바타류를 폭넓게 잡아 확실히 지운다. */
      [data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"],
      [data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"],
      [data-testid="stChatMessage"] > img:first-child,
      [data-testid="stChatMessageAvatarUser"],
      [data-testid="stChatMessageAvatarAssistant"] {
          display: none !important;
      }
      /* 아바타가 빠진 만큼 본문이 왼쪽 끝까지 차지하도록 메시지 콘텐츠 여백 제거 */
      [data-testid="stChatMessage"] > [data-testid="stChatMessageContent"],
      [data-testid="stChatMessage"] > div:last-child {
          margin-left: 0 !important;
          padding-left: 0 !important;
          width: 100% !important;
      }
      /* 말풍선 공통 */
      .cb-bubble {
          display: inline-block;
          max-width: 78%;
          padding: 11px 15px;
          border-radius: 18px;
          font-size: .92rem;
          line-height: 1.6;
          word-break: break-word;
          white-space: pre-wrap;
          box-shadow: 0 2px 8px rgba(15,23,42,.06);
      }
      /* 챗봇(왼쪽, 흰색) — 왼쪽 아래 꼬리 */
      .cb-row.bot { display:flex; justify-content:flex-start; margin:8px 0; }
      .cb-bubble.bot {
          background:#ffffff;
          border:1px solid #e5e7eb;
          color:#0f172a;
          border-bottom-left-radius:5px;
      }
      /* 사용자(오른쪽, 브랜드 그라데이션) — 오른쪽 아래 꼬리 */
      .cb-row.user { display:flex; justify-content:flex-end; margin:8px 0; }
      .cb-bubble.user {
          background:linear-gradient(135deg,#ef4444,#f97316);
          color:#ffffff;
          border:none;
          border-bottom-right-radius:5px;
      }
      /* 카드(레시피) 응답을 감싸는 와이드 버블 — 폭 제한을 풀어 카드가 넓게 펼쳐지게 */
      .cb-bubble.cards {
          display:block;
          max-width:100%;
          width:100%;
          padding:14px 16px;
      }
      .cb-bubble.cards .fc-grid { margin:0; }
    </style>
    """, unsafe_allow_html=True)


def _won(n):
    try:
        return f"₩{int(n):,}"
    except (TypeError, ValueError):
        return "-"


def _ing_name_link(name: str) -> str:
    """재료명을 네이버 쇼핑 검색 결과 페이지로 가는 링크로 감싼다(새 탭).
    name 은 이미 _esc 처리된 표시용 문자열이어야 한다.
    빈 값이면 링크 없이 그대로 반환."""
    if not name or name == "-":
        return name
    # 검색어는 표시용(이스케이프된) 값이 아니라 원문 기준으로 인코딩해야 하므로 unescape
    raw = html.unescape(name)
    url = "https://search.shopping.naver.com/search/all?query=" + quote(raw)
    return (f'<a class="fc-buy" href="{url}" target="_blank" rel="noopener" '
            f'title="네이버 쇼핑에서 \'{raw}\' 구매처 보기">{name} 🛒</a>')


def _ing_rows_html(ings: list, has_cost: bool = True) -> str:
    # 원가 계산이 없는 '레시피만' 질문이면 단가·원가 칸을 아예 빼고 '재료 | 수량'만 보여줌
    if not has_cost:
        rows = "".join(
            f"<tr><td>{_ing_name_link(_esc(it.get('name','')))}</td>"
            f"<td style='text-align:right'>{_esc(it.get('quantity') or '-')}</td></tr>"
            for it in ings
        )
        return ("<table class='fc-itable'><tr><th>재료</th>"
                "<th style='text-align:right'>수량</th></tr>"
                f"{rows}</table>")
    rows = ""
    for it in ings:
        name = _esc(it.get("name", ""))
        qty = _esc(it.get("quantity") or "-")
        ppk = it.get("price_per_kg")
        ppk_txt = f"{int(ppk):,}" if ppk else "—"
        src = it.get("source")
        cost = it.get("cost")
        if src == "non_cost":
            cost_txt = '<span style="color:#94a3b8">원가 제외</span>'
        elif cost is not None:
            cost_txt = f"<b>{int(cost):,}원</b>"
        else:
            cost_txt = '<span style="color:#cbd5e1">시세/사용량 미확인</span>'
        rows += (f"<tr><td>{_ing_name_link(name)}</td><td style='text-align:right'>{qty}</td>"
                 f"<td style='text-align:right'>{ppk_txt}</td>"
                 f"<td style='text-align:right'>{cost_txt}</td></tr>")
    return ("<table class='fc-itable'><tr><th>재료</th><th style='text-align:right'>수량</th>"
            "<th style='text-align:right'>단가(원/kg)</th><th style='text-align:right'>원가</th></tr>"
            f"{rows}</table>")


def _recipe_card_html(rc: dict, idx: int, single: bool, group: str = "rc") -> str:
    menu = _esc(rc.get("menu", "이름없음"))
    total = rc.get("total_cost")
    price = rc.get("suggested_price")
        # 가격이 검증된 대체재 전용 카드
    if rc.get("is_substitute_price_card"):
        target = _esc(rc.get("target", "원본 재료"))
        target_ppk = rc.get("target_ppk")
        candidate_ppk = rc.get("candidate_ppk")
        saving_per_kg = rc.get("saving_per_kg")
        saving_pct = rc.get("saving_pct")
        is_cheaper = rc.get("is_cheaper", False)
        difference_label = "kg당 절감액" if is_cheaper else "kg당 추가비용"
        rate_label = "예상 절감률" if is_cheaper else "예상 추가비용률"
        value_class = "fc-green" if is_cheaper else ""

        return (
            f'<div class="fc-card">'
            f'<div class="fc-head">'
            f'<div><span class="fc-pill rank">{idx}위</span>'
            f'<span class="fc-menu">{menu}</span></div>'
            f'</div>'
            f'<div class="fc-summary">'
            f'<div class="fc-srow"><span class="k">{target} 단가</span>'
            f'<span class="v">{_won(target_ppk)}/kg</span></div>'
            f'<div class="fc-srow"><span class="k">{menu} 단가</span>'
            f'<span class="v">{_won(candidate_ppk)}/kg</span></div>'
            f'<div class="fc-srow"><span class="k">{difference_label}</span>'
            f'<span class="v {value_class}">{_won(saving_per_kg)}</span></div>'
            f'<div class="fc-srow"><span class="k">{rate_label}</span>'
            f'<span class="v {value_class}">{saving_pct}%</span></div>'
            f'</div>'
            f'</div>'
        )

    pill = ('<span class="fc-pill">오늘의 추천</span>' if single
            else f'<span class="fc-pill rank">{idx}위</span>')

    # 조합(짬뽕) 메뉴: base 레시피 기반으로 만든 제안임을 표시
    combo_note = ""
    if rc.get("is_combo"):
        base = _esc(rc.get("base_menu") or "")
        base_txt = f" · {base} 기반" if base else ""
        combo_note = (f'<div class="fc-muted" style="margin-top:4px">'
                      f'✨ 조합 메뉴 제안{base_txt}</div>')

    summary = ""
    if total:
        cr = rc.get("cost_ratio_pct") or 30      # 백엔드 기준 원가율, 없으면 서비스 기본값
        mp = rc.get("margin_pct") or (100 - cr)  # 매출총이익률 = 100 - 원가율
        pricing_label = _esc(rc.get("pricing_label") or f"원가율 {cr}%")
        _est_label = '예상 원가 <span class="fc-muted">(분량 미상·2인분 추정)</span>' if rc.get("servings_estimated") else '예상 원가'
        srows = [f'<div class="fc-srow"><span class="k">{_est_label}</span><span class="v">{_won(total)}</span></div>']
        if price:
            srows.append(f'<div class="fc-srow"><span class="k">권장 판매가 ({pricing_label})</span>'
                         f'<span class="v">{_won(price)}</span></div>')
        srows.append(f'<div class="fc-srow"><span class="k">예상 마진율</span>'
                     f'<span class="v fc-green">{mp}%</span></div>')
        summary = '<div class="fc-summary">' + "".join(srows) + '</div>'

    meta = " · ".join(str(x) for x in [rc.get("servings"), rc.get("difficulty"),
                                       rc.get("cooking_time")] if x)
    body = f'<div class="fc-sec">📋 재료{f"  ({meta})" if meta else ""}</div>'
    # has_cost(원가 계산 여부)에 따라 단가·원가 칸 표시 결정. 없으면(레시피만) '재료|수량'만.
    body += _ing_rows_html(rc.get("ingredients") or [], has_cost=bool(total) or rc.get("has_cost"))

    sub = rc.get("substitute")
    if sub and sub.get("candidates"):
        cands = " · ".join(f'🍗 {_esc(c)}' for c in sub["candidates"])
        saving = sub.get("saving_pct")
        saving_txt = f'<br><span class="fc-muted">예상 절감률 {saving}%</span>' if saving is not None else ""
        body += (f'<div class="fc-sub">💡 <b>{_esc(sub.get("target","주재료"))} 비싸면 이렇게 바꿔보세요</b><br>'
                 f'<span class="fc-chip">{cands}</span>{saving_txt}</div>')

    steps = rc.get("steps") or []
    if steps:
        body += '<div class="fc-sec">👨‍🍳 조리 순서</div><ol class="fc-steps">'
        body += "".join(f"<li>{_esc(s)}</li>" for s in steps)
        body += "</ol>"

    # name 속성을 같은 그룹으로 묶으면 브라우저가 아코디언처럼 동작(하나 열면 나머지 닫힘).
    # single(단일 추천)일 때는 그냥 펼쳐둔다.
    name_attr = "" if single else f' name="{group}"'
    details = (f'<details class="fc-details"{name_attr}{" open" if single else ""}>'
               f'<summary><span class="fc-shut-label">👆 탭하면 재료·조리법·대체재 보기</span>'
               f'<span class="fc-open-label">🔼 접기</span></summary>{body}</details>')

    return (f'<div class="fc-card"><div class="fc-head">'
            f'<div>{pill}<span class="fc-menu">{menu}</span></div></div>'
            f'{combo_note}{summary}{details}</div>')


def _new_menu_html(m: dict, i: int) -> str:
    combo = f'<div class="fc-combo">{_esc(m["combo_label"])}</div>' if m.get("combo_label") else ""
    return (f'<div class="fc-nm"><span class="fc-nm-tag">✨ AI 신메뉴 제안 #{i}</span>'
            f'<span class="fc-nm-name">{_esc(m.get("name",""))}</span>{combo}'
            f'<div style="color:#64748b;font-size:.85rem">기존 재료를 활용한 조합 메뉴 제안입니다.</div></div>')


def _render_cards(card: dict, fallback_text: str, turn: int = 0):
    recipes = card.get("recipes") or []
    single = len(recipes) == 1

    # 카드 묶음마다 고유 그룹명 → 같은 묶음의 details끼리만 아코디언으로 묶인다.
    # 대화 turn 인덱스를 쓰므로 답변마다 그룹이 달라 서로 간섭하지 않는다.
    group = f"rcg-{turn}"

    cards_html = "".join(
        _recipe_card_html(rc, i, single, group) for i, rc in enumerate(recipes, 1)
    )
    # 카드들을 가로 그리드로 감싼다(한 줄에 3개, 펼치면 그 카드만 전체 폭 차지)
    cards_block = f'<div class="fc-grid">{cards_html}</div>'

    new_menus = card.get("new_menus") or []
    if new_menus:
        cards_block += "".join(_new_menu_html(m, i) for i, m in enumerate(new_menus, 1))

    # 카드 응답 전체를 챗봇 흰색 말풍선(와이드)으로 감싼다.
    st.markdown(
        f'<div class="cb-row bot"><div class="cb-bubble bot cards">'
        f'{cards_block}</div></div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# 프로필 렌더링 (공통)
# ──────────────────────────────────────────────────────────────
def _render_profile():
    if st.session_state.profile_saved:
        col_info, col_edit = st.columns([6, 1])
        with col_info:
            st.markdown(
                f"""
            <div class="profile-saved">
                <span style="font-size:18px;">✅</span>
                <span class="profile-saved-text">
                    {st.session_state.user_industry} · {st.session_state.user_region}
                </span>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col_edit:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            if st.button("수정", key="edit_profile"):
                st.session_state.profile_saved = False
                st.rerun()
        return

    st.markdown(
        """
    <div class="profile-prompt">
        <div class="profile-prompt-title">맞춤 답변을 위한 정보 입력</div>
        <div class="profile-prompt-desc">업종과 지역을 알려주시면 더 정확하게 답변할 수 있어요.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        industry = st.selectbox(
            "업종",
            ["선택하세요", "한식", "중식", "일식", "양식", "분식", "카페/디저트", "치킨/패스트푸드", "기타"],
            key="input_industry",
        )
    with col2:
        region = st.text_input("지역", placeholder="예: 서울 강남구", key="input_region")
    with col3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("저장", key="save_profile", type="primary"):
            if industry != "선택하세요" and region.strip():
                st.session_state.user_industry = industry
                st.session_state.user_region = region.strip()
                st.session_state.profile_saved = True
                st.rerun()
            else:
                st.warning("업종과 지역을 모두 입력해주세요.")


# ──────────────────────────────────────────────────────────────
# 백엔드 API 호출 통합
# ──────────────────────────────────────────────────────────────
def _fetch_response(message: str) -> tuple[str, str | None, dict | None]:
    """백엔드 API 호출. (응답 텍스트, 차트 HTML, 카드 dict) 튜플 반환."""
    try:
        history = []
        for item in st.session_state.chat_history[:-1]:
            role, msg, _, _ = _unpack(item)
            history.append({"role": role, "content": msg})

        resp = requests.post(
            f"{API_URL}/api/chatbot/chat",
            json={
                "message": message,
                "history": history,
                "industry": st.session_state.get("user_industry", ""),
                "region": st.session_state.get("user_region", ""),
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        
        response_text = data.get("response", "응답을 가져올 수 없습니다.")
        chart_html = data.get("chart_html")
        card = data.get("card")
        
        return response_text, chart_html, card
    except Exception as exc:
        # uvicorn 크래시 로그가 있으면 같이 표시 (start.sh에서 /tmp/uvicorn.log로 저장)
        uvicorn_log = ""
        try:
            with open("/tmp/uvicorn.log") as _f:
                uvicorn_log = _f.read()[-3000:]
        except Exception:
            pass
        detail = f"{exc}\n\n[uvicorn 시작 로그]\n{uvicorn_log}" if uvicorn_log else str(exc)
        return f"서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.\n\n{detail}", None, None
