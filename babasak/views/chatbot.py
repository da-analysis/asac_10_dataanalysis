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
    ("시세", "다음 주 채소 시세 전망은?"),
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


def _queue_message(text: str):
    """사용자 메시지를 히스토리에 넣고 '답변 대기' 플래그만 세운 뒤 rerun.
    실제 백엔드 호출은 화면(헤더·채팅·패널)이 그려진 뒤 채팅 영역 안에서 수행한다.
    입력창과 추천 질문 칩이 공유하는 진입점."""
    text = (text or "").strip()
    if not text:
        return
    st.session_state.chat_history.append(
        {"role": "user", "message": text, "chart_html": None, "card": None}
    )
    st.session_state["_pending_answer"] = text
    st.rerun()


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

    # ── 본문: 좌(채팅) / 우(추천질문 + Tip) 2단 ──────────────────
    col_chat, col_side = st.columns([8, 2], gap="large")

    with col_chat:
        for turn, item in enumerate(st.session_state.chat_history):
            role, message, chart_html, card = _unpack(item)
            with st.chat_message(role):
                if role == "assistant" and card and card.get("recipes"):
                    _render_cards(card, message, turn)
                else:
                    st.write(message)
                if chart_html:
                    components.html(chart_html, height=420, scrolling=False)

        # 대기 중인 질문이 있으면 채팅 영역 안에서 로딩 → 응답 받기
        pending = st.session_state.pop("_pending_answer", None)
        if pending:
            with st.chat_message("assistant"):
                with st.spinner("챗봇이 답변을 생성 중입니다..."):
                    bot_text, chart_html, bot_card = _fetch_response(pending)
            st.session_state.chat_history.append(
                {"role": "assistant", "message": bot_text,
                 "chart_html": chart_html, "card": bot_card}
            )
            st.rerun()

        # 채팅 맨 아래 스크롤 기준점(앵커). rerun 직후 여기로 자동 스크롤한다.
        st.markdown('<div id="chat-bottom-anchor"></div>', unsafe_allow_html=True)

    with col_side:
        _render_side_panel()

    # 입력창 (placeholder 제거)
    user_input = st.chat_input("")
    if user_input:
        _queue_message(user_input)

    # rerun 후 화면이 맨 위로 튀는 것을 완화: 대화가 있으면 최신 메시지로 자동 스크롤.
    # iframe에서 부모 문서 스크롤을 시도하되, 막히면(보안정책) 조용히 무시한다.
    if st.session_state.chat_history:
        components.html(
            """
            <script>
            (function () {
                try {
                    const doc = window.parent.document;
                    const anchor = doc.getElementById("chat-bottom-anchor");
                    if (anchor) {
                        // 렌더 직후 한 번 + 약간 늦게 한 번(레이아웃 안정화 후) 스크롤
                        anchor.scrollIntoView({behavior: "auto", block: "end"});
                        setTimeout(function () {
                            anchor.scrollIntoView({behavior: "smooth", block: "end"});
                        }, 150);
                    }
                } catch (e) { /* cross-origin 등으로 막히면 무시 */ }
            })();
            </script>
            """,
            height=0,
        )


def _render_side_panel():
    """우측 패널: 추천 질문 칩 + 챗봇 답변 활용 Tip."""
    st.markdown('<div class="cb-panel-title">✦ 추천 질문</div>', unsafe_allow_html=True)
    st.markdown('<div class="cb-panel-sub">이런 걸 물어보세요</div>', unsafe_allow_html=True)
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
      /* 챗봇 화면에서는 본문 폭 제한을 넓혀 채팅 영역을 더 넓게 쓴다
         (app.py 전역 max-width:1500px 를 이 페이지에 한해 확장) */
      .block-container { max-width: 1800px !important; }

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

      /* ── 우측 패널 ── */
      .cb-panel-title { font-size:1rem; font-weight:900; color:#0f172a; margin-bottom:2px; }
      .cb-panel-sub { font-size:.8rem; color:#94a3b8; margin-bottom:12px; }
      .cb-tip { background:#fffbeb; border:1px solid #fde68a; border-radius:16px;
                padding:16px 18px; }
      .cb-tip-title { font-size:.9rem; font-weight:800; color:#92400e; margin-bottom:8px; }
      .cb-tip-item { font-size:.85rem; color:#78350f; line-height:1.6; }
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
        srows = [f'<div class="fc-srow"><span class="k">예상 원가</span><span class="v">{_won(total)}</span></div>']
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

    st.markdown(cards_block, unsafe_allow_html=True)


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
