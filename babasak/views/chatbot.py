import os

import requests
import streamlit as st


API_URL = os.getenv("BACKEND_API_URL", "http://localhost:9000")


def _unpack(item):
    """chat_history 항목을 (role, message, card)로 정규화. 기존 2-tuple도 호환."""
    if isinstance(item, (list, tuple)):
        if len(item) >= 3:
            return item[0], item[1], item[2]
        if len(item) == 2:
            return item[0], item[1], None
    return "assistant", str(item), None


def render():
    if st.button("← 홈으로", key="chatbot_home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("💬 챗봇")
    st.caption("메뉴 추천, 원가 계산, 대체 재료, 시세 분석에 대해 질문할 수 있습니다.")

    _inject_card_css()
    _render_profile()
    st.divider()

    for item in st.session_state.chat_history:
        role, message, card = _unpack(item)
        with st.chat_message(role):
            if role == "assistant" and card and card.get("recipes"):
                _render_cards(card, message)
            else:
                st.write(message)

    user_input = st.chat_input("예: 감자, 양파, 돼지등뼈가 있는데 마진 좋은 메뉴 추천해줘")
    if user_input:
        st.session_state.chat_history.append(("user", user_input, None))
        with st.chat_message("user"):
            st.write(user_input)
        with st.spinner("챗봇이 답변을 생성 중입니다..."):
            bot_text, bot_card = _fetch_response(user_input)
        st.session_state.chat_history.append(("assistant", bot_text, bot_card))
        st.rerun()


# ──────────────────────────────────────────────────────────────
# 카드 렌더링
# ──────────────────────────────────────────────────────────────
def _inject_card_css():
    st.markdown("""
    <style>
      /* 목업과 거의 동일한 카드 — 전체를 HTML로 렌더 */
      .fc-card { background:#fff; border:1px solid #dbe3ef; border-radius:16px;
                 padding:18px 20px; margin:6px 0 14px; box-shadow:0 6px 18px rgba(15,23,42,.06); }
      .fc-head { display:flex; justify-content:space-between; align-items:center; }
      .fc-pill { background:#dcfce7; color:#15803d; font-size:.72rem; font-weight:800;
                 padding:3px 11px; border-radius:100px; margin-right:9px; }
      .fc-pill.rank { background:#eef2ff; color:#4f46e5; }
      .fc-menu { font-weight:800; font-size:1.12rem; color:#0f172a; }
      .fc-up { color:#16a34a; font-weight:800; }
      .fc-summary { margin-top:12px; }
      .fc-srow { display:flex; justify-content:space-between; padding:7px 0;
                 border-bottom:1px solid #f1f5f9; font-size:.95rem; }
      .fc-srow .k { color:#64748b; }
      .fc-srow .v { color:#0f172a; font-weight:700; }
      .fc-green { color:#16a34a; font-weight:800; }
      .fc-details { margin-top:6px; }
      .fc-details > summary { cursor:pointer; color:#4f6cf7; font-weight:700; font-size:.9rem;
                 list-style:none; padding:10px 0 4px; text-align:center; }
      .fc-details > summary::-webkit-details-marker { display:none; }
      .fc-sec { font-weight:800; color:#334155; margin:14px 0 6px; }
      .fc-itable { width:100%; border-collapse:collapse; font-size:.9rem; }
      .fc-itable th { background:#eef2ff; color:#475569; text-align:left; padding:7px 10px; }
      .fc-itable td { border-bottom:1px solid #f1f5f9; padding:7px 10px; }
      .fc-sub { background:#fff7ed; border:1px solid #fed7aa; border-radius:12px;
                padding:13px 15px; margin:12px 0; font-size:.9rem; }
      .fc-chip { background:#f0fdf4; border:1px solid #86efac; border-radius:9px;
                 padding:4px 11px; font-weight:800; color:#15803d; display:inline-block; }
      .fc-steps { margin:4px 0 0; padding-left:20px; }
      .fc-steps li { margin:5px 0; color:#334155; }
      .fc-nm { background:#faf5ff; border:1px solid #e9d5ff; border-radius:14px;
               padding:14px 16px; margin:8px 0 12px; }
      .fc-nm-tag { background:#ede9fe; color:#7c3aed; font-size:.7rem; font-weight:800;
                   padding:3px 10px; border-radius:100px; }
      .fc-nm-name { font-weight:800; font-size:1.05rem; color:#0f172a; margin-left:6px; }
      .fc-combo { background:#fff; border:1px dashed #cbd5e1; border-radius:10px;
                  padding:9px 12px; text-align:center; color:#475569; margin:9px 0; font-weight:600; }
    </style>
    """, unsafe_allow_html=True)


def _won(n):
    try:
        return f"₩{int(n):,}"
    except (TypeError, ValueError):
        return "-"


def _ing_rows_html(ings: list) -> str:
    rows = ""
    for it in ings:
        name = it.get("name", "")
        qty = it.get("quantity") or "-"
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
        rows += (f"<tr><td>{name}</td><td style='text-align:right'>{qty}</td>"
                 f"<td style='text-align:right'>{ppk_txt}</td>"
                 f"<td style='text-align:right'>{cost_txt}</td></tr>")
    return ("<table class='fc-itable'><tr><th>재료</th><th style='text-align:right'>수량</th>"
            "<th style='text-align:right'>단가(원/kg)</th><th style='text-align:right'>원가</th></tr>"
            f"{rows}</table>")


def _recipe_card_html(rc: dict, idx: int, single: bool) -> str:
    menu = rc.get("menu", "이름없음")
    total = rc.get("total_cost")
    price = rc.get("suggested_price")

    pill = ('<span class="fc-pill">오늘의 추천</span>' if single
            else f'<span class="fc-pill rank">{idx}위</span>')

    # 요약 (원가 있을 때만)
    summary = ""
    if total:
        srows = [f'<div class="fc-srow"><span class="k">예상 원가</span><span class="v">{_won(total)}</span></div>']
        if price:
            srows.append(f'<div class="fc-srow"><span class="k">권장 판매가 (마진 30%)</span>'
                         f'<span class="v">{_won(price)}</span></div>')
        srows.append('<div class="fc-srow"><span class="k">예상 마진율</span>'
                     '<span class="v fc-green">30% ▲</span></div>')
        summary = '<div class="fc-summary">' + "".join(srows) + '</div>'

    # 상세 (재료·대체재·조리순서)
    meta = " · ".join(str(x) for x in [rc.get("servings"), rc.get("difficulty"),
                                       rc.get("cooking_time")] if x)
    body = f'<div class="fc-sec">📋 재료{f"  ({meta})" if meta else ""}</div>'
    body += _ing_rows_html(rc.get("ingredients") or [])

    sub = rc.get("substitute")
    if sub and sub.get("candidates"):
        cands = " · ".join(f'🍗 {c}' for c in sub["candidates"])
        body += (f'<div class="fc-sub">💡 <b>{sub.get("target","주재료")} 비싸면 이렇게 바꿔보세요</b><br>'
                 f'<span class="fc-chip">{cands}</span></div>')

    steps = rc.get("steps") or []
    if steps:
        body += '<div class="fc-sec">👨‍🍳 조리 순서</div><ol class="fc-steps">'
        body += "".join(f"<li>{s}</li>" for s in steps)
        body += "</ol>"

    details = (f'<details class="fc-details"{" open" if single else ""}>'
               f'<summary>👆 탭하면 재료·조리법·대체재 보기</summary>{body}</details>')

    return (f'<div class="fc-card"><div class="fc-head">'
            f'<div>{pill}<span class="fc-menu">{menu}</span></div></div>'
            f'{summary}{details}</div>')


def _new_menu_html(m: dict, i: int) -> str:
    combo = f'<div class="fc-combo">{m["combo_label"]}</div>' if m.get("combo_label") else ""
    return (f'<div class="fc-nm"><span class="fc-nm-tag">✨ AI 신메뉴 제안 #{i}</span>'
            f'<span class="fc-nm-name">{m.get("name","")}</span>{combo}'
            f'<div style="color:#64748b;font-size:.85rem">기존 재료를 활용한 조합 메뉴 제안입니다.</div></div>')


def _render_cards(card: dict, fallback_text: str):
    recipes = card.get("recipes") or []
    single = len(recipes) == 1

    html = "".join(_recipe_card_html(rc, i, single) for i, rc in enumerate(recipes, 1))

    new_menus = card.get("new_menus") or []
    if new_menus:
        html += "".join(_new_menu_html(m, i) for i, m in enumerate(new_menus, 1))

    # 전체 카드를 한 번에 렌더 (markdown이 끼어들지 않게 줄바꿈 없이)
    st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 프로필 (ROH_67 그대로)
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


def _fetch_response(message: str):
    """백엔드 호출. (응답텍스트, card dict|None) 반환."""
    try:
        history = []
        for item in st.session_state.chat_history[:-1]:
            role, msg, _ = _unpack(item)
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
        return data.get("response", "응답을 가져올 수 없습니다."), data.get("card")
    except Exception as exc:
        return f"서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요. ({exc})", None
