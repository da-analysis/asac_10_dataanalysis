import os
import json

import requests
import streamlit as st
import streamlit.components.v1 as components


API_URL = os.getenv("BACKEND_API_URL", "http://localhost:9000")


def render():
    if st.button("← 홈으로", key="chatbot_home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("💬 챗봇")
    st.caption("메뉴 추천, 원가 계산, 대체 재료, 시세 분석에 대해 질문할 수 있습니다.")

    _render_profile()
    st.divider()

    for item in st.session_state.chat_history:
        # ═══ [차트 렌더링 지원] ═══════════════════════════════════════
        # chat_history 항목이 tuple이면 기존 (role, message) 형식.
        # dict이면 {"role": ..., "message": ..., "chart_html": ...} 형식.
        # 차트를 비활성화하려면 chart_utils.py의 ENABLE_CHART = False로 변경.
        # ═══════════════════════════════════════════════════════════════
        if isinstance(item, dict):
            role = item["role"]
            message = item["message"]
            chart_html = item.get("chart_html")
        else:
            role, message = item
            chart_html = None

        with st.chat_message(role):
            st.write(message)
            # ═══ [차트 HTML 렌더링] ═══
            if chart_html:
                components.html(chart_html, height=420, scrolling=False)

    user_input = st.chat_input("예: 감자, 양파, 돼지등뼈가 있는데 마진 좋은 메뉴 추천해줘")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        with st.spinner("챗봇이 답변을 생성 중입니다..."):
            bot_response, chart_html = _fetch_response(user_input)
        st.session_state.chat_history.append({
            "role": "assistant",
            "message": bot_response,
            "chart_html": chart_html,
        })
        st.rerun()


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


def _fetch_response(message: str) -> tuple[str, str | None]:
    """백엔드 API 호출. (응답 텍스트, 차트 HTML 또는 None) 튜플 반환."""
    try:
        history = []
        for item in st.session_state.chat_history[:-1]:
            if isinstance(item, dict):
                history.append({"role": item["role"], "content": item["message"]})
            else:
                history.append({"role": item[0], "content": item[1]})

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
        # ═══ [차트 HTML 수신] ═══
        chart_html = data.get("chart_html")
        return response_text, chart_html
    except Exception as exc:
        return f"서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요. ({exc})", None
