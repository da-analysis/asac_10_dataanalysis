import streamlit as st
import streamlit.components.v1 as components

BASE_URL = "https://dbc-b2983598-29d8.cloud.databricks.com/embed/dashboardsv3/01f1477ee2361f28a2da2e770290bfd7?o=7474645180304154"


def render():
    if st.button("← 홈으로", key="dashboard_home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("📊 분석 대시보드")

    product_name = st.text_input("상품명 필터", placeholder="예: 감자, 양파, 대파")

    url = BASE_URL
    if product_name.strip():
        url += f"&f_product_name={product_name.strip()}"

    components.iframe(url, height=600, scrolling=True)
