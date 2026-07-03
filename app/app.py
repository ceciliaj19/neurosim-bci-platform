"""NeuroLab — unified Streamlit application entry point.

Run with:
    streamlit run app/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="NeuroLab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 0 4px;">
            <span style="font-size:22px; font-weight:700;">🧠 NeuroLab</span><br>
            <span style="font-size:12px; opacity:0.45;">v0.1.0 · Open-source BCI platform</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

from pages.home import render  # noqa: E402

render()
