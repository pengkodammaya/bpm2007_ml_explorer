from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px
from config import PROCESSED_DIR
from src.io_helpers import load_df

st.set_page_config(page_title="Phase 1 ESIE", layout="wide")
st.title("Malaysia-linked External Sector Discovery Prototype")

tab1, tab2 = st.tabs(["DI Graph Summary", "Coverage Gap Scores"])

with tab1:
    try:
        df = load_df(PROCESSED_DIR / "di_graph_summary")
        st.dataframe(df.head(100), use_container_width=True)
        fig = px.histogram(df, x="out_degree", nbins=20, title="Ownership link distribution")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"No processed graph summary found yet: {e}")

with tab2:
    try:
        df = load_df(PROCESSED_DIR / "coverage_gap_scores")
        st.dataframe(df.head(100), use_container_width=True)
        fig = px.bar(df.head(20), x="legal_name", y="coverage_gap_score", title="Top coverage-gap candidates")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"No coverage scores found yet: {e}")
