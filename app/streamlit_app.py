from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config import PROCESSED_DIR
from src.country_config import list_supported_countries
from src.io_helpers import load_df


st.set_page_config(page_title="BPM2007 ML Explorer", layout="wide")
st.title("BPM2007 ML Explorer")

countries = list_supported_countries()
country = st.sidebar.selectbox("Country", countries, index=countries.index("MY") if "MY" in countries else 0)
base = PROCESSED_DIR / country.lower()


@st.cache_data(show_spinner=False)
def _load_country_outputs(country_code: str) -> dict[str, pd.DataFrame]:
    out_base = PROCESSED_DIR / country_code.lower()
    outputs: dict[str, pd.DataFrame] = {}
    for name in [
        "uie_assignments",
        "investor_economy_summary",
        "uie_review_targets",
        "coverage_gap_scores",
        "di_graph_summary",
    ]:
        try:
            outputs[name] = load_df(out_base / name)
        except Exception:
            outputs[name] = pd.DataFrame()
    return outputs


outputs = _load_country_outputs(country)
uie = outputs["uie_assignments"]
investors = outputs["investor_economy_summary"]
review_targets = outputs["uie_review_targets"]
scores = outputs["coverage_gap_scores"]
graph = outputs["di_graph_summary"]

if uie.empty and scores.empty and graph.empty:
    st.warning(f"No processed outputs found for {country}.")
    st.stop()

top = st.columns(5)
top[0].metric("Domestic entities", f"{len(uie):,}" if not uie.empty else "n/a")
top[1].metric("Assigned UIE", f"{uie['uie_country'].notna().sum():,}" if not uie.empty else "n/a")
top[2].metric("Known UIE", f"{int(uie['is_known_uie'].sum()):,}" if not uie.empty and "is_known_uie" in uie else "n/a")
top[3].metric("Inferred UIE", f"{int(uie['is_inferred_uie'].sum()):,}" if not uie.empty and "is_inferred_uie" in uie else "n/a")
top[4].metric("Review targets", f"{len(review_targets):,}" if not review_targets.empty else "n/a")

tab_review, tab_uie, tab_investors, tab_scores, tab_graph = st.tabs([
    "Review Targets",
    "UIE Assignments",
    "Investor Economy",
    "Coverage Priority",
    "Graph Summary",
])

with tab_review:
    if review_targets.empty:
        st.warning("No UIE review target table found.")
    else:
        reason_options = sorted(review_targets["review_reason"].dropna().unique())
        selected_reasons = st.multiselect("Review reason", reason_options, default=reason_options)
        status_options = sorted(review_targets["review_status"].dropna().unique()) if "review_status" in review_targets else []
        selected_statuses = st.multiselect("Review status", status_options, default=status_options)
        review_view = (
            review_targets[review_targets["review_reason"].isin(selected_reasons)]
            if selected_reasons else review_targets
        )
        if selected_statuses and "review_status" in review_view:
            review_view = review_view[review_view["review_status"].isin(selected_statuses)]
        cols = [
            "lei", "legal_name", "review_priority_score", "review_reason",
            "review_status", "entity_type", "review_note",
            "coverage_gap_priority_score", "reason_flags",
            "uie_country", "uie_source", "evidence_tier", "uie_confidence",
            "top3_countries", "top3_probabilities",
            "ctos_registered_malaysia", "ctos_name",
        ]
        st.dataframe(review_view[[c for c in cols if c in review_view.columns]], use_container_width=True)
        fig = px.bar(
            review_view.head(25),
            x="legal_name",
            y="review_priority_score",
            color="review_reason",
            title=f"{country}: top investor-economy review targets",
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

with tab_uie:
    if uie.empty:
        st.warning("No UIE assignment table found.")
    else:
        tier_filter = st.multiselect(
            "Evidence tier",
            sorted(uie["evidence_tier"].dropna().unique()),
            default=sorted(uie["evidence_tier"].dropna().unique()),
        )
        view = uie[uie["evidence_tier"].isin(tier_filter)] if tier_filter else uie
        cols = [
            "lei", "legal_name", "uie_country", "uie_source", "evidence_tier",
            "uie_confidence", "direct_parent_name", "ultimate_parent_name",
            "predicted_parent_country", "prediction_probability",
            "ctos_registered_malaysia", "ctos_name", "ctos_registration_no",
            "ctos_match_score", "override_reason", "source_url", "source_note",
        ]
        st.dataframe(view[[c for c in cols if c in view.columns]], use_container_width=True)

with tab_investors:
    if investors.empty:
        st.warning("No investor economy summary found.")
    else:
        st.dataframe(investors, use_container_width=True)
        fig = px.bar(
            investors.head(25),
            x="uie_country",
            y="total_entities",
            color="known_uie_entities",
            title=f"{country}: top investor economies by linked entities",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_scores:
    if scores.empty:
        st.warning("No coverage priority table found.")
    else:
        score_col = "coverage_gap_priority_score" if "coverage_gap_priority_score" in scores.columns else "coverage_gap_score"
        cols = [
            "lei", "legal_name", "country", "city", score_col,
            "reason_flags", "has_parent_link", "has_ultimate_link",
            "foreign_parent", "has_inferred_parent", "in_address_cluster",
            "is_non_consolidating",
        ]
        st.dataframe(scores[[c for c in cols if c in scores.columns]].head(200), use_container_width=True)
        fig = px.bar(
            scores.head(25),
            x="legal_name",
            y=score_col,
            title=f"{country}: top coverage-gap priority candidates",
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

with tab_graph:
    if graph.empty:
        st.warning("No graph summary found.")
    else:
        st.dataframe(graph.head(200), use_container_width=True)
        if "out_degree" in graph.columns:
            fig = px.histogram(graph, x="out_degree", nbins=30, title=f"{country}: ownership link distribution")
            st.plotly_chart(fig, use_container_width=True)
