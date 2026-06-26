from __future__ import annotations

import numpy as np
import pandas as pd


def compute_coverage_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a coverage-gap priority score from observable signals.

    The result is a triage ranking score, not a calibrated probability of
    missing coverage or a final Ultimate Investor Economy assignment.

    Weight formula (sums to 1.0):
    - 0.25 * size_scaled           — network prominence
    - 0.15 * appears_in_edgar      — US filing presence
    - 0.15 * foreign_parent        — confirmed foreign parent
    - 0.10 * has_inferred_parent   — Phase 1 name-matched parent
    - 0.05 * in_address_cluster    — Phase 2 address co-location
    - 0.05 * is_non_consolidating  — Phase 0.5 confirmed subsidiary
    - 0.15 * (1 - has_parent_link) — missing direct parent data
    - 0.10 * (1 - has_ultimate_link) — missing ultimate parent data
    """
    out = df.copy()

    defaults = {
        "has_parent_link": 0,
        "has_ultimate_link": 0,
        "appears_in_edgar": 0,
        "foreign_parent": 0,
        "size_proxy": 0.0,
        "has_inferred_parent": 0,
        "in_address_cluster": 0,
        "is_non_consolidating": 0,
    }
    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = val
        out[col] = out[col].fillna(val)

    int_cols = [
        "has_parent_link", "has_ultimate_link", "appears_in_edgar",
        "foreign_parent", "has_inferred_parent", "in_address_cluster",
        "is_non_consolidating",
    ]
    for col in int_cols:
        out[col] = out[col].astype(int)

    out["size_scaled"] = np.log1p(out["size_proxy"].astype(float))
    if out.empty:
        out["coverage_gap_score"] = pd.Series(dtype=float)
        out["coverage_gap_priority_score"] = pd.Series(dtype=float)
        out["reason_flags"] = pd.Series(dtype=str)
        return out

    if out["size_scaled"].nunique(dropna=True) <= 1:
        out["size_scaled"] = 0.0
    else:
        denom = max(out["size_scaled"].max(), 1.0)
        out["size_scaled"] = out["size_scaled"] / denom

    out["coverage_gap_score"] = (
        0.25 * out["size_scaled"]
        + 0.15 * out["appears_in_edgar"]
        + 0.15 * out["foreign_parent"]
        + 0.10 * out["has_inferred_parent"]
        + 0.05 * out["in_address_cluster"]
        + 0.05 * out["is_non_consolidating"]
        + 0.15 * (1 - out["has_parent_link"])
        + 0.10 * (1 - out["has_ultimate_link"])
    )
    out["coverage_gap_priority_score"] = out["coverage_gap_score"]

    def explain(row: pd.Series) -> str:
        reasons = []
        if row["size_scaled"] > 0.6:
            reasons.append("large_network_presence")
        if row["appears_in_edgar"] == 1:
            reasons.append("us_filing_presence")
        if row["foreign_parent"] == 1:
            reasons.append("foreign_parent_signal")
        if row["has_inferred_parent"] == 1:
            reasons.append("inferred_parent_match")
        if row["in_address_cluster"] == 1:
            reasons.append("shared_address_cluster")
        if row["is_non_consolidating"] == 1:
            reasons.append("non_consolidating_entity")
        if row["has_parent_link"] == 0:
            reasons.append("missing_direct_parent")
        if row["has_ultimate_link"] == 0:
            reasons.append("missing_ultimate_parent")
        return ";".join(reasons)

    out["reason_flags"] = out.apply(explain, axis=1)
    return out.sort_values("coverage_gap_score", ascending=False)
