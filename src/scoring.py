from __future__ import annotations

import numpy as np
import pandas as pd


def compute_coverage_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    defaults = {
        "has_parent_link": 0,
        "has_ultimate_link": 0,
        "appears_in_edgar": 0,
        "foreign_parent": 0,
        "size_proxy": 0.0,
    }
    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = val
        out[col] = out[col].fillna(val)

    for col in ["has_parent_link", "has_ultimate_link", "appears_in_edgar", "foreign_parent"]:
        out[col] = out[col].astype(int)

    out["size_scaled"] = np.log1p(out["size_proxy"].astype(float))
    if out.empty:
        out["coverage_gap_score"] = pd.Series(dtype=float)
        out["reason_flags"] = pd.Series(dtype=str)
        return out

    if out["size_scaled"].nunique(dropna=True) <= 1:
        out["size_scaled"] = 0.0
    else:
        denom = max(out["size_scaled"].max(), 1.0)
        out["size_scaled"] = out["size_scaled"] / denom

    out["coverage_gap_score"] = (
        0.30 * out["size_scaled"]
        + 0.20 * out["appears_in_edgar"]
        + 0.20 * out["foreign_parent"]
        + 0.15 * (1 - out["has_parent_link"])
        + 0.15 * (1 - out["has_ultimate_link"])
    )

    def explain(row: pd.Series) -> str:
        reasons = []
        if row["size_scaled"] > 0.6:
            reasons.append("large_network_presence")
        if row["appears_in_edgar"] == 1:
            reasons.append("us_filing_presence")
        if row["foreign_parent"] == 1:
            reasons.append("foreign_parent_signal")
        if row["has_parent_link"] == 0:
            reasons.append("missing_direct_parent")
        if row["has_ultimate_link"] == 0:
            reasons.append("missing_ultimate_parent")
        return ";".join(reasons)

    out["reason_flags"] = out.apply(explain, axis=1)
    return out.sort_values("coverage_gap_score", ascending=False)
