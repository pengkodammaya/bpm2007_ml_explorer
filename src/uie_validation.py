from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


BROAD_BUCKETS = {"EUROPE", "ASIA_OTHER", "MIDEAST_AFRICA", "AMERICAS_OTHER", "OFFSHORE", "OTHER"}

GROUND_TRUTH_COLUMN_CANDIDATES = {
    "lei": ["lei", "entity_lei", "LEI", "Entity LEI"],
    "ground_truth_uie_country": [
        "ground_truth_uie_country",
        "true_uie_country",
        "uie_country",
        "ultimate_investor_economy",
        "ultimate_investor_country",
        "ultimate_parent_country",
    ],
    "ground_truth_uie_name": [
        "ground_truth_uie_name",
        "true_uie_name",
        "uie_name",
        "ultimate_investor_name",
        "ultimate_parent_name",
    ],
}


def load_validation_table(path: str | Path) -> pd.DataFrame:
    """Load a validation input from parquet, CSV, or Excel."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError(f"Unsupported validation input format: {p.suffix}")


def _first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    column_set = set(columns)
    lower_map = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate in column_set:
            return candidate
        lower = candidate.lower()
        if lower in lower_map:
            return lower_map[lower]
    return None


def standardize_ground_truth_columns(ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Map common ground-truth UIE column names to canonical names."""
    mapping: dict[str, str] = {}
    for canonical, candidates in GROUND_TRUTH_COLUMN_CANDIDATES.items():
        found = _first_existing(ground_truth.columns, candidates)
        if found is not None:
            mapping[found] = canonical

    required = {"lei", "ground_truth_uie_country"}
    missing = required - set(mapping.values())
    if missing:
        raise ValueError(
            "Ground-truth file is missing required columns: "
            + ", ".join(sorted(missing))
            + ". Provide at least LEI and ground-truth UIE country."
        )

    out = ground_truth.rename(columns=mapping).copy()
    if "ground_truth_uie_name" not in out.columns:
        out["ground_truth_uie_name"] = pd.NA
    return out


def _norm_string(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.upper()


def _top3_contains(row: pd.Series) -> bool:
    truth = row.get("ground_truth_uie_country_norm")
    top3 = row.get("top3_countries")
    if not truth or pd.isna(top3):
        return False
    values = [part.strip().upper() for part in str(top3).split(";") if part.strip()]
    return truth in values


def _error_type(row: pd.Series) -> str:
    if pd.isna(row.get("pipeline_uie_source")) and pd.isna(row.get("pipeline_uie_country")):
        return "no_pipeline_match"
    if not row.get("pipeline_uie_country_norm"):
        return "pipeline_unassigned"
    if bool(row.get("is_exact_country_match")):
        return "exact_match"
    if bool(row.get("is_top3_match")):
        return "top3_only"
    if row.get("pipeline_uie_country_norm") in BROAD_BUCKETS:
        return "broad_bucket_mismatch"
    return "country_mismatch"


def _review_recommendation(row: pd.Series) -> str:
    error_type = row.get("error_type")
    source = row.get("pipeline_uie_source")
    tier = str(row.get("pipeline_evidence_tier") or "")
    if error_type == "exact_match":
        return "accept_assignment"
    if error_type == "top3_only":
        return "calibrate_model_or_threshold"
    if error_type == "pipeline_unassigned":
        return "use_ground_truth_as_candidate_override"
    if source == "phase3_jurisdiction_model":
        return "review_model_fallback"
    if tier.startswith(("A_", "B_")):
        return "audit_high_tier_conflict"
    return "manual_review"


def validate_uie_assignments(
    ground_truth: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    country: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare ground-truth UIE labels against pipeline UIE assignments."""
    truth = standardize_ground_truth_columns(ground_truth)
    truth["lei"] = truth["lei"].map(_norm_string)
    truth["ground_truth_uie_country_norm"] = truth["ground_truth_uie_country"].map(_norm_string)
    truth = truth.dropna(subset=["lei", "ground_truth_uie_country_norm"]).drop_duplicates(
        subset=["lei"],
        keep="first",
    )

    pipeline_cols = [
        "lei",
        "legal_name",
        "uie_country",
        "uie_source",
        "evidence_tier",
        "uie_confidence",
        "top3_countries",
        "top3_probabilities",
    ]
    pipe = assignments[[c for c in pipeline_cols if c in assignments.columns]].copy()
    pipe["lei"] = pipe["lei"].map(_norm_string)
    pipe = pipe.dropna(subset=["lei"]).drop_duplicates(subset=["lei"], keep="first")
    pipe = pipe.rename(
        columns={
            "uie_country": "pipeline_uie_country",
            "uie_source": "pipeline_uie_source",
            "evidence_tier": "pipeline_evidence_tier",
            "uie_confidence": "pipeline_confidence",
        }
    )
    pipe["pipeline_uie_country_norm"] = pipe.get("pipeline_uie_country", pd.Series(index=pipe.index)).map(_norm_string)

    detail = truth.merge(pipe, on="lei", how="left")
    if country:
        detail.insert(0, "host_country", country.upper())
    detail["is_exact_country_match"] = (
        detail["ground_truth_uie_country_norm"] == detail["pipeline_uie_country_norm"]
    )
    detail["is_top3_match"] = detail.apply(_top3_contains, axis=1)
    detail["error_type"] = detail.apply(_error_type, axis=1)
    detail["review_recommendation"] = detail.apply(_review_recommendation, axis=1)

    preferred_cols = [
        "host_country",
        "lei",
        "legal_name",
        "ground_truth_uie_country",
        "ground_truth_uie_name",
        "pipeline_uie_country",
        "pipeline_uie_source",
        "pipeline_evidence_tier",
        "pipeline_confidence",
        "top3_countries",
        "top3_probabilities",
        "is_exact_country_match",
        "is_top3_match",
        "error_type",
        "review_recommendation",
    ]
    for col in preferred_cols:
        if col not in detail.columns:
            detail[col] = pd.NA
    detail = detail[preferred_cols]

    summary = build_uie_validation_summary(detail)
    confusion = build_uie_confusion(detail)
    return detail, summary, confusion


def build_uie_validation_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Create a compact long-form summary table."""
    total = len(detail)
    matched = int(detail["pipeline_uie_country"].notna().sum()) if total else 0
    exact = int(detail["is_exact_country_match"].fillna(False).sum()) if total else 0
    top3 = int(detail["is_top3_match"].fillna(False).sum()) if total else 0
    rows = [
        {"section": "overall", "metric": "ground_truth_rows", "value": total},
        {"section": "overall", "metric": "matched_pipeline_rows", "value": matched},
        {"section": "overall", "metric": "exact_country_matches", "value": exact},
        {"section": "overall", "metric": "exact_country_match_rate", "value": exact / total if total else 0.0},
        {"section": "overall", "metric": "top3_matches", "value": top3},
        {"section": "overall", "metric": "top3_match_rate", "value": top3 / total if total else 0.0},
    ]

    for col, section in [
        ("pipeline_uie_source", "by_source"),
        ("pipeline_evidence_tier", "by_evidence_tier"),
        ("error_type", "by_error_type"),
    ]:
        if col not in detail.columns:
            continue
        grouped = detail.groupby(col, dropna=False)
        for key, group in grouped:
            denom = len(group)
            exact_n = int(group["is_exact_country_match"].fillna(False).sum())
            rows.append({
                "section": section,
                "metric": str(key),
                "value": denom,
                "exact_match_rate": exact_n / denom if denom else 0.0,
            })
    return pd.DataFrame(rows)


def build_uie_confusion(detail: pd.DataFrame) -> pd.DataFrame:
    """Build a ground-truth-vs-pipeline country confusion table."""
    if detail.empty:
        return pd.DataFrame(columns=["ground_truth_uie_country", "pipeline_uie_country", "count"])
    confusion = (
        detail.assign(
            ground_truth_uie_country=detail["ground_truth_uie_country"].fillna("UNKNOWN"),
            pipeline_uie_country=detail["pipeline_uie_country"].fillna("UNASSIGNED"),
        )
        .groupby(["ground_truth_uie_country", "pipeline_uie_country"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "ground_truth_uie_country", "pipeline_uie_country"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    return confusion
