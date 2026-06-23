"""Entity discovery, structural analysis, and cross-country comparison.

Provides pure-function analytics over normalized GLEIF entity DataFrames.
All functions accept DataFrames with the standard ENTITY_COLUMNS schema.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Entity Discovery — summary statistics
# ---------------------------------------------------------------------------

def entity_discovery_summary(df: pd.DataFrame) -> dict:
    """Return high-level discovery stats for a set of entities."""
    total = len(df)
    unique_leis = df["lei"].nunique()
    duplicates = total - unique_leis
    active = (df["entity_status"] == "ACTIVE").sum() if "entity_status" in df.columns else 0
    inactive = total - active

    return {
        "total_records": total,
        "unique_leis": unique_leis,
        "duplicates_removed": duplicates,
        "active_entities": int(active),
        "inactive_entities": int(inactive),
        "countries_legal": int(df["country_legal"].nunique()) if "country_legal" in df.columns else 0,
        "countries_hq": int(df["country_hq"].nunique()) if "country_hq" in df.columns else 0,
        "cities_legal": int(df["city_legal"].nunique()) if "city_legal" in df.columns else 0,
        "categories": int(df["category"].nunique()) if "category" in df.columns else 0,
        "legal_forms": int(df["legal_form"].nunique()) if "legal_form" in df.columns else 0,
        "earliest_registration": str(df["registered_at"].min()) if "registered_at" in df.columns else None,
        "latest_registration": str(df["registered_at"].max()) if "registered_at" in df.columns else None,
    }


# ---------------------------------------------------------------------------
# 2. Structural Analysis — geographic, legal form, category
# ---------------------------------------------------------------------------

def geographic_concentration(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """City-level concentration for legal address, sorted by count.

    City names are normalized to title case for consistent grouping.
    """
    if df.empty or "city_legal" not in df.columns:
        return pd.DataFrame(columns=["city_legal", "country_legal", "count", "pct"])

    tmp = df.copy()
    tmp["city_legal"] = tmp["city_legal"].str.strip().str.title()

    grouped = (
        tmp.groupby(["city_legal", "country_legal"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    grouped["pct"] = (grouped["count"] / len(df) * 100).round(1)
    return grouped


def hq_vs_legal_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """Entities where HQ country differs from legal address country."""
    if df.empty:
        return pd.DataFrame(columns=["lei", "legal_name", "country_legal", "country_hq"])

    mask = (
        df["country_legal"].notna()
        & df["country_hq"].notna()
        & (df["country_legal"] != df["country_hq"])
    )
    return df.loc[mask, ["lei", "legal_name", "country_legal", "country_hq"]].reset_index(drop=True)


def legal_form_distribution(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Distribution of legal form codes."""
    if df.empty or "legal_form" not in df.columns:
        return pd.DataFrame(columns=["legal_form", "count", "pct"])

    grouped = (
        df["legal_form"]
        .fillna("UNKNOWN")
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    grouped.columns = ["legal_form", "count"]
    grouped["pct"] = (grouped["count"] / len(df) * 100).round(1)
    return grouped


def category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of entity categories (GENERAL, FUND, BRANCH, etc.)."""
    if df.empty or "category" not in df.columns:
        return pd.DataFrame(columns=["category", "count", "pct"])

    grouped = (
        df["category"]
        .fillna("UNKNOWN")
        .value_counts()
        .reset_index()
    )
    grouped.columns = ["category", "count"]
    grouped["pct"] = (grouped["count"] / len(df) * 100).round(1)
    return grouped


def entity_status_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of entity statuses (ACTIVE, INACTIVE, etc.)."""
    if df.empty or "entity_status" not in df.columns:
        return pd.DataFrame(columns=["entity_status", "count", "pct"])

    grouped = (
        df["entity_status"]
        .fillna("UNKNOWN")
        .value_counts()
        .reset_index()
    )
    grouped.columns = ["entity_status", "count"]
    grouped["pct"] = (grouped["count"] / len(df) * 100).round(1)
    return grouped


def registration_timeline(df: pd.DataFrame, freq: str = "YS") -> pd.DataFrame:
    """Number of new LEI registrations over time, binned by frequency."""
    if df.empty or "registered_at" not in df.columns:
        return pd.DataFrame(columns=["period", "count"])

    dates = pd.to_datetime(df["registered_at"], errors="coerce").dropna()
    if dates.empty:
        return pd.DataFrame(columns=["period", "count"])

    grouped = dates.dt.to_period("Y").value_counts().sort_index().reset_index()
    grouped.columns = ["period", "count"]
    grouped["period"] = grouped["period"].astype(str)
    return grouped


# ---------------------------------------------------------------------------
# 3. Comparative Analysis — cross-country benchmarking
# ---------------------------------------------------------------------------

def country_structural_profile(df: pd.DataFrame, country_code: str) -> dict:
    """Build a structural profile dict for a single country's entities."""
    total = len(df)
    if total == 0:
        return {"country": country_code, "total_entities": 0}

    cats = category_distribution(df)
    cat_dict = dict(zip(cats["category"], cats["pct"]))

    top_city = geographic_concentration(df, top_n=1)
    top_city_name = top_city.iloc[0]["city_legal"] if not top_city.empty else None
    top_city_pct = float(top_city.iloc[0]["pct"]) if not top_city.empty else 0.0

    hq_mismatch = hq_vs_legal_mismatch(df)
    n_cities = df["city_legal"].nunique() if "city_legal" in df.columns else 0
    n_legal_forms = df["legal_form"].nunique() if "legal_form" in df.columns else 0

    return {
        "country": country_code,
        "total_entities": total,
        "active_pct": round((df["entity_status"] == "ACTIVE").mean() * 100, 1) if "entity_status" in df.columns else None,
        "n_cities": int(n_cities),
        "top_city": top_city_name,
        "top_city_concentration_pct": top_city_pct,
        "n_legal_forms": int(n_legal_forms),
        "hq_mismatch_count": len(hq_mismatch),
        "hq_mismatch_pct": round(len(hq_mismatch) / total * 100, 1),
        "pct_general": cat_dict.get("GENERAL", 0.0),
        "pct_fund": cat_dict.get("FUND", 0.0),
        "pct_branch": cat_dict.get("BRANCH", 0.0),
        "pct_sole_proprietor": cat_dict.get("SOLE_PROPRIETOR", 0.0),
    }


def compare_countries(country_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a comparison table across multiple countries.

    Parameters
    ----------
    country_frames : dict mapping ISO-2 country code -> entity DataFrame

    Returns
    -------
    DataFrame with one row per country and structural metrics as columns.
    """
    profiles = [
        country_structural_profile(df, code)
        for code, df in country_frames.items()
    ]
    return pd.DataFrame(profiles).sort_values("total_entities", ascending=False).reset_index(drop=True)


def full_structural_report(df: pd.DataFrame, country_code: str = "MY") -> dict:
    """Run all structural analyses and return as a single dict of DataFrames/dicts."""
    return {
        "discovery": entity_discovery_summary(df),
        "geographic_concentration": geographic_concentration(df),
        "hq_vs_legal_mismatch": hq_vs_legal_mismatch(df),
        "legal_form_distribution": legal_form_distribution(df),
        "category_distribution": category_distribution(df),
        "entity_status_distribution": entity_status_distribution(df),
        "registration_timeline": registration_timeline(df),
        "profile": country_structural_profile(df, country_code),
    }
