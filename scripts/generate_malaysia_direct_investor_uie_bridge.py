from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports" / "directional_uie"

ASEAN_COUNTRIES = ["BN", "ID", "KH", "LA", "MM", "MY", "PH", "SG", "TH", "TL", "VN"]
UNKNOWN_UIE = {"", "UNKNOWN", "UNASSIGNED", "NAN", "NONE"}

ROW_COLUMNS = [
    "host_country",
    "lei",
    "legal_name",
    "entity_country",
    "direct_parent_country",
    "direct_parent_name",
    "direct_parent_lei",
    "ultimate_parent_country",
    "ultimate_parent_name",
    "ultimate_parent_lei",
    "chain_parent_country",
    "chain_parent_name",
    "chain_parent_lei",
    "uie_country",
    "uie_name",
    "uie_lei",
    "uie_source",
    "evidence_tier",
    "evidence_bucket",
    "uie_confidence",
    "is_known_uie",
    "is_inferred_uie",
    "ctos_registered_malaysia",
]


def evidence_bucket(value: object) -> str:
    tier = "" if pd.isna(value) else str(value)
    if tier.startswith("A"):
        return "A_hard_or_verified"
    if tier.startswith("B"):
        return "B_parent_or_exception"
    if tier.startswith("C"):
        return "C_name_inferred"
    if tier.startswith("D"):
        return "D_address_inferred"
    if tier.startswith("E"):
        return "E_model_only"
    if tier.startswith("U"):
        return "U_unassigned"
    return "Z_other"


def norm_country(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def valid_uie(value: object) -> bool:
    return norm_country(value) not in UNKNOWN_UIE


def read_assignments(country: str = "MY") -> pd.DataFrame:
    path = DATA / "processed" / country.lower() / "uie_assignments.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing UIE assignment file: {path}")
    df = pd.read_parquet(path)
    if "host_country" not in df.columns:
        df["host_country"] = country
    df["host_country"] = df["host_country"].fillna(country).map(norm_country)
    return df


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def build_bridge(assignments: pd.DataFrame, host_country: str = "MY") -> pd.DataFrame:
    df = assignments[assignments["host_country"].eq(host_country)].copy()
    df = ensure_columns(df, ROW_COLUMNS)
    for col in [
        "entity_country",
        "direct_parent_country",
        "ultimate_parent_country",
        "chain_parent_country",
        "uie_country",
    ]:
        df[col] = df[col].map(norm_country)
    df["evidence_bucket"] = df["evidence_tier"].map(evidence_bucket)

    bridge = df[
        df["entity_country"].eq(host_country)
        & df["direct_parent_country"].isin(ASEAN_COUNTRIES)
        & df["direct_parent_country"].ne(host_country)
        & df["uie_country"].map(valid_uie)
    ].copy()
    bridge["direct_parent_is_uie"] = bridge["direct_parent_country"].eq(bridge["uie_country"])
    bridge["pass_through_to_non_direct_uie"] = ~bridge["direct_parent_is_uie"]
    bridge["uie_is_asean"] = bridge["uie_country"].isin(ASEAN_COUNTRIES)
    bridge["uie_is_non_asean"] = ~bridge["uie_is_asean"]
    bridge["uie_is_broad_bucket"] = bridge["uie_country"].isin(
        ["EUROPE", "ASIA_OTHER", "AMERICAS_OTHER", "MIDEAST_AFRICA", "OFFSHORE", "OTHER"]
    )

    output_columns = ROW_COLUMNS + [
        "direct_parent_is_uie",
        "pass_through_to_non_direct_uie",
        "uie_is_asean",
        "uie_is_non_asean",
        "uie_is_broad_bucket",
    ]
    return bridge[output_columns].sort_values(
        ["direct_parent_country", "uie_country", "evidence_bucket", "legal_name", "lei"],
        kind="stable",
    )


def summarize_counts(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols)
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            lei_count=("lei", "nunique"),
            known_uie_count=("is_known_uie", "sum"),
            inferred_uie_count=("is_inferred_uie", "sum"),
            model_only_count=("evidence_bucket", lambda s: int((s == "E_model_only").sum())),
            pass_through_count=("pass_through_to_non_direct_uie", "sum"),
            non_asean_uie_count=("uie_is_non_asean", "sum"),
            avg_uie_confidence=("uie_confidence", "mean"),
        )
        .reset_index()
    )
    out["avg_uie_confidence"] = out["avg_uie_confidence"].round(4)
    return out.sort_values(["lei_count"] + group_cols, ascending=[False] + [True] * len(group_cols))


def write_pair(df: pd.DataFrame, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_base.with_suffix(".csv"), index=False)
    df.to_parquet(path_base.with_suffix(".parquet"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Malaysia inward direct-investor-to-UIE bridge tables."
    )
    parser.add_argument("--as-of", default=date.today().isoformat(), help="Output date stamp, YYYY-MM-DD")
    parser.add_argument("--host-country", default="MY", help="Host country code, default MY")
    parser.add_argument("--top-n", type=int, default=5, help="Top direct investor countries to highlight")
    args = parser.parse_args()

    stamp = args.as_of
    host = args.host_country.upper()
    assignments = read_assignments(host)
    bridge = build_bridge(assignments, host_country=host)

    direct_summary = summarize_counts(
        bridge,
        ["host_country", "direct_parent_country"],
    )
    direct_summary["pass_through_share"] = (
        direct_summary["pass_through_count"] / direct_summary["lei_count"]
    ).round(4)
    direct_summary["non_asean_uie_share"] = (
        direct_summary["non_asean_uie_count"] / direct_summary["lei_count"]
    ).round(4)

    top_direct = direct_summary.head(args.top_n).copy()
    top_direct_countries = set(top_direct["direct_parent_country"].dropna().astype(str))
    top_bridge = bridge[bridge["direct_parent_country"].isin(top_direct_countries)].copy()

    uie_summary = summarize_counts(
        bridge,
        ["host_country", "direct_parent_country", "uie_country", "evidence_bucket"],
    )
    top_uie_summary = summarize_counts(
        top_bridge,
        ["host_country", "direct_parent_country", "uie_country", "evidence_bucket"],
    )

    out_prefix = REPORTS / f"malaysia_inward_asean_direct_investor_uie_bridge_{stamp}"
    write_pair(bridge, out_prefix)
    write_pair(direct_summary, REPORTS / f"malaysia_inward_asean_direct_investors_{stamp}")
    write_pair(uie_summary, REPORTS / f"malaysia_inward_asean_direct_investor_uie_summary_{stamp}")
    write_pair(top_bridge, REPORTS / f"malaysia_inward_top{args.top_n}_asean_direct_investor_uie_bridge_{stamp}")
    write_pair(top_uie_summary, REPORTS / f"malaysia_inward_top{args.top_n}_asean_direct_investor_uie_summary_{stamp}")

    print(f"[OK] Wrote Malaysia direct-investor UIE bridge outputs for {stamp}")
    print(f"[OK] Row-level ASEAN direct-investor bridge rows: {len(bridge):,}")
    print("[OK] Top direct investor countries:")
    if top_direct.empty:
        print("  none")
    else:
        print(top_direct.to_string(index=False))


if __name__ == "__main__":
    main()
