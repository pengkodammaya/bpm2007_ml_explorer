from __future__ import annotations

import argparse
import warnings
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

ASEAN_COUNTRIES = ["MY", "SG", "TH", "ID", "PH", "VN", "KH", "BN", "LA", "MM", "TL"]
UNKNOWN_UIE = {"", "UNKNOWN", "UNASSIGNED"}

ASEAN_NAMES = {
    "Malaysia": "MY",
    "Singapore": "SG",
    "Thailand": "TH",
    "Indonesia": "ID",
    "Philippines": "PH",
    "Vietnam": "VN",
    "Cambodia": "KH",
    "Brunei Darussalam": "BN",
    "Lao People's Democratic Republic": "LA",
    "Myanmar": "MM",
    "Timor-Leste, Democratic Republic of": "TL",
}

COUNTERPART_MAP = {
    "United States": "US",
    "United Kingdom": "GB",
    "Germany": "DE",
    "Singapore": "SG",
    "Malaysia": "MY",
    "Japan": "JP",
    "China, People's Republic of": "CN",
    "Hong Kong Special Administrative Region, People's Republic of China": "HK",
    "France": "FR",
    "Netherlands": "NL",
    "Switzerland": "CH",
    "Australia": "AU",
    "Canada": "CA",
    "India": "IN",
    "Korea, Republic of": "KR",
    "Cayman Islands": "KY",
    "British Virgin Islands": "VG",
    "Bermuda": "BM",
    "Luxembourg": "LU",
    "Ireland": "IE",
    "Sweden": "SE",
    "Denmark": "DK",
    "Norway": "NO",
    "Belgium": "BE",
    "Italy": "IT",
    "Spain": "ES",
    "Austria": "AT",
    "Finland": "FI",
    "Portugal": "PT",
    "New Zealand": "NZ",
    "Thailand": "TH",
    "Indonesia": "ID",
    "Philippines": "PH",
    "Vietnam": "VN",
    "Cambodia": "KH",
    "Brunei Darussalam": "BN",
    "Lao People's Democratic Republic": "LA",
    "Myanmar": "MM",
    "Timor-Leste, Democratic Republic of": "TL",
    "United Arab Emirates": "AE",
    "Saudi Arabia": "SA",
    "Jersey": "JE",
    "Mauritius": "MU",
    "Taiwan Province of China": "TW",
    "Brazil": "BR",
    "Guernsey": "GG",
    "Liechtenstein": "LI",
    "Turkey": "TR",
    "Russia": "RU",
    "South Africa": "ZA",
}

ASSIGNMENT_COLUMNS = [
    "direction",
    "host_country",
    "source_country",
    "destination_country",
    "lei",
    "legal_name",
    "entity_country",
    "uie_country",
    "uie_name",
    "uie_lei",
    "uie_source",
    "evidence_tier",
    "evidence_bucket",
    "uie_confidence",
    "is_known_uie",
    "is_inferred_uie",
    "direct_parent_country",
    "ultimate_parent_country",
    "chain_parent_country",
    "ctos_registered_malaysia",
]


def read_assignment(country: str) -> pd.DataFrame:
    path = DATA / "processed" / country.lower() / "uie_assignments.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["host_country"] = df.get("host_country", country).fillna(country).astype(str).str.upper()
    return df


def load_assignments(countries: list[str]) -> pd.DataFrame:
    frames = [read_assignment(country) for country in countries]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame(columns=ASSIGNMENT_COLUMNS)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
            category=FutureWarning,
        )
        out = pd.concat(frames, ignore_index=True, sort=False)
    out["host_country"] = out["host_country"].astype(str).str.upper()
    out["uie_country"] = out["uie_country"].fillna("").astype(str).str.upper()
    out["entity_country"] = out.get("entity_country", out["host_country"]).fillna(out["host_country"])
    out["evidence_bucket"] = out["evidence_tier"].map(evidence_bucket)
    return out


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


def valid_uie(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper().map(lambda x: x not in UNKNOWN_UIE)


def as_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ASSIGNMENT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[ASSIGNMENT_COLUMNS].copy()


def build_directional_assignments(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if assignments.empty:
        empty = pd.DataFrame(columns=ASSIGNMENT_COLUMNS)
        return empty, empty

    inward = assignments.copy()
    inward["direction"] = "IN"
    inward["source_country"] = inward["uie_country"].where(valid_uie(inward["uie_country"]), pd.NA)
    inward["destination_country"] = inward["host_country"]
    inward["inward_class"] = "unknown_or_unassigned"
    inward.loc[
        valid_uie(inward["uie_country"]) & inward["uie_country"].ne(inward["host_country"]),
        "inward_class",
    ] = "foreign_uie"
    inward.loc[
        valid_uie(inward["uie_country"]) & inward["uie_country"].eq(inward["host_country"]),
        "inward_class",
    ] = "domestic_uie"

    outward = assignments[
        valid_uie(assignments["uie_country"])
        & assignments["uie_country"].isin(ASEAN_COUNTRIES)
        & assignments["uie_country"].ne(assignments["host_country"])
    ].copy()
    outward["direction"] = "OUT"
    outward["source_country"] = outward["uie_country"]
    outward["destination_country"] = outward["host_country"]
    outward["outward_scope"] = "asean_observed_hosts"

    return as_output_columns(inward), as_output_columns(outward)


def summary_counts(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + [
            "lei_count",
            "known_uie_count",
            "inferred_uie_count",
            "model_only_count",
            "avg_uie_confidence",
        ])
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            lei_count=("lei", "count"),
            known_uie_count=("is_known_uie", "sum"),
            inferred_uie_count=("is_inferred_uie", "sum"),
            model_only_count=("evidence_bucket", lambda s: int((s == "E_model_only").sum())),
            avg_uie_confidence=("uie_confidence", "mean"),
        )
        .reset_index()
    )
    out["avg_uie_confidence"] = out["avg_uie_confidence"].round(4)
    return out.sort_values(["lei_count"] + group_cols, ascending=[False] + [True] * len(group_cols))


def cdis_positions(direction: str, year: str = "2023") -> pd.DataFrame:
    path = DATA / "dataset_2026-04-22T22_38_34.423483046Z_DEFAULT_INTEGRATION_IMF.STA_DIP_12.0.1.csv"
    if not path.exists():
        return pd.DataFrame(columns=["host", "parent", "fdi_usd_mn"])
    usecols = [
        "COUNTRY",
        "COUNTERPART_COUNTRY",
        "DI_DIRECTION",
        "DI_ENTITY",
        "INSTR_ASSET",
        "ACCOUNTING_ENTRY",
        year,
    ]
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    entry = "Net (liabilities less assets)" if direction == "Inward" else "Net (assets less liabilities)"
    mask = (
        df["COUNTRY"].isin(ASEAN_NAMES)
        & df["DI_DIRECTION"].eq(direction)
        & df["DI_ENTITY"].eq("All entities")
        & df["INSTR_ASSET"].eq("All financial instruments")
        & df["ACCOUNTING_ENTRY"].eq(entry)
    )
    sub = df.loc[mask].copy()
    sub["host"] = sub["COUNTRY"].map(ASEAN_NAMES)
    sub["parent"] = sub["COUNTERPART_COUNTRY"].map(COUNTERPART_MAP)
    sub[year] = pd.to_numeric(sub[year], errors="coerce")
    sub = sub.dropna(subset=["host", "parent", year])
    sub = sub[sub[year] > 0]
    return (
        sub.groupby(["host", "parent"], as_index=False)[year]
        .sum()
        .rename(columns={year: "fdi_usd_mn"})
    )


def build_inward_density(inward: pd.DataFrame) -> pd.DataFrame:
    foreign = inward[
        valid_uie(inward["uie_country"]) & inward["uie_country"].ne(inward["host_country"])
    ]
    counts = summary_counts(foreign, ["host_country"]).rename(columns={"lei_count": "foreign_uie_lei_count"})
    cdis_in = cdis_positions("Inward")
    totals = cdis_in.groupby("host", as_index=False)["fdi_usd_mn"].sum()
    out = counts.merge(totals, left_on="host_country", right_on="host", how="left")
    out["inward_fdi_usd_bn"] = out["fdi_usd_mn"] / 1000
    out["inward_lei_density"] = out["foreign_uie_lei_count"] / out["inward_fdi_usd_bn"]
    return out.drop(columns=["host", "fdi_usd_mn"], errors="ignore")


def build_outward_density(outward: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_counts = summary_counts(outward, ["source_country", "destination_country"]).rename(
        columns={"lei_count": "outward_observed_lei_count"}
    )
    cdis_out = cdis_positions("Outward").rename(
        columns={"host": "source_country", "parent": "destination_country"}
    )
    bilateral = pair_counts.merge(cdis_out, on=["source_country", "destination_country"], how="left")
    bilateral["outward_fdi_usd_bn"] = bilateral["fdi_usd_mn"] / 1000
    bilateral["outward_lei_density"] = (
        bilateral["outward_observed_lei_count"] / bilateral["outward_fdi_usd_bn"]
    )
    bilateral = bilateral.drop(columns=["fdi_usd_mn"], errors="ignore")

    source = (
        bilateral.groupby("source_country", as_index=False)
        .agg(
            destination_count=("destination_country", "nunique"),
            outward_observed_lei_count=("outward_observed_lei_count", "sum"),
            outward_fdi_usd_bn=("outward_fdi_usd_bn", "sum"),
            known_uie_count=("known_uie_count", "sum"),
            inferred_uie_count=("inferred_uie_count", "sum"),
            model_only_count=("model_only_count", "sum"),
        )
        .sort_values("outward_observed_lei_count", ascending=False)
    )
    source["asean_observed_outward_lei_density"] = (
        source["outward_observed_lei_count"] / source["outward_fdi_usd_bn"]
    )
    return bilateral, source


def write_outputs(frames: dict[str, pd.DataFrame], output_dir: Path, stamp: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(output_dir / f"{name}_{stamp}.csv", index=False)
        df.to_parquet(output_dir / f"{name}_{stamp}.parquet", index=False)


def build_results(countries: list[str]) -> dict[str, pd.DataFrame]:
    assignments = load_assignments(countries)
    inward, outward = build_directional_assignments(assignments)
    inward_foreign = inward[
        valid_uie(inward["uie_country"]) & inward["uie_country"].ne(inward["host_country"])
    ].copy()

    inward_summary = summary_counts(inward_foreign, ["host_country", "uie_country", "evidence_bucket"])
    outward_summary = summary_counts(outward, ["source_country", "destination_country", "evidence_bucket"])
    inward_density = build_inward_density(inward)
    outward_bilateral_density, outward_source_density = build_outward_density(outward)

    malaysia_inward = inward[inward["host_country"].eq("MY")].copy()
    malaysia_inward_foreign = inward_foreign[inward_foreign["host_country"].eq("MY")].copy()
    malaysia_outward = outward[outward["source_country"].eq("MY")].copy()
    malaysia_inward_summary = summary_counts(
        malaysia_inward_foreign,
        ["host_country", "uie_country", "evidence_bucket"],
    )
    malaysia_outward_summary = summary_counts(
        malaysia_outward,
        ["source_country", "destination_country", "evidence_bucket"],
    )

    frames = {
        "asean_inward_assignments": inward,
        "asean_inward_foreign_assignments": inward_foreign,
        "asean_inward_summary": inward_summary,
        "asean_inward_density": inward_density,
        "asean_outward_assignments": outward,
        "asean_outward_summary": outward_summary,
        "asean_outward_bilateral_density": outward_bilateral_density,
        "asean_outward_source_density": outward_source_density,
        "malaysia_inward_assignments": malaysia_inward,
        "malaysia_inward_foreign_assignments": malaysia_inward_foreign,
        "malaysia_inward_summary": malaysia_inward_summary,
        "malaysia_outward_assignments": malaysia_outward,
        "malaysia_outward_summary": malaysia_outward_summary,
        "malaysia_outward_bilateral_density": outward_bilateral_density[
            outward_bilateral_density["source_country"].eq("MY")
        ].copy(),
    }

    for country in countries:
        prefix = country.lower()
        country_inward = inward[inward["host_country"].eq(country)].copy()
        country_inward_foreign = inward_foreign[inward_foreign["host_country"].eq(country)].copy()
        country_outward = outward[outward["source_country"].eq(country)].copy()
        frames[f"{prefix}_inward_assignments"] = country_inward
        frames[f"{prefix}_inward_foreign_assignments"] = country_inward_foreign
        frames[f"{prefix}_inward_summary"] = summary_counts(
            country_inward_foreign,
            ["host_country", "uie_country", "evidence_bucket"],
        )
        frames[f"{prefix}_inward_density"] = inward_density[
            inward_density["host_country"].eq(country)
        ].copy()
        frames[f"{prefix}_outward_assignments"] = country_outward
        frames[f"{prefix}_outward_summary"] = summary_counts(
            country_outward,
            ["source_country", "destination_country", "evidence_bucket"],
        )
        frames[f"{prefix}_outward_bilateral_density"] = outward_bilateral_density[
            outward_bilateral_density["source_country"].eq(country)
        ].copy()

    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate directional IN/OUT UIE result tables.")
    parser.add_argument("--countries", default=",".join(ASEAN_COUNTRIES))
    parser.add_argument("--output-dir", default=str(REPORTS / "directional_uie"))
    parser.add_argument("--stamp", default=date.today().isoformat())
    args = parser.parse_args()

    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    frames = build_results(countries)
    output_dir = Path(args.output_dir)
    write_outputs(frames, output_dir, args.stamp)

    print(f"Wrote {len(frames)} directional UIE tables to {output_dir}")
    for name, df in frames.items():
        print(f"{name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
