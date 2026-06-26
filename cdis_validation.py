"""CDIS bilateral validation: GLEIF entity counts vs IMF CDIS FDI positions."""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

CDIS_FILE = "data/dataset_2026-04-22T22_38_34.423483046Z_DEFAULT_INTEGRATION_IMF.STA_DIP_12.0.1.csv"

ASEAN_NAMES = {
    "Malaysia": "MY", "Singapore": "SG", "Thailand": "TH", "Indonesia": "ID",
    "Philippines": "PH", "Vietnam": "VN", "Cambodia": "KH",
    "Brunei Darussalam": "BN", "Lao People's Democratic Republic": "LA",
    "Myanmar": "MM", "Timor-Leste, Democratic Republic of": "TL",
}

CP_MAP = {
    "United States": "US", "United Kingdom": "GB", "Germany": "DE",
    "Singapore": "SG", "Malaysia": "MY", "Japan": "JP",
    "China, People's Republic of": "CN",
    "Hong Kong Special Administrative Region, People's Republic of China": "HK",
    "France": "FR", "Netherlands": "NL", "Switzerland": "CH",
    "Australia": "AU", "Canada": "CA", "India": "IN",
    "Korea, Republic of": "KR", "Cayman Islands": "KY",
    "British Virgin Islands": "VG", "Bermuda": "BM", "Luxembourg": "LU",
    "Ireland": "IE", "Sweden": "SE", "Denmark": "DK", "Norway": "NO",
    "Belgium": "BE", "Italy": "IT", "Spain": "ES", "Austria": "AT",
    "Finland": "FI", "Portugal": "PT", "New Zealand": "NZ",
    "Thailand": "TH", "Indonesia": "ID", "Philippines": "PH",
    "Vietnam": "VN", "United Arab Emirates": "AE", "Saudi Arabia": "SA",
    "Jersey": "JE", "Mauritius": "MU", "Taiwan Province of China": "TW",
    "Brazil": "BR", "Guernsey": "GG", "Liechtenstein": "LI",
    "Turkey": "TR", "Russia": "RU", "South Africa": "ZA",
}

STRUCTURE_LABELS = {
    "LU": "Fund/SPE routing", "KY": "Offshore SPE", "VG": "Offshore SPE",
    "BM": "Offshore SPE", "JE": "Offshore SPE",
    "JP": "Large mfg/corp", "CN": "Large SOE/corp",
    "US": "Diverse MNCs", "GB": "Diverse MNCs", "DE": "Industrial MNCs",
    "IN": "SME tech/services", "SG": "Regional hub", "MY": "Regional hub",
    "HK": "Regional HQ/SPE", "FR": "Industrial MNCs", "NL": "Holding cos",
    "CH": "Industrial/pharma", "KR": "Large conglom", "CA": "Diverse MNCs",
    "AU": "Diverse", "ID": "Regional", "TH": "Regional",
}


def get_cdis(df, direction, year="2023"):
    entry = ("Net (liabilities less assets)" if direction == "Inward"
             else "Net (assets less liabilities)")
    mask = (
        df["COUNTRY"].isin(ASEAN_NAMES.keys()) &
        (df["DI_DIRECTION"] == direction) &
        (df["DI_ENTITY"] == "All entities") &
        (df["INSTR_ASSET"] == "All financial instruments") &
        (df["ACCOUNTING_ENTRY"] == entry)
    )
    sub = df[mask].copy()
    sub["host_iso2"] = sub["COUNTRY"].map(ASEAN_NAMES)
    sub["cp_iso2"] = sub["COUNTERPART_COUNTRY"].map(CP_MAP)
    sub = sub.dropna(subset=["host_iso2", "cp_iso2"])
    sub[year] = pd.to_numeric(sub[year], errors="coerce")
    sub = sub[sub[year] > 0].dropna(subset=[year])
    return (sub.groupby(["host_iso2", "cp_iso2"])[year]
            .sum().reset_index()
            .rename(columns={year: "fdi_usd_mn", "host_iso2": "host", "cp_iso2": "parent"}))


print("Loading CDIS data...")
df = pd.read_csv(CDIS_FILE, encoding="utf-8-sig", low_memory=False)
cdis_in = get_cdis(df, "Inward")
cdis_out = get_cdis(df, "Outward")

gleif = pd.read_csv("data/processed/gleif_bilateral_entity_counts.csv")
gleif = gleif.rename(columns={"host_country": "host", "parent_country": "parent"})

merged_in = gleif.merge(cdis_in, on=["host", "parent"], how="inner")

# ── TABLE 1: SPEARMAN CORRELATIONS ────────────────────────────────────────
print()
print("=" * 62)
print("TABLE 1: Spearman rho — GLEIF entity counts vs CDIS inward")
print("         FDI positions (2023, USD million)")
print("=" * 62)
print(f"{'Host':>6}  {'n pairs':>8}  {'rho':>7}  {'p-value':>9}  {'':>5}")
print("-" * 62)

for host in ["SG", "MY", "TH", "ID", "PH", "VN", "KH", "BN"]:
    sub = merged_in[merged_in["host"] == host]
    n = len(sub)
    if n >= 3:
        r, p = spearmanr(sub["entity_count"], sub["fdi_usd_mn"])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("." if p < 0.1 else "")))
        print(f"{host:>6}  {n:>8}  {r:>7.3f}  {p:>9.4f}  {sig:>5}")
    else:
        print(f"{host:>6}  {n:>8}  {'—':>7}  {'—':>9}  {'n/a':>5}")

r_all, p_all = spearmanr(merged_in["entity_count"], merged_in["fdi_usd_mn"])
sig_all = "***" if p_all < 0.001 else ""
print("-" * 62)
print(f"{'ALL':>6}  {len(merged_in):>8}  {r_all:>7.3f}  {p_all:>9.4f}  {sig_all:>5}")

# Outward
gleif_out = gleif[gleif["parent"].isin(ASEAN_NAMES.values())].copy()
gleif_out = gleif_out.rename(columns={"host": "destination", "parent": "source"})
cdis_out2 = cdis_out.rename(columns={"host": "source", "parent": "destination"})
merged_out = gleif_out.merge(cdis_out2, on=["source", "destination"], how="inner")
r_out, p_out = spearmanr(merged_out["entity_count"], merged_out["fdi_usd_mn"])
sig_out = "***" if p_out < 0.001 else ("**" if p_out < 0.01 else "")
print(f"{'OUTWARD':>6}  {len(merged_out):>8}  {r_out:>7.3f}  {p_out:>9.4f}  {sig_out:>5}")
print()
print("Signif. codes:  *** p<0.001  ** p<0.01  * p<0.05  . p<0.1")

# ── TABLE 2: ENTITY-TO-FDI RATIOS ─────────────────────────────────────────
print()
print("=" * 78)
print("TABLE 2: Entities per USD 10bn FDI stock, by counterpart economy")
print("         (inward, all ASEAN hosts pooled, 2023 CDIS)")
print("         Interpretation: high = many small entities;")
print("                         low  = few large investment vehicles")
print("=" * 78)
print(f"{'Parent':>8}  {'Entities':>9}  {'FDI ($bn)':>10}  {'Ent/$10bn':>10}  {'Hosts':>6}  Structure")
print("-" * 78)

pooled = (merged_in.groupby("parent")
          .agg(
              total_entities=("entity_count", "sum"),
              total_fdi_usd_mn=("fdi_usd_mn", "sum"),
              n_hosts=("host", "nunique"),
          )
          .reset_index())
pooled["fdi_usd_bn"] = pooled["total_fdi_usd_mn"] / 1000
pooled["entities_per_10bn"] = (pooled["total_entities"] / pooled["fdi_usd_bn"] * 10).round(1)
pooled = pooled.sort_values("total_fdi_usd_mn", ascending=False)

for _, r in pooled[pooled["total_fdi_usd_mn"] >= 500].iterrows():
    label = STRUCTURE_LABELS.get(r["parent"], "")
    print(
        f"{r['parent']:>8}  {int(r['total_entities']):>9,}  "
        f"{r['fdi_usd_bn']:>10,.1f}  {r['entities_per_10bn']:>10.1f}  "
        f"{int(r['n_hosts']):>6}  {label}"
    )

print()
print("Note: Ratio < 1.0 indicates investment concentrated in few large vehicles")
print("      (SPE/fund structures, large manufacturers).")
print("      Ratio > 5.0 indicates dispersed investment across many small entities.")

# ── TABLE 3: INTRA-ASEAN OUTWARD ──────────────────────────────────────────
print()
print("=" * 72)
print("TABLE 3: Intra-ASEAN outward — GLEIF entity counts vs CDIS outward")
print("         positions (2023, USD million)")
print("=" * 72)
print(f"{'Source':>8}  {'Dest':>6}  {'Entities':>9}  {'FDI ($bn)':>10}  {'Ent/$10bn':>10}")
print("-" * 72)
merged_out2 = merged_out.copy()
merged_out2["fdi_usd_bn"] = merged_out2["fdi_usd_mn"] / 1000
merged_out2["entities_per_10bn"] = (merged_out2["entity_count"] / merged_out2["fdi_usd_bn"] * 10).round(1)
for _, r in merged_out2.sort_values("fdi_usd_mn", ascending=False).iterrows():
    print(
        f"{r['source']:>8}  {r['destination']:>6}  {int(r['entity_count']):>9,}  "
        f"{r['fdi_usd_bn']:>10.1f}  {r['entities_per_10bn']:>10.1f}"
    )
print("-" * 72)
print(f"{'Outward rho':>8}  {r_out:.3f}  (p={p_out:.4f}, n={len(merged_out)})")
