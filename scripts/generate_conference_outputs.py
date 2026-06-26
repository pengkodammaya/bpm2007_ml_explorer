from __future__ import annotations

import csv
import argparse
import html
import math
import zipfile
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
TODAY = date.today().isoformat()

COUNTRIES = ["MY", "SG", "TH", "ID", "PH", "VN", "KH", "BN", "LA", "MM", "TL"]
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
CP_MAP = {
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


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def cdis_positions(direction: str, year: str = "2023") -> pd.DataFrame:
    path = DATA / "dataset_2026-04-22T22_38_34.423483046Z_DEFAULT_INTEGRATION_IMF.STA_DIP_12.0.1.csv"
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
        df["COUNTRY"].isin(ASEAN_NAMES.keys())
        & (df["DI_DIRECTION"] == direction)
        & (df["DI_ENTITY"] == "All entities")
        & (df["INSTR_ASSET"] == "All financial instruments")
        & (df["ACCOUNTING_ENTRY"] == entry)
    )
    sub = df.loc[mask].copy()
    sub["host"] = sub["COUNTRY"].map(ASEAN_NAMES)
    sub["parent"] = sub["COUNTERPART_COUNTRY"].map(CP_MAP)
    sub[year] = pd.to_numeric(sub[year], errors="coerce")
    sub = sub.dropna(subset=["host", "parent", year])
    sub = sub[sub[year] > 0]
    return (
        sub.groupby(["host", "parent"], as_index=False)[year]
        .sum()
        .rename(columns={year: "fdi_usd_mn"})
    )


def collect_metrics() -> dict:
    raw_rows = []
    uie_rows = []
    for country in COUNTRIES:
        code = country.lower()
        raw = read_parquet(DATA / "raw" / f"gleif_{code}_lei.parquet")
        rel = read_parquet(DATA / "interim" / f"gleif_{code}_relationships.parquet")
        related = read_parquet(DATA / "raw" / f"gleif_{code}_related_lei.parquet")
        exceptions = read_parquet(DATA / "interim" / f"gleif_{code}_reporting_exceptions.parquet")
        wip = read_parquet(DATA / "interim" / f"gleif_{code}_relationships_wip.parquet")
        scanned = read_parquet(DATA / "interim" / f"gleif_{code}_relationships_scanned.parquet")
        uie = read_parquet(DATA / "processed" / code / "uie_assignments.parquet")

        raw_rows.append(
            {
                "country": country,
                "raw_entities": len(raw),
                "relationships": len(rel),
                "related_leis": len(related),
                "reporting_exceptions": len(exceptions),
                "relationship_wip_rows": len(wip),
                "relationship_scanned": len(scanned),
            }
        )

        if not uie.empty:
            source_counts = uie["uie_source"].value_counts().to_dict() if "uie_source" in uie else {}
            foreign = int(((uie["uie_country"].notna()) & (uie["uie_country"] != country)).sum())
            known = int(uie["is_known_uie"].fillna(0).sum()) if "is_known_uie" in uie else 0
            model = int(source_counts.get("phase3_jurisdiction_model", 0))
            uie_rows.append(
                {
                    "country": country,
                    "uie_rows": len(uie),
                    "foreign_uie_count": foreign,
                    "known_uie_count": known,
                    "known_uie_share_pct": pct(known, len(uie)),
                    "model_fallback_count": model,
                    "model_fallback_share_pct": pct(model, len(uie)),
                    "gleif_ultimate": int(source_counts.get("gleif_ultimate_parent", 0)),
                    "gleif_direct": int(source_counts.get("gleif_direct_parent", 0)),
                    "address_cluster": int(source_counts.get("phase2_address_cluster", 0)),
                    "name_match": int(source_counts.get("phase1_name_match", 0)),
                    "manual_verified": int(source_counts.get("manual_verified", 0)),
                    "unassigned": int(source_counts.get("unassigned", 0)),
                }
            )

    raw_df = pd.DataFrame(raw_rows)
    uie_df = pd.DataFrame(uie_rows)

    cdis_in = cdis_positions("Inward")
    cdis_out = cdis_positions("Outward")
    inward_totals = cdis_in.groupby("host", as_index=False)["fdi_usd_mn"].sum()
    density = uie_df.merge(inward_totals, left_on="country", right_on="host", how="left")
    density["inward_fdi_usd_bn"] = density["fdi_usd_mn"] / 1000
    density["inward_lei_density"] = density["foreign_uie_count"] / density["inward_fdi_usd_bn"]
    density = density.drop(columns=["host", "fdi_usd_mn"], errors="ignore")

    gleif = pd.read_csv(DATA / "processed" / "gleif_bilateral_entity_counts.csv").rename(
        columns={"host_country": "host", "parent_country": "parent", "entity_count": "entities"}
    )
    merged_in = gleif.merge(cdis_in, on=["host", "parent"], how="inner")
    gleif_out = gleif[gleif["parent"].isin(COUNTRIES)].rename(
        columns={"host": "destination", "parent": "source"}
    )
    cdis_out_pairs = cdis_out.rename(columns={"host": "source", "parent": "destination"})
    outward_bilateral = gleif_out.merge(cdis_out_pairs, on=["source", "destination"], how="inner")
    outward_bilateral["fdi_usd_bn"] = outward_bilateral["fdi_usd_mn"] / 1000
    outward_bilateral["outward_lei_density"] = (
        outward_bilateral["entities"] / outward_bilateral["fdi_usd_bn"]
    )
    outward_source_summary = (
        outward_bilateral.groupby("source", as_index=False)
        .agg(
            destination_count=("destination", "nunique"),
            my_owned_or_source_owned_leis=("entities", "sum"),
            outward_fdi_usd_mn=("fdi_usd_mn", "sum"),
        )
        .rename(columns={"source": "country"})
    )
    outward_source_summary["outward_fdi_usd_bn"] = outward_source_summary["outward_fdi_usd_mn"] / 1000
    outward_source_summary["asean_outward_lei_density"] = (
        outward_source_summary["my_owned_or_source_owned_leis"]
        / outward_source_summary["outward_fdi_usd_bn"]
    )
    corr_rows = []
    for host, group in merged_in.groupby("host"):
        if len(group) >= 3:
            rho, pval = spearmanr(group["entities"], group["fdi_usd_mn"])
            corr_rows.append({"host": host, "n_pairs": len(group), "rho": rho, "p_value": pval})
    rho_all, p_all = spearmanr(merged_in["entities"], merged_in["fdi_usd_mn"])
    corr_rows.append({"host": "ALL", "n_pairs": len(merged_in), "rho": rho_all, "p_value": p_all})
    corr_df = pd.DataFrame(corr_rows).sort_values(["host"])

    review = read_parquet(DATA / "processed" / "my" / "uie_review_targets.parquet")
    product_vehicles = read_parquet(DATA / "processed" / "my" / "uie_product_vehicle_targets.parquet")
    benchmark = read_parquet(DATA / "processed" / "my" / "entity_match_benchmark.parquet")

    return {
        "raw": raw_df,
        "uie": uie_df,
        "density": density,
        "outward_bilateral": outward_bilateral,
        "outward_source_summary": outward_source_summary,
        "correlation": corr_df,
        "review": review,
        "product_vehicles": product_vehicles,
        "benchmark": benchmark,
    }


def pct(num: int | float, den: int | float) -> float:
    return round((num / den * 100), 1) if den else 0.0


def fmt_int(value) -> str:
    if pd.isna(value):
        return ""
    return f"{int(value):,}"


def fmt_num(value, digits: int = 2) -> str:
    if value is None or pd.isna(value) or not math.isfinite(float(value)):
        return ""
    return f"{float(value):,.{digits}f}"


def flatten_metrics(metrics: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    for name, df in metrics.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(REPORTS / f"conference_{name}_{TODAY}.csv", index=False)


def w_text(text: str) -> str:
    return escape(str(text)).replace("\n", "</w:t></w:r></w:p><w:p><w:r><w:t>")


def doc_p(text: str = "", style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t>{w_text(text)}</w:t></w:r></w:p>"


def doc_bullets(items: list[str]) -> str:
    return "".join(doc_p(f"- {item}") for item in items)


def doc_table(rows: list[list[str]]) -> str:
    xml = ["<w:tbl><w:tblPr><w:tblStyle w:val=\"TableGrid\"/><w:tblW w:w=\"0\" w:type=\"auto\"/></w:tblPr>"]
    for row in rows:
        xml.append("<w:tr>")
        for cell in row:
            xml.append(f"<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>{doc_p(cell)}</w:tc>")
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def build_word(metrics: dict) -> Path:
    raw = metrics["raw"]
    uie = metrics["uie"]
    density = metrics["density"].sort_values("raw_sort" if "raw_sort" in metrics["density"] else "country")
    outward_bilateral = metrics["outward_bilateral"]
    outward_source_summary = metrics["outward_source_summary"]
    corr = metrics["correlation"]
    review = metrics["review"]
    product_vehicles = metrics["product_vehicles"]
    benchmark = metrics["benchmark"]
    my_density = density.loc[density["country"] == "MY"].iloc[0]
    my_uie = uie.loc[uie["country"] == "MY"].iloc[0]

    raw_table = [["Host", "Raw LEIs", "Relationships", "Related LEIs", "Exceptions", "Refresh status"]]
    for _, row in raw.iterrows():
        status = "Partial refresh in progress" if row["country"] == "MY" and row["relationship_scanned"] else "Cached output"
        raw_table.append(
            [
                row["country"],
                fmt_int(row["raw_entities"]),
                fmt_int(row["relationships"]),
                fmt_int(row["related_leis"]),
                fmt_int(row["reporting_exceptions"]),
                status,
            ]
        )

    density_table = [["Host", "Foreign UIE LEIs", "Inward FDI USD bn", "Inward LEI density"]]
    for _, row in density.sort_values("foreign_uie_count", ascending=False).iterrows():
        density_table.append(
            [
                row["country"],
                fmt_int(row["foreign_uie_count"]),
                fmt_num(row["inward_fdi_usd_bn"], 1),
                fmt_num(row["inward_lei_density"], 2),
            ]
        )

    my_outward_table = [["Source", "Destination", "MY-owned LEIs", "Outward FDI USD bn", "Outward LEI density"]]
    my_outward = outward_bilateral.loc[outward_bilateral["source"] == "MY"].sort_values(
        "fdi_usd_mn",
        ascending=False,
    )
    for _, row in my_outward.iterrows():
        my_outward_table.append(
            [
                row["source"],
                row["destination"],
                fmt_int(row["entities"]),
                fmt_num(row["fdi_usd_bn"], 1),
                fmt_num(row["outward_lei_density"], 2),
            ]
        )

    outward_summary_table = [["Source", "ASEAN destinations", "Source-owned LEIs", "Outward FDI USD bn", "ASEAN outward density"]]
    for _, row in outward_source_summary.sort_values("outward_fdi_usd_bn", ascending=False).iterrows():
        outward_summary_table.append(
            [
                row["country"],
                fmt_int(row["destination_count"]),
                fmt_int(row["my_owned_or_source_owned_leis"]),
                fmt_num(row["outward_fdi_usd_bn"], 1),
                fmt_num(row["asean_outward_lei_density"], 2),
            ]
        )

    corr_table = [["Host", "Pairs", "Spearman rho", "p-value"]]
    for _, row in corr.iterrows():
        corr_table.append([row["host"], fmt_int(row["n_pairs"]), fmt_num(row["rho"], 3), fmt_num(row["p_value"], 4)])

    uie_table = [["Host", "UIE rows", "Known UIE", "Model fallback", "Address cluster", "Name match"]]
    for _, row in uie.iterrows():
        uie_table.append(
            [
                row["country"],
                fmt_int(row["uie_rows"]),
                f"{fmt_int(row['known_uie_count'])} ({fmt_num(row['known_uie_share_pct'], 1)}%)",
                f"{fmt_int(row['model_fallback_count'])} ({fmt_num(row['model_fallback_share_pct'], 1)}%)",
                fmt_int(row["address_cluster"]),
                fmt_int(row["name_match"]),
            ]
        )

    review_status = review["review_status"].value_counts().to_dict() if not review.empty else {}
    product_status = product_vehicles["review_status"].value_counts().to_dict() if not product_vehicles.empty else {}
    benchmark_flags = benchmark["review_flag"].value_counts().to_dict() if not benchmark.empty else {}

    body = []
    body.append(doc_p("LEI-Based Ultimate Investor Economy Discovery for ASEAN", "Title"))
    body.append(doc_p(f"A policy-oriented methods and preliminary-results paper. Generated {TODAY}."))
    body.append(doc_p("Executive Summary", "Heading1"))
    body.append(
        doc_p(
            "This paper develops a public-data discovery framework for identifying Ultimate Investor Economy (UIE) signals in "
            "direct-investment statistics. The exercise is motivated by a practical compilation problem: official survey and "
            "administrative systems often observe the resident entity and the immediate counterpart, but not the full cross-border "
            "ownership chain needed to identify the ultimate source of investment. The Legal Entity Identifier system offers a "
            "partial solution because it records legal entities and, in some cases, direct and ultimate parent relationships. "
            "However, parent reporting is incomplete. The contribution of this work is therefore not to treat LEI data as a "
            "complete register, but to use it as a discovery layer that prioritises where compilers should look next."
        )
    )
    body.append(
        doc_p(
            "The approach combines five forms of evidence. First, known GLEIF direct-parent and ultimate-parent relationships are "
            "taken as high-confidence ownership signals. Second, GLEIF reporting exceptions are used to distinguish entities that "
            "are structurally outside parent reporting from entities with no observable parent information. Third, name-pattern "
            "matching identifies subsidiaries whose legal names contain parent-brand tokens. Fourth, registered-address clustering "
            "propagates parent information within shared corporate-secretary and special-purpose-vehicle environments. Fifth, a "
            "jurisdiction model supplies a fallback country signal for triage, but is explicitly separated from verified evidence. "
            "The output is not a final statistical estimate. It is a ranked, auditable map of ownership evidence and uncertainty."
        )
    )
    body.append(
        doc_p(
            f"Using the current processed workspace outputs, Malaysia has {fmt_int(my_uie['uie_rows'])} UIE assignment rows. "
            f"Of these, {fmt_int(my_density['foreign_uie_count'])} are assigned to a foreign UIE country, "
            f"{fmt_int(my_uie['known_uie_count'])} are known UIE assignments, and "
            f"{fmt_int(my_uie['model_fallback_count'])} rely on the jurisdiction-model fallback. The corresponding inward LEI "
            f"density is {fmt_num(my_density['inward_lei_density'], 2)} foreign-owned LEIs per USD 1 billion of inward FDI stock. "
            "This should be interpreted as a structural coverage ratio, not as a valuation ratio."
        )
    )
    body.append(doc_p("1. Motivation and Statistical Context", "Heading1"))
    body.append(
        doc_p(
            "The introduction of BPM7 places renewed emphasis on ownership, risk transfer, and the economic interpretation of "
            "cross-border positions. For direct investment, the distinction between immediate counterpart economy and ultimate "
            "investor economy is particularly important. Immediate counterpart data describe the next jurisdiction in the ownership "
            "chain. UIE data attempt to identify the economy at the top of the chain. The latter is more informative for analysing "
            "geopolitical exposure, multinational enterprise structures, financial centres, and the role of special purpose entities."
        )
    )
    body.append(
        doc_p(
            "In practice, UIE compilation is difficult because ownership chains are fragmented across survey responses, corporate "
            "registries, tax records, securities databases, and public filings. Many structures are deliberately multi-jurisdictional. "
            "The immediate parent may be a holding company in a financial centre, while the ultimate controlling group is located "
            "elsewhere. Conversely, a resident entity may appear foreign-owned in one data source but be controlled by a domestic "
            "sovereign fund, pension fund, family group, or operating conglomerate. A useful analytical system must therefore avoid "
            "collapsing all evidence into a single black-box label. It should preserve the source, confidence, and reason for each "
            "assignment."
        )
    )
    body.append(
        doc_p(
            "This project addresses that need by building a reproducible pipeline that uses only public data. The public-data "
            "constraint is intentional. It allows the method to be audited, reproduced across countries, and used as a neutral "
            "starting point before confidential compiler data are introduced. The resulting tables are best understood as a "
            "pre-compilation intelligence layer: they do not replace official sources, but they can identify where official follow-up "
            "is likely to produce the greatest improvement in coverage."
        )
    )
    body.append(doc_p("2. Data Sources and Current Snapshot", "Heading1"))
    body.append(
        doc_p(
            "The primary source is the GLEIF API. For each host economy, the pipeline pulls country-linked LEI records, then scans "
            "each LEI for direct-parent and ultimate-parent relationships. Where parent LEIs are found, the parent records are also "
            "retrieved so that the parent economy can be assigned. The pipeline also uses GLEIF reporting-exception endpoints, which "
            "record why an entity does not report parent information. These exception records are valuable because they often reveal "
            "that a missing parent is not simply missing data; it may reflect a specific reporting rule or disclosure limitation."
        )
    )
    body.append(
        doc_p(
            "For macro validation, the exercise uses IMF direct-investment positions by counterpart economy. The validation does not "
            "compare LEI counts with FDI values one-for-one. Instead, it asks whether bilateral LEI entity counts are directionally "
            "consistent with bilateral FDI stocks, and whether density ratios provide interpretable signals about concentrated versus "
            "dispersed investment structures."
        )
    )
    body.append(doc_p("2.1 Data Provenance", "Heading1"))
    body.append(
        doc_p(
            "The LEI data used in this paper come from the GLEIF API documentation entry point at https://api.gleif.org/docs. "
            "The local pipeline uses GLEIF LEI records, parent-relationship endpoints, reporting-exception endpoints, and full-text "
            "search results. The local files feeding the current results are country-scoped parquet files under data/raw and "
            "data/interim, especially gleif_{country}_lei.parquet, gleif_{country}_relationships.parquet, "
            "gleif_{country}_related_lei.parquet, and gleif_{country}_reporting_exceptions.parquet."
        )
    )
    body.append(
        doc_p(
            "The FDI stock data come from the IMF Data portal's Direct Investment Positions by Counterpart Economy dataset "
            "(DIP, formerly CDIS), available at https://data.imf.org/en/datasets/IMF.STA:DIP. The IMF describes this dataset as "
            "covering inward direct-investment positions by immediate investor economy and outward direct-investment positions by "
            "immediate investment economy. The local extract used here is "
            "data/dataset_2026-04-22T22_38_34.423483046Z_DEFAULT_INTEGRATION_IMF.STA_DIP_12.0.1.csv."
        )
    )
    body.append(
        doc_p(
            "The density and validation tables are derived locally rather than downloaded as pre-computed statistics. The script filters "
            "the IMF file to 2023, All entities, All financial instruments, inward net liabilities less assets, and outward net assets "
            "less liabilities. It then joins those FDI positions to pipeline-derived GLEIF bilateral entity counts stored in "
            "data/processed/gleif_bilateral_entity_counts.csv. Supporting audit files are exported to reports/conference_*.csv."
        )
    )
    body.append(
        doc_p(
            "Manual UIE overrides and review statuses are local analyst files under data/manual. CTOS Malaysia data, where present, is "
            "treated only as incorporation or existence context and is not used as proof of Malaysian ownership. GEM is discussed as a "
            "future complementary evidence layer; it is not used in the current results."
        )
    )
    body.append(
        doc_p(
            "The current document uses the files available at generation time. The Malaysia raw LEI file had already been refreshed "
            "on 2026-06-22, but the Malaysia relationship refresh was still checkpointed in the background. As a result, the processed "
            "Malaysia UIE tables remain based on the last completed relationship output. This distinction matters: raw entity counts "
            "can update before relationship-derived UIE assignments are rebuilt. The report therefore presents the current results as "
            "preliminary and explicitly flags refresh status."
        )
    )
    body.append(doc_table(raw_table))
    body.append(doc_p("3. Conceptual Framework", "Heading1"))
    body.append(
        doc_p(
            "The core conceptual distinction is between evidence and inference. Evidence includes a filed GLEIF ultimate-parent "
            "relationship, a direct-parent relationship, or a source-backed analyst override. Inference includes name matching, address "
            "propagation, and jurisdiction-model fallback. The pipeline deliberately stores these categories separately so that "
            "results can be filtered according to the user's tolerance for uncertainty. For example, a central-bank compiler may use "
            "known parent relationships to update a frame immediately, but treat model-only cases as targets for review rather than "
            "as inputs to published statistics."
        )
    )
    body.append(
        doc_p(
            "The UIE assignment ladder is conservative. Manual source-backed overrides sit at the top because they represent analyst "
            "review of public evidence. GLEIF ultimate-parent relationships follow, then GLEIF direct-parent relationships. A direct "
            "parent is not always the ultimate parent, but it is still stronger evidence than a name or address heuristic. Name matches "
            "and address clusters are treated as intermediate evidence: useful for discovery, but requiring validation. The jurisdiction "
            "model sits below these because it assigns a likely economy without identifying a legal parent entity."
        )
    )
    body.append(
        doc_p(
            "This design is intended to support BPM-style statistical governance. Every UIE assignment has a source field, an evidence "
            "tier, and a confidence or priority indicator. The system can therefore produce both a broad analytical map and a narrow "
            "high-confidence subset. The broad map is useful for understanding structural exposure; the narrow subset is more suitable "
            "for formal reconciliation with confidential source data."
        )
    )
    body.append(doc_p("4. Methodology", "Heading1"))
    body.append(
        doc_p(
            "The pipeline begins by constructing a domestic LEI universe for each host economy. Each record is normalised into a "
            "common schema: LEI, legal name, legal jurisdiction, headquarters jurisdiction, legal form, registration status, city, and "
            "category fields. This standardisation is important because country-level LEI data are not homogeneous. Some economies have "
            "large numbers of financial vehicles and funds, while others have small operating-company populations. Without a common "
            "schema, comparisons across ASEAN hosts would mix data availability with true structural differences."
        )
    )
    body.append(
        doc_p(
            "The second stage scans parent relationships. For each domestic LEI, the pipeline requests direct-parent and ultimate-parent "
            "relationships from GLEIF. Where relationships exist, source and target LEIs are stored in a directed graph. Parent records "
            "are then fetched as related LEIs so that parent countries can be assigned. The graph representation is useful because it "
            "preserves ownership chains and supports later centrality, component, and propagation measures."
        )
    )
    body.append(
        doc_p(
            "The third stage uses reporting exceptions. Reporting exceptions are not UIE assignments, but they are powerful diagnostic "
            "signals. A NON_CONSOLIDATING exception indicates that an entity is a subsidiary that does not consolidate; this confirms "
            "that an ownership relationship exists even when the parent country is not disclosed. Other exception categories, such as "
            "NO_LEI, NON_PUBLIC, NATURAL_PERSONS, or NO_KNOWN_PERSON, provide insight into the nature of the information gap. In a "
            "compiler workflow, these cases should be triaged differently from ordinary missing data."
        )
    )
    body.append(
        doc_p(
            "The fourth stage adds name-pattern evidence. Parent-brand tokens are extracted from known parent names after removing legal "
            "suffixes and generic terms. These tokens are matched to domestic entity names using fuzzy matching and whole-word boundary "
            "checks. This is useful for multinational subsidiary networks where the parent brand remains visible in the local legal "
            "name. The method is intentionally conservative because false positives in brand matching can be costly: a common word or "
            "ambiguous acronym can easily resemble a parent token without implying ownership."
        )
    )
    body.append(
        doc_p(
            "The fifth stage uses address clustering. Entities are grouped by normalised registered address and postal code. Within a "
            "cluster, known parent signals can be propagated to entities that share the same address. This is especially relevant for "
            "special purpose entities, fund vehicles, and corporate-secretary environments. The method also penalises large clusters and "
            "known office-hotel patterns, because a shared address may signal administrative service provision rather than common control."
        )
    )
    body.append(
        doc_p(
            "The final stage trains a jurisdiction model to provide fallback country assignments. Features include legal-form and name "
            "patterns, geography, graph features, reporting-exception flags, and other structural indicators. The model is used as a "
            "triage device. Its output is never allowed to overwrite stronger evidence, and its probabilities are treated as uncalibrated "
            "ranking signals rather than statistical confidence intervals."
        )
    )
    body.append(doc_table(uie_table))
    body.append(doc_p("5. Ratio Measures and Macro Validation", "Heading1"))
    body.append(
        doc_p(
            "The principal ratio used in this draft is inward LEI density, defined as the count of foreign-owned or foreign-UIE LEIs in "
            "the host economy divided by inward FDI stock measured in USD billions. The ratio is designed to answer a structural "
            "question: how many LEI-bearing entities are observed per unit of inward investment? A high ratio may indicate a dispersed "
            "population of smaller subsidiaries. A low ratio may indicate large positions concentrated in a few investment vehicles, "
            "holding companies, or capital-intensive operating groups."
        )
    )
    body.append(
        doc_p(
            f"For Malaysia, the current processed result is {fmt_int(my_density['foreign_uie_count'])} foreign-UIE LEIs over "
            f"USD {fmt_num(my_density['inward_fdi_usd_bn'], 1)} billion of inward FDI stock. This gives an inward LEI density of "
            f"{fmt_num(my_density['inward_lei_density'], 2)} LEIs per USD 1 billion. The measure should not be interpreted as "
            "coverage completeness by itself, because LEI adoption differs across sectors and countries. It is best used comparatively, "
            "together with the evidence-tier distribution and the review queue."
        )
    )
    body.append(doc_table(density_table))
    body.append(
        doc_p(
            "The outward analogue is outward LEI density, defined as the count of source-economy-owned LEIs observed abroad divided by "
            "the source economy's outward FDI stock to the same destination, measured in USD billions. In this draft the outward measure "
            "is intentionally scoped to matched intra-ASEAN bilateral pairs. This avoids combining an ASEAN-only LEI numerator with a "
            "global outward-FDI denominator. The measure should therefore be read as an ASEAN-observed outward density, not as total "
            "global outward coverage."
        )
    )
    body.append(
        doc_p(
            "For Malaysia, the current matched outward pairs cover MY-owned LEIs observed in Singapore, Thailand, Vietnam and the "
            "Philippines. These figures are useful for comparing bilateral structures within ASEAN, but they do not include MY-owned "
            "LEIs outside ASEAN. A global outward density would require a global crawl of MY-owned LEIs abroad or another comprehensive "
            "cross-border ownership register."
        )
    )
    body.append(doc_table(my_outward_table))
    body.append(doc_table(outward_summary_table))
    body.append(
        doc_p(
            "A second validation exercise compares bilateral GLEIF entity counts with IMF direct-investment positions by counterpart "
            "economy. The Spearman rank correlation is used because the relationship between entity counts and value stocks is unlikely "
            "to be linear. A country with ten small subsidiaries and a country with one large holding company may have the same FDI "
            "value but very different entity counts. The question is whether countries with larger bilateral FDI positions tend, in "
            "rank terms, to have more observed LEI entities."
        )
    )
    body.append(doc_table(corr_table))
    body.append(doc_p("6. Current Empirical Findings", "Heading1"))
    body.append(
        doc_p(
            "Three findings stand out. First, the public LEI universe is highly uneven across ASEAN. Singapore is the dominant LEI host "
            "in the current workspace, reflecting its role as a regional financial and corporate hub. Malaysia and Thailand form the "
            "next tier, while several smaller ASEAN economies have sparse LEI populations. This unevenness is itself analytically "
            "important: LEI-based methods are most informative where LEI adoption is broad enough to reveal structure."
        )
    )
    body.append(
        doc_p(
            "Second, known parent coverage remains limited compared with the total LEI universe. This is the central empirical constraint "
            "of the project. GLEIF parent relationships are high-quality when present, but they do not cover the majority of entities. "
            "The value of the pipeline therefore comes from combining high-confidence parent records with exception analysis and "
            "carefully labelled inference. In the Malaysia output, only a minority of assignments are known UIE evidence, while the "
            "model fallback covers a large share of entities. That is useful for prioritisation, but not yet sufficient for final "
            "statistical allocation."
        )
    )
    body.append(
        doc_p(
            "Third, the review queue reveals where methodological improvement pays off fastest. The Malaysia main review queue now "
            f"contains {len(review):,} operating, financial, telecommunications and manual-review rows. A separate product-vehicle "
            f"audit file contains {len(product_vehicles):,} rows, of which "
            f"{product_status.get('product_vehicle_rule_applied', 0):,} have the fund/product rule applied. This prevents investment "
            "products and fund vehicles from crowding the operating-company UIE review queue while preserving them for audit and "
            "methodological reporting."
        )
    )
    body.append(
        doc_p(
            f"The GLEIF search benchmark reinforces this point. The benchmark generated {len(benchmark):,} candidate rows for top "
            f"Malaysia review targets. In {benchmark_flags.get('self_entity_found', 0):,} cases, GLEIF search primarily rediscovered "
            "the same domestic entity. In "
            f"{benchmark_flags.get('candidate_country_differs_from_current_uie', 0):,} cases, it produced a candidate whose country "
            "differed from the current UIE assignment. This suggests that GLEIF's own search is useful as an entity-resolution tool, "
            "but weak as standalone evidence of ultimate ownership. It should be used to generate candidate legal entities for review, "
            "not to replace the evidence ladder."
        )
    )
    body.append(doc_p("7. Interpretation for Compilers", "Heading1"))
    body.append(
        doc_p(
            "The results support a practical division of labour between machine learning and statistical compilation. Machine learning "
            "is most useful upstream, where it can organise public evidence, rank gaps, and suggest candidate ownership links. The "
            "statistical compiler remains responsible for deciding whether an inferred signal is admissible for official use, how it "
            "interacts with confidential source data, and whether the resulting classification is consistent with BPM7 principles."
        )
    )
    body.append(
        doc_p(
            "In this framing, the pipeline offers three operational benefits. It can improve coverage by identifying entities with "
            "known parent evidence or strong exception signals. It can improve efficiency by ranking entities for follow-up instead of "
            "treating all missing parent data equally. It can improve coherence by comparing micro-level entity structures with macro "
            "FDI positions. None of these benefits require the model to be treated as an oracle. They require the model to be auditable, "
            "stable, and clearly separated from higher-confidence evidence."
        )
    )
    body.append(
        doc_p(
            "For Malaysia, the current findings suggest that a meaningful share of the ownership map can be organised from public data, "
            "but that the marginal work now lies in reducing false precision. Fund vehicles, nominee structures, asset managers, and "
            "domestically controlled strategic entities require different treatment from ordinary subsidiaries. A UIE system that fails "
            "to distinguish these cases may produce a complete-looking table that is analytically misleading."
        )
    )
    body.append(doc_p("8. Governance, Auditability and Use of Complementary Data", "Heading1"))
    body.append(
        doc_p(
            "A BIS-style application of this work should emphasise governance. Each assignment should retain provenance: source URL or "
            "API endpoint, evidence tier, timestamp, and review status. Manual overrides should be rare, source-backed, and separated "
            "from model outputs. Model-only assignments should be flagged as triage results. This design allows the same dataset to "
            "support multiple uses: broad analytical mapping, targeted survey follow-up, and high-confidence statistical reconciliation."
        )
    )
    body.append(
        doc_p(
            "Complementary datasets should be added selectively. Global Energy Monitor is a useful example. It can provide source-backed "
            "ownership and asset information for power plants, coal assets, LNG infrastructure, and other energy-sector entities. It "
            "should not be treated as a general company matcher. Instead, it should enter as a sector-specific evidence layer for "
            "energy and heavy-industry cases where public asset ownership may reveal group control more clearly than LEI parent records."
        )
    )
    body.append(doc_p("9. Limitations", "Heading1"))
    body.append(
        doc_bullets(
            [
                "The current results are preliminary because the refreshed ASEAN GLEIF crawl is still running in the background.",
                "LEI adoption is uneven across sectors and economies; absence from LEI data is not absence from the economy.",
                "GLEIF parent relationships are voluntary or rule-dependent in ways that create non-random missingness.",
                "Address clustering can confuse common service-provider addresses with common ownership unless penalised and reviewed.",
                "Jurisdiction-model probabilities are uncalibrated and should be used for ranking, not for direct publication.",
                "CDIS validation is macro-plausibility validation; it does not provide legal-entity ground truth.",
                "The current pipeline does not yet implement a full BPM7 enforcement layer for valuation, ownership thresholds, and stock-flow reconciliation.",
            ]
        )
    )
    body.append(doc_p("10. Work Programme Before Final Release", "Heading1"))
    body.append(
        doc_bullets(
            [
                "Complete refreshed GLEIF entity, relationship and reporting-exception pulls for Malaysia and the rest of ASEAN.",
                "Rebuild graph metrics, UIE assignments, inward and outward density ratios, and CDIS validation tables from the refreshed snapshot.",
                "Keep the fund/product vehicle rule under review as new ASEAN outputs are rebuilt.",
                "Add Global Energy Monitor as a sector-specific, source-backed evidence layer for energy and heavy industry.",
                "Expand manual overrides only for high-impact cases with clear public evidence and review notes.",
                "Produce a high-confidence subset separate from the full analytical/discovery dataset.",
            ]
        )
    )
    body.append(doc_p("Conclusion", "Heading1"))
    body.append(
        doc_p(
            "The main conclusion is that public LEI data can materially improve the organisation of UIE discovery, but only if the "
            "outputs are treated as evidence-ranked signals rather than final classifications. The method is most valuable as an "
            "intermediate layer between raw public data and official compilation. It provides a transparent way to identify known "
            "ownership links, diagnose missingness, prioritise review, and compare micro-level structure with macro-level FDI stocks. "
            "For Malaysia and ASEAN, the early results are strong enough to justify continued development, while also making clear that "
            "the next gains will come from better vehicle classification, refreshed crawls, complementary source layers, and disciplined "
            "governance of manual overrides."
        )
    )

    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(body)
        + '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>'
        "</w:body></w:document>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="36"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>'
        "</w:styles>"
    )
    path = REPORTS / f"asean_uie_bis_style_writeup_outward_density_provenance_fund_rule_{TODAY}.docx"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", DOCX_CONTENT_TYPES)
        z.writestr("_rels/.rels", DOCX_RELS)
        z.writestr("word/document.xml", document)
        z.writestr("word/styles.xml", styles)
    return path


DOCX_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

DOCX_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def a_text(text: str) -> str:
    return escape(str(text))


def ppt_shape(shape_id: int, x: int, y: int, cx: int, cy: int, text: str, font_size: int = 2200, bold: bool = False) -> str:
    runs = []
    for line in str(text).split("\n"):
        b = "<a:b/>" if bold else ""
        runs.append(
            f'<a:p><a:r><a:rPr lang="en-US" sz="{font_size}">{b}</a:rPr><a:t>{a_text(line)}</a:t></a:r></a:p>'
        )
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
  <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{"".join(runs)}</p:txBody>
</p:sp>"""


def ppt_slide(
    title: str,
    bullets: list[str],
    footer: str = "",
    body_font: int = 1850,
    title_font: int = 3200,
) -> str:
    shapes = [
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>',
        ppt_shape(2, 500000, 260000, 11500000, 800000, title, title_font, True),
    ]
    bullet_text = "\n".join(f"- {b}" for b in bullets)
    shapes.append(ppt_shape(3, 700000, 1150000, 11200000, 5250000, bullet_text, body_font))
    if footer:
        shapes.append(ppt_shape(4, 700000, 6500000, 11200000, 400000, footer, 1300))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="F7F8FA"/></a:solidFill></p:bgPr></p:bg><p:spTree>{''.join(shapes)}</p:spTree></p:cSld>
<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def build_powerpoint(metrics: dict) -> Path:
    raw = metrics["raw"]
    uie = metrics["uie"]
    density = metrics["density"]
    outward_bilateral = metrics["outward_bilateral"]
    outward_source_summary = metrics["outward_source_summary"]
    corr = metrics["correlation"]
    review = metrics["review"]
    product_vehicles = metrics["product_vehicles"]
    benchmark = metrics["benchmark"]
    my = density.loc[density["country"] == "MY"].iloc[0]
    my_raw = raw.loc[raw["country"] == "MY"].iloc[0]
    my_uie = uie.loc[uie["country"] == "MY"].iloc[0]
    sg = raw.loc[raw["country"] == "SG"].iloc[0]

    top_density = density.sort_values("foreign_uie_count", ascending=False).head(5)
    density_lines = [
        f"{r.country}: {fmt_int(r.foreign_uie_count)} foreign UIE LEIs, density {fmt_num(r.inward_lei_density, 2)} / USD 1bn"
        for r in top_density.itertuples(index=False)
    ]
    my_outward = outward_bilateral.loc[outward_bilateral["source"] == "MY"].sort_values(
        "fdi_usd_mn",
        ascending=False,
    )
    my_outward_lines = [
        f"MY -> {r.destination}: {fmt_int(r.entities)} LEIs; USD {fmt_num(r.fdi_usd_bn, 1)}bn; density {fmt_num(r.outward_lei_density, 2)} / USD 1bn"
        for r in my_outward.itertuples(index=False)
    ]
    outward_summary_lines = [
        f"{r.country}: {fmt_int(r.my_owned_or_source_owned_leis)} LEIs across {fmt_int(r.destination_count)} ASEAN destinations; density {fmt_num(r.asean_outward_lei_density, 2)} / USD 1bn"
        for r in outward_source_summary.sort_values("outward_fdi_usd_bn", ascending=False).itertuples(index=False)
    ]
    corr_lines = [
        f"{r.host}: rho {fmt_num(r.rho, 3)} across {fmt_int(r.n_pairs)} pairs"
        for r in corr.sort_values("rho", ascending=False).head(5).itertuples(index=False)
    ]
    raw_lines = [
        f"{r.country}: {fmt_int(r.raw_entities)} LEIs; {fmt_int(r.relationships)} relationship rows; {fmt_int(r.reporting_exceptions)} exceptions"
        for r in raw.sort_values("raw_entities", ascending=False).head(6).itertuples(index=False)
    ]
    uie_lines = [
        f"{r.country}: known {fmt_int(r.known_uie_count)} ({fmt_num(r.known_uie_share_pct, 1)}%); model fallback {fmt_int(r.model_fallback_count)} ({fmt_num(r.model_fallback_share_pct, 1)}%)"
        for r in uie.sort_values("uie_rows", ascending=False).head(6).itertuples(index=False)
    ]
    review_status = review["review_status"].value_counts().to_dict() if not review.empty else {}
    product_status = product_vehicles["review_status"].value_counts().to_dict() if not product_vehicles.empty else {}
    benchmark_flags = benchmark["review_flag"].value_counts().to_dict() if not benchmark.empty else {}
    footer = f"Preliminary public-data discovery outputs | generated {TODAY}"

    slides = [
        ppt_slide(
            "LEI-Based Ultimate Investor Economy Discovery for ASEAN",
            [
                "A public-data discovery layer for BPM7 direct-investment ownership analysis",
                "Focus: evidence-ranked UIE signals, not direct estimation of FDI values",
                "Case study emphasis: Malaysia in comparative ASEAN context",
                "Current caveat: refreshed GLEIF crawls are still running; results use available processed outputs",
            ],
            footer,
        ),
        ppt_slide(
            "Why UIE Is Hard To Compile",
            [
                "Immediate counterpart data do not necessarily reveal ultimate control",
                "Ownership chains can pass through holding companies, funds, SPEs and regional hubs",
                "Survey frames often observe resident entities but not full cross-border parent chains",
                "Public parent data are sparse and missingness is non-random",
                "Need: transparent triage that separates evidence, inference and review priorities",
            ],
            footer,
        ),
        ppt_slide(
            "Definitions For First-Time Readers",
            [
                "LEI: Legal Entity Identifier, a public identifier for legal entities",
                "GLEIF: Global Legal Entity Identifier Foundation, the source of LEI records and relationship data",
                "UIE: Ultimate Investor Economy, the economy of the ultimate owner/controller in a direct-investment chain",
                "Immediate counterpart economy is not always the UIE",
                "CDIS/DIP: IMF direct-investment positions by counterpart economy, used here as macro validation",
            ],
            footer,
        ),
        ppt_slide(
            "What This Exercise Is Not",
            [
                "Not a replacement for official survey or administrative source data",
                "Not a valuation model for FDI positions",
                "Not proof that every model-assigned UIE is correct",
                "Not a claim that LEI adoption is complete or uniform across ASEAN",
                "It is a transparent discovery and prioritisation layer for compiler review",
            ],
            footer,
        ),
        ppt_slide(
            "Core Contribution",
            [
                "Use LEI data as a discovery graph rather than a complete ownership register",
                "Build a country-parameterised pipeline for ASEAN hosts",
                "Assign UIE with an explicit evidence ladder and provenance fields",
                "Retain weak/model evidence as triage, not as final statistical classification",
                "Compare micro entity structure with macro CDIS/DIP positions through density ratios",
            ],
            footer,
        ),
        ppt_slide(
            "Data Architecture",
            [
                "GLEIF LEI records: legal entity universe by host economy",
                "GLEIF parent relationships: direct-parent and ultimate-parent graph edges",
                "GLEIF reporting exceptions: reason why parent is not reported",
                "Related LEI fetch: parent-node country and legal-name enrichment",
                "Review files: manual overrides and analyst queue statuses",
                "IMF CDIS/DIP: bilateral FDI stock benchmark for macro plausibility",
            ],
            footer,
        ),
        ppt_slide(
            "Data Provenance: Public Sources",
            [
                "GLEIF API: https://api.gleif.org/docs",
                "GLEIF inputs used: LEI records, parent relationships, reporting exceptions, full-text search",
                "IMF DIP/CDIS: https://data.imf.org/en/datasets/IMF.STA:DIP",
                "IMF inputs used: 2023 Direct Investment Positions by Counterpart Economy",
                "DIP/CDIS provides inward positions by immediate investor economy and outward positions by immediate investment economy",
            ],
            footer,
            body_font=1600,
        ),
        ppt_slide(
            "Data Provenance: Local Files",
            [
                "GLEIF raw files: data/raw/gleif_{country}_lei.parquet",
                "GLEIF relationships: data/interim/gleif_{country}_relationships.parquet",
                "Related parent LEIs: data/raw/gleif_{country}_related_lei.parquet",
                "Reporting exceptions: data/interim/gleif_{country}_reporting_exceptions.parquet",
                "FDI extract: data/dataset_2026-04-22T22_38_34.423483046Z_DEFAULT_INTEGRATION_IMF.STA_DIP_12.0.1.csv",
                "Bilateral LEI counts: data/processed/gleif_bilateral_entity_counts.csv",
            ],
            footer,
            body_font=1450,
        ),
        ppt_slide(
            "Data Provenance: Derived Outputs",
            [
                "UIE assignments: data/processed/{country}/uie_assignments.parquet",
                "Review queue: data/processed/my/uie_review_targets.parquet",
                "Product-vehicle audit: data/processed/my/uie_product_vehicle_targets.parquet",
                "GLEIF search benchmark: data/processed/my/entity_match_benchmark.parquet",
                "Report audit exports: reports/conference_*.csv",
                "Manual overrides: data/manual/uie_overrides.csv",
                "Manual review statuses: data/manual/uie_review_status.csv",
            ],
            footer,
            body_font=1500,
        ),
        ppt_slide(
            "Method: Evidence Ladder",
            [
                "1. Source-backed manual override: analyst verified public evidence",
                "2. GLEIF ultimate parent: strongest machine-readable parent evidence",
                "3. GLEIF direct parent: known immediate parent, not always ultimate owner",
                "4. High-confidence name match: brand-bearing subsidiary signal",
                "5. Address-cluster propagation: useful but penalised for office-hotel clusters",
                "6. Jurisdiction model fallback: triage signal only",
                "7. Reporting exception without country: subsidiary/context signal",
            ],
            footer,
            body_font=1700,
        ),
        ppt_slide(
            "Method: Why Exceptions Matter",
            [
                "A missing GLEIF parent link is not one type of missing data",
                "NON_CONSOLIDATING can confirm subsidiary status without revealing parent country",
                "NO_LEI suggests a parent exists but lacks an LEI",
                "NON_PUBLIC and NO_KNOWN_PERSON imply governance/disclosure limits",
                "Exception reason changes the appropriate compiler follow-up path",
            ],
            footer,
        ),
        ppt_slide(
            "Current ASEAN Data Snapshot",
            raw_lines
            + [
                f"Malaysia refresh note: raw entities now {fmt_int(my_raw['raw_entities'])}; relationship scan checkpointed at {fmt_int(my_raw['relationship_scanned'])}",
                "Final processed UIE tables should be rebuilt after background crawls complete",
            ],
            footer,
            body_font=1650,
        ),
        ppt_slide(
            "UIE Assignment Mix",
            uie_lines
            + [
                "Interpretation: model fallback dominates where known parent evidence is sparse",
                "A high model share is useful for prioritisation but not equivalent to verified UIE coverage",
            ],
            footer,
            body_font=1600,
        ),
        ppt_slide(
            "Malaysia Current Position",
            [
                f"Processed UIE rows: {fmt_int(my_uie['uie_rows'])}",
                f"Foreign-UIE entities: {fmt_int(my['foreign_uie_count'])}",
                f"Known UIE assignments: {fmt_int(my_uie['known_uie_count'])} ({fmt_num(my_uie['known_uie_share_pct'], 1)}%)",
                f"Model fallback assignments: {fmt_int(my_uie['model_fallback_count'])} ({fmt_num(my_uie['model_fallback_share_pct'], 1)}%)",
                f"Address-cluster assignments: {fmt_int(my_uie['address_cluster'])}; name-match assignments: {fmt_int(my_uie['name_match'])}",
                "Reading: useful public-data map, but final findings must privilege known/source-backed evidence",
            ],
            footer,
        ),
        ppt_slide(
            "Ratio Measure: Inward LEI Density",
            [
                "Definition: foreign-UIE LEI count in host / inward FDI stock in USD billions",
                f"Malaysia: {fmt_int(my['foreign_uie_count'])} foreign-UIE LEIs / USD {fmt_num(my['inward_fdi_usd_bn'], 1)}bn",
                f"Malaysia density: {fmt_num(my['inward_lei_density'], 2)} LEIs per USD 1bn",
                "High density: many observed entities per unit of FDI stock",
                "Low density: investment may be concentrated in fewer large vehicles or groups",
                "Caveat: density reflects LEI adoption and entity structure, not statistical coverage alone",
            ],
            footer,
        ),
        ppt_slide(
            "Ratio Measure: Outward LEI Density",
            [
                "Definition: source-owned LEIs observed abroad / outward FDI to the same destination in USD billions",
                "Current scope: matched intra-ASEAN bilateral pairs only",
                "Reason for scope: avoids ASEAN-only LEI numerator divided by global outward-FDI denominator",
                "Interpretation: observed entity intensity of outward investment into each ASEAN host",
                "Caveat: not a global outward density unless the numerator is also global",
            ],
            footer,
        ),
        ppt_slide(
            "Malaysia Outward Density: ASEAN-Observed",
            my_outward_lines
            + [
                "These are bilateral MY -> ASEAN ratios, not Malaysia's total global outward density",
                "A global denominator would require a global MY-owned LEI numerator",
            ],
            footer,
            body_font=1600,
        ),
        ppt_slide(
            "ASEAN Outward Density Summary",
            outward_summary_lines[:8]
            + [
                "Source totals are sums over matched intra-ASEAN outward pairs only",
                "Use for structure comparison, not for full outward coverage claims",
            ],
            footer,
            body_font=1500,
        ),
        ppt_slide("Largest Current Host Results", density_lines, footer, body_font=1650),
        ppt_slide(
            "CDIS/DIP Validation",
            corr_lines
            + [
                "Pooled inward rank correlation is positive and statistically significant",
                "Purpose is macro plausibility, not legal-entity ground truth",
                "Entity counts should correlate directionally with FDI stocks, but not linearly",
            ],
            footer,
        ),
        ppt_slide(
            "Malaysia Review Queue",
            [
                f"Main review queue: {len(review):,} operating/financial/manual-review entities",
                f"Ordinary todos in main queue: {review_status.get('todo', 0):,}",
                f"Verified/source-backed rows in main queue: {review_status.get('verified', 0):,}",
                f"Separate product-vehicle audit: {len(product_vehicles):,} rows",
                f"Product rule applied: {product_status.get('product_vehicle_rule_applied', 0):,}",
                "Implication: product vehicles no longer crowd the operating-company UIE queue",
            ],
            footer,
        ),
        ppt_slide(
            "GLEIF Search Benchmark",
            [
                f"Benchmark rows: {len(benchmark):,} from top Malaysia review targets",
                f"Self-entity found: {benchmark_flags.get('self_entity_found', 0):,}",
                f"Candidate country differs from current UIE: {benchmark_flags.get('candidate_country_differs_from_current_uie', 0):,}",
                "Interpretation: useful for entity resolution and candidate generation",
                "Not sufficient as standalone UIE evidence because search often returns the resident entity itself",
            ],
            footer,
        ),
        ppt_slide(
            "Compiler Use Case",
            [
                "High-confidence subset: manual overrides + GLEIF ultimate/direct parents",
                "Review subset: name matches, address clusters, exception classes, high-priority model cases",
                "Macro-check subset: compare bilateral entity counts and density with CDIS positions",
                "Governance requirement: preserve source, evidence tier, timestamp and review status",
                "Statistical role: pre-compilation intelligence layer feeding official reconciliation",
            ],
            footer,
        ),
        ppt_slide(
            "Interpretation of Findings",
            [
                "GLEIF is valuable as a map of observable legal-entity structure",
                "Known parent evidence is credible but incomplete",
                "Model fallback gives broad coverage but should be read as triage",
                "Malaysia's fund/product vehicle bottleneck is now isolated in a separate audit file",
                "Sector evidence such as GEM is promising for energy/heavy industry, not as a general company matcher",
            ],
            footer,
        ),
        ppt_slide(
            "Limitations",
            [
                "Current results are preliminary while refreshed crawls are still running",
                "LEI adoption differs by sector and jurisdiction",
                "GLEIF parent reporting has non-random gaps",
                "Address clustering can reflect service-provider addresses rather than common control",
                "Model probabilities are uncalibrated ranking signals",
                "CDIS validation is macro plausibility, not entity-level validation",
            ],
            footer,
        ),
        ppt_slide(
            "Next Work Programme",
            [
                "Complete refreshed GLEIF crawls and rebuild all outputs",
                "Review product-vehicle rule after refreshed ASEAN outputs are rebuilt",
                "Add GEM as auditable sector-specific evidence layer",
                "Produce separate high-confidence and full-discovery result tables",
                "Refresh density and CDIS validation after crawl completion",
                "Prepare final BIS-style paper and conference deck from refreshed snapshot",
            ],
            footer,
        ),
    ]

    path = REPORTS / f"asean_uie_bis_style_deck_sceptical_readers_provenance_fund_rule_{TODAY}.pptx"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ppt_content_types(len(slides)))
        z.writestr("_rels/.rels", PPT_ROOT_RELS)
        z.writestr("ppt/presentation.xml", ppt_presentation(len(slides)))
        z.writestr("ppt/_rels/presentation.xml.rels", ppt_rels(len(slides)))
        z.writestr("ppt/slideMasters/slideMaster1.xml", PPT_SLIDE_MASTER)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", PPT_SLIDE_MASTER_RELS)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", PPT_SLIDE_LAYOUT)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", PPT_SLIDE_LAYOUT_RELS)
        z.writestr("ppt/theme/theme1.xml", PPT_THEME)
        z.writestr("docProps/core.xml", core_props())
        z.writestr("docProps/app.xml", app_props(len(slides)))
        for idx, slide in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{idx}.xml", slide)
            z.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", PPT_SLIDE_RELS)
    return path


def ppt_content_types(n: int) -> str:
    slides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, n + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
{slides}
</Types>"""


PPT_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def ppt_presentation(n: int) -> str:
    ids = "\n".join(f'<p:sldId id="{255+i}" r:id="rId{i}"/>' for i in range(1, n + 1))
    master_rid = f"rId{n + 1}"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="{master_rid}"/></p:sldMasterIdLst>
<p:sldIdLst>{ids}</p:sldIdLst>
<p:sldSz cx="12192000" cy="6858000" type="wide"/>
<p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""


def ppt_rels(n: int) -> str:
    rels = "\n".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, n + 1)
    )
    rels += (
        f'\n<Relationship Id="rId{n + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" '
        'Target="slideMasters/slideMaster1.xml"/>'
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>"""


PPT_SLIDE_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>"""

PPT_SLIDE_MASTER = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""

PPT_SLIDE_MASTER_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""

PPT_SLIDE_LAYOUT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
<p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""

PPT_SLIDE_LAYOUT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""

PPT_THEME = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Conference">
<a:themeElements>
<a:clrScheme name="Conference"><a:dk1><a:srgbClr val="1F2937"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="334155"/></a:dk2><a:lt2><a:srgbClr val="E5E7EB"/></a:lt2><a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="059669"/></a:accent2><a:accent3><a:srgbClr val="D97706"/></a:accent3><a:accent4><a:srgbClr val="7C3AED"/></a:accent4><a:accent5><a:srgbClr val="DC2626"/></a:accent5><a:accent6><a:srgbClr val="0891B2"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme>
<a:fontScheme name="Office"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
<a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
</a:themeElements>
<a:objectDefaults/><a:extraClrSchemeLst/>
</a:theme>"""


def core_props() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>LEI-Based UIE Discovery for ASEAN</dc:title>
<dc:creator>Codex</dc:creator>
<cp:lastModifiedBy>Codex</cp:lastModifiedBy>
<dcterms:created xsi:type="dcterms:W3CDTF">{TODAY}T00:00:00Z</dcterms:created>
<dcterms:modified xsi:type="dcterms:W3CDTF">{TODAY}T00:00:00Z</dcterms:modified>
</cp:coreProperties>"""


def app_props(n: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
<Application>Codex OOXML Generator</Application><Slides>{n}</Slides>
</Properties>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate conference paper/deck outputs from current pipeline results.")
    parser.add_argument("--word-only", action="store_true", help="Generate only the Word-style writeup")
    parser.add_argument("--deck-only", action="store_true", help="Generate only the PowerPoint deck")
    args = parser.parse_args()
    if args.word_only and args.deck_only:
        raise SystemExit("--word-only and --deck-only cannot both be set")

    REPORTS.mkdir(exist_ok=True)
    metrics = collect_metrics()
    # Preserve country display order in CSV output.
    raw_order = {country: idx for idx, country in enumerate(COUNTRIES)}
    for key in ["raw", "uie", "density"]:
        if key in metrics and "country" in metrics[key]:
            metrics[key]["country_order"] = metrics[key]["country"].map(raw_order)
            metrics[key] = metrics[key].sort_values("country_order").drop(columns=["country_order"])
    flatten_metrics(metrics)
    if not args.deck_only:
        docx = build_word(metrics)
        print(docx)
    if not args.word_only:
        pptx = build_powerpoint(metrics)
        print(pptx)


if __name__ == "__main__":
    main()
