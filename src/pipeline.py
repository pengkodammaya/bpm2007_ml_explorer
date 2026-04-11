from __future__ import annotations

import pandas as pd
from config import RAW_DIR, INTERIM_DIR, PROCESSED_DIR
from src.gleif import (
    ENTITY_COLUMNS,
    RELATIONSHIP_COLUMNS,
    fetch_lei_records_by_lei,
    fetch_malaysia_lei_records,
    fetch_relationships_for_lei,
    normalize_entity_records,
    normalize_relationship_records,
)
from src.graph_build import build_di_graph, graph_summary, parent_flags, add_graph_features
from src.io_helpers import save_df, save_csv, load_df, load_optional_df
from src.edgar import fetch_company_tickers, match_entities_to_edgar
from src.scoring import compute_coverage_score


def _combine_entity_sets(*frames: pd.DataFrame) -> pd.DataFrame:
    normalized = [normalize_entity_records(frame) for frame in frames if frame is not None and not frame.empty]
    if not normalized:
        return pd.DataFrame(columns=ENTITY_COLUMNS)
    return (
        pd.concat(normalized, ignore_index=True)
        .dropna(subset=["lei"])
        .drop_duplicates(subset=["lei"], keep="first")
        .reset_index(drop=True)
    )


def _add_size_proxy(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    defaults = {
        "in_degree": 0,
        "out_degree": 0,
        "weakly_connected_component_size": 1,
        "pagerank": 0.0,
        "degree_centrality": 0.0,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default)

    out["size_proxy"] = (
        out["in_degree"].astype(float)
        + out["out_degree"].astype(float)
        + out["weakly_connected_component_size"].astype(float)
        + (out["pagerank"].astype(float) * 1000)
        + (out["degree_centrality"].astype(float) * 100)
    )
    return out


def _as_int_flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _foreign_parent_flags(relationships: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    relationships = normalize_relationship_records(relationships)
    if relationships.empty:
        return pd.DataFrame(columns=["lei", "foreign_parent"])

    entity_countries = (
        normalize_entity_records(entities)[["lei", "country_legal"]]
        .dropna(subset=["lei"])
        .drop_duplicates(subset=["lei"])
        .rename(columns={"lei": "target_lei", "country_legal": "target_country"})
    )

    relationship_targets = relationships[["source_lei", "target_lei", "relationship_type"]].merge(
        entity_countries,
        on="target_lei",
        how="left",
    )

    return (
        relationship_targets.loc[
            relationship_targets["relationship_type"].isin(["direct_parent", "ultimate_parent"])
            & relationship_targets["target_country"].notna()
            & (relationship_targets["target_country"] != "MY"),
            ["source_lei"],
        ]
        .drop_duplicates()
        .assign(foreign_parent=1)
        .rename(columns={"source_lei": "lei"})
    )


def run_gleif_pull(
    max_relationship_entities: int = 100,
    lei_max_pages: int = 50,
    lei_page_size: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    entities = normalize_entity_records(fetch_malaysia_lei_records(max_pages=lei_max_pages, page_size=lei_page_size))
    save_df(entities, RAW_DIR / "gleif_malaysia_lei")

    rels = []
    for lei in entities["lei"].dropna().unique()[:max_relationship_entities]:
        try:
            df = fetch_relationships_for_lei(lei)
            if not df.empty:
                rels.append(df)
        except Exception as e:
            print(f"[WARN] relationship fetch failed for {lei}: {e}")

    relationships = normalize_relationship_records(pd.concat(rels, ignore_index=True) if rels else pd.DataFrame())
    save_df(relationships, INTERIM_DIR / "gleif_malaysia_relationships")

    target_leis = relationships["target_lei"].dropna().unique()
    source_leis = set(entities["lei"].dropna().astype(str))
    related_leis = [lei for lei in target_leis if str(lei) not in source_leis]
    related_entities = fetch_lei_records_by_lei(related_leis)
    save_df(related_entities, RAW_DIR / "gleif_related_lei")

    return entities, relationships


def run_graph_and_scoring(enrich_edgar: bool = True) -> pd.DataFrame:
    malaysia_entities = normalize_entity_records(load_df(RAW_DIR / "gleif_malaysia_lei"))
    related_entities = normalize_entity_records(load_optional_df(RAW_DIR / "gleif_related_lei", ENTITY_COLUMNS))
    entities = _combine_entity_sets(malaysia_entities, related_entities)
    relationships = normalize_relationship_records(load_optional_df(INTERIM_DIR / "gleif_malaysia_relationships", RELATIONSHIP_COLUMNS))

    g = build_di_graph(entities, relationships)
    summary = graph_summary(g)
    summary = add_graph_features(g, summary)

    flags = parent_flags(relationships)
    summary = summary.merge(flags, on="lei", how="left")
    summary["has_parent_link"] = _as_int_flag(summary["has_parent_link"])
    summary["has_ultimate_link"] = _as_int_flag(summary["has_ultimate_link"])

    summary = _add_size_proxy(summary)

    if enrich_edgar:
        try:
            tickers = fetch_company_tickers()
            summary = match_entities_to_edgar(summary, tickers)
        except Exception as e:
            print(f"[WARN] EDGAR enrichment failed: {e}")
            summary["appears_in_edgar"] = 0
            summary["cik"] = None
            summary["ticker"] = None
            summary["title"] = None
    else:
        summary["appears_in_edgar"] = 0
        summary["cik"] = None
        summary["ticker"] = None
        summary["title"] = None

    summary = summary.merge(_foreign_parent_flags(relationships, entities), on="lei", how="left")
    summary["foreign_parent"] = _as_int_flag(summary["foreign_parent"])

    scored = compute_coverage_score(summary)

    save_df(summary, PROCESSED_DIR / "di_graph_summary")
    save_df(scored, PROCESSED_DIR / "coverage_gap_scores")
    save_csv(scored.head(200), PROCESSED_DIR / "coverage_gap_scores_top200.csv")

    return scored


def run_mock_pipeline() -> pd.DataFrame:
    entities = normalize_entity_records(pd.DataFrame([
        {"lei": "A1", "legal_name": "MALAYSIA HOLDINGS BERHAD", "country_legal": "MY", "city_legal": "KUALA LUMPUR", "category": "GENERAL", "legal_form": "BHD"},
        {"lei": "A2", "legal_name": "ASEAN ENERGY SDN BHD", "country_legal": "MY", "city_legal": "KUALA LUMPUR", "category": "GENERAL", "legal_form": "SDN"},
        {"lei": "P1", "legal_name": "SINGAPORE PARENT LTD", "country_legal": "SG", "city_legal": "SINGAPORE", "category": "GENERAL", "legal_form": "LTD"},
        {"lei": "P2", "legal_name": "GLOBAL ULTIMATE INC", "country_legal": "US", "city_legal": "NEW YORK", "category": "GENERAL", "legal_form": "INC"},
        {"lei": "A3", "legal_name": "PALM EXPORTS BERHAD", "country_legal": "MY", "city_legal": "SHAH ALAM", "category": "GENERAL", "legal_form": "BHD"},
    ]))
    relationships = normalize_relationship_records(pd.DataFrame([
        {"source_lei": "A1", "target_lei": "P1", "relationship_type": "direct_parent", "relationship_status": "ACTIVE", "accounting_standard": "IFRS", "period_end": None},
        {"source_lei": "A1", "target_lei": "P2", "relationship_type": "ultimate_parent", "relationship_status": "ACTIVE", "accounting_standard": "IFRS", "period_end": None},
        {"source_lei": "A2", "target_lei": "P1", "relationship_type": "direct_parent", "relationship_status": "ACTIVE", "accounting_standard": "IFRS", "period_end": None},
    ]))

    save_df(entities, RAW_DIR / "gleif_malaysia_lei")
    save_df(pd.DataFrame(columns=ENTITY_COLUMNS), RAW_DIR / "gleif_related_lei")
    save_df(relationships, INTERIM_DIR / "gleif_malaysia_relationships")

    g = build_di_graph(entities, relationships)
    summary = graph_summary(g)
    summary = add_graph_features(g, summary)

    flags = parent_flags(relationships)
    summary = summary.merge(flags, on="lei", how="left")
    summary["has_parent_link"] = _as_int_flag(summary["has_parent_link"])
    summary["has_ultimate_link"] = _as_int_flag(summary["has_ultimate_link"])
    summary["appears_in_edgar"] = summary["legal_name"].eq("GLOBAL ULTIMATE INC").astype(int)
    summary = summary.merge(_foreign_parent_flags(relationships, entities), on="lei", how="left")
    summary["foreign_parent"] = _as_int_flag(summary["foreign_parent"])
    summary = _add_size_proxy(summary)

    scored = compute_coverage_score(summary)
    save_df(summary, PROCESSED_DIR / "di_graph_summary")
    save_df(scored, PROCESSED_DIR / "coverage_gap_scores")
    save_csv(scored.head(200), PROCESSED_DIR / "coverage_gap_scores_top200.csv")

    return scored
