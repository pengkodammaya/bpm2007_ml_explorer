from __future__ import annotations

import pandas as pd
from config import RAW_DIR, INTERIM_DIR, PROCESSED_DIR
from src.gleif import fetch_malaysia_lei_records, fetch_relationships_for_lei
from src.graph_build import build_di_graph, graph_summary, parent_flags, add_graph_features
from src.io_helpers import save_df, save_csv, load_df
from src.edgar import fetch_company_tickers, match_entities_to_edgar
from src.scoring import compute_coverage_score


def run_gleif_pull(max_relationship_entities: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    entities = fetch_malaysia_lei_records()
    save_df(entities, RAW_DIR / "gleif_malaysia_lei")

    rels = []
    for lei in entities["lei"].dropna().unique()[:max_relationship_entities]:
        try:
            df = fetch_relationships_for_lei(lei)
            if not df.empty:
                rels.append(df)
        except Exception as e:
            print(f"[WARN] relationship fetch failed for {lei}: {e}")

    relationships = pd.concat(rels, ignore_index=True) if rels else pd.DataFrame()
    save_df(relationships, INTERIM_DIR / "gleif_malaysia_relationships")
    return entities, relationships


def run_graph_and_scoring() -> pd.DataFrame:
    entities = load_df(RAW_DIR / "gleif_malaysia_lei")
    relationships = load_df(INTERIM_DIR / "gleif_malaysia_relationships")

    g = build_di_graph(entities, relationships)
    summary = graph_summary(g)
    summary = add_graph_features(g, summary)

    flags = parent_flags(relationships)
    summary = summary.merge(flags, on="lei", how="left")
    summary["has_parent_link"] = summary["has_parent_link"].fillna(0).astype(int)
    summary["has_ultimate_link"] = summary["has_ultimate_link"].fillna(0).astype(int)

    summary["size_proxy"] = (
        summary["in_degree"].fillna(0)
        + summary["out_degree"].fillna(0)
        + summary["weakly_connected_component_size"].fillna(1)
        + (summary["pagerank"].fillna(0) * 1000)
        + (summary["degree_centrality"].fillna(0) * 100)
    )

    try:
        tickers = fetch_company_tickers()
        summary = match_entities_to_edgar(summary, tickers)
    except Exception as e:
        print(f"[WARN] EDGAR enrichment failed: {e}")
        summary["appears_in_edgar"] = 0
        summary["cik"] = None
        summary["ticker"] = None
        summary["title"] = None

    relationship_targets = relationships[["source_lei", "target_lei", "relationship_type"]].copy()
    target_country = entities[["lei", "country_legal"]].rename(columns={"lei": "target_lei", "country_legal": "target_country"})
    relationship_targets = relationship_targets.merge(target_country, on="target_lei", how="left")

    foreign_parent = (
        relationship_targets.loc[
            relationship_targets["relationship_type"].isin(["direct_parent", "ultimate_parent"])
            & relationship_targets["target_country"].notna()
            & (relationship_targets["target_country"] != "MY"),
            ["source_lei"]
        ]
        .drop_duplicates()
        .assign(foreign_parent=1)
        .rename(columns={"source_lei": "lei"})
    )

    summary = summary.merge(foreign_parent, on="lei", how="left")
    summary["foreign_parent"] = summary["foreign_parent"].fillna(0).astype(int)

    scored = compute_coverage_score(summary)

    save_df(summary, PROCESSED_DIR / "di_graph_summary")
    save_df(scored, PROCESSED_DIR / "coverage_gap_scores")
    save_csv(scored.head(200), PROCESSED_DIR / "coverage_gap_scores_top200.csv")

    return scored


def run_mock_pipeline() -> pd.DataFrame:
    entities = pd.DataFrame([
        {"lei": "A1", "legal_name": "MALAYSIA HOLDINGS BERHAD", "country_legal": "MY", "city_legal": "KUALA LUMPUR", "category": "GENERAL", "legal_form": "BHD"},
        {"lei": "A2", "legal_name": "ASEAN ENERGY SDN BHD", "country_legal": "MY", "city_legal": "KUALA LUMPUR", "category": "GENERAL", "legal_form": "SDN"},
        {"lei": "P1", "legal_name": "SINGAPORE PARENT LTD", "country_legal": "SG", "city_legal": "SINGAPORE", "category": "GENERAL", "legal_form": "LTD"},
        {"lei": "P2", "legal_name": "GLOBAL ULTIMATE INC", "country_legal": "US", "city_legal": "NEW YORK", "category": "GENERAL", "legal_form": "INC"},
        {"lei": "A3", "legal_name": "PALM EXPORTS BERHAD", "country_legal": "MY", "city_legal": "SHAH ALAM", "category": "GENERAL", "legal_form": "BHD"},
    ])
    relationships = pd.DataFrame([
        {"source_lei": "A1", "target_lei": "P1", "relationship_type": "direct_parent", "relationship_status": "ACTIVE", "accounting_standard": "IFRS", "period_end": None},
        {"source_lei": "A1", "target_lei": "P2", "relationship_type": "ultimate_parent", "relationship_status": "ACTIVE", "accounting_standard": "IFRS", "period_end": None},
        {"source_lei": "A2", "target_lei": "P1", "relationship_type": "direct_parent", "relationship_status": "ACTIVE", "accounting_standard": "IFRS", "period_end": None},
    ])

    save_df(entities, RAW_DIR / "gleif_malaysia_lei")
    save_df(relationships, INTERIM_DIR / "gleif_malaysia_relationships")

    g = build_di_graph(entities, relationships)
    summary = graph_summary(g)
    summary = add_graph_features(g, summary)

    flags = parent_flags(relationships)
    summary = summary.merge(flags, on="lei", how="left")
    summary["has_parent_link"] = summary["has_parent_link"].fillna(0).astype(int)
    summary["has_ultimate_link"] = summary["has_ultimate_link"].fillna(0).astype(int)
    summary["appears_in_edgar"] = [0, 0, 0, 1, 0]
    summary["foreign_parent"] = summary["country"].fillna("MY").ne("MY").astype(int)
    summary["size_proxy"] = (
        summary["in_degree"].fillna(0)
        + summary["out_degree"].fillna(0)
        + summary["weakly_connected_component_size"].fillna(1)
        + (summary["pagerank"].fillna(0) * 1000)
        + (summary["degree_centrality"].fillna(0) * 100)
    )

    scored = compute_coverage_score(summary)
    save_df(summary, PROCESSED_DIR / "di_graph_summary")
    save_df(scored, PROCESSED_DIR / "coverage_gap_scores")
    save_csv(scored.head(200), PROCESSED_DIR / "coverage_gap_scores_top200.csv")

    return scored
