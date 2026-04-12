from __future__ import annotations

import pandas as pd
from config import RAW_DIR, INTERIM_DIR, PROCESSED_DIR, INFERENCE_DIR
from src.gleif import (
    ENTITY_COLUMNS,
    RELATIONSHIP_COLUMNS,
    fetch_country_lei_records,
    fetch_lei_records_by_lei,
    fetch_malaysia_lei_records,
    fetch_relationships_for_lei,
    normalize_entity_records,
    normalize_relationship_records,
)
from src.entity_analysis import full_structural_report, compare_countries
from src.graph_build import build_di_graph, graph_summary, parent_flags, add_graph_features
from src.io_helpers import save_df, save_csv, load_df, load_optional_df
from src.edgar import fetch_company_tickers, match_entities_to_edgar
from src.scoring import compute_coverage_score
from src.reporting_exceptions import fetch_reporting_exceptions, enrich_with_exception_flags, EXCEPTION_COLUMNS
from src.name_inference import (
    extract_parent_brand_tokens,
    fuzzy_match_names,
    build_phase1_inferred_edges,
    phase1_summary,
)


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


def run_entity_analysis(
    country: str = "MY",
    max_pages: int = 50,
    page_size: int = 200,
    skip_pull: bool = False,
) -> dict:
    """Fetch entities for a country and run full structural analysis."""
    country = country.upper()
    cache_path = RAW_DIR / f"gleif_{country.lower()}_lei"
    # Backwards compatibility: MY data was previously saved as gleif_malaysia_lei
    legacy_path = RAW_DIR / "gleif_malaysia_lei" if country == "MY" else None

    if skip_pull:
        try:
            entities = normalize_entity_records(load_df(cache_path))
        except FileNotFoundError:
            if legacy_path:
                entities = normalize_entity_records(load_df(legacy_path))
            else:
                raise
        print(f"[INFO] Loaded {len(entities):,} cached entities for {country}")
    else:
        print(f"[INFO] Fetching LEI records for country={country}...")
        entities = normalize_entity_records(
            fetch_country_lei_records(country, max_pages=max_pages, page_size=page_size)
        )
        save_df(entities, cache_path)
        print(f"[INFO] Fetched {len(entities):,} entities for {country}")

    report = full_structural_report(entities, country)

    # Save analysis artifacts
    analysis_dir = PROCESSED_DIR / "entity_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for key in ["geographic_concentration", "legal_form_distribution",
                "category_distribution", "entity_status_distribution",
                "registration_timeline", "hq_vs_legal_mismatch"]:
        df = report[key]
        if isinstance(df, pd.DataFrame) and not df.empty:
            save_csv(df, analysis_dir / f"{country.lower()}_{key}.csv")

    return report


def run_comparative_analysis(
    countries: list[str],
    max_pages: int = 50,
    page_size: int = 200,
    skip_pull: bool = False,
) -> pd.DataFrame:
    """Fetch entities for multiple countries and compare structural profiles."""
    country_frames: dict[str, pd.DataFrame] = {}

    for country in countries:
        country = country.upper()
        cache_path = RAW_DIR / f"gleif_{country.lower()}_lei"

        if skip_pull:
            try:
                entities = normalize_entity_records(load_df(cache_path))
                print(f"[INFO] Loaded {len(entities):,} cached entities for {country}")
            except Exception:
                print(f"[WARN] No cached data for {country}, fetching...")
                entities = normalize_entity_records(
                    fetch_country_lei_records(country, max_pages=max_pages, page_size=page_size)
                )
                save_df(entities, cache_path)
                print(f"[INFO] Fetched {len(entities):,} entities for {country}")
        else:
            print(f"[INFO] Fetching LEI records for country={country}...")
            entities = normalize_entity_records(
                fetch_country_lei_records(country, max_pages=max_pages, page_size=page_size)
            )
            save_df(entities, cache_path)
            print(f"[INFO] Fetched {len(entities):,} entities for {country}")

        country_frames[country] = entities

    comparison = compare_countries(country_frames)

    analysis_dir = PROCESSED_DIR / "entity_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    save_csv(comparison, analysis_dir / "cross_country_comparison.csv")

    return comparison


def run_gleif_pull(
    max_relationship_entities: int = 100,
    lei_max_pages: int = 50,
    lei_page_size: int = 200,
    scan_all: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    entities = normalize_entity_records(fetch_malaysia_lei_records(max_pages=lei_max_pages, page_size=lei_page_size))
    save_df(entities, RAW_DIR / "gleif_malaysia_lei")

    leis = entities["lei"].dropna().unique()
    if not scan_all:
        leis = leis[:max_relationship_entities]

    rels = []
    total = len(leis)
    found = 0
    for i, lei in enumerate(leis):
        try:
            df = fetch_relationships_for_lei(lei)
            if not df.empty:
                rels.append(df)
                found += 1
                print(f"[{i+1}/{total}] {lei} - parent found ({found} total)", flush=True)
            elif (i + 1) % 200 == 0:
                print(f"[{i+1}/{total}] scanning... {found} with parents so far", flush=True)
        except Exception as e:
            print(f"[WARN] relationship fetch failed for {lei}: {e}")

    print(f"[INFO] Relationship scan complete: {found}/{total} entities have parent data")

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


# ---------------------------------------------------------------------------
# Inference pipeline — phases 0.5, 1, (2, 3 to follow)
# ---------------------------------------------------------------------------

def run_reporting_exceptions(skip_pull: bool = False) -> pd.DataFrame:
    """Phase 0.5: Fetch reporting exceptions for entities without known parents."""
    cache_path = INTERIM_DIR / "gleif_reporting_exceptions"

    if skip_pull:
        try:
            exceptions = load_df(cache_path)
            print(f"[INFO] Loaded {len(exceptions):,} cached reporting exceptions")
            return exceptions
        except FileNotFoundError:
            print("[WARN] No cached exceptions, fetching from API...")

    entities = normalize_entity_records(load_df(RAW_DIR / "gleif_malaysia_lei"))
    relationships = normalize_relationship_records(
        load_optional_df(INTERIM_DIR / "gleif_malaysia_relationships", RELATIONSHIP_COLUMNS)
    )

    # Only scan entities without known parent relationships
    known_leis = set(relationships["source_lei"].dropna().unique()) if not relationships.empty else set()
    leis_to_scan = [lei for lei in entities["lei"].dropna().unique() if lei not in known_leis]

    print(f"[INFO] Scanning {len(leis_to_scan):,} entities for reporting exceptions...")
    exceptions = fetch_reporting_exceptions(leis_to_scan)
    save_df(exceptions, cache_path)

    # Summary
    if not exceptions.empty:
        reason_counts = exceptions["exception_reason"].value_counts()
        print(f"\n--- Reporting Exception Summary ---")
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}")
    else:
        print("[INFO] No reporting exceptions found")

    return exceptions


def run_phase1_name_inference(
    threshold: int = 80,
    skip_pull: bool = False,
) -> pd.DataFrame:
    """Phase 1: Name pattern extraction and fuzzy matching."""
    # Load required data
    entities = normalize_entity_records(load_df(RAW_DIR / "gleif_malaysia_lei"))
    parents = normalize_entity_records(load_optional_df(RAW_DIR / "gleif_related_lei", ENTITY_COLUMNS))
    relationships = normalize_relationship_records(
        load_optional_df(INTERIM_DIR / "gleif_malaysia_relationships", RELATIONSHIP_COLUMNS)
    )

    if parents.empty:
        print("[WARN] No parent entities loaded. Run the full pipeline first.")
        return pd.DataFrame()

    # Extract brand tokens from known parents
    brand_tokens = extract_parent_brand_tokens(parents)
    print(f"[INFO] Extracted {len(brand_tokens)} brand tokens from {len(parents)} parent entities")

    if brand_tokens.empty:
        print("[WARN] No valid brand tokens extracted")
        return pd.DataFrame()

    # Get known LEIs to exclude
    known_leis = set(relationships["source_lei"].dropna().unique()) if not relationships.empty else set()

    # Run fuzzy matching
    print(f"[INFO] Fuzzy matching {len(entities):,} entities against {len(brand_tokens)} brand tokens (threshold={threshold})...")
    matches = fuzzy_match_names(entities, brand_tokens, known_leis=known_leis, threshold=threshold)

    # Build inferred edges
    edges = build_phase1_inferred_edges(matches, min_score=threshold)

    # Save artifacts
    save_df(matches, INFERENCE_DIR / "phase1_name_matches")
    save_df(edges, INFERENCE_DIR / "phase1_inferred_edges")

    if not matches.empty:
        save_csv(matches, INFERENCE_DIR / "phase1_name_matches.csv")

    # Print summary
    summary = phase1_summary(matches)
    print(f"\n--- Phase 1: Name Inference Summary ---")
    print(f"  Total matches:      {summary['total_matches']}")
    print(f"  High confidence:    {summary.get('high_confidence', 0)} (score >= 90)")
    print(f"  Medium confidence:  {summary.get('medium_confidence', 0)} (score 80-89)")
    print(f"  Unique brands:      {summary.get('unique_parent_brands', 0)}")
    if summary['total_matches'] > 0:
        print(f"  Average score:      {summary.get('avg_score', 0)}")

    return matches


def run_full_inference_pipeline(
    threshold: int = 80,
    skip_pull: bool = False,
) -> dict:
    """Run all inference phases sequentially.

    Returns a dict with results from each phase.
    """
    results = {}

    print("=" * 70)
    print("PHASE 0.5: REPORTING EXCEPTIONS")
    print("=" * 70)
    results["exceptions"] = run_reporting_exceptions(skip_pull=skip_pull)

    print("\n" + "=" * 70)
    print("PHASE 1: NAME PATTERN MATCHING")
    print("=" * 70)
    results["name_matches"] = run_phase1_name_inference(threshold=threshold, skip_pull=skip_pull)

    # Phases 2 and 3 will be added here
    print("\n" + "=" * 70)
    print("INFERENCE PIPELINE COMPLETE")
    print("=" * 70)

    total_inferred = len(results.get("name_matches", pd.DataFrame()))
    total_exceptions = len(results.get("exceptions", pd.DataFrame()))
    print(f"  Reporting exceptions found:  {total_exceptions}")
    print(f"  Name-inferred subsidiaries:  {total_inferred}")

    return results
