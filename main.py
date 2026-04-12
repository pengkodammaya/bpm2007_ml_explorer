from __future__ import annotations

import argparse
from src.pipeline import (
    run_entity_analysis,
    run_comparative_analysis,
    run_gleif_pull,
    run_graph_and_scoring,
    run_mock_pipeline,
    run_reporting_exceptions,
    run_phase1_name_inference,
    run_phase2_address_clustering,
    run_phase3_jurisdiction_prediction,
    run_phase4_graph_prediction,
    run_full_inference_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="BPM7 ESIE inference pipeline")
    parser.add_argument("--country", type=str, default="MY",
                        help="ISO-2 country code (default: MY). E.g. MY, SG, PH, TH, ID")
    parser.add_argument("--relationships-limit", type=int, default=100)
    parser.add_argument("--lei-max-pages", type=int, default=50)
    parser.add_argument("--lei-page-size", type=int, default=200)
    parser.add_argument("--skip-edgar", action="store_true")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--scan-all", action="store_true", help="Scan all entities for parent relationships (not just first N)")
    parser.add_argument("--use-mock-data", action="store_true")

    # Entity analysis options
    parser.add_argument(
        "--entity-analysis",
        metavar="COUNTRY",
        help="Run entity discovery and structural analysis for a country (ISO-2 code, e.g. MY, SG, US)",
    )
    parser.add_argument(
        "--compare-countries",
        metavar="CODES",
        help="Comma-separated ISO-2 codes for cross-country comparison (e.g. MY,SG,TH,ID,PH)",
    )

    # Inference pipeline options
    parser.add_argument("--run-inference", action="store_true", help="Run full parent inference pipeline (phases 0.5-4)")
    parser.add_argument("--phase1-only", action="store_true", help="Run only Phase 1 name inference")
    parser.add_argument("--phase2-only", action="store_true", help="Run only Phase 2 address clustering")
    parser.add_argument("--phase3-only", action="store_true", help="Run only Phase 3 jurisdiction prediction")
    parser.add_argument("--phase4-only", action="store_true", help="Run only Phase 4 graph link prediction")
    parser.add_argument("--exceptions-only", action="store_true", help="Run only Phase 0.5 reporting exceptions")
    parser.add_argument("--fuzzy-threshold", type=int, default=80, help="Phase 1 fuzzy match threshold (0-100, default 80)")
    parser.add_argument("--min-cluster-size", type=int, default=3, help="Phase 2 minimum address cluster size (default 3)")

    args = parser.parse_args()
    country = args.country.upper()

    # --- Entity analysis mode ---
    if args.entity_analysis:
        report = run_entity_analysis(
            country=args.entity_analysis,
            max_pages=args.lei_max_pages,
            page_size=args.lei_page_size,
            skip_pull=args.skip_pull,
        )
        _print_entity_report(report)
        return

    # --- Cross-country comparison mode ---
    if args.compare_countries:
        countries = [c.strip() for c in args.compare_countries.split(",")]
        comparison = run_comparative_analysis(
            countries=countries,
            max_pages=args.lei_max_pages,
            page_size=args.lei_page_size,
            skip_pull=args.skip_pull,
        )
        print("\n" + "=" * 70)
        print("CROSS-COUNTRY STRUCTURAL COMPARISON")
        print("=" * 70)
        print(comparison.to_string(index=False))
        return

    # --- Inference pipeline modes ---
    if args.run_inference:
        run_full_inference_pipeline(
            threshold=args.fuzzy_threshold,
            skip_pull=args.skip_pull,
            min_cluster=args.min_cluster_size,
            country=country,
        )
        return

    if args.exceptions_only:
        run_reporting_exceptions(skip_pull=args.skip_pull, country=country)
        return

    if args.phase2_only:
        run_phase2_address_clustering(
            skip_pull=args.skip_pull,
            min_cluster=args.min_cluster_size,
            country=country,
        )
        return

    if args.phase4_only:
        run_phase4_graph_prediction(skip_pull=args.skip_pull, country=country)
        return

    if args.phase3_only:
        run_phase3_jurisdiction_prediction(skip_pull=args.skip_pull, country=country)
        return

    if args.phase1_only:
        run_phase1_name_inference(
            threshold=args.fuzzy_threshold,
            skip_pull=args.skip_pull,
            country=country,
        )
        return

    # --- Original pipeline modes ---
    if args.use_mock_data:
        print("[INFO] Running mock pipeline...")
        scored = run_mock_pipeline()
        print(scored.head(20).to_string(index=False))
        return

    if not args.skip_pull:
        print(f"[INFO] Pulling GLEIF {country} entities and relationships...")
        entities, relationships = run_gleif_pull(
            max_relationship_entities=args.relationships_limit,
            lei_max_pages=args.lei_max_pages,
            lei_page_size=args.lei_page_size,
            scan_all=args.scan_all,
            country=country,
        )
        print(f"[INFO] Entities: {len(entities):,}")
        print(f"[INFO] Relationships: {len(relationships):,}")

    print("[INFO] Building graph and scoring...")
    scored = run_graph_and_scoring(enrich_edgar=not args.skip_edgar, country=country)
    print(scored.head(20).to_string(index=False))


def _print_entity_report(report: dict) -> None:
    """Pretty-print a structural analysis report to stdout."""
    disc = report["discovery"]
    profile = report["profile"]

    print("\n" + "=" * 70)
    print(f"ENTITY DISCOVERY & STRUCTURAL ANALYSIS: {profile['country']}")
    print("=" * 70)

    print(f"\n--- Discovery Summary ---")
    print(f"  Total entities:       {disc['total_records']:,}")
    print(f"  Unique LEIs:          {disc['unique_leis']:,}")
    print(f"  Active:               {disc['active_entities']:,}")
    print(f"  Inactive:             {disc['inactive_entities']:,}")
    print(f"  Legal form types:     {disc['legal_forms']}")
    print(f"  Cities (legal addr):  {disc['cities_legal']}")
    print(f"  Earliest registration: {disc['earliest_registration']}")
    print(f"  Latest registration:   {disc['latest_registration']}")

    print(f"\n--- Category Distribution ---")
    print(report["category_distribution"].to_string(index=False))

    print(f"\n--- Entity Status ---")
    print(report["entity_status_distribution"].to_string(index=False))

    print(f"\n--- Top Geographic Concentrations ---")
    print(report["geographic_concentration"].to_string(index=False))

    print(f"\n--- Legal Form Distribution (top 20) ---")
    print(report["legal_form_distribution"].to_string(index=False))

    print(f"\n--- Registration Timeline ---")
    print(report["registration_timeline"].to_string(index=False))

    mismatch = report["hq_vs_legal_mismatch"]
    print(f"\n--- HQ vs Legal Address Mismatch ---")
    print(f"  {len(mismatch)} entities ({profile['hq_mismatch_pct']}%) have HQ in a different country")
    if not mismatch.empty:
        print(mismatch.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
