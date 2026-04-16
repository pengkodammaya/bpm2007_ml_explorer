"""Recoverable ASEAN pipeline runner.

Runs entity pull, relationship scan, and inference pipeline for each
ASEAN country.  Each stage is checkpointed — if the script crashes or
is interrupted, re-running it skips completed stages automatically.

Usage:
    python run_asean.py                 # run all pending countries
    python run_asean.py --status        # show progress
    python run_asean.py --country SG    # run only Singapore
    python run_asean.py --skip SG       # run all except Singapore
    python run_asean.py --inference-only # skip entity/relationship pull, run inference on cached data
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import PROCESSED_DIR, country_paths

CHECKPOINT_FILE = PROCESSED_DIR / "asean_run_checkpoint.json"

ASEAN_COUNTRIES = ["MY", "SG", "TH", "ID", "PH", "VN", "KH", "BN", "LA", "MM", "TL"]

# Stages in execution order
STAGES = ["entity_pull", "inference", "scoring"]


def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {}


def _save_checkpoint(data: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(data, indent=2))


def _mark(checkpoint: dict, country: str, stage: str, status: str, detail: str = "") -> dict:
    key = f"{country}_{stage}"
    checkpoint[key] = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": detail,
    }
    _save_checkpoint(checkpoint)
    return checkpoint


def _is_done(checkpoint: dict, country: str, stage: str) -> bool:
    key = f"{country}_{stage}"
    return checkpoint.get(key, {}).get("status") == "done"


def _has_cached_data(country: str) -> bool:
    """Check if entity data already exists for this country."""
    paths = country_paths(country)
    primary = paths["raw_entities"].with_suffix(".parquet")
    legacy = paths.get("legacy", {}).get("raw_entities")
    if primary.exists():
        return True
    if legacy and legacy.with_suffix(".parquet").exists():
        return True
    return False


def run_entity_pull(country: str, checkpoint: dict) -> dict:
    """Stage 1: Fetch entities, scan relationships, fetch parent entities."""
    if _is_done(checkpoint, country, "entity_pull"):
        print(f"  [{country}] entity_pull: already done, skipping")
        return checkpoint

    # Check if data already cached (e.g. from a prior manual run)
    if _has_cached_data(country):
        paths = country_paths(country)
        rels_path = paths["interim_relationships"].with_suffix(".parquet")
        related_path = paths["raw_related"].with_suffix(".parquet")
        legacy_rels = paths.get("legacy", {}).get("interim_relationships")
        legacy_related = paths.get("legacy", {}).get("raw_related")
        has_rels = rels_path.exists() or (legacy_rels and legacy_rels.with_suffix(".parquet").exists())
        has_related = related_path.exists() or (legacy_related and legacy_related.with_suffix(".parquet").exists())

        if has_rels and has_related:
            print(f"  [{country}] entity_pull: cached data found, marking done")
            return _mark(checkpoint, country, "entity_pull", "done", "cached")
        elif has_rels and not has_related:
            # Relationships exist but parent entities not yet fetched — do just that
            print(f"  [{country}] entity_pull: relationships cached, fetching parent entities...")
            import pandas as pd
            from src.gleif import fetch_lei_records_by_lei
            from src.io_helpers import save_df, load_df
            try:
                rels = load_df(rels_path if rels_path.exists() else legacy_rels)
                entities = load_df(paths["raw_entities"] if paths["raw_entities"].with_suffix(".parquet").exists()
                                   else paths.get("legacy", {}).get("raw_entities"))
                target_leis = rels["target_lei"].dropna().unique()
                source_leis = set(entities["lei"].dropna().astype(str))
                related_leis = [lei for lei in target_leis if str(lei) not in source_leis]
                related = fetch_lei_records_by_lei(related_leis)
                save_df(related, paths["raw_related"])
                detail = f"cached, fetched {len(related)} parent entities"
                print(f"  [{country}] entity_pull: done ({detail})")
                return _mark(checkpoint, country, "entity_pull", "done", detail)
            except Exception as e:
                print(f"  [{country}] entity_pull: parent fetch failed - {e}, running full pull")

    from src.pipeline import run_gleif_pull
    print(f"  [{country}] entity_pull: starting...")
    checkpoint = _mark(checkpoint, country, "entity_pull", "running")

    try:
        entities, relationships = run_gleif_pull(
            lei_max_pages=100,
            lei_page_size=200,
            scan_all=True,
            country=country,
        )
        detail = f"entities={len(entities)}, relationships={len(relationships)}"
        checkpoint = _mark(checkpoint, country, "entity_pull", "done", detail)
        print(f"  [{country}] entity_pull: done ({detail})")
    except Exception as e:
        checkpoint = _mark(checkpoint, country, "entity_pull", "failed", str(e))
        print(f"  [{country}] entity_pull: FAILED - {e}")

    return checkpoint


def run_inference(country: str, checkpoint: dict) -> dict:
    """Stage 2: Run full inference pipeline (phases 0.5-4)."""
    if _is_done(checkpoint, country, "inference"):
        print(f"  [{country}] inference: already done, skipping")
        return checkpoint

    from src.pipeline import run_full_inference_pipeline
    print(f"  [{country}] inference: starting...")
    checkpoint = _mark(checkpoint, country, "inference", "running")

    try:
        results = run_full_inference_pipeline(
            threshold=80,
            skip_pull=True,
            min_cluster=3,
            country=country,
        )
        exc_count = len(results.get("exceptions", []))
        name_count = len(results.get("name_matches", []))
        cluster_count = len(results.get("address_clusters", []))
        detail = f"exceptions={exc_count}, name_matches={name_count}, clusters={cluster_count}"
        checkpoint = _mark(checkpoint, country, "inference", "done", detail)
        print(f"  [{country}] inference: done ({detail})")
    except Exception as e:
        checkpoint = _mark(checkpoint, country, "inference", "failed", str(e))
        print(f"  [{country}] inference: FAILED - {e}")

    return checkpoint


def run_scoring(country: str, checkpoint: dict) -> dict:
    """Stage 3: Run graph scoring with inference signals."""
    if _is_done(checkpoint, country, "scoring"):
        print(f"  [{country}] scoring: already done, skipping")
        return checkpoint

    from src.pipeline import run_graph_and_scoring
    print(f"  [{country}] scoring: starting...")
    checkpoint = _mark(checkpoint, country, "scoring", "running")

    try:
        scored = run_graph_and_scoring(enrich_edgar=False, country=country)
        detail = f"scored={len(scored)}, max_score={scored['coverage_gap_score'].max():.3f}"
        checkpoint = _mark(checkpoint, country, "scoring", "done", detail)
        print(f"  [{country}] scoring: done ({detail})")
    except Exception as e:
        checkpoint = _mark(checkpoint, country, "scoring", "failed", str(e))
        print(f"  [{country}] scoring: FAILED - {e}")

    return checkpoint


def print_status(checkpoint: dict) -> None:
    """Print the current progress of all countries."""
    print("\n" + "=" * 70)
    print("ASEAN PIPELINE STATUS")
    print("=" * 70)
    print(f"{'Country':<8} {'entity_pull':<20} {'inference':<20} {'scoring':<20}")
    print("-" * 68)
    for country in ASEAN_COUNTRIES:
        row = []
        for stage in STAGES:
            key = f"{country}_{stage}"
            entry = checkpoint.get(key, {})
            status = entry.get("status", "pending")
            row.append(status)
        print(f"{country:<8} {row[0]:<20} {row[1]:<20} {row[2]:<20}")

    # Summary counts
    total = len(ASEAN_COUNTRIES)
    done = sum(
        1 for c in ASEAN_COUNTRIES
        if all(_is_done(checkpoint, c, s) for s in STAGES)
    )
    print(f"\nCompleted: {done}/{total} countries")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recoverable ASEAN pipeline runner")
    parser.add_argument("--status", action="store_true", help="Show progress and exit")
    parser.add_argument("--country", type=str, help="Run only this country (ISO-2)")
    parser.add_argument("--skip", type=str, help="Comma-separated countries to skip")
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip entity pull, run inference + scoring on cached data")
    parser.add_argument("--reset", type=str,
                        help="Reset a country's checkpoint (e.g. --reset SG or --reset SG_inference)")
    args = parser.parse_args()

    checkpoint = _load_checkpoint()

    if args.status:
        print_status(checkpoint)
        return

    if args.reset:
        token = args.reset.strip()
        if "_" in token:
            # Reset specific stage — e.g. "SG_inference"
            # Normalise: country part uppercase, stage part lowercase
            parts = token.split("_", 1)
            key = f"{parts[0].upper()}_{parts[1].lower()}"
            if key in checkpoint:
                del checkpoint[key]
                _save_checkpoint(checkpoint)
                print(f"Reset {key}")
            else:
                print(f"No checkpoint entry for {key}")
        else:
            # Reset all stages for a country
            country = token.upper()
            for stage in STAGES:
                key = f"{country}_{stage}"
                checkpoint.pop(key, None)
            _save_checkpoint(checkpoint)
            print(f"Reset all stages for {country}")
        return

    # Determine which countries to run
    if args.country:
        countries = [args.country.upper()]
    else:
        countries = list(ASEAN_COUNTRIES)

    skip_set = set()
    if args.skip:
        skip_set = {c.strip().upper() for c in args.skip.split(",")}

    countries = [c for c in countries if c not in skip_set]

    print("=" * 70)
    print(f"ASEAN PIPELINE RUN — {len(countries)} countries")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for i, country in enumerate(countries):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(countries)}] {country}")
        print(f"{'='*70}")

        t0 = time.time()

        if not args.inference_only:
            checkpoint = run_entity_pull(country, checkpoint)
            # Don't proceed if entity pull failed
            if checkpoint.get(f"{country}_entity_pull", {}).get("status") == "failed":
                print(f"  [{country}] Skipping remaining stages due to entity_pull failure")
                continue

        checkpoint = run_inference(country, checkpoint)
        checkpoint = run_scoring(country, checkpoint)

        elapsed = time.time() - t0
        print(f"  [{country}] Total time: {elapsed/60:.1f} min")

    print("\n" + "=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)
    print_status(checkpoint)


if __name__ == "__main__":
    main()
