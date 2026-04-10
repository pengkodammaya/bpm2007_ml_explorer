from __future__ import annotations

import argparse
from src.pipeline import run_gleif_pull, run_graph_and_scoring, run_mock_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 ESIE pipeline")
    parser.add_argument("--relationships-limit", type=int, default=100)
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--use-mock-data", action="store_true")
    args = parser.parse_args()

    if args.use_mock_data:
        print("[INFO] Running mock pipeline...")
        scored = run_mock_pipeline()
        print(scored.head(20).to_string(index=False))
        return

    if not args.skip_pull:
        print("[INFO] Pulling GLEIF Malaysia entities and relationships...")
        entities, relationships = run_gleif_pull(max_relationship_entities=args.relationships_limit)
        print(f"[INFO] Entities: {len(entities):,}")
        print(f"[INFO] Relationships: {len(relationships):,}")

    print("[INFO] Building graph and scoring...")
    scored = run_graph_and_scoring()
    print(scored.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
