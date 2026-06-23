# BPM2007 ML Explorer

ASEAN external-sector discovery prototype for LEI-based ownership graphing,
Ultimate Investor Economy (UIE) assignment, and coverage-gap triage.

## What it does
- Pulls country-linked LEI entities from GLEIF
- Pulls direct/ultimate parent relationships where available
- Builds a direct-investment ownership graph
- Generates domestic-entity UIE assignment tables
- Separates known UIE evidence from inferred/model fallback evidence
- Applies auditable manual UIE overrides for analyst-verified cases
- Optionally matches Malaysia LEIs to the public CTOS directory as a local
  incorporation/SSM existence signal
- Enriches with basic EDGAR presence
- Produces a coverage-gap priority score for analyst triage

The pipeline is parameterised by ISO-2 country code. Explicit country
configuration currently covers ASEAN: `MY`, `SG`, `TH`, `ID`, `PH`, `VN`,
`KH`, `BN`, `LA`, `MM`, and `TL`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
python main.py --use-mock-data
```

## Live mode

```bash
python main.py --country MY --relationships-limit 100
```

Live mode pulls country LEI records from GLEIF, then fetches relationship
target LEIs so foreign parent signals can be scored.

Before producing final tables, refresh the GLEIF snapshot explicitly:

```bash
python main.py --country MY --force-pull --scan-all --skip-edgar
python main.py --country MY --exceptions-only --force-pull
```

For a small smoke test:

```bash
python main.py --country MY --lei-max-pages 1 --relationships-limit 5 --skip-edgar
```

To use cached data only and regenerate graph/scoring/UIE outputs:

```bash
python main.py --country MY --skip-pull --skip-edgar
```

To run a cautious CTOS Malaysia pilot fetch and match:

```bash
python main.py --ctos-my --ctos-letters A --ctos-pages 1 --ctos-details-limit 2
```

CTOS matches are saved to `data/processed/my/ctos_entity_matches.*` and merged
into `uie_assignments` as `ctos_registered_malaysia`. This is an incorporation
signal only; it does not prove Malaysian ownership or set UIE to Malaysia.

Manual UIE overrides live in `data/manual/uie_overrides.csv`. They are intended
for analyst-verified public evidence where inferred signals are weaker than a
known ownership/control fact. Overrides take precedence over GLEIF/inference
signals, retain a source URL/note, and are marked as `manual_verified`.

Analyst queue status notes live in `data/manual/uie_review_status.csv`. They do
not change UIE assignment; they classify rows in `uie_review_targets` with
fields such as `review_status`, `entity_type`, and `review_note` so fund
vehicles, nominee/custodian vehicles, and operating companies can be handled
with different review rules.

## Streamlit app

Run the pipeline first for one or more countries, then launch:

```bash
streamlit run app/streamlit_app.py
```

## Tests

```bash
python -m unittest discover -s tests
```

## Outputs
- `data/raw/gleif_{country}_lei.parquet` or `.pkl`
- `data/raw/gleif_{country}_related_lei.parquet` or `.pkl`
- `data/interim/gleif_{country}_relationships.parquet` or `.pkl`
- `data/processed/{country}/di_graph_summary.parquet` or `.pkl`
- `data/processed/{country}/coverage_gap_scores.parquet` or `.pkl`
- `data/processed/{country}/coverage_gap_scores_top200.csv`
- `data/processed/{country}/uie_assignments.parquet` or `.pkl`
- `data/processed/{country}/uie_assignments.csv`
- `data/processed/{country}/uie_review_targets.parquet` or `.pkl`
- `data/processed/{country}/uie_review_targets.csv`
- `data/processed/{country}/uie_product_vehicle_targets.parquet` or `.pkl`
- `data/processed/{country}/uie_product_vehicle_targets.csv`
- `data/processed/{country}/entity_match_benchmark.parquet` or `.pkl` when
  the GLEIF search benchmark is run
- `data/processed/{country}/entity_match_benchmark.csv` when the GLEIF search
  benchmark is run
- `data/processed/{country}/investor_economy_summary.parquet` or `.pkl`
- `data/processed/{country}/investor_economy_summary.csv`
- `data/processed/my/ctos_entity_matches.parquet` or `.pkl` when CTOS enrichment is run

## Goal
Prototype discovery layer for BPM7 external-sector compilation:
- Build country-linked DI ownership graphs from public GLEIF data
- Assign best-available UIE signals to domestic entities
- Rank coverage-gap candidates for analyst follow-up
- Compare aggregate GLEIF relationship signals with CDIS patterns

See `docs/project_plan.md` for the current roadmap, including the planned GLEIF
search benchmark, Global Energy Monitor evidence layer, fund/product vehicle
rule, and source-backed manual override workflow.

## Evidence tiers

UIE assignment uses a precedence order:

1. Analyst-reviewed manual override
2. GLEIF ultimate parent relationship
3. GLEIF direct parent relationship
4. Phase 1 high-confidence name match
5. Phase 2 address-cluster inferred parent
6. Phase 3 jurisdiction model fallback
7. Reporting exception subsidiary signal with no country assignment
8. Unassigned

Phase 3 model probabilities are uncalibrated ranking signals, not validated
confidence. Phase 4 link-prediction validation uses edge-holdout CV so held-out
positive edges are removed before graph features are computed.
