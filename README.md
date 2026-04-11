# BPM2007 ML Explorer

Malaysia-linked external sector discovery prototype.

## What it does
- Pulls Malaysia-linked LEI entities from GLEIF
- Pulls direct/ultimate parent relationships where available
- Builds a DI ownership graph
- Enriches with basic EDGAR presence
- Produces a first-pass coverage-gap score

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
python main.py --relationships-limit 100
```

Live mode pulls Malaysia LEI records from GLEIF, then fetches relationship target LEIs so foreign parent signals can be scored.

For a small smoke test:

```bash
python main.py --lei-max-pages 1 --relationships-limit 5 --skip-edgar
```

## Streamlit app

Run the pipeline first, then launch:

```bash
streamlit run app/streamlit_app.py
```

## Tests

```bash
python -m unittest discover -s tests
```

## Outputs
- `data/raw/gleif_malaysia_lei.parquet` or `.pkl`
- `data/raw/gleif_related_lei.parquet` or `.pkl`
- `data/interim/gleif_malaysia_relationships.parquet` or `.pkl`
- `data/processed/di_graph_summary.parquet` or `.pkl`
- `data/processed/coverage_gap_scores.parquet` or `.pkl`

## Goal
Phase 1 prototype for Malaysia-linked external sector discovery:
- Build DI ownership graph (GLEIF)
- Simulate portfolio allocation (IMF PIP/CPIS stub included)
- Detect coverage gaps
