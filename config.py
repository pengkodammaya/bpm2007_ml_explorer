from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

INFERENCE_DIR = PROCESSED_DIR / "inference"

for p in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, INFERENCE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

USER_AGENT = "phase1-esie/0.1 research prototype"


# ---------------------------------------------------------------------------
# Country-scoped paths
# ---------------------------------------------------------------------------

# Legacy path mappings for Malaysia (old naming convention)
_MY_LEGACY = {
    "raw_entities": RAW_DIR / "gleif_malaysia_lei",
    "raw_related": RAW_DIR / "gleif_related_lei",
    "interim_relationships": INTERIM_DIR / "gleif_malaysia_relationships",
    "interim_addresses": INTERIM_DIR / "gleif_full_addresses",
    "interim_exceptions": INTERIM_DIR / "gleif_reporting_exceptions",
    "inference_dir": INFERENCE_DIR,
}


def country_paths(country: str) -> dict[str, Path]:
    """Return a dict of standard I/O paths scoped to *country*.

    For MY the legacy paths are included as fallbacks so that existing
    cached Malaysian data is found without re-fetching.
    """
    c = country.upper()
    cl = c.lower()
    paths = {
        "raw_entities": RAW_DIR / f"gleif_{cl}_lei",
        "raw_related": RAW_DIR / f"gleif_{cl}_related_lei",
        "interim_relationships": INTERIM_DIR / f"gleif_{cl}_relationships",
        "interim_addresses": INTERIM_DIR / f"gleif_{cl}_full_addresses",
        "interim_exceptions": INTERIM_DIR / f"gleif_{cl}_reporting_exceptions",
        "inference_dir": INFERENCE_DIR / cl,
    }
    if c == "MY":
        paths["legacy"] = _MY_LEGACY
    return paths
GLEIF_BASE = "https://api.gleif.org/api/v1"
IMF_PIP_BASE = "https://api.imf.org/external/sdmx/3.0/data"
