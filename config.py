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
GLEIF_BASE = "https://api.gleif.org/api/v1"
IMF_PIP_BASE = "https://api.imf.org/external/sdmx/3.0/data"
