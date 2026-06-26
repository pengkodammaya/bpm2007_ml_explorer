"""Phase 0.5: Fetch and analyze GLEIF reporting exceptions.

Entities that file a reporting exception (especially NON_CONSOLIDATING)
are confirmed subsidiaries that simply don't report parent data under
consolidation rules. This is a high-confidence signal, not an inference.
"""
from __future__ import annotations

import pandas as pd
from src.utils import get_json, HTTPFetchError
from config import GLEIF_BASE, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}

EXCEPTION_COLUMNS = ["lei", "exception_reason", "exception_reference"]


def fetch_reporting_exception(lei: str) -> dict | None:
    """Fetch the direct-parent-reporting-exception for a single LEI."""
    url = f"{GLEIF_BASE}/lei-records/{lei}/direct-parent-reporting-exception"
    try:
        payload = get_json(url, headers=HEADERS, allow_404=True)
    except HTTPFetchError:
        return None

    item = payload.get("data")
    if not item or not isinstance(item, dict):
        return None

    attrs = item.get("attributes", {}) or {}

    return {
        "lei": lei,
        "exception_reason": attrs.get("reason"),
        "exception_reference": attrs.get("reference"),
    }


def fetch_reporting_exceptions(
    leis: list[str],
    checkpoint_path=None,
    scanned_path=None,
    batch_size: int = 500,
) -> pd.DataFrame:
    """Bulk fetch reporting exceptions for a list of LEIs.

    Returns a DataFrame with columns: [lei, exception_reason, exception_reference].
    Only entities that have an exception filed are included.

    Incremental checkpointing (when paths provided):
    - ``checkpoint_path``: parquet of accumulated exception rows. Resume on re-run.
    - ``scanned_path``: parquet tracking every LEI already probed (hit or miss),
      so we don't re-scan entities that returned no exception.
    """
    from pathlib import Path

    rows: list[dict] = []
    scanned: set[str] = set()
    total = len(leis)

    # Resume from checkpoints
    if checkpoint_path is not None:
        cp = Path(checkpoint_path)
        if cp.exists():
            try:
                existing = pd.read_parquet(cp)
                if not existing.empty:
                    rows = existing.to_dict("records")
            except Exception:
                pass
    if scanned_path is not None:
        sp = Path(scanned_path)
        if sp.exists():
            try:
                scanned = set(pd.read_parquet(sp)["lei"].dropna().astype(str))
            except Exception:
                pass

    # Always treat already-found LEIs as scanned
    scanned |= {r["lei"] for r in rows if r.get("lei")}

    if scanned:
        print(f"[INFO] Resuming exception scan: {len(scanned):,} already scanned, {len(rows):,} exceptions found", flush=True)

    remaining = [lei for lei in leis if str(lei) not in scanned]
    found = len(rows)

    for i, lei in enumerate(remaining):
        result = fetch_reporting_exception(lei)
        scanned.add(str(lei))
        if result and result.get("exception_reason"):
            rows.append(result)
            found += 1

        done = len(scanned)
        if done % 200 == 0:
            print(f"[{done}/{total}] exceptions scanned... {found} found so far", flush=True)

        if (i + 1) % batch_size == 0:
            _flush_exception_checkpoint(rows, checkpoint_path)
            _flush_scanned_checkpoint(scanned, scanned_path)

    # Final save
    _flush_exception_checkpoint(rows, checkpoint_path)
    _flush_scanned_checkpoint(scanned, scanned_path)

    print(f"[INFO] Exception scan complete: {found}/{total} entities have reporting exceptions", flush=True)

    if not rows:
        return pd.DataFrame(columns=EXCEPTION_COLUMNS)
    return pd.DataFrame(rows).reindex(columns=EXCEPTION_COLUMNS)


def _flush_exception_checkpoint(rows: list[dict], path) -> None:
    if path is None or not rows:
        return
    from pathlib import Path
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).reindex(columns=EXCEPTION_COLUMNS).to_parquet(p, index=False)
    except Exception as e:
        print(f"[WARN] Exception checkpoint save failed: {e}", flush=True)


def _flush_scanned_checkpoint(scanned: set[str], path) -> None:
    if path is None or not scanned:
        return
    from pathlib import Path
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"lei": sorted(scanned)}).to_parquet(p, index=False)
    except Exception as e:
        print(f"[WARN] Scanned checkpoint save failed: {e}", flush=True)


def enrich_with_exception_flags(
    entities_df: pd.DataFrame,
    exceptions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge exception data onto entities, adding boolean flag columns.

    Adds:
    - is_non_consolidating: True if exception reason is NON_CONSOLIDATING
    - has_exception_filed: True if any reporting exception exists
    """
    out = entities_df.copy()

    if exceptions_df.empty or "lei" not in exceptions_df.columns:
        out["is_non_consolidating"] = 0
        out["has_exception_filed"] = 0
        return out

    exc = exceptions_df[["lei", "exception_reason"]].drop_duplicates(subset=["lei"])

    out = out.merge(exc, on="lei", how="left")
    out["is_non_consolidating"] = (out["exception_reason"] == "NON_CONSOLIDATING").astype(int)
    out["has_exception_filed"] = out["exception_reason"].notna().astype(int)
    out.drop(columns=["exception_reason"], inplace=True, errors="ignore")

    return out
