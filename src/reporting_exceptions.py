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
    exception = attrs.get("exception", {}) or {}

    return {
        "lei": lei,
        "exception_reason": exception.get("reason"),
        "exception_reference": exception.get("reference"),
    }


def fetch_reporting_exceptions(leis: list[str]) -> pd.DataFrame:
    """Bulk fetch reporting exceptions for a list of LEIs.

    Returns a DataFrame with columns: [lei, exception_reason, exception_reference].
    Only entities that have an exception filed are included.
    """
    rows: list[dict] = []
    total = len(leis)
    found = 0

    for i, lei in enumerate(leis):
        result = fetch_reporting_exception(lei)
        if result and result.get("exception_reason"):
            rows.append(result)
            found += 1

        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{total}] exceptions scanned... {found} found so far", flush=True)

    print(f"[INFO] Exception scan complete: {found}/{total} entities have reporting exceptions", flush=True)

    if not rows:
        return pd.DataFrame(columns=EXCEPTION_COLUMNS)
    return pd.DataFrame(rows).reindex(columns=EXCEPTION_COLUMNS)


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
