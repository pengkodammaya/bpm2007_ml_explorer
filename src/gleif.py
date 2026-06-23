from __future__ import annotations

import pandas as pd
from collections.abc import Iterable
from typing import Any
from src.utils import get_json, HTTPFetchError
from config import GLEIF_BASE, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}

ENTITY_COLUMNS = [
    "lei",
    "legal_name",
    "other_names",
    "transliterated_other_names",
    "category",
    "legal_form",
    "entity_status",
    "registered_at",
    "last_update_at",
    "country_legal",
    "city_legal",
    "country_hq",
    "city_hq",
]

LEI_SEARCH_COLUMNS = [
    "query",
    "rank",
    "lei",
    "legal_name",
    "country_legal",
    "country_hq",
    "category",
    "entity_status",
    "registration_status",
    "total_results",
]

RELATIONSHIP_COLUMNS = [
    "source_lei",
    "target_lei",
    "relationship_type",
    "relationship_status",
    "accounting_standard",
    "period_end",
    "valid_from",
    "valid_to",
]


def normalize_entity_records(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(columns=ENTITY_COLUMNS)


def normalize_relationship_records(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(columns=RELATIONSHIP_COLUMNS)


def fetch_lei_page(params: dict[str, Any]) -> dict[str, Any]:
    return get_json(f"{GLEIF_BASE}/lei-records", params=params, headers=HEADERS)


def search_lei_records(
    query: str,
    *,
    page_size: int = 5,
    country: str | None = None,
) -> pd.DataFrame:
    """Search GLEIF LEI records using the API full-text filter.

    This is an entity-resolution/candidate-generation helper. It does not imply
    ownership or UIE by itself.
    """
    query = str(query or "").strip()
    if not query:
        return pd.DataFrame(columns=LEI_SEARCH_COLUMNS)

    params: dict[str, Any] = {
        "filter[fulltext]": query,
        "page[size]": page_size,
        "page[number]": 1,
    }
    if country:
        params["filter[entity.legalAddress.country]"] = country.upper()

    payload = fetch_lei_page(params)
    total_results = (
        payload.get("meta", {})
        .get("pagination", {})
        .get("total", 0)
    )
    rows = []
    for rank, item in enumerate(payload.get("data", []) or [], start=1):
        parsed = _parse_lei_record(item)
        registration = (item.get("attributes", {}) or {}).get("registration", {}) or {}
        rows.append({
            "query": query,
            "rank": rank,
            "lei": parsed.get("lei"),
            "legal_name": parsed.get("legal_name"),
            "country_legal": parsed.get("country_legal"),
            "country_hq": parsed.get("country_hq"),
            "category": parsed.get("category"),
            "entity_status": parsed.get("entity_status"),
            "registration_status": registration.get("status"),
            "total_results": total_results,
        })
    return pd.DataFrame(rows, columns=LEI_SEARCH_COLUMNS)


def fetch_lei_record(lei: str) -> dict[str, Any] | None:
    payload = get_json(f"{GLEIF_BASE}/lei-records/{lei}", headers=HEADERS, allow_404=True)
    item = payload.get("data")
    if not item:
        return None
    return _parse_lei_record(item)


def _parse_lei_record(item: dict[str, Any]) -> dict[str, Any]:
    attrs = item.get("attributes", {}) or {}
    entity = attrs.get("entity", {}) or {}
    legal_addr = entity.get("legalAddress", {}) or {}
    headquarters = entity.get("headquartersAddress", {}) or {}
    legal_name_obj = entity.get("legalName") or {}
    registration = attrs.get("registration", {}) or {}

    return {
        "lei": attrs.get("lei"),
        "legal_name": legal_name_obj.get("name"),
        "other_names": entity.get("otherNames"),
        "transliterated_other_names": entity.get("transliteratedOtherNames"),
        "category": entity.get("category"),
        "legal_form": (entity.get("legalForm") or {}).get("id"),
        "entity_status": entity.get("status"),
        "registered_at": registration.get("initialRegistrationDate"),
        "last_update_at": registration.get("lastUpdateDate"),
        "country_legal": legal_addr.get("country"),
        "city_legal": legal_addr.get("city"),
        "country_hq": headquarters.get("country"),
        "city_hq": headquarters.get("city"),
    }


def _fetch_filtered_lei_records(
    base_filters: dict[str, str],
    max_pages: int = 50,
    page_size: int = 200,
    label: str = "",
) -> tuple[list[dict[str, Any]], bool]:
    """Fetch LEI records with arbitrary filters.

    Returns (rows, hit_limit) where *hit_limit* is True when the last
    requested page still returned data, indicating there may be more
    records beyond the GLEIF pagination cap.
    """
    rows: list[dict[str, Any]] = []
    hit_limit = False

    for page_number in range(1, max_pages + 1):
        params = {
            **base_filters,
            "page[size]": page_size,
            "page[number]": page_number,
        }
        try:
            payload = fetch_lei_page(params)
        except HTTPFetchError as e:
            # GLEIF returns 400 when page exceeds available range
            if "400" in str(e):
                hit_limit = True
                break
            raise
        data = payload.get("data", [])
        if not data:
            break
        rows.extend(_parse_lei_record(item) for item in data)

        if page_number == max_pages and len(data) == page_size:
            hit_limit = True

    if label and rows:
        print(f"  [{label}] fetched {len(rows)} records (hit_limit={hit_limit})")
    return rows, hit_limit


# Categories used to split large country fetches when the page limit is hit
_ENTITY_CATEGORIES = ["FUND", "BRANCH", "SOLE_PROPRIETOR", "GENERAL"]


def fetch_country_lei_records(country: str = "MY", max_pages: int = 50, page_size: int = 200) -> pd.DataFrame:
    """Fetch all LEI records for *country*, auto-splitting if the GLEIF
    pagination limit (default 50 pages × 200 = 10 000) is exceeded.

    Split strategy:
    1. Try a single unfiltered query.
    2. If the page limit is hit, split by entity status (ACTIVE / INACTIVE).
    3. If any status bucket still hits the limit, further split by category.
    """
    c = country.upper()
    base = {"filter[entity.legalAddress.country]": c}

    # --- Attempt 1: single query ---
    rows, hit_limit = _fetch_filtered_lei_records(
        base, max_pages=max_pages, page_size=page_size, label=f"{c}"
    )
    if not hit_limit:
        return normalize_entity_records(pd.DataFrame(rows)).dropna(subset=["lei"]).drop_duplicates(subset=["lei"])

    print(f"[INFO] {c}: page limit reached ({len(rows)} records). Splitting by entity status...")

    # --- Attempt 2: split by entity status ---
    all_rows: list[dict[str, Any]] = []
    for status in ["ACTIVE", "INACTIVE"]:
        filters = {**base, "filter[entity.status]": status}
        s_rows, s_hit = _fetch_filtered_lei_records(
            filters, max_pages=max_pages, page_size=page_size,
            label=f"{c}/{status}",
        )
        if not s_hit:
            all_rows.extend(s_rows)
            continue

        # --- Attempt 3: split by category within this status ---
        print(f"[INFO] {c}/{status}: still exceeds limit ({len(s_rows)}). Splitting by category...")
        for cat in _ENTITY_CATEGORIES:
            cat_filters = {**filters, "filter[entity.category]": cat}
            c_rows, c_hit = _fetch_filtered_lei_records(
                cat_filters, max_pages=max_pages, page_size=page_size,
                label=f"{c}/{status}/{cat}",
            )
            if c_hit:
                print(f"[WARN] {c}/{status}/{cat}: still exceeds limit ({len(c_rows)} records). Some entities may be missing.")
            all_rows.extend(c_rows)

    return normalize_entity_records(pd.DataFrame(all_rows)).dropna(subset=["lei"]).drop_duplicates(subset=["lei"])


def fetch_malaysia_lei_records(max_pages: int = 50, page_size: int = 200) -> pd.DataFrame:
    return fetch_country_lei_records("MY", max_pages=max_pages, page_size=page_size)


def fetch_lei_records_by_lei(leis: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for lei in leis:
        if pd.isna(lei):
            continue
        lei_str = str(lei).strip()
        if not lei_str or lei_str in seen:
            continue
        seen.add(lei_str)
        try:
            record = fetch_lei_record(lei_str)
        except HTTPFetchError as e:
            print(f"[WARN] LEI fetch failed for {lei_str}: {e}")
            continue
        if record:
            rows.append(record)

    return normalize_entity_records(pd.DataFrame(rows)).dropna(subset=["lei"]).drop_duplicates(subset=["lei"])


def fetch_relationships_for_lei(lei: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    mapping = {
        "direct-parent-relationship": "direct_parent",
        "ultimate-parent-relationship": "ultimate_parent",
    }

    for endpoint, rel_label in mapping.items():
        url = f"{GLEIF_BASE}/lei-records/{lei}/{endpoint}"
        try:
            payload = get_json(url, headers=HEADERS, allow_404=True)
        except HTTPFetchError:
            continue

        item = payload.get("data")
        if not item or not isinstance(item, dict):
            continue

        attrs = item.get("attributes", {}) or {}
        rel = attrs.get("relationship", {}) or {}
        start = rel.get("startNode", {}) or {}
        end = rel.get("endNode", {}) or {}

        periods = rel.get("periods") or []
        latest_period = periods[-1] if periods else {}

        rows.append({
            "source_lei": start.get("id"),
            "target_lei": end.get("id"),
            "relationship_type": rel_label,
            "relationship_status": rel.get("status"),
            "accounting_standard": latest_period.get("accountingStandard"),
            "period_end": latest_period.get("endDate"),
            "valid_from": attrs.get("validFrom"),
            "valid_to": attrs.get("validTo"),
        })

    return normalize_relationship_records(pd.DataFrame(rows))
