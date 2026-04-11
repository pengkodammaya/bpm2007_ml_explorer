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


def fetch_malaysia_lei_records(max_pages: int = 50, page_size: int = 200) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for page_number in range(1, max_pages + 1):
        params = {
            "filter[entity.legalAddress.country]": "MY",
            "page[size]": page_size,
            "page[number]": page_number,
        }
        payload = fetch_lei_page(params)
        data = payload.get("data", [])
        if not data:
            break

        rows.extend(_parse_lei_record(item) for item in data)

    return normalize_entity_records(pd.DataFrame(rows)).dropna(subset=["lei"]).drop_duplicates(subset=["lei"])


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
