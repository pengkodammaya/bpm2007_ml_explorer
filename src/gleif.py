from __future__ import annotations

import pandas as pd
from typing import Any
from src.utils import get_json, HTTPFetchError
from config import GLEIF_BASE, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}


def fetch_lei_page(params: dict[str, Any]) -> dict[str, Any]:
    return get_json(f"{GLEIF_BASE}/lei-records", params=params, headers=HEADERS)


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

    return pd.DataFrame(rows).drop_duplicates(subset=["lei"])


def fetch_relationships_for_lei(lei: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    mapping = {
        "is-directly-consolidated-by": "direct_parent",
        "is-ultimately-consolidated-by": "ultimate_parent",
    }

    for endpoint, rel_label in mapping.items():
        url = f"{GLEIF_BASE}/lei-records/{lei}/{endpoint}"
        try:
            payload = get_json(url, headers=HEADERS, allow_404=True)
        except HTTPFetchError:
            continue

        for item in payload.get("data", []):
            attrs = item.get("attributes", {}) or {}
            relationships = item.get("relationships", {}) or {}
            start = ((relationships.get("startNode") or {}).get("data") or {})
            end = ((relationships.get("endNode") or {}).get("data") or {})

            rows.append({
                "source_lei": start.get("id"),
                "target_lei": end.get("id"),
                "relationship_type": rel_label,
                "relationship_status": attrs.get("relationshipStatus"),
                "accounting_standard": attrs.get("accountingStandard"),
                "period_end": attrs.get("periodEnd"),
                "valid_from": attrs.get("validFrom"),
                "valid_to": attrs.get("validTo"),
            })

    return pd.DataFrame(rows)
