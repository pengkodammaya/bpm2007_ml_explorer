"""Country-specific configuration for the inference pipeline.

Centralises per-country knowledge — legal suffix features, address
abbreviations, office-hotel markers — so the pipeline is parameterised
by ISO-2 country code rather than littered with hardcoded values.

Adding a new country:
    1. Create a CountryConfig instance below.
    2. Register it in _REGISTRY.
    3. Run `python main.py --run-inference --country XX`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CountryConfig:
    """Per-country configuration for the inference pipeline."""

    code: str
    """ISO-2 country code."""

    address_abbreviations: dict[str, str] = field(default_factory=dict)
    """Street-address abbreviation expansions for normalisation."""

    office_hotel_markers: list[str] = field(default_factory=list)
    """Substrings that identify corporate-secretary / office-hotel addresses."""

    name_features: list[tuple[str, str]] = field(default_factory=list)
    """(feature_column_name, regex_pattern) pairs for legal-name feature
    engineering in the jurisdiction predictor."""

    extra_legal_suffixes: list[str] = field(default_factory=list)
    """Additional regex fragments appended to the base LEGAL_SUFFIX_PATTERN
    when extracting brand tokens from parent company names."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_MY = CountryConfig(
    code="MY",
    address_abbreviations={
        "JLN": "JALAN",
        "TMN": "TAMAN",
        "KG": "KAMPUNG",
        "BDR": "BANDAR",
        "PERSIARAN": "PERSIARAN",
        "LRG": "LORONG",
        "BTU": "BATU",
        "SRI": "SERI",
    },
    office_hotel_markers=[
        "LABUAN",
        "WISMA UOA",
        "MENARA UOA",
    ],
    name_features=[
        ("has_sdn_bhd", r"\bSDN\b"),
        ("has_berhad", r"\bBERHAD\b"),
        ("has_local_country", r"\bMALAYSIA\b"),
    ],
)

_SG = CountryConfig(
    code="SG",
    address_abbreviations={
        "BLK": "BLOCK",
        "RD": "ROAD",
        "ST": "STREET",
        "AVE": "AVENUE",
        "DR": "DRIVE",
        "CRES": "CRESCENT",
    },
    office_hotel_markers=[
        "RAFFLES PLACE",
        "MARINA BAY",
        "SUNTEC TOWER",
        "CENTENNIAL TOWER",
        "CAPITAL TOWER",
    ],
    name_features=[
        ("has_pte_ltd", r"\bPTE\b"),
        ("has_local_country", r"\bSINGAPORE\b"),
    ],
    extra_legal_suffixes=[
        r"PTE\s*\.?\s*LTD",
    ],
)

_PH = CountryConfig(
    code="PH",
    address_abbreviations={
        "BGY": "BARANGAY",
        "BRGY": "BARANGAY",
        "STO": "SANTO",
        "STA": "SANTA",
        "AVE": "AVENUE",
        "BLVD": "BOULEVARD",
    },
    office_hotel_markers=[
        "MAKATI",
        "BGC",
        "BONIFACIO GLOBAL CITY",
        "AYALA AVENUE",
        "RCBC PLAZA",
    ],
    name_features=[
        ("has_inc_ph", r"\bINC(?:ORPORATED)?\b"),
        ("has_corp_ph", r"\bCORP(?:ORATION)?\b"),
        ("has_local_country", r"\bPHILIPPINE|PILIPINAS\b"),
    ],
)

_TH = CountryConfig(
    code="TH",
    address_abbreviations={
        "SOI": "SOI",
        "MOO": "MOO",
    },
    office_hotel_markers=[
        "SATHORN",
        "SILOM",
    ],
    name_features=[
        ("has_co_ltd", r"\bCO\.\s*LTD\b"),
        ("has_local_country", r"\bTHAILAND\b"),
    ],
)

_ID = CountryConfig(
    code="ID",
    address_abbreviations={
        "JL": "JALAN",
        "GG": "GANG",
    },
    office_hotel_markers=[
        "SUDIRMAN",
        "MEGA KUNINGAN",
    ],
    name_features=[
        ("has_pt", r"\bPT\b"),
        ("has_tbk", r"\bTBK\b"),
        ("has_local_country", r"\bINDONESIA\b"),
    ],
)

_REGISTRY: dict[str, CountryConfig] = {
    "MY": _MY,
    "SG": _SG,
    "PH": _PH,
    "TH": _TH,
    "ID": _ID,
}

_DEFAULT = CountryConfig(
    code="XX",
    name_features=[
        ("has_local_country", r"\bNEVER_MATCH_PLACEHOLDER\b"),
    ],
)


def get_country_config(country: str) -> CountryConfig:
    """Return the configuration for *country* (ISO-2), or a safe default."""
    return _REGISTRY.get(country.upper(), CountryConfig(code=country.upper()))


def list_supported_countries() -> list[str]:
    """Return sorted list of country codes with explicit configs."""
    return sorted(_REGISTRY.keys())
