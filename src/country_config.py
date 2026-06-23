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

_VN = CountryConfig(
    code="VN",
    address_abbreviations={
        "TP": "THANH PHO",       # City
        "Q": "QUAN",             # District
        "P": "PHUONG",           # Ward
        "TX": "THI XA",          # Town
        "TT": "THI TRAN",        # Township
        "DUONG": "DUONG",        # Street (already full form)
    },
    office_hotel_markers=[
        "DISTRICT 1",
        "QUAN 1",
        "BITEXCO",
        "SAIGON TRADE CENTER",
        "LOTTE CENTER",
        "KEANGNAM",
        "LANDMARK 72",
    ],
    name_features=[
        ("has_co_ltd", r"\bCO\.\s*LTD\b"),
        ("has_jsc", r"\bJSC\b"),        # Joint Stock Company
        ("has_llc_vn", r"\bLLC\b"),
        ("has_local_country", r"\bVIET\s*NAM|VIETNAM\b"),
    ],
    extra_legal_suffixes=[
        r"JSC",
        r"TNHH",               # Trach Nhiem Huu Han (Limited Liability)
    ],
)

_KH = CountryConfig(
    code="KH",
    address_abbreviations={
        "ST": "STREET",
        "BLV": "BOULEVARD",
        "KH": "KHAN",           # District
        "SK": "SANGKAT",        # Commune
    },
    office_hotel_markers=[
        "PHNOM PENH TOWER",
        "EXCHANGE SQUARE",
        "CANADIA TOWER",
        "VATTANAC CAPITAL",
    ],
    name_features=[
        ("has_co_ltd", r"\bCO\.\s*LTD\b"),
        ("has_plc_kh", r"\bPLC\b"),
        ("has_local_country", r"\bCAMBODIA|KAMPUCHEA\b"),
    ],
)

_BN = CountryConfig(
    code="BN",
    address_abbreviations={
        "JLN": "JALAN",
        "KG": "KAMPONG",
        "SPG": "SIMPANG",
    },
    office_hotel_markers=[
        "BANDAR SERI BEGAWAN",
    ],
    name_features=[
        ("has_sdn_bhd", r"\bSDN\s*BHD\b"),
        ("has_local_country", r"\bBRUNEI\b"),
    ],
)

_LA = CountryConfig(
    code="LA",
    address_abbreviations={},
    office_hotel_markers=[
        "VIENTIANE",
    ],
    name_features=[
        ("has_co_ltd", r"\bCO\.\s*LTD\b"),
        ("has_local_country", r"\bLAO\b"),
    ],
)

_MM = CountryConfig(
    code="MM",
    address_abbreviations={},
    office_hotel_markers=[
        "YANGON",
    ],
    name_features=[
        ("has_co_ltd", r"\bCO\.\s*LTD\b"),
        ("has_local_country", r"\bMYANMAR\b"),
    ],
)

_TL = CountryConfig(
    code="TL",
    address_abbreviations={
        "RUA": "RUA",           # Street (Portuguese, already full form)
        "AV": "AVENIDA",        # Avenue
    },
    office_hotel_markers=[
        "DILI",
        "TIMOR PLAZA",
    ],
    name_features=[
        ("has_lda", r"\bLDA\b"),          # Limitada (Portuguese limited company)
        ("has_unipessoal", r"\bUNIPESSOAL\b"),
        ("has_local_country", r"\bTIMOR|TIMOR.LESTE\b"),
    ],
)

_REGISTRY: dict[str, CountryConfig] = {
    "MY": _MY,
    "SG": _SG,
    "PH": _PH,
    "TH": _TH,
    "ID": _ID,
    "VN": _VN,
    "KH": _KH,
    "BN": _BN,
    "LA": _LA,
    "MM": _MM,
    "TL": _TL,
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
