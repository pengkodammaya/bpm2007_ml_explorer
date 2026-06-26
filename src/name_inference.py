"""Phase 1: Name pattern extraction and fuzzy matching for parent inference.

Extracts brand tokens from known foreign parent entities and fuzzy-matches
them against domestic entity names to identify likely subsidiaries.
Country-agnostic — works with any nation's entity data.
"""
from __future__ import annotations

import re
import pandas as pd
from rapidfuzz import fuzz
from src.edgar import clean_company_name
from src.country_config import get_country_config

# Legal suffixes to strip when extracting brand tokens
LEGAL_SUFFIX_PATTERN = re.compile(
    r"\b("
    r"LTD|LIMITED|INC|INCORPORATED|CORP|CORPORATION|"
    r"GMBH|AG|SA|SE|NV|BV|PLC|LLC|LP|LLP|"
    r"SDN\s*BHD|BERHAD|BHD|"
    r"PTE|PTY|CO|COMPANY|GROUP|HOLDINGS|HOLDING|"
    r"S\s*A|S\s*R\s*L|SPA|OYJ|ASA|AB|KK|"
    r"AKTIENGESELLSCHAFT|"
    r"SOCIETE\s*DES\s*PRODUITS"
    r")\b",
    re.IGNORECASE,
)

# Minimum brand token length to avoid false positives from short names
MIN_BRAND_TOKEN_LENGTH = 4

# Common geographic/generic terms that cause false positive matches
# when they appear as the primary brand token content
GENERIC_TOKENS = {
    # Geographic
    "ASIA", "ASIA PACIFIC", "PACIFIC", "GLOBAL", "INTERNATIONAL",
    "MALAYSIA", "SINGAPORE", "INDONESIA", "THAILAND", "PHILIPPINES",
    "VIETNAM", "CAMBODIA", "LAOS", "MYANMAR", "BRUNEI", "BURMA",
    # Industry / function (singular AND plural — token_set_ratio doesn't stem)
    "SERVICE", "SERVICES",
    "CAPITAL", "FINANCIAL", "FINANCE",
    "INVESTMENT", "INVESTMENTS",
    "MANAGEMENT", "MANAGER", "MANAGERS",
    "TRADING", "TRADE",
    "TECHNOLOGY", "TECHNOLOGIES", "TECH",
    "ENERGY", "POWER",
    "PROPERTY", "PROPERTIES",
    "DEVELOPMENT", "DEVELOPMENTS",
    "SOLUTION", "SOLUTIONS",
    "INDUSTRY", "INDUSTRIES",
    "RESOURCE", "RESOURCES",
    "VENTURE", "VENTURES",
    "PARTNER", "PARTNERS", "PARTNERSHIP",
    "BANK", "BANKING",
    "CITY", "URBAN",
    # Investment-vehicle / fund-y words
    "FUND", "FUNDS",
    "TRUST", "TRUSTS",
    "EQUITY", "EQUITIES",
    "ASSET", "ASSETS",
    "ADVISOR", "ADVISORS", "ADVISERS",
    "PORTFOLIO", "PORTFOLIOS",
    "GROWTH", "INCOME", "VALUE",
    # Holding-co words
    "HOLDING", "HOLDINGS",
    "ENTERPRISE", "ENTERPRISES",
    "CORPORATE",
}


def extract_parent_brand_tokens(
    parents_df: pd.DataFrame,
    country: str = "MY",
) -> pd.DataFrame:
    """Extract core brand tokens from parent entity legal names.

    Parameters
    ----------
    parents_df : DataFrame with at least 'lei' and 'legal_name' columns.
    country : ISO-2 code — selects country-specific extra suffixes.

    Returns
    -------
    DataFrame with columns [parent_lei, parent_name, brand_token].
    """
    if parents_df.empty or "legal_name" not in parents_df.columns:
        return pd.DataFrame(columns=["parent_lei", "parent_name", "brand_token"])

    suffix_pat = get_suffix_pattern(country)

    rows: list[dict] = []
    for _, row in parents_df.iterrows():
        lei = row.get("lei")
        name = row.get("legal_name")
        if not lei or not name:
            continue

        token = _extract_brand_token(name, suffix_pat)
        if len(token) >= MIN_BRAND_TOKEN_LENGTH and not _is_generic_token(token):
            rows.append({
                "parent_lei": lei,
                "parent_name": name,
                "brand_token": token,
            })

    return pd.DataFrame(rows).drop_duplicates(subset=["brand_token"])


def get_suffix_pattern(country: str = "MY") -> re.Pattern:
    """Return the legal suffix regex, augmented with country-specific extras."""
    cfg = get_country_config(country)
    if not cfg.extra_legal_suffixes:
        return LEGAL_SUFFIX_PATTERN
    extras = "|".join(cfg.extra_legal_suffixes)
    combined = LEGAL_SUFFIX_PATTERN.pattern.rstrip(")\\b") + "|" + extras + r")\b"
    return re.compile(combined, re.IGNORECASE)


def _extract_brand_token(
    name: str,
    suffix_pattern: re.Pattern | None = None,
) -> str:
    """Extract the core brand name by removing legal suffixes and cleaning."""
    if suffix_pattern is None:
        suffix_pattern = LEGAL_SUFFIX_PATTERN
    cleaned = clean_company_name(name)
    # Remove legal suffixes
    token = suffix_pattern.sub("", cleaned)
    # Remove extra whitespace
    token = re.sub(r"\s+", " ", token).strip()
    return token


def _is_generic_token(token: str) -> bool:
    """Check if a brand token is too generic to produce reliable matches.

    A token is generic if ALL of its words are in the generic set,
    or if removing generic words leaves nothing meaningful.
    """
    words = token.upper().split()
    non_generic = [w for w in words if w not in GENERIC_TOKENS]
    # If no non-generic words remain, the whole token is generic
    if not non_generic:
        return True
    # If the remaining non-generic part is too short
    remaining = " ".join(non_generic)
    if len(remaining) < MIN_BRAND_TOKEN_LENGTH:
        return True
    return False


def _is_whole_word_match(brand: str, entity_name: str) -> bool:
    """Check if the brand token appears as whole words in the entity name.

    'SHELL' matches 'SHELL REFINING SDN BHD' but not 'NUTSHELL CORP'.
    'NIUM' should NOT match 'ALUMINIUM'.
    """
    pattern = r"\b" + re.escape(brand) + r"\b"
    return bool(re.search(pattern, entity_name, re.IGNORECASE))


def _brand_words_present(brand: str, entity_name: str) -> bool:
    """Check if the distinctive words of the brand appear in the entity name.

    For multi-word brands like 'DEUTSCHE BANK', checks that the key
    differentiating word (not a generic term) is present.
    """
    brand_words = set(brand.upper().split())
    entity_words = set(entity_name.upper().split())
    distinctive_words = brand_words - GENERIC_TOKENS
    if not distinctive_words:
        # All brand words are generic — require full match
        return brand.upper() in entity_name.upper()
    # At least one distinctive brand word must appear in entity name
    return bool(distinctive_words & entity_words)


def fuzzy_match_names(
    entities_df: pd.DataFrame,
    brand_tokens_df: pd.DataFrame,
    known_leis: set[str] | None = None,
    threshold: int = 90,
    country: str = "MY",
) -> pd.DataFrame:
    """Fuzzy-match entity names against parent brand tokens.

    Parameters
    ----------
    entities_df : Domestic entities DataFrame.
    brand_tokens_df : Output from extract_parent_brand_tokens().
    known_leis : LEIs that already have known parents (excluded from results).
    threshold : Minimum fuzzy match score (0-100).
    country : ISO-2 code — selects country-specific suffix pattern.

    Returns
    -------
    DataFrame with columns [lei, matched_parent_lei, brand_token, match_score, match_field].
    """
    if entities_df.empty or brand_tokens_df.empty:
        return pd.DataFrame(columns=["lei", "matched_parent_lei", "brand_token", "match_score", "match_field"])

    known_leis = known_leis or set()
    matches: list[dict] = []
    suffix_pat = get_suffix_pattern(country)

    # Pre-clean entity names
    entities = entities_df.copy()
    entities["name_clean"] = entities["legal_name"].map(clean_company_name)

    tokens = brand_tokens_df.to_dict("records")
    total = len(entities)

    for i, (_, entity) in enumerate(entities.iterrows()):
        lei = entity.get("lei")
        if not lei or lei in known_leis:
            continue

        entity_name = entity.get("name_clean", "")
        if not entity_name:
            continue

        best_match = None
        best_score = 0

        for token_row in tokens:
            brand = token_row["brand_token"]

            # Primary check: does the brand appear as a whole-word substring?
            # This is the highest-precision signal.
            # Use word boundary check to avoid "NIUM" matching "ALUMINIUM"
            if _is_whole_word_match(brand, entity_name):
                score = 100
            else:
                # Fallback: fuzzy match on the extracted brand token
                entity_token = _extract_brand_token(entity_name, suffix_pat)
                score = fuzz.token_set_ratio(brand, entity_token)

                # Penalize fuzzy matches where the brand isn't actually
                # a distinct component of the entity name
                if score >= threshold and not _brand_words_present(brand, entity_name):
                    score = max(score - 15, 0)

            if score > best_score and score >= threshold:
                best_score = score
                best_match = token_row

        if best_match:
            matches.append({
                "lei": lei,
                "matched_parent_lei": best_match["parent_lei"],
                "brand_token": best_match["brand_token"],
                "match_score": best_score,
                "match_field": "legal_name",
            })

        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{total}] fuzzy matching... {len(matches)} matches so far", flush=True)

    print(f"[INFO] Fuzzy matching complete: {len(matches)} matches from {total} entities", flush=True)

    if not matches:
        return pd.DataFrame(columns=["lei", "matched_parent_lei", "brand_token", "match_score", "match_field"])

    result = pd.DataFrame(matches)
    # Keep only the best match per entity
    result = result.sort_values("match_score", ascending=False).drop_duplicates(subset=["lei"], keep="first")
    return result.reset_index(drop=True)


def build_phase1_inferred_edges(
    matches_df: pd.DataFrame,
    min_score: int = 80,
) -> pd.DataFrame:
    """Convert fuzzy matches into relationship-format edges.

    Parameters
    ----------
    matches_df : Output from fuzzy_match_names().
    min_score : Minimum match score to include.

    Returns
    -------
    DataFrame with RELATIONSHIP_COLUMNS-compatible schema plus inference metadata.
    """
    from src.gleif import RELATIONSHIP_COLUMNS

    if matches_df.empty:
        return pd.DataFrame(columns=RELATIONSHIP_COLUMNS + ["inference_source", "inference_score"])

    filtered = matches_df[matches_df["match_score"] >= min_score].copy()

    if filtered.empty:
        return pd.DataFrame(columns=RELATIONSHIP_COLUMNS + ["inference_source", "inference_score"])

    edges = pd.DataFrame({
        "source_lei": filtered["lei"],
        "target_lei": filtered["matched_parent_lei"],
        "relationship_type": "inferred_name_match",
        "relationship_status": "INFERRED",
        "accounting_standard": None,
        "period_end": None,
        "valid_from": None,
        "valid_to": None,
        "inference_source": "name_fuzzy_match",
        "inference_score": filtered["match_score"].values,
    })

    return edges.reset_index(drop=True)


def phase1_summary(matches_df: pd.DataFrame) -> dict:
    """Generate a summary dict for Phase 1 results."""
    if matches_df.empty:
        return {"total_matches": 0, "high_confidence": 0, "medium_confidence": 0}

    return {
        "total_matches": len(matches_df),
        "high_confidence": int((matches_df["match_score"] >= 90).sum()),
        "medium_confidence": int(
            ((matches_df["match_score"] >= 80) & (matches_df["match_score"] < 90)).sum()
        ),
        "unique_parent_brands": int(matches_df["brand_token"].nunique()),
        "avg_score": round(matches_df["match_score"].mean(), 1),
        "score_distribution": matches_df["match_score"].describe().to_dict(),
    }
