"""Phase 2: Address parsing and geospatial clustering for parent inference.

Fetches full street addresses from GLEIF API, normalizes them, and clusters
entities sharing the same registered address. Shared corporate-secretary or
office-hotel addresses are strong indicators of SPE structures or shared parents.
"""
from __future__ import annotations

import re
import pandas as pd
from src.utils import get_json, HTTPFetchError
from src.country_config import get_country_config
from config import GLEIF_BASE, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}

ADDRESS_COLUMNS = [
    "lei", "address_line1", "address_line2", "postal_code",
    "region", "city", "country",
]

# Malaysian address abbreviation expansions
MY_ABBREVIATIONS = {
    "JLN": "JALAN",
    "JLN.": "JALAN",
    "LRG": "LORONG",
    "LRG.": "LORONG",
    "TMN": "TAMAN",
    "TMN.": "TAMAN",
    "KG": "KAMPUNG",
    "KG.": "KAMPUNG",
    "SRI": "SERI",
    "BT": "BUKIT",
    "BT.": "BUKIT",
    "NO": "NO",
    "NO.": "NO",
}

# Known office-hotel / corporate secretarial addresses in Malaysia
# These are addresses where many shell companies or SPEs register
KNOWN_OFFICE_HOTELS = [
    # Labuan IBFC — Malaysia's offshore financial center
    "LABUAN",
    # Common corporate secretarial buildings
    "WISMA UOA",
    "MENARA UOA",
    # Add more as discovered from the data
]


def fetch_single_address(lei: str) -> dict | None:
    """Fetch the full legal address for a single LEI."""
    url = f"{GLEIF_BASE}/lei-records/{lei}"
    try:
        payload = get_json(url, headers=HEADERS, allow_404=True)
    except HTTPFetchError:
        return None

    data = payload.get("data")
    if not data:
        return None

    entity = data.get("attributes", {}).get("entity", {}) or {}
    addr = entity.get("legalAddress", {}) or {}
    lines = addr.get("addressLines", []) or []

    return {
        "lei": lei,
        "address_line1": lines[0] if len(lines) > 0 else None,
        "address_line2": " | ".join(lines[1:]) if len(lines) > 1 else None,
        "postal_code": addr.get("postalCode"),
        "region": addr.get("region"),
        "city": addr.get("city"),
        "country": addr.get("country"),
    }


def fetch_full_addresses(leis: list[str]) -> pd.DataFrame:
    """Bulk fetch full addresses for a list of LEIs.

    Returns DataFrame with ADDRESS_COLUMNS.
    """
    rows: list[dict] = []
    total = len(leis)

    for i, lei in enumerate(leis):
        result = fetch_single_address(lei)
        if result:
            rows.append(result)

        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{total}] addresses fetched...", flush=True)

    print(f"[INFO] Address fetch complete: {len(rows)}/{total} addresses retrieved", flush=True)

    if not rows:
        return pd.DataFrame(columns=ADDRESS_COLUMNS)
    return pd.DataFrame(rows).reindex(columns=ADDRESS_COLUMNS)


def normalize_address(
    addr: str | None,
    abbreviations: dict[str, str] | None = None,
) -> str:
    """Normalize an address string for clustering.

    Uppercases, expands abbreviations, collapses whitespace, strips punctuation.

    Parameters
    ----------
    abbreviations : address abbreviation expansions.  Defaults to
        ``MY_ABBREVIATIONS`` for backward compatibility.
    """
    if not addr or pd.isna(addr):
        return ""

    if abbreviations is None:
        abbreviations = MY_ABBREVIATIONS

    result = str(addr).upper().strip()

    # Expand abbreviations (word-boundary aware)
    for abbr, full in abbreviations.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        result = re.sub(pattern, full, result)

    # Remove punctuation except hyphens (common in building names)
    result = re.sub(r"[^A-Z0-9\s\-]", " ", result)
    # Collapse whitespace
    result = re.sub(r"\s+", " ", result).strip()

    return result


def build_address_key(
    row: pd.Series,
    abbreviations: dict[str, str] | None = None,
) -> str:
    """Build a clustering key from address components.

    Uses normalized first address line + postal code for grouping.
    This catches entities at the exact same building/suite.
    """
    line1 = normalize_address(row.get("address_line1"), abbreviations)
    postal = str(row.get("postal_code", "")).strip()

    if not line1:
        return ""
    return f"{line1}|{postal}"


def cluster_by_address(
    addresses_df: pd.DataFrame,
    min_cluster_size: int = 3,
    country: str = "MY",
) -> pd.DataFrame:
    """Group entities by shared registered address.

    Parameters
    ----------
    addresses_df : DataFrame with ADDRESS_COLUMNS.
    min_cluster_size : Minimum entities per cluster to be flagged.
    country : ISO-2 code — selects address abbreviations and office-hotel
        markers from ``country_config``.

    Returns
    -------
    DataFrame with columns [lei, address_cluster_id, cluster_size,
    cluster_address_key, is_office_hotel].
    """
    output_cols = ["lei", "address_cluster_id", "cluster_size",
                   "cluster_address_key", "is_office_hotel"]

    if addresses_df.empty:
        return pd.DataFrame(columns=output_cols)

    cfg = get_country_config(country)
    abbreviations = cfg.address_abbreviations or MY_ABBREVIATIONS
    markers = cfg.office_hotel_markers or KNOWN_OFFICE_HOTELS

    df = addresses_df.copy()
    df["address_key"] = df.apply(
        lambda row: build_address_key(row, abbreviations), axis=1
    )

    # Drop empty keys
    df = df[df["address_key"].str.len() > 0]

    if df.empty:
        return pd.DataFrame(columns=output_cols)

    # Count entities per address key
    key_counts = df["address_key"].value_counts()
    cluster_keys = key_counts[key_counts >= min_cluster_size].index

    if cluster_keys.empty:
        return pd.DataFrame(columns=output_cols)

    # Assign cluster IDs
    cluster_map = {key: i for i, key in enumerate(cluster_keys)}

    clustered = df[df["address_key"].isin(cluster_keys)].copy()
    clustered["address_cluster_id"] = clustered["address_key"].map(cluster_map)
    clustered["cluster_size"] = clustered["address_key"].map(key_counts)
    clustered["cluster_address_key"] = clustered["address_key"]
    clustered["is_office_hotel"] = clustered["address_key"].apply(
        lambda k: _is_office_hotel(k, markers)
    )

    return clustered[output_cols].reset_index(drop=True)


def _is_office_hotel(
    address_key: str,
    markers: list[str] | None = None,
) -> bool:
    """Check if an address matches known office-hotel or corp-sec locations."""
    if markers is None:
        markers = KNOWN_OFFICE_HOTELS
    upper = address_key.upper()
    return any(marker in upper for marker in markers)


def infer_shared_parent_from_cluster(
    clusters_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
) -> pd.DataFrame:
    """Within each cluster, propagate known parent signals to other members.

    If at least one entity in a cluster has a known parent, infer that
    other cluster members may share the same parent or parent jurisdiction.

    Returns DataFrame with [lei, inferred_parent_lei, inference_source, confidence].
    """
    output_cols = ["lei", "inferred_parent_lei", "inference_source", "confidence"]

    if clusters_df.empty or relationships_df.empty:
        return pd.DataFrame(columns=output_cols)

    # Find which clustered entities have known parents
    known_parents = relationships_df[["source_lei", "target_lei"]].drop_duplicates()
    known_parent_leis = set(known_parents["source_lei"].dropna())

    inferred: list[dict] = []

    for cluster_id in clusters_df["address_cluster_id"].unique():
        cluster = clusters_df[clusters_df["address_cluster_id"] == cluster_id]
        cluster_leis = set(cluster["lei"])

        # Find entities in this cluster that have known parents
        anchors = cluster_leis & known_parent_leis
        if not anchors:
            continue

        # Get the parent LEIs for anchor entities
        anchor_parents = known_parents[known_parents["source_lei"].isin(anchors)]

        # For non-anchor entities in the cluster, infer the parent
        unknowns = cluster_leis - known_parent_leis
        for lei in unknowns:
            # Use the most common parent in the cluster as the inference
            for _, parent_row in anchor_parents.iterrows():
                inferred.append({
                    "lei": lei,
                    "inferred_parent_lei": parent_row["target_lei"],
                    "inference_source": "address_cluster",
                    "confidence": 0.6,  # moderate confidence
                })

    if not inferred:
        return pd.DataFrame(columns=output_cols)

    result = pd.DataFrame(inferred)
    # Keep highest-confidence inference per entity
    result = result.sort_values("confidence", ascending=False).drop_duplicates(
        subset=["lei"], keep="first"
    )
    return result[output_cols].reset_index(drop=True)


def phase2_summary(clusters_df: pd.DataFrame) -> dict:
    """Generate a summary dict for Phase 2 results."""
    if clusters_df.empty:
        return {
            "total_clustered_entities": 0,
            "num_clusters": 0,
            "office_hotel_entities": 0,
        }

    return {
        "total_clustered_entities": len(clusters_df),
        "num_clusters": int(clusters_df["address_cluster_id"].nunique()),
        "largest_cluster": int(clusters_df["cluster_size"].max()),
        "office_hotel_entities": int(clusters_df["is_office_hotel"].sum()),
        "avg_cluster_size": round(clusters_df.groupby("address_cluster_id").size().mean(), 1),
    }
