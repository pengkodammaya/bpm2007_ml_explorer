from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from config import RAW_DIR, INTERIM_DIR, PROCESSED_DIR, INFERENCE_DIR, DATA_DIR, country_paths
from src.gleif import (
    ENTITY_COLUMNS,
    RELATIONSHIP_COLUMNS,
    fetch_country_lei_records,
    fetch_lei_records_by_lei,
    fetch_malaysia_lei_records,
    fetch_relationships_for_lei,
    normalize_entity_records,
    normalize_relationship_records,
    search_lei_records,
)
from src.entity_analysis import full_structural_report, compare_countries
from src.graph_build import build_di_graph, graph_summary, parent_flags, add_graph_features
from src.io_helpers import save_df, save_csv, load_df, load_optional_df
from src.edgar import fetch_company_tickers, match_entities_to_edgar
from src.scoring import compute_coverage_score
from src.reporting_exceptions import fetch_reporting_exceptions, enrich_with_exception_flags, EXCEPTION_COLUMNS
from src.name_inference import (
    extract_parent_brand_tokens,
    fuzzy_match_names,
    build_phase1_inferred_edges,
    phase1_summary,
)
from src.address_cluster import (
    fetch_full_addresses,
    cluster_by_address,
    infer_shared_parent_from_cluster,
    phase2_summary,
    ADDRESS_COLUMNS,
)
from src.jurisdiction_predictor import (
    prepare_training_features,
    train_jurisdiction_model,
    predict_parent_jurisdiction,
    save_model_artifacts,
    phase3_summary,
)
from src.graph_link_prediction import (
    run_phase4_analysis,
    phase4_summary,
)
from src.ctos import (
    CTOS_MATCH_COLUMNS,
    fetch_ctos_company_directory,
    load_ctos_company_snapshot,
    match_entities_to_ctos,
)


REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _should_use_cached_file(path: str | Path, force_refresh: bool = False) -> bool:
    """Return true when a non-empty cache file should be reused."""
    p = Path(path)
    return (not force_refresh) and p.exists() and p.stat().st_size > 0


def save_uie_assignment_snapshot(
    uie_assignments: pd.DataFrame,
    country: str,
    run_timestamp: datetime | None = None,
) -> Path:
    """Save a repo-trackable timestamped CSV snapshot of UIE assignments."""
    timestamp = run_timestamp or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp_label = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    country_label = country.upper()
    snapshot_dir = REPORTS_DIR / "uie_snapshots" / country_label.lower()
    snapshot_path = snapshot_dir / f"uie_assignments_{country_label}_{timestamp_label}.csv"
    return save_csv(uie_assignments, snapshot_path)


def _load_with_fallback(
    paths: dict, key: str, columns: list[str] | None = None
) -> pd.DataFrame:
    """Load from primary path, falling back to legacy path for MY data."""
    primary = paths[key]
    legacy = paths.get("legacy", {}).get(key)
    try:
        return load_df(primary)
    except FileNotFoundError:
        if legacy:
            try:
                return load_df(legacy)
            except FileNotFoundError:
                pass
        if columns is not None:
            return pd.DataFrame(columns=columns)
        raise


def _combine_entity_sets(*frames: pd.DataFrame) -> pd.DataFrame:
    normalized = [normalize_entity_records(frame) for frame in frames if frame is not None and not frame.empty]
    if not normalized:
        return pd.DataFrame(columns=ENTITY_COLUMNS)
    return (
        pd.concat(normalized, ignore_index=True)
        .dropna(subset=["lei"])
        .drop_duplicates(subset=["lei"], keep="first")
        .reset_index(drop=True)
    )


def _add_size_proxy(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    defaults = {
        "in_degree": 0,
        "out_degree": 0,
        "weakly_connected_component_size": 1,
        "pagerank": 0.0,
        "degree_centrality": 0.0,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default)

    out["size_proxy"] = (
        out["in_degree"].astype(float)
        + out["out_degree"].astype(float)
        + out["weakly_connected_component_size"].astype(float)
        + (out["pagerank"].astype(float) * 1000)
        + (out["degree_centrality"].astype(float) * 100)
    )
    return out


def _as_int_flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _foreign_parent_flags(
    relationships: pd.DataFrame,
    entities: pd.DataFrame,
    country: str = "MY",
) -> pd.DataFrame:
    relationships = normalize_relationship_records(relationships)
    if relationships.empty:
        return pd.DataFrame(columns=["lei", "foreign_parent"])

    entity_countries = (
        normalize_entity_records(entities)[["lei", "country_legal"]]
        .dropna(subset=["lei"])
        .drop_duplicates(subset=["lei"])
        .rename(columns={"lei": "target_lei", "country_legal": "target_country"})
    )

    relationship_targets = relationships[["source_lei", "target_lei", "relationship_type"]].merge(
        entity_countries,
        on="target_lei",
        how="left",
    )

    return (
        relationship_targets.loc[
            relationship_targets["relationship_type"].isin(["direct_parent", "ultimate_parent"])
            & relationship_targets["target_country"].notna()
            & (relationship_targets["target_country"] != country),
            ["source_lei"],
        ]
        .drop_duplicates()
        .assign(foreign_parent=1)
        .rename(columns={"source_lei": "lei"})
    )


def _first_relationship_by_type(
    relationships: pd.DataFrame,
    relationship_type: str,
) -> pd.DataFrame:
    rows = relationships.loc[
        relationships["relationship_type"] == relationship_type,
        ["source_lei", "target_lei"],
    ].dropna(subset=["source_lei", "target_lei"])
    if rows.empty:
        return pd.DataFrame(columns=["lei", f"{relationship_type}_lei"])
    return (
        rows.drop_duplicates(subset=["source_lei"], keep="first")
        .rename(columns={"source_lei": "lei", "target_lei": f"{relationship_type}_lei"})
        .reset_index(drop=True)
    )


def _preferred_upstream_relationships(relationships: pd.DataFrame) -> pd.DataFrame:
    """Select one upstream parent per LEI, preferring ultimate over direct."""
    rels = normalize_relationship_records(relationships)
    if rels.empty:
        return pd.DataFrame(columns=[
            "direct_parent_lei",
            "chain_parent_lei",
            "chain_parent_relationship_type",
        ])

    rows = rels.loc[
        rels["relationship_type"].isin(["ultimate_parent", "direct_parent"]),
        ["source_lei", "target_lei", "relationship_type"],
    ].dropna(subset=["source_lei", "target_lei"])
    if rows.empty:
        return pd.DataFrame(columns=[
            "direct_parent_lei",
            "chain_parent_lei",
            "chain_parent_relationship_type",
        ])

    priority = {"ultimate_parent": 0, "direct_parent": 1}
    rows = rows.assign(_priority=rows["relationship_type"].map(priority).fillna(9))
    return (
        rows.sort_values(["source_lei", "_priority"])
        .drop_duplicates(subset=["source_lei"], keep="first")
        .rename(columns={
            "source_lei": "direct_parent_lei",
            "target_lei": "chain_parent_lei",
            "relationship_type": "chain_parent_relationship_type",
        })
        [["direct_parent_lei", "chain_parent_lei", "chain_parent_relationship_type"]]
        .reset_index(drop=True)
    )


def _load_uie_chain_context(country: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cross-country relationship/entity context for upstream UIE chains.

    The host-country assignment remains scoped to the requested country, but
    parent chains can pass through a foreign holding-company jurisdiction such
    as SG before reaching the actual ultimate investor economy.
    """
    relationship_frames: list[pd.DataFrame] = []
    for pattern in ["gleif_*_relationships.parquet", "gleif_*_relationships_wip.parquet"]:
        for path in INTERIM_DIR.glob(pattern):
            try:
                relationship_frames.append(load_df(path))
            except Exception as e:
                print(f"[WARN] Could not load UIE chain relationship context {path.name}: {e}")

    entity_frames: list[pd.DataFrame] = []
    for pattern in ["gleif_*_lei.parquet", "gleif_*_related_lei.parquet"]:
        for path in RAW_DIR.glob(pattern):
            try:
                entity_frames.append(load_df(path))
            except Exception as e:
                print(f"[WARN] Could not load UIE chain entity context {path.name}: {e}")

    chain_relationships = (
        normalize_relationship_records(pd.concat(relationship_frames, ignore_index=True))
        if relationship_frames else pd.DataFrame(columns=RELATIONSHIP_COLUMNS)
    )
    if not chain_relationships.empty:
        chain_relationships = chain_relationships.drop_duplicates(
            subset=["source_lei", "target_lei", "relationship_type"],
            keep="first",
        )

    chain_entities = _combine_entity_sets(*entity_frames)
    print(
        f"[INFO] [{country.upper()}] UIE chain context: "
        f"{len(chain_relationships):,} relationship rows, {len(chain_entities):,} entity rows"
    )
    return chain_relationships, chain_entities


UIE_OVERRIDE_COLUMNS = [
    "host_country",
    "lei",
    "legal_name",
    "uie_lei",
    "uie_name",
    "uie_country",
    "uie_source",
    "evidence_tier",
    "uie_confidence",
    "override_reason",
    "source_url",
    "source_note",
    "active",
]

UIE_REVIEW_STATUS_COLUMNS = [
    "host_country",
    "lei",
    "legal_name",
    "review_status",
    "entity_type",
    "review_note",
    "active",
]


def load_uie_overrides(country: str = "MY") -> pd.DataFrame:
    """Load analyst-reviewed UIE overrides for a host country."""
    path = DATA_DIR / "manual" / "uie_overrides.csv"
    if not path.exists():
        return pd.DataFrame(columns=UIE_OVERRIDE_COLUMNS)

    overrides = pd.read_csv(path)
    for col in UIE_OVERRIDE_COLUMNS:
        if col not in overrides.columns:
            overrides[col] = pd.NA

    active = overrides["active"].fillna(1).astype(str).str.lower().isin(["1", "true", "yes", "y"])
    host = overrides["host_country"].fillna("").astype(str).str.upper().eq(country.upper())
    overrides = overrides[active & host].copy()
    overrides["uie_source"] = overrides["uie_source"].fillna("manual_verified")
    overrides["evidence_tier"] = overrides["evidence_tier"].fillna("A_manual_verified_uie")
    overrides["uie_confidence"] = pd.to_numeric(
        overrides["uie_confidence"],
        errors="coerce",
    ).fillna(0.95)
    return overrides[UIE_OVERRIDE_COLUMNS].drop_duplicates(subset=["lei"], keep="first")


def load_uie_review_statuses(country: str = "MY") -> pd.DataFrame:
    """Load analyst review statuses for UIE target triage."""
    path = DATA_DIR / "manual" / "uie_review_status.csv"
    if not path.exists():
        return pd.DataFrame(columns=UIE_REVIEW_STATUS_COLUMNS)

    statuses = pd.read_csv(path)
    for col in UIE_REVIEW_STATUS_COLUMNS:
        if col not in statuses.columns:
            statuses[col] = pd.NA

    active = statuses["active"].fillna(1).astype(str).str.lower().isin(["1", "true", "yes", "y"])
    host = statuses["host_country"].fillna("").astype(str).str.upper().eq(country.upper())
    return (
        statuses[active & host][UIE_REVIEW_STATUS_COLUMNS]
        .drop_duplicates(subset=["lei"], keep="first")
        .reset_index(drop=True)
    )


def build_uie_assignments(
    domestic_entities: pd.DataFrame,
    relationships: pd.DataFrame,
    all_entities: pd.DataFrame,
    *,
    country: str = "MY",
    chain_relationships: pd.DataFrame | None = None,
    exceptions: pd.DataFrame | None = None,
    name_matches: pd.DataFrame | None = None,
    address_inferred: pd.DataFrame | None = None,
    jurisdiction_predictions: pd.DataFrame | None = None,
    ctos_matches: pd.DataFrame | None = None,
    manual_overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assign a best-available Ultimate Investor Economy signal per host entity.

    The ownership graph can include foreign parent nodes, but this table is
    intentionally domestic-entity scoped: one row per host-country LEI with
    parent evidence and a source/tier for the selected UIE.
    """
    domestic = normalize_entity_records(domestic_entities)
    rels = normalize_relationship_records(relationships)
    chain_rels = normalize_relationship_records(
        chain_relationships if chain_relationships is not None else relationships
    )
    entities = normalize_entity_records(all_entities)
    host = country.upper()

    out = domestic.loc[domestic["country_legal"].eq(host)].copy()
    if out.empty:
        out = domestic.copy()

    out = out[["lei", "legal_name", "country_legal"]].rename(
        columns={"country_legal": "entity_country"}
    )
    out.insert(0, "host_country", host)

    entity_lookup = (
        entities[["lei", "legal_name", "country_legal"]]
        .dropna(subset=["lei"])
        .drop_duplicates(subset=["lei"], keep="first")
        .rename(columns={"legal_name": "lookup_name", "country_legal": "lookup_country"})
    )

    for rel_type, prefix in [
        ("direct_parent", "direct_parent"),
        ("ultimate_parent", "ultimate_parent"),
    ]:
        rel_df = _first_relationship_by_type(rels, rel_type)
        out = out.merge(rel_df, on="lei", how="left")
        out = out.merge(
            entity_lookup.rename(
                columns={
                    "lei": f"{prefix}_lei",
                    "lookup_name": f"{prefix}_name",
                    "lookup_country": f"{prefix}_country",
                }
            ),
            on=f"{prefix}_lei",
            how="left",
        )

    chain_parent_links = _preferred_upstream_relationships(chain_rels)
    if not chain_parent_links.empty:
        out = out.merge(chain_parent_links, on="direct_parent_lei", how="left")
        out = out.merge(
            entity_lookup.rename(
                columns={
                    "lei": "chain_parent_lei",
                    "lookup_name": "chain_parent_name",
                    "lookup_country": "chain_parent_country",
                }
            ),
            on="chain_parent_lei",
            how="left",
        )
        same_as_direct = out["chain_parent_lei"].eq(out["direct_parent_lei"]).fillna(False)
        same_as_domestic = out["chain_parent_lei"].eq(out["lei"]).fillna(False)
        out.loc[same_as_direct | same_as_domestic, [
            "chain_parent_lei",
            "chain_parent_relationship_type",
            "chain_parent_name",
            "chain_parent_country",
        ]] = pd.NA
    else:
        out["chain_parent_lei"] = None
        out["chain_parent_relationship_type"] = None
        out["chain_parent_name"] = None
        out["chain_parent_country"] = None

    if exceptions is not None and not exceptions.empty and "lei" in exceptions.columns:
        exc = exceptions[["lei", "exception_reason"]].drop_duplicates(subset=["lei"])
        out = out.merge(exc, on="lei", how="left")
    else:
        out["exception_reason"] = None
    out["is_non_consolidating"] = (
        out["exception_reason"].eq("NON_CONSOLIDATING").fillna(False).astype(int)
    )

    if name_matches is not None and not name_matches.empty and "lei" in name_matches.columns:
        name_cols = [
            c for c in ["lei", "matched_parent_lei", "brand_token", "match_score"]
            if c in name_matches.columns
        ]
        names = name_matches[name_cols].copy()
        if "match_score" in names.columns:
            names = names.sort_values("match_score", ascending=False)
        names = names.drop_duplicates(subset=["lei"], keep="first")
        out = out.merge(names, on="lei", how="left")
        out = out.merge(
            entity_lookup.rename(
                columns={
                    "lei": "matched_parent_lei",
                    "lookup_name": "matched_parent_name",
                    "lookup_country": "matched_parent_country",
                }
            ),
            on="matched_parent_lei",
            how="left",
        )
    else:
        out["matched_parent_lei"] = None
        out["brand_token"] = None
        out["match_score"] = pd.NA
        out["matched_parent_name"] = None
        out["matched_parent_country"] = None

    if address_inferred is not None and not address_inferred.empty and "lei" in address_inferred.columns:
        addr_cols = [
            c for c in ["lei", "inferred_parent_lei", "confidence"]
            if c in address_inferred.columns
        ]
        addr = address_inferred[addr_cols].copy()
        if "confidence" in addr.columns:
            addr = addr.sort_values("confidence", ascending=False)
        addr = addr.drop_duplicates(subset=["lei"], keep="first")
        out = out.merge(addr, on="lei", how="left")
        out = out.merge(
            entity_lookup.rename(
                columns={
                    "lei": "inferred_parent_lei",
                    "lookup_name": "address_parent_name",
                    "lookup_country": "address_parent_country",
                }
            ),
            on="inferred_parent_lei",
            how="left",
        )
    else:
        out["inferred_parent_lei"] = None
        out["confidence"] = pd.NA
        out["address_parent_name"] = None
        out["address_parent_country"] = None

    if (
        jurisdiction_predictions is not None
        and not jurisdiction_predictions.empty
        and "lei" in jurisdiction_predictions.columns
    ):
        pred_cols = [
            c for c in [
                "lei", "predicted_parent_country", "prediction_probability",
                "confidence_gap", "top3_countries", "top3_probabilities",
            ]
            if c in jurisdiction_predictions.columns
        ]
        preds = jurisdiction_predictions[pred_cols].drop_duplicates(subset=["lei"])
        out = out.merge(preds, on="lei", how="left")
    else:
        out["predicted_parent_country"] = None
        out["prediction_probability"] = pd.NA
        out["confidence_gap"] = pd.NA
        out["top3_countries"] = None
        out["top3_probabilities"] = None

    if ctos_matches is not None and not ctos_matches.empty and "lei" in ctos_matches.columns:
        ctos_cols = [c for c in CTOS_MATCH_COLUMNS if c in ctos_matches.columns]
        ctos = ctos_matches[ctos_cols].copy()
        if "ctos_match_score" in ctos.columns:
            ctos = ctos.sort_values("ctos_match_score", ascending=False)
        ctos = ctos.drop_duplicates(subset=["lei"], keep="first")
        ctos = ctos.drop(columns=["legal_name"], errors="ignore")
        out = out.merge(ctos, on="lei", how="left")
    else:
        out["ctos_name"] = None
        out["ctos_url"] = None
        out["ctos_registration_no"] = None
        out["ctos_match_score"] = pd.NA
        out["ctos_match_method"] = None
        out["ctos_registered_malaysia"] = 0
    out["ctos_registered_malaysia"] = out["ctos_registered_malaysia"].fillna(0).astype(int)

    if manual_overrides is not None and not manual_overrides.empty and "lei" in manual_overrides.columns:
        override_cols = [
            c for c in [
                "lei", "uie_lei", "uie_name", "uie_country", "uie_source",
                "evidence_tier", "uie_confidence", "override_reason",
                "source_url", "source_note",
            ]
            if c in manual_overrides.columns
        ]
        overrides = manual_overrides[override_cols].copy()
        overrides = overrides.rename(columns={
            "uie_lei": "override_uie_lei",
            "uie_name": "override_uie_name",
            "uie_country": "override_uie_country",
            "uie_source": "override_uie_source",
            "evidence_tier": "override_evidence_tier",
            "uie_confidence": "override_uie_confidence",
        })
        overrides = overrides.drop_duplicates(subset=["lei"], keep="first")
        out = out.merge(overrides, on="lei", how="left")
    else:
        out["override_uie_lei"] = None
        out["override_uie_name"] = None
        out["override_uie_country"] = None
        out["override_uie_source"] = None
        out["override_evidence_tier"] = None
        out["override_uie_confidence"] = pd.NA
        out["override_reason"] = None
        out["source_url"] = None
        out["source_note"] = None

    def choose_uie(row: pd.Series) -> pd.Series:
        if pd.notna(row.get("override_uie_country")):
            confidence = pd.to_numeric(
                pd.Series([row.get("override_uie_confidence")]), errors="coerce"
            ).iloc[0]
            confidence = float(confidence) if pd.notna(confidence) else 0.95
            return pd.Series({
                "uie_lei": row.get("override_uie_lei"),
                "uie_name": row.get("override_uie_name"),
                "uie_country": row.get("override_uie_country"),
                "uie_source": (
                    row.get("override_uie_source")
                    if pd.notna(row.get("override_uie_source"))
                    else "manual_verified"
                ),
                "evidence_tier": (
                    row.get("override_evidence_tier")
                    if pd.notna(row.get("override_evidence_tier"))
                    else "A_manual_verified_uie"
                ),
                "uie_confidence": round(confidence, 4),
            })
        if pd.notna(row.get("ultimate_parent_country")):
            return pd.Series({
                "uie_lei": row.get("ultimate_parent_lei"),
                "uie_name": row.get("ultimate_parent_name"),
                "uie_country": row.get("ultimate_parent_country"),
                "uie_source": "gleif_ultimate_parent",
                "evidence_tier": "A_known_ultimate_parent",
                "uie_confidence": 1.0,
            })
        if pd.notna(row.get("chain_parent_country")):
            rel_type = row.get("chain_parent_relationship_type")
            source = (
                "gleif_upstream_ultimate_parent"
                if rel_type == "ultimate_parent"
                else "gleif_upstream_direct_parent"
            )
            tier = (
                "A_known_upstream_ultimate_parent"
                if rel_type == "ultimate_parent"
                else "B_known_upstream_direct_parent"
            )
            confidence = 0.97 if rel_type == "ultimate_parent" else 0.9
            return pd.Series({
                "uie_lei": row.get("chain_parent_lei"),
                "uie_name": row.get("chain_parent_name"),
                "uie_country": row.get("chain_parent_country"),
                "uie_source": source,
                "evidence_tier": tier,
                "uie_confidence": confidence,
            })
        if pd.notna(row.get("direct_parent_country")):
            return pd.Series({
                "uie_lei": row.get("direct_parent_lei"),
                "uie_name": row.get("direct_parent_name"),
                "uie_country": row.get("direct_parent_country"),
                "uie_source": "gleif_direct_parent",
                "evidence_tier": "B_known_direct_parent",
                "uie_confidence": 0.85,
            })
        if pd.notna(row.get("matched_parent_country")):
            score = pd.to_numeric(pd.Series([row.get("match_score")]), errors="coerce").iloc[0]
            confidence = float(score) / 100 if pd.notna(score) else 0.8
            tier = "C_name_match_high" if confidence >= 0.9 else "D_name_match_medium"
            return pd.Series({
                "uie_lei": row.get("matched_parent_lei"),
                "uie_name": row.get("matched_parent_name"),
                "uie_country": row.get("matched_parent_country"),
                "uie_source": "phase1_name_match",
                "evidence_tier": tier,
                "uie_confidence": round(confidence, 4),
            })
        if pd.notna(row.get("address_parent_country")):
            confidence = pd.to_numeric(pd.Series([row.get("confidence")]), errors="coerce").iloc[0]
            confidence = float(confidence) if pd.notna(confidence) else 0.6
            return pd.Series({
                "uie_lei": row.get("inferred_parent_lei"),
                "uie_name": row.get("address_parent_name"),
                "uie_country": row.get("address_parent_country"),
                "uie_source": "phase2_address_cluster",
                "evidence_tier": "D_address_cluster",
                "uie_confidence": round(confidence, 4),
            })
        if pd.notna(row.get("predicted_parent_country")):
            confidence = pd.to_numeric(
                pd.Series([row.get("prediction_probability")]), errors="coerce"
            ).iloc[0]
            confidence = float(confidence) if pd.notna(confidence) else 0.0
            return pd.Series({
                "uie_lei": None,
                "uie_name": None,
                "uie_country": row.get("predicted_parent_country"),
                "uie_source": "phase3_jurisdiction_model",
                "evidence_tier": "E_model_predicted_jurisdiction",
                "uie_confidence": round(confidence, 4),
            })
        if row.get("is_non_consolidating") == 1:
            return pd.Series({
                "uie_lei": None,
                "uie_name": None,
                "uie_country": None,
                "uie_source": "reporting_exception_no_country",
                "evidence_tier": "B_subsidiary_signal_no_uie",
                "uie_confidence": 0.7,
            })
        return pd.Series({
            "uie_lei": None,
            "uie_name": None,
            "uie_country": None,
            "uie_source": "unassigned",
            "evidence_tier": "U_unassigned",
            "uie_confidence": 0.0,
        })

    out = pd.concat([out, out.apply(choose_uie, axis=1)], axis=1)
    out["is_known_uie"] = out["uie_source"].isin(
        [
            "manual_verified",
            "gleif_ultimate_parent",
            "gleif_upstream_ultimate_parent",
            "gleif_direct_parent",
            "gleif_upstream_direct_parent",
        ]
    ).astype(int)
    out["is_inferred_uie"] = out["uie_source"].isin(
        ["phase1_name_match", "phase2_address_cluster", "phase3_jurisdiction_model"]
    ).astype(int)

    preferred_cols = [
        "host_country", "lei", "legal_name", "entity_country",
        "direct_parent_lei", "direct_parent_name", "direct_parent_country",
        "ultimate_parent_lei", "ultimate_parent_name", "ultimate_parent_country",
        "chain_parent_lei", "chain_parent_name", "chain_parent_country",
        "chain_parent_relationship_type",
        "uie_lei", "uie_name", "uie_country", "uie_source", "evidence_tier",
        "uie_confidence", "is_known_uie", "is_inferred_uie",
        "is_non_consolidating", "exception_reason",
        "matched_parent_lei", "matched_parent_name", "matched_parent_country",
        "brand_token", "match_score",
        "inferred_parent_lei", "address_parent_name", "address_parent_country",
        "confidence", "predicted_parent_country", "prediction_probability",
        "confidence_gap", "top3_countries", "top3_probabilities",
        "ctos_registered_malaysia", "ctos_name", "ctos_registration_no",
        "ctos_match_score", "ctos_match_method", "ctos_url",
        "override_reason", "source_url", "source_note",
    ]
    for col in preferred_cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out[preferred_cols].sort_values(
        ["evidence_tier", "uie_confidence", "legal_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_uie_review_targets(
    scored: pd.DataFrame,
    uie_assignments: pd.DataFrame,
    *,
    review_statuses: pd.DataFrame | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Build a prioritized review queue for uncertain investor-economy calls."""
    if scored.empty:
        return pd.DataFrame()

    scored_cols = [
        c for c in [
            "lei", "legal_name", "country", "city", "coverage_gap_priority_score",
            "coverage_gap_score", "reason_flags", "size_proxy", "has_parent_link",
            "has_ultimate_link", "foreign_parent", "is_non_consolidating",
            "has_exception_filed", "has_inferred_parent", "in_address_cluster",
        ]
        if c in scored.columns
    ]
    uie_cols = [
        c for c in [
            "lei", "uie_country", "uie_source", "evidence_tier", "uie_confidence",
            "is_known_uie", "is_inferred_uie", "direct_parent_name",
            "direct_parent_country", "ultimate_parent_name", "ultimate_parent_country",
            "predicted_parent_country", "prediction_probability", "confidence_gap",
            "top3_countries", "top3_probabilities", "ctos_registered_malaysia",
            "ctos_name", "ctos_match_score", "ctos_match_method",
            "override_reason", "source_url", "source_note",
        ]
        if c in uie_assignments.columns
    ]

    out = scored[scored_cols].merge(
        uie_assignments[uie_cols].drop_duplicates(subset=["lei"]),
        on="lei",
        how="left",
    )

    if review_statuses is not None and not review_statuses.empty and "lei" in review_statuses.columns:
        status_cols = [
            c for c in ["lei", "review_status", "entity_type", "review_note"]
            if c in review_statuses.columns
        ]
        out = out.merge(
            review_statuses[status_cols].drop_duplicates(subset=["lei"], keep="first"),
            on="lei",
            how="left",
        )
    else:
        out["review_status"] = pd.NA
        out["entity_type"] = pd.NA
        out["review_note"] = pd.NA

    base = pd.to_numeric(
        out.get("coverage_gap_priority_score", out.get("coverage_gap_score", 0)),
        errors="coerce",
    ).fillna(0.0)
    source = out.get("uie_source", pd.Series("", index=out.index)).fillna("")
    source_penalty = source.map({
        "unassigned": 0.25,
        "reporting_exception_no_country": 0.20,
        "phase3_jurisdiction_model": 0.18,
        "phase2_address_cluster": 0.12,
        "phase1_name_match": 0.08,
        "gleif_direct_parent": 0.03,
        "gleif_ultimate_parent": 0.0,
        "manual_verified": 0.0,
    }).fillna(0.10)
    ctos_bonus = pd.to_numeric(
        out.get("ctos_registered_malaysia", pd.Series(0, index=out.index)),
        errors="coerce",
    ).fillna(0).clip(0, 1) * 0.02

    out["entity_type"] = out.apply(_infer_uie_entity_type, axis=1)
    product_vehicle_penalty = out["entity_type"].eq("fund_or_product_vehicle").astype(float) * 0.20
    nominee_vehicle_penalty = out["entity_type"].eq("nominee_or_custodian").astype(float) * 0.12
    out["review_priority_score"] = (
        base + source_penalty + ctos_bonus - product_vehicle_penalty - nominee_vehicle_penalty
    ).clip(lower=0.0, upper=1.0).round(4)

    def reason(row: pd.Series) -> str:
        entity_type = str(row.get("entity_type") or "")
        if entity_type == "fund_or_product_vehicle":
            return "fund_product_vehicle_rule_applied"
        if entity_type == "nominee_or_custodian":
            return "nominee_or_custodian_rule_applied"
        src = row.get("uie_source")
        if src == "phase3_jurisdiction_model":
            return "validate_model_only_investor_economy"
        if src == "phase2_address_cluster":
            return "verify_address_cluster_parent"
        if src == "phase1_name_match":
            return "verify_name_matched_parent"
        if src == "reporting_exception_no_country":
            return "review_non_consolidating_no_uie"
        if src == "unassigned" or pd.isna(src):
            return "find_parent_or_investor_economy"
        if pd.to_numeric(row.get("coverage_gap_priority_score"), errors="coerce") >= 0.4:
            return "known_uie_but_high_coverage_gap"
        return "lower_priority_known_uie"

    out["review_reason"] = out.apply(reason, axis=1)
    out["review_status"] = out.apply(_infer_uie_review_status, axis=1)
    out["review_sort_bucket"] = out["entity_type"].isin([
        "fund_or_product_vehicle",
        "nominee_or_custodian",
    ]).astype(int)

    focus = (
        out["uie_source"].isin([
            "unassigned",
            "reporting_exception_no_country",
            "phase3_jurisdiction_model",
            "phase2_address_cluster",
            "phase1_name_match",
        ])
        | (pd.to_numeric(out.get("coverage_gap_priority_score", 0), errors="coerce").fillna(0) >= 0.4)
    )
    out = out[focus].sort_values(
        ["review_sort_bucket", "review_priority_score", "coverage_gap_priority_score", "legal_name"],
        ascending=[True, False, False, True],
    )

    preferred_cols = [
        "lei", "legal_name", "review_priority_score", "review_reason",
        "review_status", "entity_type", "review_note",
        "coverage_gap_priority_score", "reason_flags",
        "uie_country", "uie_source", "evidence_tier", "uie_confidence",
        "predicted_parent_country", "prediction_probability", "confidence_gap",
        "top3_countries", "top3_probabilities",
        "direct_parent_name", "direct_parent_country",
        "ultimate_parent_name", "ultimate_parent_country",
        "ctos_registered_malaysia", "ctos_name", "ctos_match_score", "ctos_match_method",
        "override_reason", "source_url", "source_note",
        "city", "size_proxy", "has_parent_link", "has_ultimate_link",
        "foreign_parent", "is_non_consolidating", "has_exception_filed",
        "has_inferred_parent", "in_address_cluster",
    ]
    for col in preferred_cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out[preferred_cols].head(limit).reset_index(drop=True)


def _infer_uie_entity_type(row: pd.Series) -> str:
    existing = row.get("entity_type")
    if pd.notna(existing) and str(existing).strip():
        return str(existing)

    name = str(row.get("legal_name") or "").upper()
    if any(token in name for token in [" NOMINEE", "NOMINEES", " CUSTODIAN", "TRUSTEE"]):
        return "nominee_or_custodian"
    if _is_fund_or_product_vehicle_name(name):
        return "fund_or_product_vehicle"
    if str(row.get("uie_source") or "") == "manual_verified":
        return "manual_verified_entity"
    if "BANK" in name:
        return "financial_institution"
    return "operating_or_holding_company"


def _is_fund_or_product_vehicle_name(name: str) -> bool:
    normalized = f" {name.upper().replace('.', ' ')} "
    compact = " ".join(normalized.split())
    markers = [
        " FUND",
        " FUNDS",
        " PORTFOLIO",
        " PRS ",
        " MONEY MARKET",
        " BOND",
        " SUKUK",
        " EQUITY",
        " EQUITIES",
        " INCOME",
        " BALANCED",
        " ETF",
        " AMANAH SAHAM",
        " UNIT TRUST",
        " WHOLESALE",
        " MULTI ASSET",
        " MIXED ASSET",
        " ASIA EQUITIES",
        " GLOBAL TACTICAL",
        " SHARIAH ",
        " DANA ",
    ]
    if any(marker in normalized for marker in markers):
        return True
    fund_prefixes = [
        "ASN ",
        "ASB ",
        "ASM ",
        "AMGLOBAL ",
        "AMDYNAMIC ",
        "AMCIO SERIES",
    ]
    return any(compact.startswith(prefix) for prefix in fund_prefixes)


def _infer_uie_review_status(row: pd.Series) -> str:
    existing = row.get("review_status")
    if pd.notna(existing) and str(existing).strip():
        return str(existing)

    if str(row.get("uie_source") or "") == "manual_verified":
        return "verified"
    entity_type = str(row.get("entity_type") or "")
    if entity_type == "fund_or_product_vehicle":
        return "product_vehicle_rule_applied"
    if entity_type == "nominee_or_custodian":
        return "nominee_or_custodian_rule_applied"
    return "todo"


def build_gleif_search_benchmark(
    review_targets: pd.DataFrame,
    search_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine UIE review targets with GLEIF search candidates."""
    columns = [
        "lei", "legal_name", "review_priority_score", "review_status",
        "entity_type", "current_uie_country", "current_uie_source",
        "gleif_candidate_lei", "gleif_candidate_name", "gleif_candidate_country",
        "gleif_candidate_hq_country", "gleif_candidate_rank",
        "gleif_total_results", "gleif_search_query", "gleif_match_method",
        "review_flag",
    ]
    if review_targets.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    for _, target in review_targets.iterrows():
        query = str(target.get("legal_name") or "").strip()
        candidates = search_results.get(query, pd.DataFrame())
        if candidates.empty:
            rows.append(_gleif_benchmark_row(target, None, query, "no_candidate"))
            continue
        for _, candidate in candidates.iterrows():
            rows.append(_gleif_benchmark_row(target, candidate, query, None))
    return pd.DataFrame(rows, columns=columns)


def _gleif_benchmark_row(
    target: pd.Series,
    candidate: pd.Series | None,
    query: str,
    forced_flag: str | None,
) -> dict:
    if candidate is None:
        return {
            "lei": target.get("lei"),
            "legal_name": target.get("legal_name"),
            "review_priority_score": target.get("review_priority_score"),
            "review_status": target.get("review_status"),
            "entity_type": target.get("entity_type"),
            "current_uie_country": target.get("uie_country"),
            "current_uie_source": target.get("uie_source"),
            "gleif_candidate_lei": None,
            "gleif_candidate_name": None,
            "gleif_candidate_country": None,
            "gleif_candidate_hq_country": None,
            "gleif_candidate_rank": pd.NA,
            "gleif_total_results": 0,
            "gleif_search_query": query,
            "gleif_match_method": "fulltext",
            "review_flag": forced_flag or "no_candidate",
        }

    target_lei = str(target.get("lei") or "")
    candidate_lei = str(candidate.get("lei") or "")
    candidate_country = candidate.get("country_legal")
    current_uie = target.get("uie_country")
    if candidate_lei and candidate_lei == target_lei:
        flag = "self_entity_found"
    elif pd.notna(current_uie) and pd.notna(candidate_country) and current_uie != candidate_country:
        flag = "candidate_country_differs_from_current_uie"
    else:
        flag = "candidate_for_review"

    return {
        "lei": target.get("lei"),
        "legal_name": target.get("legal_name"),
        "review_priority_score": target.get("review_priority_score"),
        "review_status": target.get("review_status"),
        "entity_type": target.get("entity_type"),
        "current_uie_country": current_uie,
        "current_uie_source": target.get("uie_source"),
        "gleif_candidate_lei": candidate.get("lei"),
        "gleif_candidate_name": candidate.get("legal_name"),
        "gleif_candidate_country": candidate_country,
        "gleif_candidate_hq_country": candidate.get("country_hq"),
        "gleif_candidate_rank": candidate.get("rank"),
        "gleif_total_results": candidate.get("total_results"),
        "gleif_search_query": query,
        "gleif_match_method": "fulltext",
        "review_flag": flag,
    }


def run_entity_analysis(
    country: str = "MY",
    max_pages: int = 50,
    page_size: int = 200,
    skip_pull: bool = False,
) -> dict:
    """Fetch entities for a country and run full structural analysis."""
    country = country.upper()
    cache_path = RAW_DIR / f"gleif_{country.lower()}_lei"
    # Backwards compatibility: MY data was previously saved as gleif_malaysia_lei
    legacy_path = RAW_DIR / "gleif_malaysia_lei" if country == "MY" else None

    if skip_pull:
        try:
            entities = normalize_entity_records(load_df(cache_path))
        except FileNotFoundError:
            if legacy_path:
                entities = normalize_entity_records(load_df(legacy_path))
            else:
                raise
        print(f"[INFO] Loaded {len(entities):,} cached entities for {country}")
    else:
        print(f"[INFO] Fetching LEI records for country={country}...")
        entities = normalize_entity_records(
            fetch_country_lei_records(country, max_pages=max_pages, page_size=page_size)
        )
        save_df(entities, cache_path)
        print(f"[INFO] Fetched {len(entities):,} entities for {country}")

    report = full_structural_report(entities, country)

    # Save analysis artifacts
    analysis_dir = PROCESSED_DIR / "entity_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for key in ["geographic_concentration", "legal_form_distribution",
                "category_distribution", "entity_status_distribution",
                "registration_timeline", "hq_vs_legal_mismatch"]:
        df = report[key]
        if isinstance(df, pd.DataFrame) and not df.empty:
            save_csv(df, analysis_dir / f"{country.lower()}_{key}.csv")

    return report


def run_comparative_analysis(
    countries: list[str],
    max_pages: int = 50,
    page_size: int = 200,
    skip_pull: bool = False,
) -> pd.DataFrame:
    """Fetch entities for multiple countries and compare structural profiles."""
    country_frames: dict[str, pd.DataFrame] = {}

    for country in countries:
        country = country.upper()
        cache_path = RAW_DIR / f"gleif_{country.lower()}_lei"

        if skip_pull:
            try:
                entities = normalize_entity_records(load_df(cache_path))
                print(f"[INFO] Loaded {len(entities):,} cached entities for {country}")
            except Exception:
                print(f"[WARN] No cached data for {country}, fetching...")
                entities = normalize_entity_records(
                    fetch_country_lei_records(country, max_pages=max_pages, page_size=page_size)
                )
                save_df(entities, cache_path)
                print(f"[INFO] Fetched {len(entities):,} entities for {country}")
        else:
            print(f"[INFO] Fetching LEI records for country={country}...")
            entities = normalize_entity_records(
                fetch_country_lei_records(country, max_pages=max_pages, page_size=page_size)
            )
            save_df(entities, cache_path)
            print(f"[INFO] Fetched {len(entities):,} entities for {country}")

        country_frames[country] = entities

    comparison = compare_countries(country_frames)

    analysis_dir = PROCESSED_DIR / "entity_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    save_csv(comparison, analysis_dir / "cross_country_comparison.csv")

    return comparison


def run_ctos_malaysia_enrichment(
    *,
    letters: list[str] | None = None,
    pages_per_letter: int = 1,
    fetch_details_limit: int = 0,
    match_threshold: int = 95,
    sleep_s: float = 1.0,
    timeout: int = 60,
    max_retries: int = 3,
    max_consecutive_failures: int = 5,
    skip_pull: bool = False,
    source_parquet: str | Path | None = None,
    use_duckdb: bool = True,
    fuzzy: bool = True,
) -> pd.DataFrame:
    """Fetch/match CTOS Malaysia directory entries against Malaysia LEIs.

    CTOS is used only as an incorporation/registration signal. It does not
    determine ownership nationality or UIE.
    """
    paths = country_paths("MY")
    raw_path = RAW_DIR / "ctos_my_companies"
    match_path = PROCESSED_DIR / "my" / "ctos_entity_matches"

    entities = normalize_entity_records(_load_with_fallback(paths, "raw_entities"))

    if source_parquet is not None:
        ctos_companies = load_ctos_company_snapshot(source_parquet, use_duckdb=use_duckdb)
        save_df(ctos_companies, raw_path)
        save_csv(ctos_companies, RAW_DIR / "ctos_my_companies.csv")
    elif skip_pull:
        ctos_companies = load_df(raw_path)
    else:
        ctos_companies = fetch_ctos_company_directory(
            letters=letters,
            pages_per_letter=pages_per_letter,
            fetch_details_limit=fetch_details_limit,
            sleep_s=sleep_s,
            checkpoint_path=raw_path.with_suffix(".parquet"),
            resume=True,
            timeout=timeout,
            max_retries=max_retries,
            max_consecutive_failures=max_consecutive_failures,
        )
        save_df(ctos_companies, raw_path)
        save_csv(ctos_companies, RAW_DIR / "ctos_my_companies.csv")

    matches = match_entities_to_ctos(
        entities,
        ctos_companies,
        threshold=match_threshold,
        fuzzy=fuzzy,
    )
    save_df(matches, match_path)
    save_csv(matches, PROCESSED_DIR / "my" / "ctos_entity_matches.csv")

    print(f"[INFO] CTOS companies loaded: {len(ctos_companies):,}")
    print(f"[INFO] Malaysia LEIs matched to CTOS: {len(matches):,}")
    if not matches.empty:
        print(matches.head(20).to_string(index=False))
    return matches


def run_gleif_search_benchmark(
    *,
    country: str = "MY",
    limit: int = 50,
    page_size: int = 5,
) -> pd.DataFrame:
    """Benchmark GLEIF full-text search against top UIE review targets."""
    country = country.upper()
    out_dir = PROCESSED_DIR / country.lower()
    review_path = out_dir / "uie_review_targets"
    try:
        review_targets = load_df(review_path)
    except FileNotFoundError:
        review_targets = pd.read_csv(out_dir / "uie_review_targets.csv")

    if review_targets.empty:
        benchmark = build_gleif_search_benchmark(review_targets, {})
        save_df(benchmark, out_dir / "entity_match_benchmark")
        save_csv(benchmark, out_dir / "entity_match_benchmark.csv")
        return benchmark

    review_targets = review_targets.head(limit).copy()
    search_results: dict[str, pd.DataFrame] = {}
    for i, row in review_targets.iterrows():
        query = str(row.get("legal_name") or "").strip()
        if not query:
            continue
        try:
            result = search_lei_records(query, page_size=page_size)
        except Exception as exc:
            print(f"[WARN] GLEIF search failed for {query}: {exc}", flush=True)
            result = pd.DataFrame()
        search_results[query] = result
        print(f"[{len(search_results)}/{len(review_targets)}] GLEIF search: {query}", flush=True)

    benchmark = build_gleif_search_benchmark(review_targets, search_results)
    save_df(benchmark, out_dir / "entity_match_benchmark")
    save_csv(benchmark, out_dir / "entity_match_benchmark.csv")
    print(f"[INFO] GLEIF search benchmark rows: {len(benchmark):,}", flush=True)
    return benchmark


def run_gleif_pull(
    max_relationship_entities: int = 100,
    lei_max_pages: int = 50,
    lei_page_size: int = 200,
    scan_all: bool = False,
    country: str = "MY",
    rel_checkpoint_every: int = 500,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch entities and scan parent relationships for *country*.

    Skips entity re-fetch if a non-empty cached file already exists unless
    *force_refresh* is set.
    Checkpoints the relationship scan every *rel_checkpoint_every* entities
    so a crash resumes rather than restarts from zero.
    """
    paths = country_paths(country)
    entities_path = Path(str(paths["raw_entities"]) + ".parquet")

    # --- Entity fetch (skip if already cached) ---
    if _should_use_cached_file(entities_path, force_refresh=force_refresh):
        print(f"[INFO] [{country}] Entity file already exists ({entities_path.name}), skipping re-fetch.", flush=True)
        entities = normalize_entity_records(pd.read_parquet(entities_path))
    else:
        if force_refresh and entities_path.exists():
            print(f"[INFO] [{country}] Force refresh enabled; re-fetching entity records.", flush=True)
        entities = normalize_entity_records(
            fetch_country_lei_records(country, max_pages=lei_max_pages, page_size=lei_page_size)
        )
        save_df(entities, paths["raw_entities"])

    leis = entities["lei"].dropna().unique()
    if not scan_all:
        leis = leis[:max_relationship_entities]

    # --- Relationship scan with mid-process checkpointing ---
    wip_path = Path(str(paths["interim_relationships"]) + "_wip.parquet")
    scanned_path = Path(str(paths["interim_relationships"]) + "_scanned.parquet")
    rels: list[pd.DataFrame] = []
    scanned_leis: set[str] = set()

    if force_refresh:
        for checkpoint_path in [wip_path, scanned_path]:
            if checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                    print(f"[INFO] [{country}] Removed old relationship checkpoint: {checkpoint_path.name}", flush=True)
                except Exception as e:
                    print(f"[WARN] [{country}] Could not remove relationship checkpoint {checkpoint_path.name}: {e}", flush=True)

    # Resume from checkpoint if available
    if _should_use_cached_file(wip_path, force_refresh=force_refresh):
        try:
            wip_df = pd.read_parquet(wip_path)
            if not wip_df.empty:
                rels.append(wip_df)
                scanned_leis = set(wip_df["source_lei"].dropna().astype(str).unique())
                print(f"[INFO] [{country}] Loaded relationship checkpoint: "
                      f"{len(wip_df)} relationships cached", flush=True)
        except Exception as e:
            print(f"[WARN] [{country}] Could not load relationship checkpoint: {e}", flush=True)
    if _should_use_cached_file(scanned_path, force_refresh=force_refresh):
        try:
            scanned_df = pd.read_parquet(scanned_path)
            scanned_leis.update(scanned_df["lei"].dropna().astype(str).unique())
            print(f"[INFO] [{country}] Loaded scanned-LEI checkpoint: "
                  f"{len(scanned_leis)} LEIs already scanned", flush=True)
        except Exception as e:
            print(f"[WARN] [{country}] Could not load scanned-LEI checkpoint: {e}", flush=True)
    elif scanned_leis:
        print(f"[INFO] [{country}] Resuming from legacy relationship checkpoint: "
              f"{len(scanned_leis)} LEIs with parent rows already scanned", flush=True)

    remaining = [lei for lei in leis if str(lei) not in scanned_leis]
    total = len(leis)
    found = sum(1 for df in rels for _ in [df]) if rels else 0  # count entities with parents
    # recount properly
    found = len(set(pd.concat(rels)["source_lei"].dropna()) if rels else [])

    print(f"[INFO] [{country}] Scanning {len(remaining):,} remaining entities "
          f"({len(scanned_leis):,} already done, {total:,} total)...", flush=True)

    batch_rels: list[pd.DataFrame] = []

    for i, lei in enumerate(remaining):
        try:
            df = fetch_relationships_for_lei(lei)
            if not df.empty:
                batch_rels.append(df)
                rels.append(df)
                found += 1
                scanned_leis.add(str(lei))
                overall = len(scanned_leis)
                print(f"[{overall}/{total}] {lei} - parent found ({found} total)", flush=True)
            else:
                scanned_leis.add(str(lei))
                overall = len(scanned_leis)
                if overall % 200 == 0:
                    print(f"[{overall}/{total}] scanning... {found} with parents so far", flush=True)
        except Exception as e:
            print(f"[WARN] relationship fetch failed for {lei}: {e}")

        # Save checkpoint every rel_checkpoint_every entities
        if (i + 1) % rel_checkpoint_every == 0:
            if rels:
                _flush_rel_checkpoint(rels, wip_path)
            _flush_scanned_lei_checkpoint(scanned_leis, scanned_path)

    print(f"[INFO] Relationship scan complete: {found}/{total} entities have parent data", flush=True)

    relationships = normalize_relationship_records(
        pd.concat(rels, ignore_index=True) if rels else pd.DataFrame()
    )
    save_df(relationships, paths["interim_relationships"])

    # Clean up WIP checkpoint now that final file is saved
    for checkpoint_path in [wip_path, scanned_path]:
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
            except Exception:
                pass

    target_leis = relationships["target_lei"].dropna().unique()
    source_leis = set(entities["lei"].dropna().astype(str))
    related_leis = [lei for lei in target_leis if str(lei) not in source_leis]
    related_entities = fetch_lei_records_by_lei(related_leis)
    save_df(related_entities, paths["raw_related"])

    return entities, relationships


def _flush_rel_checkpoint(rels: list[pd.DataFrame], path) -> None:
    """Write accumulated relationship DataFrames to a WIP checkpoint file."""
    from pathlib import Path
    try:
        combined = normalize_relationship_records(
            pd.concat(rels, ignore_index=True).drop_duplicates(
                subset=["source_lei", "target_lei", "relationship_type"]
            )
        )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(p, index=False)
        print(f"[INFO] Relationship checkpoint saved: {len(combined)} rows -> {p.name}", flush=True)
    except Exception as e:
        print(f"[WARN] Relationship checkpoint save failed: {e}", flush=True)


def _flush_scanned_lei_checkpoint(scanned_leis: set[str], path) -> None:
    """Write scanned relationship LEIs, including no-parent misses, to a checkpoint."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"lei": sorted(scanned_leis)}).to_parquet(p, index=False)
        print(f"[INFO] Relationship scanned-LEI checkpoint saved: {len(scanned_leis)} rows -> {p.name}", flush=True)
    except Exception as e:
        print(f"[WARN] Relationship scanned-LEI checkpoint save failed: {e}", flush=True)


def run_graph_and_scoring(enrich_edgar: bool = True, country: str = "MY") -> pd.DataFrame:
    paths = country_paths(country)
    inf_dir = paths["inference_dir"]
    inf_dir.mkdir(parents=True, exist_ok=True)

    domestic_entities = normalize_entity_records(_load_with_fallback(paths, "raw_entities"))
    related_entities = normalize_entity_records(
        _load_with_fallback(paths, "raw_related", columns=ENTITY_COLUMNS)
    )
    entities = _combine_entity_sets(domestic_entities, related_entities)
    relationships = normalize_relationship_records(
        _load_with_fallback(paths, "interim_relationships", columns=RELATIONSHIP_COLUMNS)
    )
    chain_relationships, chain_entities = _load_uie_chain_context(country)
    uie_entities = _combine_entity_sets(entities, chain_entities)

    g = build_di_graph(entities, relationships)
    summary = graph_summary(g)
    summary = add_graph_features(g, summary)

    flags = parent_flags(relationships)
    summary = summary.merge(flags, on="lei", how="left")
    summary["has_parent_link"] = _as_int_flag(summary["has_parent_link"])
    summary["has_ultimate_link"] = _as_int_flag(summary["has_ultimate_link"])

    summary = _add_size_proxy(summary)

    if enrich_edgar:
        try:
            tickers = fetch_company_tickers()
            summary = match_entities_to_edgar(summary, tickers)
        except Exception as e:
            print(f"[WARN] EDGAR enrichment failed: {e}")
            summary["appears_in_edgar"] = 0
            summary["cik"] = None
            summary["ticker"] = None
            summary["title"] = None
    else:
        summary["appears_in_edgar"] = 0
        summary["cik"] = None
        summary["ticker"] = None
        summary["title"] = None

    summary = summary.merge(
        _foreign_parent_flags(relationships, entities, country=country), on="lei", how="left"
    )
    summary["foreign_parent"] = _as_int_flag(summary["foreign_parent"])

    # --- Merge inference signals (if available) ---

    # Phase 0.5: reporting exception flags
    exceptions = pd.DataFrame()
    try:
        exceptions = _load_with_fallback(paths, "interim_exceptions")
        summary = enrich_with_exception_flags(summary, exceptions)
        nc_count = int(summary["is_non_consolidating"].sum())
        print(f"[INFO] Phase 0.5 enrichment: {nc_count} non-consolidating entities flagged")
    except FileNotFoundError:
        summary["is_non_consolidating"] = 0
        summary["has_exception_filed"] = 0

    # Phase 1: name-inferred parent flag
    try:
        phase1 = _load_with_fallback(paths, "inference_dir")  # won't work as-is, use direct
    except (FileNotFoundError, Exception):
        phase1 = pd.DataFrame()
    # Try inference dir for phase1 name matches
    for p1_path in [inf_dir / "phase1_name_matches", paths.get("legacy", {}).get("inference_dir", inf_dir) / "phase1_name_matches"]:
        try:
            phase1 = load_df(p1_path)
            break
        except FileNotFoundError:
            continue
    if not phase1.empty and "lei" in phase1.columns:
        inferred_leis = set(phase1["lei"].dropna().unique())
        summary["has_inferred_parent"] = summary["lei"].isin(inferred_leis).astype(int)
        print(f"[INFO] Phase 1 enrichment: {len(inferred_leis)} name-inferred parents merged")
    else:
        summary["has_inferred_parent"] = 0

    # Phase 2: address cluster flag
    phase2 = pd.DataFrame()
    for p2_path in [inf_dir / "phase2_address_clusters", paths.get("legacy", {}).get("inference_dir", inf_dir) / "phase2_address_clusters"]:
        try:
            phase2 = load_df(p2_path)
            break
        except FileNotFoundError:
            continue
    if not phase2.empty and "lei" in phase2.columns:
        clustered_leis = set(phase2["lei"].dropna().unique())
        summary["in_address_cluster"] = summary["lei"].isin(clustered_leis).astype(int)
        print(f"[INFO] Phase 2 enrichment: {len(clustered_leis)} address-clustered entities merged")
    else:
        summary["in_address_cluster"] = 0

    # Phase 2 inferred parent edges, used for UIE assignment rather than scoring.
    phase2_inferred = pd.DataFrame()
    for p2_edge_path in [
        inf_dir / "phase2_inferred_edges",
        paths.get("legacy", {}).get("inference_dir", inf_dir) / "phase2_inferred_edges",
    ]:
        try:
            phase2_inferred = load_df(p2_edge_path)
            break
        except FileNotFoundError:
            continue

    # Phase 3 predicted parent jurisdiction, used as a low-tier UIE fallback.
    phase3 = pd.DataFrame()
    for p3_path in [
        inf_dir / "phase3_jurisdiction_predictions",
        paths.get("legacy", {}).get("inference_dir", inf_dir) / "phase3_jurisdiction_predictions",
    ]:
        try:
            phase3 = load_df(p3_path)
            break
        except FileNotFoundError:
            continue

    # Malaysia-only CTOS incorporation/SSM existence signal, if available.
    ctos_matches = pd.DataFrame()
    if country.upper() == "MY":
        try:
            ctos_matches = load_df(PROCESSED_DIR / "my" / "ctos_entity_matches")
        except FileNotFoundError:
            ctos_matches = pd.DataFrame()
    manual_overrides = load_uie_overrides(country)
    review_statuses = load_uie_review_statuses(country)

    scored = compute_coverage_score(summary)

    out_dir = PROCESSED_DIR / country.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    uie_assignments = build_uie_assignments(
        domestic_entities,
        relationships,
        uie_entities,
        country=country,
        chain_relationships=chain_relationships,
        exceptions=exceptions,
        name_matches=phase1,
        address_inferred=phase2_inferred,
        jurisdiction_predictions=phase3,
        ctos_matches=ctos_matches,
        manual_overrides=manual_overrides,
    )
    save_df(uie_assignments, out_dir / "uie_assignments")
    save_csv(uie_assignments, out_dir / "uie_assignments.csv")
    save_uie_assignment_snapshot(uie_assignments, country)

    assigned_uie = uie_assignments[uie_assignments["uie_country"].notna()].copy()
    if assigned_uie.empty:
        investor_summary = pd.DataFrame(columns=[
            "host_country", "uie_country", "total_entities",
            "known_uie_entities", "inferred_uie_entities", "avg_confidence",
        ])
    else:
        investor_summary = (
            assigned_uie.groupby(["host_country", "uie_country"], dropna=False)
            .agg(
                total_entities=("lei", "count"),
                known_uie_entities=("is_known_uie", "sum"),
                inferred_uie_entities=("is_inferred_uie", "sum"),
                avg_confidence=("uie_confidence", "mean"),
            )
            .reset_index()
            .sort_values(["total_entities", "uie_country"], ascending=[False, True])
        )
        investor_summary["avg_confidence"] = investor_summary["avg_confidence"].round(4)
    save_df(investor_summary, out_dir / "investor_economy_summary")
    save_csv(investor_summary, out_dir / "investor_economy_summary.csv")

    all_uie_review_targets = build_uie_review_targets(
        scored,
        uie_assignments,
        review_statuses=review_statuses,
        limit=max(len(scored), 500),
    )
    uie_product_vehicle_targets = all_uie_review_targets[
        all_uie_review_targets["entity_type"].eq("fund_or_product_vehicle")
    ].reset_index(drop=True)
    uie_review_targets = all_uie_review_targets[
        ~all_uie_review_targets["entity_type"].eq("fund_or_product_vehicle")
    ].head(500).reset_index(drop=True)
    save_df(uie_review_targets, out_dir / "uie_review_targets")
    save_csv(uie_review_targets, out_dir / "uie_review_targets.csv")
    save_df(uie_product_vehicle_targets, out_dir / "uie_product_vehicle_targets")
    save_csv(uie_product_vehicle_targets, out_dir / "uie_product_vehicle_targets.csv")

    save_df(summary, out_dir / "di_graph_summary")
    save_df(scored, out_dir / "coverage_gap_scores")
    save_csv(scored.head(200), out_dir / "coverage_gap_scores_top200.csv")

    return scored


def run_mock_pipeline() -> pd.DataFrame:
    entities = normalize_entity_records(pd.DataFrame([
        {"lei": "A1", "legal_name": "MALAYSIA HOLDINGS BERHAD", "country_legal": "MY", "city_legal": "KUALA LUMPUR", "category": "GENERAL", "legal_form": "BHD"},
        {"lei": "A2", "legal_name": "ASEAN ENERGY SDN BHD", "country_legal": "MY", "city_legal": "KUALA LUMPUR", "category": "GENERAL", "legal_form": "SDN"},
        {"lei": "P1", "legal_name": "SINGAPORE PARENT LTD", "country_legal": "SG", "city_legal": "SINGAPORE", "category": "GENERAL", "legal_form": "LTD"},
        {"lei": "P2", "legal_name": "GLOBAL ULTIMATE INC", "country_legal": "US", "city_legal": "NEW YORK", "category": "GENERAL", "legal_form": "INC"},
        {"lei": "A3", "legal_name": "PALM EXPORTS BERHAD", "country_legal": "MY", "city_legal": "SHAH ALAM", "category": "GENERAL", "legal_form": "BHD"},
    ]))
    relationships = normalize_relationship_records(pd.DataFrame([
        {"source_lei": "A1", "target_lei": "P1", "relationship_type": "direct_parent", "relationship_status": "ACTIVE", "accounting_standard": "IFRS", "period_end": None},
        {"source_lei": "A1", "target_lei": "P2", "relationship_type": "ultimate_parent", "relationship_status": "ACTIVE", "accounting_standard": "IFRS", "period_end": None},
        {"source_lei": "A2", "target_lei": "P1", "relationship_type": "direct_parent", "relationship_status": "ACTIVE", "accounting_standard": "IFRS", "period_end": None},
    ]))

    save_df(entities, RAW_DIR / "gleif_malaysia_lei")
    save_df(pd.DataFrame(columns=ENTITY_COLUMNS), RAW_DIR / "gleif_related_lei")
    save_df(relationships, INTERIM_DIR / "gleif_malaysia_relationships")

    g = build_di_graph(entities, relationships)
    summary = graph_summary(g)
    summary = add_graph_features(g, summary)

    flags = parent_flags(relationships)
    summary = summary.merge(flags, on="lei", how="left")
    summary["has_parent_link"] = _as_int_flag(summary["has_parent_link"])
    summary["has_ultimate_link"] = _as_int_flag(summary["has_ultimate_link"])
    summary["appears_in_edgar"] = summary["legal_name"].eq("GLOBAL ULTIMATE INC").astype(int)
    summary = summary.merge(_foreign_parent_flags(relationships, entities), on="lei", how="left")
    summary["foreign_parent"] = _as_int_flag(summary["foreign_parent"])
    summary = _add_size_proxy(summary)

    scored = compute_coverage_score(summary)
    save_df(summary, PROCESSED_DIR / "di_graph_summary")
    save_df(scored, PROCESSED_DIR / "coverage_gap_scores")
    save_csv(scored.head(200), PROCESSED_DIR / "coverage_gap_scores_top200.csv")

    return scored


# ---------------------------------------------------------------------------
# Inference pipeline — phases 0.5, 1, (2, 3 to follow)
# ---------------------------------------------------------------------------

def run_reporting_exceptions(
    skip_pull: bool = False,
    country: str = "MY",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Phase 0.5: Fetch reporting exceptions for entities without known parents."""
    paths = country_paths(country)
    cache_path = paths["interim_exceptions"]

    if skip_pull and not force_refresh:
        try:
            exceptions = _load_with_fallback(paths, "interim_exceptions")
            print(f"[INFO] Loaded {len(exceptions):,} cached reporting exceptions")
            return exceptions
        except FileNotFoundError:
            print("[WARN] No cached exceptions, fetching from API...")

    entities = normalize_entity_records(_load_with_fallback(paths, "raw_entities"))
    relationships = normalize_relationship_records(
        _load_with_fallback(paths, "interim_relationships", columns=RELATIONSHIP_COLUMNS)
    )

    # Only scan entities without known parent relationships
    known_leis = set(relationships["source_lei"].dropna().unique()) if not relationships.empty else set()
    leis_to_scan = [lei for lei in entities["lei"].dropna().unique() if lei not in known_leis]

    print(f"[INFO] Scanning {len(leis_to_scan):,} entities for reporting exceptions...")
    cache_pq = cache_path.with_suffix(".parquet")
    scanned_pq = cache_path.with_name(cache_path.name + "_scanned").with_suffix(".parquet")
    if force_refresh:
        for checkpoint_path in [cache_pq, scanned_pq]:
            if checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                    print(f"[INFO] Removed cached reporting-exception checkpoint: {checkpoint_path.name}")
                except Exception as e:
                    print(f"[WARN] Could not remove reporting-exception checkpoint {checkpoint_path.name}: {e}")
    exceptions = fetch_reporting_exceptions(
        leis_to_scan,
        checkpoint_path=cache_pq,
        scanned_path=scanned_pq,
    )
    save_df(exceptions, cache_path)

    # Summary
    if not exceptions.empty:
        reason_counts = exceptions["exception_reason"].value_counts()
        print(f"\n--- Reporting Exception Summary ---")
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}")
    else:
        print("[INFO] No reporting exceptions found")

    return exceptions


def run_phase1_name_inference(
    threshold: int = 90,
    skip_pull: bool = False,
    country: str = "MY",
) -> pd.DataFrame:
    """Phase 1: Name pattern extraction and fuzzy matching."""
    paths = country_paths(country)
    inf_dir = paths["inference_dir"]
    inf_dir.mkdir(parents=True, exist_ok=True)

    # Load required data
    entities = normalize_entity_records(_load_with_fallback(paths, "raw_entities"))
    parents = normalize_entity_records(
        _load_with_fallback(paths, "raw_related", columns=ENTITY_COLUMNS)
    )
    relationships = normalize_relationship_records(
        _load_with_fallback(paths, "interim_relationships", columns=RELATIONSHIP_COLUMNS)
    )

    if parents.empty:
        print("[WARN] No parent entities loaded. Run the full pipeline first.")
        return pd.DataFrame()

    # Extract brand tokens from known parents
    brand_tokens = extract_parent_brand_tokens(parents, country=country)
    print(f"[INFO] Extracted {len(brand_tokens)} brand tokens from {len(parents)} parent entities")

    if brand_tokens.empty:
        print("[WARN] No valid brand tokens extracted")
        return pd.DataFrame()

    # Get known LEIs to exclude
    known_leis = set(relationships["source_lei"].dropna().unique()) if not relationships.empty else set()

    # Run fuzzy matching
    print(f"[INFO] Fuzzy matching {len(entities):,} entities against {len(brand_tokens)} brand tokens (threshold={threshold})...")
    matches = fuzzy_match_names(entities, brand_tokens, known_leis=known_leis, threshold=threshold, country=country)

    # Build inferred edges
    edges = build_phase1_inferred_edges(matches, min_score=threshold)

    # Save artifacts
    save_df(matches, inf_dir / "phase1_name_matches")
    save_df(edges, inf_dir / "phase1_inferred_edges")

    if not matches.empty:
        save_csv(matches, inf_dir / "phase1_name_matches.csv")

    # Print summary
    summary = phase1_summary(matches)
    print(f"\n--- Phase 1: Name Inference Summary ---")
    print(f"  Total matches:      {summary['total_matches']}")
    print(f"  High confidence:    {summary.get('high_confidence', 0)} (score >= 90)")
    print(f"  Medium confidence:  {summary.get('medium_confidence', 0)} (score 80-89)")
    print(f"  Unique brands:      {summary.get('unique_parent_brands', 0)}")
    if summary['total_matches'] > 0:
        print(f"  Average score:      {summary.get('avg_score', 0)}")

    return matches


def run_phase2_address_clustering(
    skip_pull: bool = False,
    min_cluster: int = 3,
    country: str = "MY",
) -> pd.DataFrame:
    """Phase 2: Address parsing and geospatial clustering."""
    paths = country_paths(country)
    inf_dir = paths["inference_dir"]
    inf_dir.mkdir(parents=True, exist_ok=True)
    address_cache = paths["interim_addresses"]

    # Load or fetch addresses
    if skip_pull:
        try:
            addresses = _load_with_fallback(paths, "interim_addresses")
            print(f"[INFO] Loaded {len(addresses):,} cached addresses")
        except FileNotFoundError:
            print("[WARN] No cached addresses, fetching from API...")
            skip_pull = False

    if not skip_pull:
        entities = normalize_entity_records(_load_with_fallback(paths, "raw_entities"))
        leis = entities["lei"].dropna().unique().tolist()
        print(f"[INFO] Fetching full addresses for {len(leis):,} entities...")
        addresses = fetch_full_addresses(leis, checkpoint_path=address_cache.with_suffix(".parquet"))
        save_df(addresses, address_cache)

    # Cluster by address
    print(f"[INFO] Clustering addresses (min_cluster_size={min_cluster})...")
    clusters = cluster_by_address(addresses, min_cluster_size=min_cluster, country=country)

    # Infer shared parents within clusters
    relationships = normalize_relationship_records(
        _load_with_fallback(paths, "interim_relationships", columns=RELATIONSHIP_COLUMNS)
    )
    inferred = infer_shared_parent_from_cluster(clusters, relationships)

    # Save artifacts
    save_df(clusters, inf_dir / "phase2_address_clusters")
    save_df(inferred, inf_dir / "phase2_inferred_edges")

    if not clusters.empty:
        save_csv(clusters, inf_dir / "phase2_address_clusters.csv")

    # Print summary
    summary = phase2_summary(clusters)
    print(f"\n--- Phase 2: Address Clustering Summary ---")
    print(f"  Clustered entities:  {summary['total_clustered_entities']}")
    print(f"  Number of clusters:  {summary['num_clusters']}")
    if summary['total_clustered_entities'] > 0:
        print(f"  Largest cluster:     {summary.get('largest_cluster', 0)}")
        print(f"  Avg cluster size:    {summary.get('avg_cluster_size', 0)}")
        print(f"  Office-hotel entities: {summary.get('office_hotel_entities', 0)}")
    print(f"  Inferred parent links: {len(inferred)}")

    return clusters


def run_phase3_jurisdiction_prediction(skip_pull: bool = False, country: str = "MY") -> pd.DataFrame:
    """Phase 3: Train jurisdiction predictor and predict parent countries."""
    paths = country_paths(country)
    inf_dir = paths["inference_dir"]
    inf_dir.mkdir(parents=True, exist_ok=True)

    # Load all required data
    entities = normalize_entity_records(_load_with_fallback(paths, "raw_entities"))
    parents = normalize_entity_records(
        _load_with_fallback(paths, "raw_related", columns=ENTITY_COLUMNS)
    )
    relationships = normalize_relationship_records(
        _load_with_fallback(paths, "interim_relationships", columns=RELATIONSHIP_COLUMNS)
    )

    # Optional enrichments
    out_dir = PROCESSED_DIR / country.lower()
    try:
        graph_summary_df = load_df(out_dir / "di_graph_summary")
    except FileNotFoundError:
        graph_summary_df = None

    try:
        exceptions = _load_with_fallback(paths, "interim_exceptions")
    except FileNotFoundError:
        exceptions = None

    # Phase 1 pseudo-labels: high-confidence name matches as extra training signal
    try:
        pseudo_labels = load_df(inf_dir / "phase1_name_matches")
    except FileNotFoundError:
        pseudo_labels = None

    if relationships.empty or parents.empty:
        print("[WARN] No relationship or parent data available. Cannot train model.")
        return pd.DataFrame()

    # Prepare training data
    print("[INFO] Preparing training features...", flush=True)
    X_train, y_train, label_encoder = prepare_training_features(
        entities, relationships, parents, graph_summary_df, exceptions,
        pseudo_labels_df=pseudo_labels, country=country
    )

    if X_train.empty:
        print("[WARN] No training data could be prepared")
        return pd.DataFrame()

    print(f"[INFO] Training set: {len(X_train)} samples, {y_train.nunique()} classes", flush=True)
    print(f"[INFO] Classes: {list(label_encoder.classes_)}", flush=True)

    # Train model
    model, cv_metrics = train_jurisdiction_model(X_train, y_train)

    # Predict on ALL entities (including labeled, for validation)
    from src.jurisdiction_predictor import _build_features
    all_features = _build_features(entities["lei"], entities, graph_summary_df, exceptions, country=country)

    # Align columns with training set
    for col in X_train.columns:
        if col not in all_features.columns:
            all_features[col] = 0
    all_features = all_features[["lei"] + list(X_train.columns)]

    leis = all_features["lei"]
    X_all = all_features.drop(columns=["lei"])

    print(f"[INFO] Predicting jurisdiction for {len(X_all):,} entities...", flush=True)
    predictions = predict_parent_jurisdiction(model, label_encoder, X_all, leis)

    # Save artifacts
    save_model_artifacts(model, cv_metrics, list(X_train.columns), label_encoder, inf_dir)
    save_df(predictions, inf_dir / "phase3_jurisdiction_predictions")
    save_csv(predictions, inf_dir / "phase3_jurisdiction_predictions.csv")

    # Print summary
    summary = phase3_summary(predictions, cv_metrics)
    print(f"\n--- Phase 3: Jurisdiction Prediction Summary ---")
    print(f"  Model type:          {summary.get('model_type', 'N/A')}")
    print(f"  CV accuracy:         {summary.get('cv_accuracy_mean', 0):.3f} +/- {summary.get('cv_accuracy_std', 0):.3f}")
    if "cv_top3_accuracy_mean" in summary:
        print(f"  Top-{summary.get('cv_top_k', 3)} CV accuracy: {summary.get('cv_top3_accuracy_mean', 0):.3f}")
    if "majority_class_baseline_accuracy" in summary:
        print(f"  Majority baseline:   {summary.get('majority_class_baseline_accuracy', 0):.3f}")
    print(f"  Total predictions:   {summary.get('total_predictions', 0)}")
    print(f"  Top predicted:       {summary.get('top_predicted_country', 'N/A')}")
    print(f"  Avg model probability: {summary.get('avg_confidence', 0):.3f}")
    print(
        f"  High-probability:    "
        f"{summary.get('uncalibrated_high_probability_predictions', 0)} "
        f"(uncalibrated prob >= 0.5)"
    )

    # Show prediction distribution
    if not predictions.empty:
        print(f"\n  Predicted parent jurisdiction distribution:")
        dist = predictions["predicted_parent_country"].value_counts().head(10)
        for country, count in dist.items():
            print(f"    {country}: {count}")

    return predictions


def run_phase4_graph_prediction(skip_pull: bool = False, country: str = "MY") -> dict:
    """Phase 4: Graph-based link prediction and label propagation."""
    paths = country_paths(country)
    inf_dir = paths["inference_dir"]
    inf_dir.mkdir(parents=True, exist_ok=True)

    # Load required data
    entities = normalize_entity_records(_load_with_fallback(paths, "raw_entities"))
    parents = normalize_entity_records(
        _load_with_fallback(paths, "raw_related", columns=ENTITY_COLUMNS)
    )
    relationships = normalize_relationship_records(
        _load_with_fallback(paths, "interim_relationships", columns=RELATIONSHIP_COLUMNS)
    )

    if relationships.empty or parents.empty:
        print("[WARN] No relationship or parent data available. Cannot run graph link prediction.")
        return {}

    # Build enriched graph — include inferred edges from Phase 1 and 2
    all_relationships = relationships.copy()
    for phase_file in ["phase1_inferred_edges", "phase2_inferred_edges"]:
        # Try country-scoped path first, then legacy
        for search_dir in [inf_dir, paths.get("legacy", {}).get("inference_dir", inf_dir)]:
            try:
                inferred = load_df(search_dir / phase_file)
                if not inferred.empty:
                    for col in RELATIONSHIP_COLUMNS:
                        if col not in inferred.columns:
                            inferred[col] = None
                    all_relationships = pd.concat(
                        [all_relationships, inferred[RELATIONSHIP_COLUMNS]],
                        ignore_index=True,
                    )
                    print(f"[INFO] Added {len(inferred)} inferred edges from {phase_file}")
                break
            except FileNotFoundError:
                continue

    all_relationships = all_relationships.drop_duplicates(
        subset=["source_lei", "target_lei"], keep="first"
    )

    # Build graph with all edges
    all_entities = _combine_entity_sets(entities, parents)
    g = build_di_graph(all_entities, all_relationships)
    print(f"[INFO] Enriched graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    # Run Phase 4 analysis
    results = run_phase4_analysis(
        g, entities, all_relationships, parents,
        neg_ratio=3,
        min_link_probability=0.3,
    )

    # Save artifacts
    link_preds = results.get("link_predictions", pd.DataFrame())
    propagated = results.get("propagated_labels", pd.DataFrame())
    feat_imp = results.get("feature_importance", pd.DataFrame())

    if not link_preds.empty:
        save_df(link_preds, inf_dir / "phase4_link_predictions")
        save_csv(link_preds, inf_dir / "phase4_link_predictions.csv")

    if not propagated.empty:
        save_df(propagated, inf_dir / "phase4_propagated_labels")
        save_csv(propagated, inf_dir / "phase4_propagated_labels.csv")

    if not feat_imp.empty:
        save_csv(feat_imp, inf_dir / "phase4_feature_importance.csv")

    # Save metrics
    metrics = results.get("metrics", {})
    if metrics and not metrics.get("skipped"):
        pd.DataFrame([metrics]).to_csv(inf_dir / "phase4_metrics.csv", index=False)

    # Print summary
    summary = phase4_summary(results)
    print(f"\n--- Phase 4: Graph Link Prediction Summary ---")
    print(f"  Graph nodes:           {summary.get('n_nodes', 0)}")
    print(f"  Graph edges:           {summary.get('n_edges', 0)}")
    print(f"  Labeled ratio:         {summary.get('labeled_ratio', 0):.3f}")
    print(f"  Status:                {summary.get('status', 'unknown')}")

    if summary.get("status") == "completed":
        print(f"  Model type:            {summary.get('model_type', 'N/A')}")
        print(f"  CV AUC:                {summary.get('cv_auc', 0):.3f}")
        print(f"  Link predictions:      {summary.get('link_predictions_count', 0)}")
        print(f"  High confidence links: {summary.get('high_confidence_links', 0)}")
        print(f"  Propagated labels:     {summary.get('propagated_labels_count', 0)}")

        if not link_preds.empty and "predicted_parent_country" in link_preds.columns:
            print(f"\n  Predicted parent country distribution (top 10):")
            dist = link_preds["predicted_parent_country"].value_counts().head(10)
            for country, count in dist.items():
                print(f"    {country}: {count}")

        if not propagated.empty:
            print(f"\n  Propagated jurisdiction distribution (top 10):")
            dist = propagated["propagated_country"].value_counts().head(10)
            for country, count in dist.items():
                print(f"    {country}: {count}")
    elif summary.get("status") == "skipped":
        print(f"  Reason:                {summary.get('skip_reason', 'unknown')}")

    return results


def run_full_inference_pipeline(
    threshold: int = 90,
    skip_pull: bool = False,
    min_cluster: int = 3,
    country: str = "MY",
    force_pull: bool = False,
) -> dict:
    """Run all inference phases sequentially.

    Returns a dict with results from each phase.
    """
    results = {}

    print("=" * 70)
    print(f"PHASE 0.5: REPORTING EXCEPTIONS  [{country}]")
    print("=" * 70)
    results["exceptions"] = run_reporting_exceptions(
        skip_pull=skip_pull,
        country=country,
        force_refresh=force_pull,
    )

    print("\n" + "=" * 70)
    print(f"PHASE 1: NAME PATTERN MATCHING  [{country}]")
    print("=" * 70)
    results["name_matches"] = run_phase1_name_inference(threshold=threshold, skip_pull=skip_pull, country=country)

    print("\n" + "=" * 70)
    print(f"PHASE 2: ADDRESS CLUSTERING  [{country}]")
    print("=" * 70)
    results["address_clusters"] = run_phase2_address_clustering(skip_pull=skip_pull, min_cluster=min_cluster, country=country)

    print("\n" + "=" * 70)
    print(f"PHASE 3: JURISDICTION PREDICTION  [{country}]")
    print("=" * 70)
    results["predictions"] = run_phase3_jurisdiction_prediction(skip_pull=True, country=country)

    print("\n" + "=" * 70)
    print(f"PHASE 4: GRAPH LINK PREDICTION  [{country}]")
    print("=" * 70)
    results["graph_predictions"] = run_phase4_graph_prediction(skip_pull=True, country=country)

    print("\n" + "=" * 70)
    print("INFERENCE PIPELINE COMPLETE")
    print("=" * 70)

    total_name = len(results.get("name_matches", pd.DataFrame()))
    total_exceptions = len(results.get("exceptions", pd.DataFrame()))
    total_clustered = len(results.get("address_clusters", pd.DataFrame()))
    total_predicted = len(results.get("predictions", pd.DataFrame()))
    graph_results = results.get("graph_predictions", {})
    total_link_preds = len(graph_results.get("link_predictions", pd.DataFrame()))
    total_propagated = len(graph_results.get("propagated_labels", pd.DataFrame()))
    print(f"  Reporting exceptions found:  {total_exceptions}")
    print(f"  Name-inferred subsidiaries:  {total_name}")
    print(f"  Address-clustered entities:  {total_clustered}")
    print(f"  Jurisdiction predictions:    {total_predicted}")
    print(f"  Graph link predictions:      {total_link_preds}")
    print(f"  Propagated labels:           {total_propagated}")

    return results
