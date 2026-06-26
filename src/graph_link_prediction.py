"""Phase 4: Graph-based link prediction for parent relationship inference.

Uses graph topology (networkx link prediction metrics + structural
similarity) to predict missing parent–child edges.  This complements
Phase 3's tabular jurisdiction predictor by leveraging the ownership
graph structure that feature-based models cannot capture.

Design is country-agnostic — works with any nation's entity graph.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Graph readiness assessment
# ---------------------------------------------------------------------------

def assess_graph_readiness(
    g: nx.DiGraph,
    relationships: pd.DataFrame,
    entities: pd.DataFrame,
    *,
    min_edges: int = 50,
    min_labeled_ratio: float = 0.05,
    min_parent_nodes: int = 10,
) -> dict:
    """Check whether the ownership graph is dense enough for link prediction.

    Thresholds are intentionally low — even sparse graphs can yield useful
    structural signals when combined with other phases.

    Returns a dict with readiness flags and diagnostics.
    """
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    density = nx.density(g) if n_nodes > 1 else 0.0

    # Count domestic entities and parent (target) entities
    domestic_leis = set(entities["lei"].dropna().unique())
    if not relationships.empty:
        labeled_leis = set(relationships["source_lei"].dropna().unique()) & domestic_leis
        parent_leis = set(relationships["target_lei"].dropna().unique()) - domestic_leis
    else:
        labeled_leis = set()
        parent_leis = set()

    labeled_ratio = len(labeled_leis) / max(len(domestic_leis), 1)

    # Connected components
    ug = g.to_undirected()
    n_components = nx.number_connected_components(ug) if n_nodes > 0 else 0
    largest_cc = max(len(c) for c in nx.connected_components(ug)) if n_nodes > 0 else 0

    ready = (
        n_edges >= min_edges
        and labeled_ratio >= min_labeled_ratio
        and len(parent_leis) >= min_parent_nodes
    )

    return {
        "ready": ready,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": round(density, 6),
        "n_domestic_entities": len(domestic_leis),
        "n_labeled_entities": len(labeled_leis),
        "labeled_ratio": round(labeled_ratio, 4),
        "n_parent_nodes": len(parent_leis),
        "n_components": n_components,
        "largest_component": largest_cc,
        "min_edges_threshold": min_edges,
        "min_labeled_ratio_threshold": min_labeled_ratio,
        "min_parent_nodes_threshold": min_parent_nodes,
    }


# ---------------------------------------------------------------------------
# Topological feature extraction for node pairs
# ---------------------------------------------------------------------------

def _safe_jaccard(ug: nx.Graph, u: str, v: str) -> float:
    """Jaccard coefficient between two nodes (undirected)."""
    try:
        preds = nx.jaccard_coefficient(ug, [(u, v)])
        for _, _, score in preds:
            return float(score)
    except (nx.NetworkXError, ZeroDivisionError):
        return 0.0
    return 0.0


def _safe_adamic_adar(ug: nx.Graph, u: str, v: str) -> float:
    """Adamic-Adar index between two nodes."""
    try:
        preds = nx.adamic_adar_index(ug, [(u, v)])
        for _, _, score in preds:
            return float(score)
    except (nx.NetworkXError, ZeroDivisionError):
        return 0.0
    return 0.0


def _safe_preferential_attachment(ug: nx.Graph, u: str, v: str) -> float:
    """Preferential attachment score between two nodes."""
    try:
        preds = nx.preferential_attachment(ug, [(u, v)])
        for _, _, score in preds:
            return float(score)
    except (nx.NetworkXError, ZeroDivisionError):
        return 0.0
    return 0.0


def _common_neighbors_count(ug: nx.Graph, u: str, v: str) -> int:
    """Count common neighbors between two nodes."""
    try:
        return len(list(nx.common_neighbors(ug, u, v)))
    except nx.NetworkXError:
        return 0


def _same_component(ug: nx.Graph, u: str, v: str) -> int:
    """Check if two nodes are in the same connected component."""
    try:
        return int(nx.has_path(ug, u, v))
    except (nx.NetworkXError, nx.NodeNotFound):
        return 0


def compute_pair_features(
    g: nx.DiGraph,
    pairs: list[tuple[str, str]],
    pagerank: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute topological features for a list of (entity, parent) pairs.

    Features:
    - jaccard_coefficient: neighborhood overlap
    - adamic_adar: weighted common neighbor index
    - preferential_attachment: degree product
    - common_neighbors: raw count
    - same_component: whether connected in undirected graph
    - source_out_degree / target_in_degree: local connectivity
    - source_pagerank / target_pagerank: global importance
    - component_size_ratio: relative component sizes
    """
    ug = g.to_undirected()

    if pagerank is None:
        pagerank = nx.pagerank(g) if len(g) > 0 else {}

    # Pre-compute component memberships
    node_component_size: dict[str, int] = {}
    for comp in nx.connected_components(ug):
        size = len(comp)
        for node in comp:
            node_component_size[node] = size

    rows = []
    for source, target in pairs:
        row = {
            "source_lei": source,
            "target_lei": target,
            "jaccard_coefficient": _safe_jaccard(ug, source, target),
            "adamic_adar": _safe_adamic_adar(ug, source, target),
            "preferential_attachment": _safe_preferential_attachment(ug, source, target),
            "common_neighbors": _common_neighbors_count(ug, source, target),
            "same_component": _same_component(ug, source, target),
            "source_out_degree": g.out_degree(source) if source in g else 0,
            "target_in_degree": g.in_degree(target) if target in g else 0,
            "source_pagerank": pagerank.get(source, 0.0),
            "target_pagerank": pagerank.get(target, 0.0),
            "source_component_size": node_component_size.get(source, 1),
            "target_component_size": node_component_size.get(target, 1),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Training pair generation
# ---------------------------------------------------------------------------

_FEATURE_COLS = [
    "jaccard_coefficient", "adamic_adar", "preferential_attachment",
    "common_neighbors", "same_component",
    "source_out_degree", "target_in_degree",
    "source_pagerank", "target_pagerank",
    "source_component_size", "target_component_size",
]


def _edge_holdout_cv_scores(
    model: Any,
    g: nx.DiGraph,
    training_pairs: pd.DataFrame,
    cv: StratifiedKFold,
) -> np.ndarray:
    """Compute ROC-AUC with validation positive edges removed from the graph."""
    y = training_pairs["label"].astype(int).to_numpy()
    scores: list[float] = []

    for train_idx, test_idx in cv.split(training_pairs, y):
        train_pairs = training_pairs.iloc[train_idx].reset_index(drop=True)
        test_pairs = training_pairs.iloc[test_idx].reset_index(drop=True)

        fold_graph = g.copy()
        heldout_edges = [
            (row["source_lei"], row["target_lei"])
            for _, row in test_pairs[test_pairs["label"].eq(1)].iterrows()
        ]
        fold_graph.remove_edges_from(heldout_edges)

        train_pair_list = list(zip(train_pairs["source_lei"], train_pairs["target_lei"]))
        test_pair_list = list(zip(test_pairs["source_lei"], test_pairs["target_lei"]))

        X_train = compute_pair_features(fold_graph, train_pair_list)[_FEATURE_COLS].fillna(0)
        X_test = compute_pair_features(fold_graph, test_pair_list)[_FEATURE_COLS].fillna(0)
        y_train = train_pairs["label"].astype(int).to_numpy()
        y_test = test_pairs["label"].astype(int).to_numpy()

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=_FEATURE_COLS
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=_FEATURE_COLS
        )

        fold_model = clone(model)
        fold_model.fit(X_train_scaled, y_train)
        pos_idx = list(fold_model.classes_).index(1)
        y_score = fold_model.predict_proba(X_test_scaled)[:, pos_idx]
        scores.append(float(roc_auc_score(y_test, y_score)))

    return np.array(scores)


def generate_training_pairs(
    relationships: pd.DataFrame,
    domestic_leis: list[str],
    parent_leis: list[str],
    neg_ratio: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate positive and negative pairs for training.

    Positive pairs: known (source_lei, target_lei) from relationships.
    Negative pairs: random (domestic_lei, parent_lei) that don't exist.

    Parameters
    ----------
    neg_ratio : number of negative samples per positive sample.
    """
    rng = np.random.RandomState(random_state)

    # Positive pairs from known relationships
    if relationships.empty:
        return pd.DataFrame(columns=["source_lei", "target_lei", "label"])

    positives = (
        relationships[["source_lei", "target_lei"]]
        .dropna()
        .drop_duplicates()
    )
    # Filter to domestic→parent pairs only
    domestic_set = set(domestic_leis)
    parent_set = set(parent_leis)
    positives = positives[
        positives["source_lei"].isin(domestic_set)
        & positives["target_lei"].isin(parent_set)
    ].copy()

    if positives.empty:
        return pd.DataFrame(columns=["source_lei", "target_lei", "label"])

    positives["label"] = 1
    known_pairs = set(zip(positives["source_lei"], positives["target_lei"]))

    # Negative pairs: random entity-parent combinations that don't exist
    neg_count = len(positives) * neg_ratio
    neg_rows = []
    domestic_arr = list(domestic_set)
    parent_arr = list(parent_set)
    attempts = 0
    max_attempts = neg_count * 10

    while len(neg_rows) < neg_count and attempts < max_attempts:
        s = rng.choice(domestic_arr)
        t = rng.choice(parent_arr)
        if (s, t) not in known_pairs:
            neg_rows.append({"source_lei": s, "target_lei": t, "label": 0})
            known_pairs.add((s, t))  # prevent duplicates
        attempts += 1

    negatives = pd.DataFrame(neg_rows)

    return pd.concat([positives, negatives], ignore_index=True).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Link prediction model
# ---------------------------------------------------------------------------

def train_link_predictor(
    g: nx.DiGraph,
    training_pairs: pd.DataFrame,
) -> tuple[Any, dict, StandardScaler]:
    """Train a classifier to predict parent–child links from graph topology.

    Returns (model, metrics_dict, scaler).
    """
    if training_pairs.empty or "label" not in training_pairs.columns:
        return None, {"error": "no training data"}, StandardScaler()

    # Compute features
    pairs = list(zip(training_pairs["source_lei"], training_pairs["target_lei"]))
    features = compute_pair_features(g, pairs)

    X = features[_FEATURE_COLS].fillna(0)
    y = training_pairs["label"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=_FEATURE_COLS)

    if len(X_scaled) < 10:
        print("[WARN] Very small training set for link prediction", flush=True)

    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=4, random_state=42, class_weight="balanced"
    )

    label_counts = pd.Series(y).value_counts()
    if len(label_counts) < 2 or label_counts.min() < 2:
        print("[WARN] Too few positive/negative pairs for link-prediction CV", flush=True)
        gb_scores = np.array([np.nan])
        rf_scores = np.array([np.nan])
        model = gb
        model_name = "GradientBoosting"
        scores = gb_scores
        n_splits = 0
    else:
        # Cross-validate with edge holdout so graph features cannot see
        # the validation positive edge they are meant to predict.
        n_splits = min(5, max(2, int(label_counts.min())))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        gb_scores = _edge_holdout_cv_scores(gb, g, training_pairs, cv)
        rf_scores = _edge_holdout_cv_scores(rf, g, training_pairs, cv)

        if np.nanmean(gb_scores) >= np.nanmean(rf_scores):
            model = gb
            model_name = "GradientBoosting"
            scores = gb_scores
        else:
            model = rf
            model_name = "RandomForest"
            scores = rf_scores

    # Fit the final production model on the full graph and all training pairs.
    model.fit(X_scaled, y)

    metrics = {
        "model_type": model_name,
        "cv_auc_mean": round(float(np.nanmean(scores)), 4),
        "cv_auc_std": round(float(np.nanstd(scores)), 4),
        "n_positive_pairs": int(y.sum()),
        "n_negative_pairs": int(len(y) - y.sum()),
        "n_folds": n_splits,
        "gb_auc_mean": round(float(np.nanmean(gb_scores)), 4),
        "rf_auc_mean": round(float(np.nanmean(rf_scores)), 4),
        "cv_method": "edge_holdout_remove_validation_positives",
    }

    print(
        f"[INFO] Link predictor: {model_name} "
        f"(edge-holdout CV AUC: {np.nanmean(scores):.3f} +/- {np.nanstd(scores):.3f})",
        flush=True,
    )

    return model, metrics, scaler


def predict_links(
    model: Any,
    scaler: StandardScaler,
    g: nx.DiGraph,
    candidate_pairs: list[tuple[str, str]],
    min_probability: float = 0.3,
) -> pd.DataFrame:
    """Score candidate entity–parent pairs and return likely links.

    Parameters
    ----------
    candidate_pairs : list of (domestic_lei, parent_lei) to evaluate.
    min_probability : minimum predicted probability to include.

    Returns DataFrame with [source_lei, target_lei, link_probability, rank].
    """
    if model is None or not candidate_pairs:
        return pd.DataFrame(columns=[
            "source_lei", "target_lei", "link_probability",
            "predicted_parent_country", "rank",
        ])

    features = compute_pair_features(g, candidate_pairs)
    X = features[_FEATURE_COLS].fillna(0)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=_FEATURE_COLS)

    probas = model.predict_proba(X_scaled)
    # Probability of class 1 (positive link)
    pos_idx = list(model.classes_).index(1)
    link_probs = probas[:, pos_idx]

    results = features[["source_lei", "target_lei"]].copy()
    results["link_probability"] = np.round(link_probs, 4)

    # Filter by threshold and rank
    results = results[results["link_probability"] >= min_probability].copy()
    results = results.sort_values("link_probability", ascending=False)
    results["rank"] = range(1, len(results) + 1)

    return results.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Label propagation for jurisdiction
# ---------------------------------------------------------------------------

def propagate_jurisdiction_labels(
    g: nx.DiGraph,
    known_labels: dict[str, str],
    max_iterations: int = 10,
) -> pd.DataFrame:
    """Propagate parent jurisdiction labels through the graph.

    Uses the ownership graph structure: if entity A is connected to parent P
    with known country, entities in the same connected component can inherit
    that label weighted by graph distance.

    Unlike Phase 3's tabular model, this captures structural proximity.

    Returns DataFrame with [lei, propagated_country, confidence, hops].
    """
    if not known_labels or len(g) == 0:
        return pd.DataFrame(columns=["lei", "propagated_country", "confidence", "hops"])

    ug = g.to_undirected()
    results: list[dict] = []

    for component in nx.connected_components(ug):
        # Collect known labels within this component
        component_labels: dict[str, str] = {}
        for node in component:
            if node in known_labels:
                component_labels[node] = known_labels[node]

        if not component_labels:
            continue

        # Vote: most common jurisdiction in the component
        from collections import Counter
        label_counts = Counter(component_labels.values())
        dominant_label = label_counts.most_common(1)[0][0]
        total_labeled = sum(label_counts.values())

        # For each unlabeled node, compute shortest path to a labeled node
        for node in component:
            if node in known_labels:
                continue

            min_hops = float("inf")
            nearest_label = None
            for labeled_node, label in component_labels.items():
                try:
                    hops = nx.shortest_path_length(ug, node, labeled_node)
                    if hops < min_hops:
                        min_hops = hops
                        nearest_label = label
                except nx.NetworkXNoPath:
                    continue

            if nearest_label is not None and min_hops < float("inf"):
                # Confidence decays with distance, boosted by label consensus
                consensus = label_counts[nearest_label] / total_labeled
                distance_decay = 1.0 / (1.0 + min_hops)
                confidence = round(consensus * distance_decay, 4)

                results.append({
                    "lei": node,
                    "propagated_country": nearest_label,
                    "confidence": confidence,
                    "hops": int(min_hops),
                })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("confidence", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Full Phase 4 orchestration
# ---------------------------------------------------------------------------

def run_phase4_analysis(
    g: nx.DiGraph,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    parent_entities: pd.DataFrame,
    *,
    neg_ratio: int = 3,
    min_link_probability: float = 0.3,
) -> dict:
    """Run the complete Phase 4 graph link prediction pipeline.

    Returns a dict with:
    - readiness: graph readiness assessment
    - link_predictions: predicted parent links (or empty if graph too sparse)
    - propagated_labels: jurisdiction labels propagated through graph
    - metrics: model performance metrics
    - feature_importance: which graph features matter most
    """
    domestic_leis = entities["lei"].dropna().unique().tolist()
    domestic_set = set(domestic_leis)

    if not relationships.empty:
        parent_leis = [
            lei for lei in relationships["target_lei"].dropna().unique()
            if lei not in domestic_set
        ]
    else:
        parent_leis = []

    # 1. Assess readiness
    readiness = assess_graph_readiness(g, relationships, entities)

    if not readiness["ready"]:
        print(f"[WARN] Graph not ready for link prediction:", flush=True)
        print(f"       edges={readiness['n_edges']} (need {readiness['min_edges_threshold']})", flush=True)
        print(f"       labeled_ratio={readiness['labeled_ratio']:.3f} (need {readiness['min_labeled_ratio_threshold']})", flush=True)
        print(f"       parent_nodes={readiness['n_parent_nodes']} (need {readiness['min_parent_nodes_threshold']})", flush=True)
        return {
            "readiness": readiness,
            "link_predictions": pd.DataFrame(),
            "propagated_labels": pd.DataFrame(),
            "metrics": {"skipped": True, "reason": "graph_not_ready"},
            "feature_importance": pd.DataFrame(),
        }

    # 2. Generate training pairs
    print(f"[INFO] Generating training pairs (neg_ratio={neg_ratio})...", flush=True)
    training_pairs = generate_training_pairs(
        relationships, domestic_leis, parent_leis, neg_ratio=neg_ratio
    )
    print(f"[INFO] Training pairs: {len(training_pairs)} ({int(training_pairs['label'].sum())} positive)", flush=True)

    # 3. Train link predictor
    model, metrics, scaler = train_link_predictor(g, training_pairs)

    # 4. Feature importance
    feature_importance = pd.DataFrame()
    if model is not None and hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame({
            "feature": _FEATURE_COLS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

    # 5. Generate candidate pairs for prediction
    #    Unlabeled domestic entities × all parent entities
    known_sources = set(relationships["source_lei"].dropna()) if not relationships.empty else set()
    unlabeled_leis = [lei for lei in domestic_leis if lei not in known_sources]

    print(f"[INFO] Scoring {len(unlabeled_leis)} unlabeled × {len(parent_leis)} parents "
          f"= up to {len(unlabeled_leis) * len(parent_leis):,} candidate pairs...", flush=True)

    # Cap candidate pairs to avoid memory issues
    max_candidates = 500_000
    candidate_pairs: list[tuple[str, str]] = []
    for source in unlabeled_leis:
        for target in parent_leis:
            candidate_pairs.append((source, target))
            if len(candidate_pairs) >= max_candidates:
                break
        if len(candidate_pairs) >= max_candidates:
            print(f"[WARN] Capped at {max_candidates:,} candidate pairs", flush=True)
            break

    # 6. Predict links
    link_predictions = pd.DataFrame()
    if model is not None and candidate_pairs:
        link_predictions = predict_links(
            model, scaler, g, candidate_pairs,
            min_probability=min_link_probability,
        )

        # Enrich with parent country
        if not link_predictions.empty and not parent_entities.empty:
            parent_country_map = dict(zip(
                parent_entities["lei"],
                parent_entities["country_legal"],
            ))
            link_predictions["predicted_parent_country"] = (
                link_predictions["target_lei"].map(parent_country_map)
            )

        # Keep only best prediction per entity
        if not link_predictions.empty:
            link_predictions = (
                link_predictions
                .sort_values("link_probability", ascending=False)
                .drop_duplicates(subset=["source_lei"], keep="first")
                .reset_index(drop=True)
            )
            link_predictions["rank"] = range(1, len(link_predictions) + 1)

    # 7. Label propagation
    parent_country_labels: dict[str, str] = {}
    if not parent_entities.empty:
        for _, row in parent_entities.iterrows():
            if pd.notna(row.get("lei")) and pd.notna(row.get("country_legal")):
                parent_country_labels[row["lei"]] = row["country_legal"]

    propagated = propagate_jurisdiction_labels(g, parent_country_labels)

    return {
        "readiness": readiness,
        "link_predictions": link_predictions,
        "propagated_labels": propagated,
        "metrics": metrics,
        "feature_importance": feature_importance,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def phase4_summary(results: dict) -> dict:
    """Generate a summary dict for Phase 4 results."""
    readiness = results.get("readiness", {})
    metrics = results.get("metrics", {})
    link_preds = results.get("link_predictions", pd.DataFrame())
    propagated = results.get("propagated_labels", pd.DataFrame())

    summary = {
        "graph_ready": readiness.get("ready", False),
        "n_nodes": readiness.get("n_nodes", 0),
        "n_edges": readiness.get("n_edges", 0),
        "labeled_ratio": readiness.get("labeled_ratio", 0),
        "n_parent_nodes": readiness.get("n_parent_nodes", 0),
    }

    if metrics.get("skipped"):
        summary["status"] = "skipped"
        summary["skip_reason"] = metrics.get("reason", "unknown")
        return summary

    summary["status"] = "completed"
    summary["model_type"] = metrics.get("model_type", "N/A")
    summary["cv_auc"] = metrics.get("cv_auc_mean", 0)
    summary["link_predictions_count"] = len(link_preds)
    summary["propagated_labels_count"] = len(propagated)

    if not link_preds.empty:
        summary["avg_link_probability"] = round(link_preds["link_probability"].mean(), 4)
        summary["high_confidence_links"] = int((link_preds["link_probability"] >= 0.5).sum())
        if "predicted_parent_country" in link_preds.columns:
            top = link_preds["predicted_parent_country"].value_counts().head(1)
            if not top.empty:
                summary["top_predicted_parent_country"] = top.index[0]

    if not propagated.empty:
        summary["avg_propagation_confidence"] = round(propagated["confidence"].mean(), 4)
        summary["propagated_high_confidence"] = int((propagated["confidence"] >= 0.3).sum())
        top_prop = propagated["propagated_country"].value_counts().head(1)
        if not top_prop.empty:
            summary["top_propagated_country"] = top_prop.index[0]

    return summary
