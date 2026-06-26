"""Phase 3: Jurisdiction predictor for parent company country inference.

Trains a classifier to predict the most likely parent country for entities
without known parent relationships. Uses GradientBoosting from scikit-learn.
Outputs probability distributions over jurisdictions — directly maps to
BPM7's Ultimate Investor Economy (UIE) classification.
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.country_config import get_country_config


# Minimum training samples per country to keep as a distinct class.
# Raised to 15: at MIN=5, rare-country classes with 1-4 samples break
# StratifiedKFold for large registries (SG has 39 post-bucket classes).
MIN_COUNTRY_SAMPLES = 15

# Regional buckets for rare countries
REGION_MAP = {
    # Europe
    "DE": "EUROPE", "GB": "EUROPE", "FR": "EUROPE", "NL": "EUROPE",
    "CH": "EUROPE", "SE": "EUROPE", "DK": "EUROPE", "IT": "EUROPE",
    "IE": "EUROPE", "BE": "EUROPE", "AT": "EUROPE", "ES": "EUROPE",
    "NO": "EUROPE", "FI": "EUROPE", "LU": "EUROPE", "GG": "EUROPE",
    "JE": "EUROPE",
    # Asia-Pacific (excluding SG/JP which may have enough samples)
    "IN": "ASIA_OTHER", "HK": "ASIA_OTHER", "KR": "ASIA_OTHER",
    "TW": "ASIA_OTHER", "TH": "ASIA_OTHER", "PH": "ASIA_OTHER",
    "ID": "ASIA_OTHER", "VN": "ASIA_OTHER", "BD": "ASIA_OTHER",
    "NZ": "ASIA_OTHER", "AU": "ASIA_OTHER",
    # Middle East & Africa
    "AE": "MIDEAST_AFRICA", "SA": "MIDEAST_AFRICA", "KW": "MIDEAST_AFRICA",
    "QA": "MIDEAST_AFRICA", "BH": "MIDEAST_AFRICA", "DJ": "MIDEAST_AFRICA",
    "ZA": "MIDEAST_AFRICA",
    # Americas (excluding US)
    "CA": "AMERICAS_OTHER", "BR": "AMERICAS_OTHER", "MX": "AMERICAS_OTHER",
    # Offshore
    "KY": "OFFSHORE", "BM": "OFFSHORE", "VG": "OFFSHORE", "MU": "OFFSHORE",
    "JE": "OFFSHORE",
}


def _bucket_country(country: str, country_counts: dict[str, int]) -> str:
    """Map a country to itself if it has enough samples, else to a regional bucket."""
    if country_counts.get(country, 0) >= MIN_COUNTRY_SAMPLES:
        return country
    return REGION_MAP.get(country, "OTHER")


def _apply_second_pass_bucketing(labels: pd.Series) -> pd.Series:
    """Collapse any post-first-pass bucket that still has < 2 samples into OTHER.

    StratifiedKFold requires every class to have at least n_splits samples.
    A single-sample class makes CV impossible even after the first bucketing pass.
    This guarantees every surviving class has ≥ 2 samples.
    """
    counts = labels.value_counts()
    singletons = set(counts[counts < 2].index)
    if not singletons:
        return labels
    return labels.apply(lambda x: "OTHER" if x in singletons else x)


def prepare_training_features(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    parent_entities_df: pd.DataFrame,
    graph_summary_df: pd.DataFrame | None = None,
    exceptions_df: pd.DataFrame | None = None,
    pseudo_labels_df: pd.DataFrame | None = None,
    country: str = "MY",
) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Build feature matrix and target variable from labeled entities.

    Parameters
    ----------
    entities_df : Domestic entities.
    relationships_df : Known relationships (source_lei → target_lei).
    parent_entities_df : Parent entities (with country_legal data).
    graph_summary_df : Optional graph features to merge.
    exceptions_df : Optional reporting exception data.
    pseudo_labels_df : Optional high-confidence Phase 1 matches with columns
        [lei, matched_parent_lei, match_score]. Rows with score ≥ 95 are
        added to the training set as pseudo-labels.

    Returns
    -------
    (X, y, label_encoder) where X is the feature DataFrame, y is the
    bucketed parent country, and label_encoder maps back to country names.
    """
    if relationships_df.empty:
        return pd.DataFrame(), pd.Series(dtype=str), LabelEncoder()

    # Map source_lei -> target_lei -> target country
    parent_countries = parent_entities_df[["lei", "country_legal"]].rename(
        columns={"lei": "target_lei", "country_legal": "parent_country"}
    ).dropna().drop_duplicates(subset=["target_lei"])

    labeled = (
        relationships_df[["source_lei", "target_lei"]]
        .merge(parent_countries, on="target_lei", how="inner")
        .drop_duplicates(subset=["source_lei"], keep="first")
        .rename(columns={"source_lei": "lei"})
    )

    if labeled.empty:
        return pd.DataFrame(), pd.Series(dtype=str), LabelEncoder()

    # --- Pseudo-labels from Phase 1 high-confidence matches ---
    if pseudo_labels_df is not None and not pseudo_labels_df.empty:
        HIGH_CONF = 95
        high_conf = pseudo_labels_df[pseudo_labels_df["match_score"] >= HIGH_CONF].copy()
        high_conf = high_conf.rename(columns={"matched_parent_lei": "target_lei", "lei": "source_lei"})
        pseudo = (
            high_conf[["source_lei", "target_lei"]]
            .merge(parent_countries, on="target_lei", how="inner")
            .drop_duplicates(subset=["source_lei"], keep="first")
            .rename(columns={"source_lei": "lei"})
        )
        # Don't overwrite existing ground-truth labels
        existing_leis = set(labeled["lei"])
        pseudo = pseudo[~pseudo["lei"].isin(existing_leis)]
        if not pseudo.empty:
            labeled = pd.concat([labeled, pseudo], ignore_index=True)
            print(f"[INFO] Phase 3: added {len(pseudo)} pseudo-labels from Phase 1 (score >= {HIGH_CONF})", flush=True)

    # Bucket rare countries — first pass
    country_counts = labeled["parent_country"].value_counts().to_dict()
    labeled["parent_bucket"] = labeled["parent_country"].apply(
        lambda c: _bucket_country(c, country_counts)
    )

    # Second pass: collapse any post-bucketing class with < 2 samples into OTHER
    labeled["parent_bucket"] = _apply_second_pass_bucketing(labeled["parent_bucket"])

    n_classes = labeled["parent_bucket"].nunique()
    print(f"[INFO] Phase 3 training: {len(labeled)} samples, {n_classes} classes after bucketing", flush=True)

    # Build features for labeled entities
    features = _build_features(labeled["lei"], entities_df, graph_summary_df, exceptions_df, country=country)

    # Align features with labels
    merged = features.merge(labeled[["lei", "parent_bucket"]], on="lei", how="inner")
    y = merged.pop("parent_bucket")
    X = merged.drop(columns=["lei"])

    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y), index=y.index, name="parent_bucket")

    return X, y_encoded, le


def _build_features(
    leis: pd.Series,
    entities_df: pd.DataFrame,
    graph_summary_df: pd.DataFrame | None = None,
    exceptions_df: pd.DataFrame | None = None,
    country: str = "MY",
) -> pd.DataFrame:
    """Build feature columns for a set of LEIs."""
    df = entities_df[entities_df["lei"].isin(leis)].copy()

    features = pd.DataFrame({"lei": df["lei"].values})
    cfg = get_country_config(country)

    # Legal form — one-hot top forms, rest as "OTHER"
    if "legal_form" in df.columns:
        top_forms = df["legal_form"].value_counts().head(10).index
        features["legal_form_cat"] = df["legal_form"].where(
            df["legal_form"].isin(top_forms), "OTHER"
        ).values
    else:
        features["legal_form_cat"] = "UNKNOWN"

    # Category — one-hot
    if "category" in df.columns:
        features["category_cat"] = df["category"].fillna("UNKNOWN").values
    else:
        features["category_cat"] = "UNKNOWN"

    # City — frequency encoded
    if "city_legal" in df.columns:
        city_counts = entities_df["city_legal"].value_counts().to_dict()
        features["city_frequency"] = df["city_legal"].map(city_counts).fillna(0).values
    else:
        features["city_frequency"] = 0

    # Registration year
    if "registered_at" in df.columns:
        dates = pd.to_datetime(df["registered_at"], errors="coerce")
        features["registration_year"] = dates.dt.year.fillna(0).astype(int).values
    else:
        features["registration_year"] = 0

    # Name features — country-specific legal-name patterns from config
    if "legal_name" in df.columns:
        names = df["legal_name"].fillna("")
        features["name_token_count"] = names.str.split().str.len().fillna(0).astype(int).values
        features["name_length"] = names.str.len().fillna(0).astype(int).values
        upper_names = names.str.upper()
        for feat_name, pattern in cfg.name_features:
            features[feat_name] = upper_names.str.contains(pattern, regex=True).astype(int).values
    else:
        features["name_token_count"] = 0
        features["name_length"] = 0
        for feat_name, _ in cfg.name_features:
            features[feat_name] = 0

    # Graph features
    if graph_summary_df is not None and not graph_summary_df.empty:
        graph_cols = ["lei", "in_degree", "out_degree", "pagerank",
                      "degree_centrality", "weakly_connected_component_size"]
        available = [c for c in graph_cols if c in graph_summary_df.columns]
        if "lei" in available:
            features = features.merge(
                graph_summary_df[available], on="lei", how="left"
            )
            for col in available:
                if col != "lei":
                    features[col] = features[col].fillna(0)

    # Exception flags
    if exceptions_df is not None and not exceptions_df.empty:
        exc = exceptions_df[["lei", "exception_reason"]].drop_duplicates(subset=["lei"])
        features = features.merge(exc, on="lei", how="left")
        features["is_non_consolidating"] = (
            features.get("exception_reason") == "NON_CONSOLIDATING"
        ).astype(int)
        features["has_exception"] = features["exception_reason"].notna().astype(int)
        features.drop(columns=["exception_reason"], inplace=True, errors="ignore")
    else:
        features["is_non_consolidating"] = 0
        features["has_exception"] = 0

    # One-hot encode categorical columns
    features = pd.get_dummies(features, columns=["legal_form_cat", "category_cat"], dtype=int)

    return features


def _cross_validate_jurisdiction_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> dict:
    """Evaluate a classifier with top-1/top-k CV metrics and confusion counts."""
    classes = np.array(sorted(pd.Series(y).unique()))
    top_k = min(3, len(classes))
    top1_scores: list[float] = []
    topk_scores: list[float] = []
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx].to_numpy()

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        proba = fold_model.predict_proba(X_test)
        model_classes = np.array(fold_model.classes_)

        pred = model_classes[np.argmax(proba, axis=1)]
        top_order = np.argsort(proba, axis=1)[:, -top_k:]
        top_candidates = model_classes[top_order]

        top1_scores.append(float(np.mean(pred == y_test)))
        topk_scores.append(float(np.mean([
            true_label in candidate_row
            for true_label, candidate_row in zip(y_test, top_candidates)
        ])))
        y_true_all.extend(int(v) for v in y_test)
        y_pred_all.extend(int(v) for v in pred)

    matrix = confusion_matrix(y_true_all, y_pred_all, labels=classes)
    confusion_rows = []
    for i, actual in enumerate(classes):
        for j, predicted in enumerate(classes):
            count = int(matrix[i, j])
            if count:
                confusion_rows.append({
                    "actual_class": int(actual),
                    "predicted_class": int(predicted),
                    "count": count,
                })

    return {
        "top1_scores": np.array(top1_scores),
        "topk_scores": np.array(topk_scores),
        "top_k": top_k,
        "confusion_matrix": pd.DataFrame(
            confusion_rows,
            columns=["actual_class", "predicted_class", "count"],
        ),
    }


def train_jurisdiction_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[GradientBoostingClassifier | RandomForestClassifier, dict]:
    """Train a jurisdiction classifier with cross-validation.

    Tries GradientBoosting and RandomForest, returns the better one.

    Returns (model, cv_metrics_dict).
    """
    if len(X) < 20:
        print("[WARN] Very small training set — results will be unreliable", flush=True)

    n_classes = y.nunique()
    min_class_count = int(min(y.value_counts()))
    majority_baseline = round(float(y.value_counts(normalize=True).max()), 4)

    # GradientBoosting requires at least 2 classes.  If we have only 1,
    # return a trivial constant predictor that always outputs that class.
    if n_classes < 2:
        print(f"[WARN] Only {n_classes} class in training data — returning constant predictor.", flush=True)
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X, y)
        metrics = {
            "model_type": "DummyConstant",
            "cv_accuracy_mean": 1.0,
            "cv_accuracy_std": 0.0,
            "n_training_samples": len(X),
            "n_classes": int(n_classes),
            "n_folds": 0,
            "gb_cv_mean": float("nan"),
            "rf_cv_mean": float("nan"),
            "majority_class_baseline_accuracy": majority_baseline,
            "cv_accuracy_lift_over_baseline": 0.0,
            "cv_top3_accuracy_mean": 1.0,
            "cv_top3_accuracy_std": 0.0,
            "cv_top_k": 1,
            "_cv_confusion_matrix": pd.DataFrame([{
                "actual_class": int(y.iloc[0]) if len(y) else 0,
                "predicted_class": int(y.iloc[0]) if len(y) else 0,
                "count": int(len(y)),
            }]),
            "note": "single_class_constant_predictor",
        }
        return model, metrics

    # If any class has fewer than 2 samples, CV is impossible — train
    # directly without cross-validation.
    if min_class_count < 2 or len(X) < 4:
        print(f"[WARN] Too few samples for CV (min_class_count={min_class_count}, n={len(X)}). "
              "Training without cross-validation.", flush=True)
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
        )
        model.fit(X, y)
        metrics = {
            "model_type": "GradientBoosting",
            "cv_accuracy_mean": float("nan"),
            "cv_accuracy_std": float("nan"),
            "n_training_samples": len(X),
            "n_classes": int(n_classes),
            "n_folds": 0,
            "gb_cv_mean": float("nan"),
            "rf_cv_mean": float("nan"),
            "majority_class_baseline_accuracy": majority_baseline,
            "cv_accuracy_lift_over_baseline": float("nan"),
            "cv_top3_accuracy_mean": float("nan"),
            "cv_top3_accuracy_std": float("nan"),
            "cv_top_k": min(3, int(n_classes)),
            "note": "no_cv_too_few_samples",
        }
        return model, metrics

    n_splits = min(5, min_class_count)
    n_splits = max(2, n_splits)  # at least 2-fold

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # GradientBoosting
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    gb_eval = _cross_validate_jurisdiction_model(gb, X, y, cv)
    gb_scores = gb_eval["top1_scores"]

    # RandomForest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    rf_eval = _cross_validate_jurisdiction_model(rf, X, y, cv)
    rf_scores = rf_eval["top1_scores"]

    # Pick the better model
    if gb_scores.mean() >= rf_scores.mean():
        model = gb
        model_name = "GradientBoosting"
        scores = gb_scores
        eval_result = gb_eval
    else:
        model = rf
        model_name = "RandomForest"
        scores = rf_scores
        eval_result = rf_eval

    # Fit on full training set
    model.fit(X, y)

    metrics = {
        "model_type": model_name,
        "cv_accuracy_mean": round(float(scores.mean()), 4),
        "cv_accuracy_std": round(float(scores.std()), 4),
        "n_training_samples": len(X),
        "n_classes": int(n_classes),
        "n_folds": n_splits,
        "gb_cv_mean": round(float(gb_scores.mean()), 4),
        "rf_cv_mean": round(float(rf_scores.mean()), 4),
        "majority_class_baseline_accuracy": majority_baseline,
        "cv_accuracy_lift_over_baseline": round(float(scores.mean()) - majority_baseline, 4),
        "cv_top3_accuracy_mean": round(float(eval_result["topk_scores"].mean()), 4),
        "cv_top3_accuracy_std": round(float(eval_result["topk_scores"].std()), 4),
        "cv_top_k": int(eval_result["top_k"]),
        "gb_top3_mean": round(float(gb_eval["topk_scores"].mean()), 4),
        "rf_top3_mean": round(float(rf_eval["topk_scores"].mean()), 4),
        "_cv_confusion_matrix": eval_result["confusion_matrix"],
    }

    print(f"[INFO] Best model: {model_name} (CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f})", flush=True)
    print(f"[INFO] Top-{eval_result['top_k']} CV accuracy: {eval_result['topk_scores'].mean():.3f}", flush=True)
    print(f"[INFO] GB: {gb_scores.mean():.3f}, RF: {rf_scores.mean():.3f}", flush=True)

    return model, metrics


def predict_parent_jurisdiction(
    model,
    label_encoder: LabelEncoder,
    X_new: pd.DataFrame,
    leis: pd.Series,
) -> pd.DataFrame:
    """Predict parent jurisdiction for unlabeled entities.

    Returns DataFrame with [lei, predicted_parent_country, prediction_probability,
    top3_countries, top3_probabilities].
    """
    if X_new.empty:
        return pd.DataFrame(columns=[
            "lei", "predicted_parent_country", "prediction_probability",
            "confidence_gap", "top3_countries", "top3_probabilities",
        ])

    proba = model.predict_proba(X_new)
    classes = label_encoder.classes_

    predictions = []
    for i in range(len(X_new)):
        prob_row = proba[i]
        sorted_idx = np.argsort(prob_row)[::-1]

        top1_class = classes[sorted_idx[0]]
        top1_prob = float(prob_row[sorted_idx[0]])
        top2_prob = float(prob_row[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0

        top3_classes = [classes[j] for j in sorted_idx[:3]]
        top3_probs = [round(float(prob_row[j]), 4) for j in sorted_idx[:3]]

        predictions.append({
            "lei": leis.iloc[i],
            "predicted_parent_country": top1_class,
            "prediction_probability": round(top1_prob, 4),
            "confidence_gap": round(top1_prob - top2_prob, 4),  # high = confident, low = ambiguous
            "top3_countries": ";".join(top3_classes),
            "top3_probabilities": ";".join(str(p) for p in top3_probs),
        })

    return pd.DataFrame(predictions)


def save_model_artifacts(
    model,
    cv_metrics: dict,
    feature_names: list[str],
    label_encoder: LabelEncoder,
    output_dir: Path,
) -> None:
    """Save model, metrics, and feature importance to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, output_dir / "phase3_model.joblib")
    joblib.dump(label_encoder, output_dir / "phase3_label_encoder.joblib")

    # Save CV metrics
    confusion = cv_metrics.get("_cv_confusion_matrix")
    metrics_for_csv = {
        key: value for key, value in cv_metrics.items()
        if not key.startswith("_")
    }
    pd.DataFrame([metrics_for_csv]).to_csv(output_dir / "phase3_cv_metrics.csv", index=False)

    if isinstance(confusion, pd.DataFrame) and not confusion.empty:
        confusion_out = confusion.copy()
        confusion_out["actual_country"] = label_encoder.inverse_transform(
            confusion_out["actual_class"].astype(int)
        )
        confusion_out["predicted_country"] = label_encoder.inverse_transform(
            confusion_out["predicted_class"].astype(int)
        )
        confusion_out = confusion_out[
            ["actual_country", "predicted_country", "count", "actual_class", "predicted_class"]
        ].sort_values(["actual_country", "count"], ascending=[True, False])
        confusion_out.to_csv(output_dir / "phase3_cv_confusion_matrix.csv", index=False)

    # Save feature importance
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(output_dir / "phase3_feature_importance.csv", index=False)


def phase3_summary(predictions_df: pd.DataFrame, cv_metrics: dict) -> dict:
    """Generate a summary dict for Phase 3 results."""
    if predictions_df.empty:
        return {"total_predictions": 0, **cv_metrics}

    high_probability = int((predictions_df["prediction_probability"] >= 0.5).sum())
    return {
        "total_predictions": len(predictions_df),
        "top_predicted_country": predictions_df["predicted_parent_country"].value_counts().index[0],
        "avg_confidence": round(predictions_df["prediction_probability"].mean(), 4),
        "high_confidence_predictions": high_probability,
        "uncalibrated_high_probability_predictions": high_probability,
        "probability_note": "Model probabilities are uncalibrated ranking signals, not validated confidence.",
        **cv_metrics,
    }
