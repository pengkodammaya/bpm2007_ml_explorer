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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.country_config import get_country_config


# Minimum training samples per country to keep as a distinct class
MIN_COUNTRY_SAMPLES = 5

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


def prepare_training_features(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    parent_entities_df: pd.DataFrame,
    graph_summary_df: pd.DataFrame | None = None,
    exceptions_df: pd.DataFrame | None = None,
    country: str = "MY",
) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Build feature matrix and target variable from labeled entities.

    Parameters
    ----------
    entities_df : Malaysian entities.
    relationships_df : Known relationships.
    parent_entities_df : Parent entities (with country data).
    graph_summary_df : Optional graph features to merge.
    exceptions_df : Optional reporting exception data.

    Returns
    -------
    (X, y, label_encoder) where X is the feature DataFrame, y is the
    bucketed parent country, and label_encoder maps back to country names.
    """
    # Get entities that have known parents
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

    # Bucket rare countries
    country_counts = labeled["parent_country"].value_counts().to_dict()
    labeled["parent_bucket"] = labeled["parent_country"].apply(
        lambda c: _bucket_country(c, country_counts)
    )

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
    n_splits = min(5, min(y.value_counts()))
    n_splits = max(2, n_splits)  # at least 2-fold

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # GradientBoosting
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    gb_scores = cross_val_score(gb, X, y, cv=cv, scoring="accuracy")

    # RandomForest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")

    # Pick the better model
    if gb_scores.mean() >= rf_scores.mean():
        model = gb
        model_name = "GradientBoosting"
        scores = gb_scores
    else:
        model = rf
        model_name = "RandomForest"
        scores = rf_scores

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
    }

    print(f"[INFO] Best model: {model_name} (CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f})", flush=True)
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
            "top3_countries", "top3_probabilities",
        ])

    proba = model.predict_proba(X_new)
    classes = label_encoder.classes_

    predictions = []
    for i in range(len(X_new)):
        prob_row = proba[i]
        sorted_idx = np.argsort(prob_row)[::-1]

        top1_class = classes[sorted_idx[0]]
        top1_prob = prob_row[sorted_idx[0]]

        top3_classes = [classes[j] for j in sorted_idx[:3]]
        top3_probs = [round(float(prob_row[j]), 4) for j in sorted_idx[:3]]

        predictions.append({
            "lei": leis.iloc[i],
            "predicted_parent_country": top1_class,
            "prediction_probability": round(float(top1_prob), 4),
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
    pd.DataFrame([cv_metrics]).to_csv(output_dir / "phase3_cv_metrics.csv", index=False)

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

    return {
        "total_predictions": len(predictions_df),
        "top_predicted_country": predictions_df["predicted_parent_country"].value_counts().index[0],
        "avg_confidence": round(predictions_df["prediction_probability"].mean(), 4),
        "high_confidence_predictions": int((predictions_df["prediction_probability"] >= 0.5).sum()),
        **cv_metrics,
    }
