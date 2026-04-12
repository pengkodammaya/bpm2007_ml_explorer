from __future__ import annotations

import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.jurisdiction_predictor import (
    _bucket_country,
    prepare_training_features,
    _build_features,
    train_jurisdiction_model,
    predict_parent_jurisdiction,
    phase3_summary,
)


def _sample_entities():
    return pd.DataFrame([
        {"lei": "L1", "legal_name": "ACME MALAYSIA SDN BHD", "category": "GENERAL", "legal_form": "XEOV", "entity_status": "ACTIVE", "registered_at": "2020-01-15", "country_legal": "MY", "city_legal": "KUALA LUMPUR", "country_hq": "MY", "city_hq": "KUALA LUMPUR"},
        {"lei": "L2", "legal_name": "BETA FUND BERHAD", "category": "FUND", "legal_form": "7OYN", "entity_status": "ACTIVE", "registered_at": "2019-06-01", "country_legal": "MY", "city_legal": "KUALA LUMPUR", "country_hq": "MY", "city_hq": "KUALA LUMPUR"},
        {"lei": "L3", "legal_name": "GAMMA SDN BHD", "category": "GENERAL", "legal_form": "XEOV", "entity_status": "ACTIVE", "registered_at": "2021-03-10", "country_legal": "MY", "city_legal": "PENANG", "country_hq": "MY", "city_hq": "PENANG"},
        {"lei": "L4", "legal_name": "DELTA HOLDINGS BERHAD", "category": "GENERAL", "legal_form": "7OYN", "entity_status": "ACTIVE", "registered_at": "2018-08-20", "country_legal": "MY", "city_legal": "JOHOR BAHRU", "country_hq": "MY", "city_hq": "JOHOR BAHRU"},
        {"lei": "L5", "legal_name": "EPSILON TECH SDN BHD", "category": "GENERAL", "legal_form": "XEOV", "entity_status": "ACTIVE", "registered_at": "2022-01-01", "country_legal": "MY", "city_legal": "KUALA LUMPUR", "country_hq": "MY", "city_hq": "KUALA LUMPUR"},
    ])


def _sample_relationships():
    return pd.DataFrame([
        {"source_lei": "L1", "target_lei": "P1"},
        {"source_lei": "L2", "target_lei": "P2"},
        {"source_lei": "L3", "target_lei": "P1"},
        {"source_lei": "L4", "target_lei": "P3"},
        {"source_lei": "L5", "target_lei": "P1"},
    ])


def _sample_parents():
    return pd.DataFrame([
        {"lei": "P1", "country_legal": "US"},
        {"lei": "P2", "country_legal": "GB"},
        {"lei": "P3", "country_legal": "US"},
    ])


class BucketCountryTests(unittest.TestCase):

    def test_high_count_country_keeps_itself(self):
        counts = {"US": 10, "GB": 3}
        self.assertEqual(_bucket_country("US", counts), "US")

    def test_low_count_country_bucketed(self):
        counts = {"NZ": 1}
        self.assertEqual(_bucket_country("NZ", counts), "ASIA_OTHER")

    def test_unknown_country_becomes_other(self):
        counts = {"XX": 1}
        self.assertEqual(_bucket_country("XX", counts), "OTHER")


class BuildFeaturesTests(unittest.TestCase):

    def test_feature_columns_present(self):
        entities = _sample_entities()
        leis = entities["lei"]
        features = _build_features(leis, entities)
        self.assertIn("lei", features.columns)
        self.assertIn("city_frequency", features.columns)
        self.assertIn("registration_year", features.columns)
        self.assertIn("name_token_count", features.columns)
        # Country-specific features for MY (default)
        self.assertIn("has_sdn_bhd", features.columns)
        self.assertIn("has_berhad", features.columns)
        self.assertIn("has_local_country", features.columns)

    def test_feature_shape(self):
        entities = _sample_entities()
        features = _build_features(entities["lei"], entities)
        self.assertEqual(len(features), 5)


class PrepareTrainingTests(unittest.TestCase):

    def test_produces_features_and_labels(self):
        X, y, le = prepare_training_features(
            _sample_entities(),
            _sample_relationships(),
            _sample_parents(),
        )
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))

    def test_empty_relationships(self):
        X, y, le = prepare_training_features(
            _sample_entities(),
            pd.DataFrame(columns=["source_lei", "target_lei"]),
            _sample_parents(),
        )
        self.assertTrue(X.empty)


class TrainModelTests(unittest.TestCase):

    def test_trains_and_returns_metrics(self):
        X, y, le = prepare_training_features(
            _sample_entities(),
            _sample_relationships(),
            _sample_parents(),
        )
        if X.empty:
            self.skipTest("No training data")
        model, metrics = train_jurisdiction_model(X, y)
        self.assertIn("cv_accuracy_mean", metrics)
        self.assertIn("model_type", metrics)
        self.assertIsNotNone(model)


class PredictTests(unittest.TestCase):

    def test_predictions_have_correct_columns(self):
        X, y, le = prepare_training_features(
            _sample_entities(),
            _sample_relationships(),
            _sample_parents(),
        )
        if X.empty:
            self.skipTest("No training data")
        model, _ = train_jurisdiction_model(X, y)
        preds = predict_parent_jurisdiction(model, le, X, pd.Series(["L1", "L2", "L3", "L4", "L5"][:len(X)]))
        self.assertIn("predicted_parent_country", preds.columns)
        self.assertIn("prediction_probability", preds.columns)
        self.assertIn("top3_countries", preds.columns)

    def test_empty_input(self):
        le = LabelEncoder()
        le.fit(["US", "GB"])
        preds = predict_parent_jurisdiction(None, le, pd.DataFrame(), pd.Series(dtype=str))
        self.assertTrue(preds.empty)


class SummaryTests(unittest.TestCase):

    def test_summary_with_data(self):
        preds = pd.DataFrame([
            {"lei": "L1", "predicted_parent_country": "US", "prediction_probability": 0.7, "top3_countries": "US;GB;SG", "top3_probabilities": "0.7;0.2;0.1"},
        ])
        metrics = {"model_type": "GradientBoosting", "cv_accuracy_mean": 0.85}
        summary = phase3_summary(preds, metrics)
        self.assertEqual(summary["total_predictions"], 1)
        self.assertEqual(summary["top_predicted_country"], "US")

    def test_empty_summary(self):
        preds = pd.DataFrame(columns=["lei", "predicted_parent_country", "prediction_probability", "top3_countries", "top3_probabilities"])
        summary = phase3_summary(preds, {"model_type": "none"})
        self.assertEqual(summary["total_predictions"], 0)


if __name__ == "__main__":
    unittest.main()
