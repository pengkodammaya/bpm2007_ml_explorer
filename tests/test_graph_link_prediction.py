from __future__ import annotations

import unittest
import pandas as pd
import networkx as nx
from src.graph_link_prediction import (
    assess_graph_readiness,
    compute_pair_features,
    generate_training_pairs,
    train_link_predictor,
    predict_links,
    propagate_jurisdiction_labels,
    phase4_summary,
)


def _build_test_graph():
    """Build a small ownership graph for testing."""
    g = nx.DiGraph()
    # Domestic entities
    for lei in ["D1", "D2", "D3", "D4", "D5"]:
        g.add_node(lei, country="MY", legal_name=f"Entity {lei}")
    # Parent entities
    for lei, country in [("P1", "US"), ("P2", "GB"), ("P3", "SG")]:
        g.add_node(lei, country=country, legal_name=f"Parent {lei}")
    # Known edges
    g.add_edge("D1", "P1", relationship_type="direct_parent")
    g.add_edge("D2", "P1", relationship_type="direct_parent")
    g.add_edge("D3", "P2", relationship_type="direct_parent")
    return g


def _test_entities():
    return pd.DataFrame([
        {"lei": "D1"}, {"lei": "D2"}, {"lei": "D3"},
        {"lei": "D4"}, {"lei": "D5"},
    ])


def _test_relationships():
    return pd.DataFrame([
        {"source_lei": "D1", "target_lei": "P1", "relationship_type": "direct_parent"},
        {"source_lei": "D2", "target_lei": "P1", "relationship_type": "direct_parent"},
        {"source_lei": "D3", "target_lei": "P2", "relationship_type": "direct_parent"},
    ])


def _test_parents():
    return pd.DataFrame([
        {"lei": "P1", "country_legal": "US"},
        {"lei": "P2", "country_legal": "GB"},
        {"lei": "P3", "country_legal": "SG"},
    ])


class AssessReadinessTests(unittest.TestCase):

    def test_sparse_graph_not_ready(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        entities = pd.DataFrame([{"lei": "A"}])
        rels = pd.DataFrame([{"source_lei": "A", "target_lei": "B"}])
        result = assess_graph_readiness(g, rels, entities, min_edges=10)
        self.assertFalse(result["ready"])

    def test_dense_enough_graph_ready(self):
        g = _build_test_graph()
        result = assess_graph_readiness(
            g, _test_relationships(), _test_entities(),
            min_edges=2, min_labeled_ratio=0.01, min_parent_nodes=1,
        )
        self.assertTrue(result["ready"])
        self.assertEqual(result["n_edges"], 3)
        self.assertGreater(result["labeled_ratio"], 0)

    def test_empty_graph(self):
        g = nx.DiGraph()
        entities = pd.DataFrame(columns=["lei"])
        rels = pd.DataFrame(columns=["source_lei", "target_lei"])
        result = assess_graph_readiness(g, rels, entities)
        self.assertFalse(result["ready"])
        self.assertEqual(result["n_nodes"], 0)


class ComputePairFeaturesTests(unittest.TestCase):

    def test_feature_columns_present(self):
        g = _build_test_graph()
        features = compute_pair_features(g, [("D1", "P1"), ("D4", "P2")])
        self.assertIn("jaccard_coefficient", features.columns)
        self.assertIn("adamic_adar", features.columns)
        self.assertIn("preferential_attachment", features.columns)
        self.assertIn("common_neighbors", features.columns)
        self.assertIn("same_component", features.columns)
        self.assertEqual(len(features), 2)

    def test_connected_pair_has_nonzero_features(self):
        g = _build_test_graph()
        features = compute_pair_features(g, [("D1", "P1")])
        # D1 is directly connected to P1, so same_component should be 1
        self.assertEqual(features.iloc[0]["same_component"], 1)

    def test_empty_pairs(self):
        g = _build_test_graph()
        features = compute_pair_features(g, [])
        self.assertTrue(features.empty)


class GenerateTrainingPairsTests(unittest.TestCase):

    def test_generates_positive_and_negative(self):
        pairs = generate_training_pairs(
            _test_relationships(),
            ["D1", "D2", "D3", "D4", "D5"],
            ["P1", "P2", "P3"],
            neg_ratio=2,
        )
        self.assertGreater(len(pairs), 0)
        self.assertEqual(pairs["label"].sum(), 3)  # 3 positive
        self.assertEqual((pairs["label"] == 0).sum(), 6)  # 3 * 2 negative

    def test_empty_relationships(self):
        pairs = generate_training_pairs(
            pd.DataFrame(columns=["source_lei", "target_lei"]),
            ["D1"], ["P1"],
        )
        self.assertTrue(pairs.empty)


class TrainLinkPredictorTests(unittest.TestCase):

    def test_trains_and_returns_metrics(self):
        g = _build_test_graph()
        pairs = generate_training_pairs(
            _test_relationships(),
            ["D1", "D2", "D3", "D4", "D5"],
            ["P1", "P2", "P3"],
            neg_ratio=3,
        )
        model, metrics, scaler = train_link_predictor(g, pairs)
        self.assertIsNotNone(model)
        self.assertIn("cv_auc_mean", metrics)
        self.assertIn("model_type", metrics)

    def test_empty_training_data(self):
        g = _build_test_graph()
        model, metrics, _ = train_link_predictor(
            g, pd.DataFrame(columns=["source_lei", "target_lei", "label"])
        )
        self.assertIsNone(model)
        self.assertIn("error", metrics)


class PredictLinksTests(unittest.TestCase):

    def test_predictions_have_correct_columns(self):
        g = _build_test_graph()
        pairs = generate_training_pairs(
            _test_relationships(),
            ["D1", "D2", "D3", "D4", "D5"],
            ["P1", "P2", "P3"],
            neg_ratio=3,
        )
        model, _, scaler = train_link_predictor(g, pairs)
        preds = predict_links(
            model, scaler, g,
            [("D4", "P1"), ("D5", "P2")],
            min_probability=0.0,
        )
        self.assertIn("source_lei", preds.columns)
        self.assertIn("target_lei", preds.columns)
        self.assertIn("link_probability", preds.columns)

    def test_no_model_returns_empty(self):
        preds = predict_links(None, None, nx.DiGraph(), [])
        self.assertTrue(preds.empty)


class PropagateLabelsTests(unittest.TestCase):

    def test_propagates_to_connected_nodes(self):
        g = _build_test_graph()
        labels = {"P1": "US", "P2": "GB"}
        result = propagate_jurisdiction_labels(g, labels)
        # D1 and D2 are connected to P1 (US), D3 to P2 (GB)
        # D4 and D5 are isolated, should not appear
        self.assertGreater(len(result), 0)
        self.assertIn("propagated_country", result.columns)
        self.assertIn("confidence", result.columns)
        self.assertIn("hops", result.columns)

    def test_empty_labels(self):
        g = _build_test_graph()
        result = propagate_jurisdiction_labels(g, {})
        self.assertTrue(result.empty)

    def test_empty_graph(self):
        result = propagate_jurisdiction_labels(nx.DiGraph(), {"P1": "US"})
        self.assertTrue(result.empty)


class SummaryTests(unittest.TestCase):

    def test_skipped_summary(self):
        results = {
            "readiness": {"ready": False, "n_nodes": 5, "n_edges": 1,
                          "labeled_ratio": 0.01, "n_parent_nodes": 0},
            "link_predictions": pd.DataFrame(),
            "propagated_labels": pd.DataFrame(),
            "metrics": {"skipped": True, "reason": "graph_not_ready"},
            "feature_importance": pd.DataFrame(),
        }
        s = phase4_summary(results)
        self.assertEqual(s["status"], "skipped")

    def test_completed_summary(self):
        results = {
            "readiness": {"ready": True, "n_nodes": 100, "n_edges": 50,
                          "labeled_ratio": 0.10, "n_parent_nodes": 20},
            "link_predictions": pd.DataFrame([
                {"source_lei": "D4", "target_lei": "P1",
                 "link_probability": 0.7, "predicted_parent_country": "US", "rank": 1},
            ]),
            "propagated_labels": pd.DataFrame([
                {"lei": "D4", "propagated_country": "US", "confidence": 0.5, "hops": 2},
            ]),
            "metrics": {"model_type": "GradientBoosting", "cv_auc_mean": 0.85},
            "feature_importance": pd.DataFrame(),
        }
        s = phase4_summary(results)
        self.assertEqual(s["status"], "completed")
        self.assertEqual(s["link_predictions_count"], 1)
        self.assertEqual(s["propagated_labels_count"], 1)


if __name__ == "__main__":
    unittest.main()
