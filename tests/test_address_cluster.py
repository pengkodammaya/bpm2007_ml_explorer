from __future__ import annotations

import unittest
import pandas as pd
from src.address_cluster import (
    normalize_address,
    build_address_key,
    cluster_by_address,
    _is_office_hotel,
    infer_shared_parent_from_cluster,
    phase2_summary,
)


class NormalizeAddressTests(unittest.TestCase):

    def test_uppercase_and_strip(self):
        self.assertEqual(normalize_address("  jalan bukit  "), "JALAN BUKIT")

    def test_abbreviation_expansion(self):
        result = normalize_address("JLN. SULTAN")
        self.assertIn("JALAN", result)
        self.assertNotIn("JLN", result)

    def test_taman_expansion(self):
        result = normalize_address("TMN MELAWATI")
        self.assertIn("TAMAN", result)

    def test_none_input(self):
        self.assertEqual(normalize_address(None), "")

    def test_empty_input(self):
        self.assertEqual(normalize_address(""), "")

    def test_punctuation_removed(self):
        result = normalize_address("NO. 6, CHANGKAT SEMANTAN")
        self.assertNotIn(",", result)


class ClusterTests(unittest.TestCase):

    def _sample_addresses(self):
        return pd.DataFrame([
            {"lei": "L1", "address_line1": "SUITE 14-3, WISMA UOA", "address_line2": None, "postal_code": "50490", "region": "MY-14", "city": "KL", "country": "MY"},
            {"lei": "L2", "address_line1": "SUITE 14-3, WISMA UOA", "address_line2": None, "postal_code": "50490", "region": "MY-14", "city": "KL", "country": "MY"},
            {"lei": "L3", "address_line1": "SUITE 14-3, WISMA UOA", "address_line2": None, "postal_code": "50490", "region": "MY-14", "city": "KL", "country": "MY"},
            {"lei": "L4", "address_line1": "MENARA HONG LEONG", "address_line2": None, "postal_code": "50200", "region": "MY-14", "city": "KL", "country": "MY"},
        ])

    def test_cluster_groups_same_address(self):
        result = cluster_by_address(self._sample_addresses(), min_cluster_size=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["address_cluster_id"].nunique(), 1)

    def test_min_cluster_size_respected(self):
        result = cluster_by_address(self._sample_addresses(), min_cluster_size=4)
        # Only 3 at the same address, so no clusters with min=4
        self.assertEqual(len(result), 0)

    def test_office_hotel_labeling(self):
        result = cluster_by_address(self._sample_addresses(), min_cluster_size=3)
        # WISMA UOA is in KNOWN_OFFICE_HOTELS
        self.assertTrue(result["is_office_hotel"].all())

    def test_empty_input(self):
        result = cluster_by_address(pd.DataFrame(columns=["lei", "address_line1", "address_line2", "postal_code", "region", "city", "country"]))
        self.assertTrue(result.empty)

    def test_labuan_office_hotel(self):
        self.assertTrue(_is_office_hotel("LEVEL 15 MAIN OFFICE TOWER LABUAN|87000"))


class InferSharedParentTests(unittest.TestCase):

    def test_propagates_parent_within_cluster(self):
        clusters = pd.DataFrame([
            {"lei": "L1", "address_cluster_id": 0, "cluster_size": 3, "cluster_address_key": "X|123", "is_office_hotel": False},
            {"lei": "L2", "address_cluster_id": 0, "cluster_size": 3, "cluster_address_key": "X|123", "is_office_hotel": False},
            {"lei": "L3", "address_cluster_id": 0, "cluster_size": 3, "cluster_address_key": "X|123", "is_office_hotel": False},
        ])
        relationships = pd.DataFrame([
            {"source_lei": "L1", "target_lei": "P1"},
        ])
        result = infer_shared_parent_from_cluster(clusters, relationships)
        # L2 and L3 should get inferred parent P1
        self.assertEqual(len(result), 2)
        inferred_leis = set(result["lei"])
        self.assertIn("L2", inferred_leis)
        self.assertIn("L3", inferred_leis)

    def test_no_anchor_no_inference(self):
        clusters = pd.DataFrame([
            {"lei": "L1", "address_cluster_id": 0, "cluster_size": 3, "cluster_address_key": "X|123", "is_office_hotel": False},
            {"lei": "L2", "address_cluster_id": 0, "cluster_size": 3, "cluster_address_key": "X|123", "is_office_hotel": False},
        ])
        relationships = pd.DataFrame([
            {"source_lei": "L99", "target_lei": "P1"},  # Not in cluster
        ])
        result = infer_shared_parent_from_cluster(clusters, relationships)
        self.assertEqual(len(result), 0)

    def test_empty_inputs(self):
        result = infer_shared_parent_from_cluster(
            pd.DataFrame(columns=["lei", "address_cluster_id", "cluster_size", "cluster_address_key", "is_office_hotel"]),
            pd.DataFrame(columns=["source_lei", "target_lei"]),
        )
        self.assertTrue(result.empty)


class SummaryTests(unittest.TestCase):

    def test_summary_stats(self):
        clusters = pd.DataFrame([
            {"lei": "L1", "address_cluster_id": 0, "cluster_size": 3, "cluster_address_key": "X|123", "is_office_hotel": True},
            {"lei": "L2", "address_cluster_id": 0, "cluster_size": 3, "cluster_address_key": "X|123", "is_office_hotel": True},
            {"lei": "L3", "address_cluster_id": 0, "cluster_size": 3, "cluster_address_key": "X|123", "is_office_hotel": True},
        ])
        summary = phase2_summary(clusters)
        self.assertEqual(summary["total_clustered_entities"], 3)
        self.assertEqual(summary["num_clusters"], 1)
        self.assertEqual(summary["office_hotel_entities"], 3)

    def test_empty_summary(self):
        summary = phase2_summary(pd.DataFrame(columns=["lei", "address_cluster_id", "cluster_size", "cluster_address_key", "is_office_hotel"]))
        self.assertEqual(summary["total_clustered_entities"], 0)


if __name__ == "__main__":
    unittest.main()
