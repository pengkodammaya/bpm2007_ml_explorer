from __future__ import annotations

import unittest
import pandas as pd
from src.name_inference import (
    extract_parent_brand_tokens,
    fuzzy_match_names,
    build_phase1_inferred_edges,
    _extract_brand_token,
    phase1_summary,
)


class BrandTokenExtractionTests(unittest.TestCase):

    def test_strips_simple_suffix(self):
        self.assertEqual(_extract_brand_token("NESTLE S.A."), "NESTLE")

    def test_strips_inc(self):
        token = _extract_brand_token("MONDELEZ INTERNATIONAL, INC.")
        self.assertIn("MONDELEZ", token)
        self.assertNotIn("INC", token)

    def test_strips_gmbh(self):
        token = _extract_brand_token("SIEMENS GMBH")
        self.assertEqual(token, "SIEMENS")

    def test_multi_word_brand(self):
        token = _extract_brand_token("DEUTSCHE BANK AG")
        self.assertEqual(token, "DEUTSCHE BANK")

    def test_strips_plc(self):
        token = _extract_brand_token("SHELL PLC")
        self.assertEqual(token, "SHELL")

    def test_strips_sdn_bhd(self):
        token = _extract_brand_token("ACME SDN BHD")
        self.assertEqual(token, "ACME")

    def test_extract_parent_brand_tokens_dataframe(self):
        parents = pd.DataFrame([
            {"lei": "P1", "legal_name": "NESTLE S.A."},
            {"lei": "P2", "legal_name": "SHELL PLC"},
        ])
        result = extract_parent_brand_tokens(parents)
        self.assertEqual(len(result), 2)
        tokens = set(result["brand_token"])
        self.assertIn("NESTLE", tokens)
        self.assertIn("SHELL", tokens)

    def test_short_brand_token_filtered(self):
        parents = pd.DataFrame([
            {"lei": "P1", "legal_name": "AB LTD"},  # "AB" is too short
        ])
        result = extract_parent_brand_tokens(parents)
        self.assertEqual(len(result), 0)

    def test_empty_input(self):
        result = extract_parent_brand_tokens(pd.DataFrame(columns=["lei", "legal_name"]))
        self.assertTrue(result.empty)


class FuzzyMatchTests(unittest.TestCase):

    def _brands(self):
        return pd.DataFrame([
            {"parent_lei": "P1", "parent_name": "NESTLE S.A.", "brand_token": "NESTLE"},
            {"parent_lei": "P2", "parent_name": "SIEMENS AG", "brand_token": "SIEMENS"},
        ])

    def test_obvious_subsidiary_match(self):
        entities = pd.DataFrame([
            {"lei": "E1", "legal_name": "NESTLE MANUFACTURING (MALAYSIA) SDN. BHD."},
        ])
        result = fuzzy_match_names(entities, self._brands(), threshold=80)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["matched_parent_lei"], "P1")
        self.assertGreaterEqual(result.iloc[0]["match_score"], 80)

    def test_excludes_known_parents(self):
        entities = pd.DataFrame([
            {"lei": "E1", "legal_name": "NESTLE MANUFACTURING (MALAYSIA) SDN. BHD."},
        ])
        result = fuzzy_match_names(entities, self._brands(), known_leis={"E1"}, threshold=80)
        self.assertEqual(len(result), 0)

    def test_threshold_filtering(self):
        entities = pd.DataFrame([
            {"lei": "E1", "legal_name": "COMPLETELY UNRELATED COMPANY SDN BHD"},
        ])
        result = fuzzy_match_names(entities, self._brands(), threshold=80)
        self.assertEqual(len(result), 0)

    def test_empty_entities(self):
        result = fuzzy_match_names(
            pd.DataFrame(columns=["lei", "legal_name"]),
            self._brands(),
            threshold=80,
        )
        self.assertTrue(result.empty)

    def test_empty_brands(self):
        entities = pd.DataFrame([{"lei": "E1", "legal_name": "ACME SDN BHD"}])
        result = fuzzy_match_names(
            entities,
            pd.DataFrame(columns=["parent_lei", "parent_name", "brand_token"]),
            threshold=80,
        )
        self.assertTrue(result.empty)


class InferredEdgeTests(unittest.TestCase):

    def test_builds_edges(self):
        matches = pd.DataFrame([
            {"lei": "E1", "matched_parent_lei": "P1", "brand_token": "NESTLE",
             "match_score": 95, "match_field": "legal_name"},
        ])
        edges = build_phase1_inferred_edges(matches, min_score=80)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges.iloc[0]["source_lei"], "E1")
        self.assertEqual(edges.iloc[0]["target_lei"], "P1")
        self.assertEqual(edges.iloc[0]["relationship_type"], "inferred_name_match")
        self.assertEqual(edges.iloc[0]["relationship_status"], "INFERRED")

    def test_filters_by_min_score(self):
        matches = pd.DataFrame([
            {"lei": "E1", "matched_parent_lei": "P1", "brand_token": "NESTLE",
             "match_score": 75, "match_field": "legal_name"},
        ])
        edges = build_phase1_inferred_edges(matches, min_score=80)
        self.assertEqual(len(edges), 0)

    def test_empty_matches(self):
        edges = build_phase1_inferred_edges(
            pd.DataFrame(columns=["lei", "matched_parent_lei", "brand_token", "match_score", "match_field"]),
        )
        self.assertTrue(edges.empty)


class SummaryTests(unittest.TestCase):

    def test_summary_stats(self):
        matches = pd.DataFrame([
            {"lei": "E1", "matched_parent_lei": "P1", "brand_token": "NESTLE", "match_score": 95, "match_field": "legal_name"},
            {"lei": "E2", "matched_parent_lei": "P1", "brand_token": "NESTLE", "match_score": 85, "match_field": "legal_name"},
            {"lei": "E3", "matched_parent_lei": "P2", "brand_token": "SHELL", "match_score": 82, "match_field": "legal_name"},
        ])
        summary = phase1_summary(matches)
        self.assertEqual(summary["total_matches"], 3)
        self.assertEqual(summary["high_confidence"], 1)
        self.assertEqual(summary["medium_confidence"], 2)
        self.assertEqual(summary["unique_parent_brands"], 2)

    def test_empty_summary(self):
        summary = phase1_summary(pd.DataFrame(columns=["lei", "matched_parent_lei", "brand_token", "match_score", "match_field"]))
        self.assertEqual(summary["total_matches"], 0)


if __name__ == "__main__":
    unittest.main()
