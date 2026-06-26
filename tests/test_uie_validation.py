from __future__ import annotations

import unittest

import pandas as pd

from src.uie_validation import validate_uie_assignments


class UIEGroundTruthValidationTests(unittest.TestCase):
    def test_validates_exact_top3_and_mismatch_cases(self) -> None:
        ground_truth = pd.DataFrame([
            {"lei": "L1", "ground_truth_uie_country": "US", "ground_truth_uie_name": "Alpha Inc"},
            {"lei": "L2", "ground_truth_uie_country": "CN", "ground_truth_uie_name": "Beta Ltd"},
            {"lei": "L3", "ground_truth_uie_country": "JP"},
        ])
        assignments = pd.DataFrame([
            {
                "lei": "L1",
                "legal_name": "Alpha Malaysia",
                "uie_country": "US",
                "uie_source": "gleif_ultimate_parent",
                "evidence_tier": "A_known_ultimate_parent",
                "uie_confidence": 1.0,
                "top3_countries": "US;GB;DE",
            },
            {
                "lei": "L2",
                "legal_name": "Beta Malaysia",
                "uie_country": "SG",
                "uie_source": "phase3_jurisdiction_model",
                "evidence_tier": "E_model_predicted_jurisdiction",
                "uie_confidence": 0.42,
                "top3_countries": "SG;CN;US",
            },
            {
                "lei": "L3",
                "legal_name": "Gamma Malaysia",
                "uie_country": "EUROPE",
                "uie_source": "phase3_jurisdiction_model",
                "evidence_tier": "E_model_predicted_jurisdiction",
                "uie_confidence": 0.51,
                "top3_countries": "EUROPE;US;GB",
            },
        ])

        detail, summary, confusion = validate_uie_assignments(
            ground_truth,
            assignments,
            country="MY",
        )

        by_lei = detail.set_index("lei")
        self.assertTrue(bool(by_lei.loc["L1", "is_exact_country_match"]))
        self.assertEqual(by_lei.loc["L1", "error_type"], "exact_match")
        self.assertFalse(bool(by_lei.loc["L2", "is_exact_country_match"]))
        self.assertTrue(bool(by_lei.loc["L2", "is_top3_match"]))
        self.assertEqual(by_lei.loc["L2", "error_type"], "top3_only")
        self.assertEqual(by_lei.loc["L3", "error_type"], "broad_bucket_mismatch")
        self.assertEqual(by_lei.loc["L3", "review_recommendation"], "review_model_fallback")

        exact_rate = summary.loc[
            summary["metric"].eq("exact_country_match_rate"),
            "value",
        ].iloc[0]
        self.assertAlmostEqual(float(exact_rate), 1 / 3)
        self.assertEqual(int(confusion["count"].sum()), 3)


if __name__ == "__main__":
    unittest.main()
