import unittest

import pandas as pd

from src.scoring import compute_coverage_score


class ScoringTests(unittest.TestCase):
    def test_compute_coverage_score_handles_empty_frame(self) -> None:
        scored = compute_coverage_score(pd.DataFrame())

        self.assertTrue(scored.empty)
        self.assertIn("coverage_gap_score", scored.columns)
        self.assertIn("coverage_gap_priority_score", scored.columns)
        self.assertIn("reason_flags", scored.columns)

    def test_constant_size_proxy_is_not_large_presence(self) -> None:
        scored = compute_coverage_score(
            pd.DataFrame(
                [
                    {"lei": "A1", "size_proxy": 6.0},
                    {"lei": "A2", "size_proxy": 6.0},
                ]
            )
        )

        self.assertEqual(scored["size_scaled"].tolist(), [0.0, 0.0])
        self.assertNotIn("large_network_presence", ";".join(scored["reason_flags"]))

    def test_inference_signals_contribute_to_score(self) -> None:
        """Entities with inference signals should score higher than those without."""
        scored = compute_coverage_score(
            pd.DataFrame([
                {
                    "lei": "A1", "size_proxy": 1.0,
                    "has_inferred_parent": 1, "in_address_cluster": 1,
                    "is_non_consolidating": 1,
                },
                {
                    "lei": "A2", "size_proxy": 1.0,
                    "has_inferred_parent": 0, "in_address_cluster": 0,
                    "is_non_consolidating": 0,
                },
            ])
        )
        score_a1 = scored.loc[scored["lei"] == "A1", "coverage_gap_score"].iloc[0]
        score_a2 = scored.loc[scored["lei"] == "A2", "coverage_gap_score"].iloc[0]
        priority_a1 = scored.loc[scored["lei"] == "A1", "coverage_gap_priority_score"].iloc[0]
        self.assertGreater(score_a1, score_a2)
        self.assertEqual(score_a1, priority_a1)

    def test_inference_reason_flags(self) -> None:
        scored = compute_coverage_score(
            pd.DataFrame([{
                "lei": "X1", "size_proxy": 0.0,
                "has_inferred_parent": 1, "in_address_cluster": 1,
                "is_non_consolidating": 1,
            }])
        )
        flags = scored.iloc[0]["reason_flags"]
        self.assertIn("inferred_parent_match", flags)
        self.assertIn("shared_address_cluster", flags)
        self.assertIn("non_consolidating_entity", flags)

    def test_missing_inference_columns_default_to_zero(self) -> None:
        """Scoring should work without inference columns present."""
        scored = compute_coverage_score(
            pd.DataFrame([{"lei": "Z1", "size_proxy": 5.0}])
        )
        self.assertEqual(scored.iloc[0]["has_inferred_parent"], 0)
        self.assertEqual(scored.iloc[0]["in_address_cluster"], 0)
        self.assertEqual(scored.iloc[0]["is_non_consolidating"], 0)


if __name__ == "__main__":
    unittest.main()
