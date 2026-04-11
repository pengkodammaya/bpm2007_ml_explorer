import unittest

import pandas as pd

from src.scoring import compute_coverage_score


class ScoringTests(unittest.TestCase):
    def test_compute_coverage_score_handles_empty_frame(self) -> None:
        scored = compute_coverage_score(pd.DataFrame())

        self.assertTrue(scored.empty)
        self.assertIn("coverage_gap_score", scored.columns)
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


if __name__ == "__main__":
    unittest.main()
