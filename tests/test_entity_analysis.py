from __future__ import annotations

import unittest
import pandas as pd
from src.entity_analysis import (
    entity_discovery_summary,
    geographic_concentration,
    hq_vs_legal_mismatch,
    legal_form_distribution,
    category_distribution,
    entity_status_distribution,
    registration_timeline,
    country_structural_profile,
    compare_countries,
    full_structural_report,
)


def _sample_entities() -> pd.DataFrame:
    return pd.DataFrame([
        {"lei": "L1", "legal_name": "ACME SDN BHD", "category": "GENERAL", "legal_form": "8888",
         "entity_status": "ACTIVE", "registered_at": "2020-01-15", "country_legal": "MY",
         "city_legal": "KUALA LUMPUR", "country_hq": "MY", "city_hq": "KUALA LUMPUR"},
        {"lei": "L2", "legal_name": "BETA FUND", "category": "FUND", "legal_form": "9999",
         "entity_status": "ACTIVE", "registered_at": "2021-06-01", "country_legal": "MY",
         "city_legal": "KUALA LUMPUR", "country_hq": "SG", "city_hq": "SINGAPORE"},
        {"lei": "L3", "legal_name": "GAMMA BRANCH", "category": "BRANCH", "legal_form": "8888",
         "entity_status": "INACTIVE", "registered_at": "2019-03-10", "country_legal": "MY",
         "city_legal": "PENANG", "country_hq": "MY", "city_hq": "PENANG"},
    ])


class EntityAnalysisTests(unittest.TestCase):

    def test_discovery_summary(self):
        df = _sample_entities()
        result = entity_discovery_summary(df)
        self.assertEqual(result["total_records"], 3)
        self.assertEqual(result["unique_leis"], 3)
        self.assertEqual(result["active_entities"], 2)
        self.assertEqual(result["inactive_entities"], 1)
        self.assertEqual(result["legal_forms"], 2)

    def test_geographic_concentration(self):
        df = _sample_entities()
        result = geographic_concentration(df)
        self.assertEqual(result.iloc[0]["city_legal"], "Kuala Lumpur")
        self.assertEqual(result.iloc[0]["count"], 2)

    def test_hq_mismatch(self):
        df = _sample_entities()
        result = hq_vs_legal_mismatch(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["lei"], "L2")

    def test_legal_form_distribution(self):
        df = _sample_entities()
        result = legal_form_distribution(df)
        self.assertEqual(result.iloc[0]["legal_form"], "8888")
        self.assertEqual(result.iloc[0]["count"], 2)

    def test_category_distribution(self):
        df = _sample_entities()
        result = category_distribution(df)
        self.assertEqual(len(result), 3)

    def test_entity_status_distribution(self):
        df = _sample_entities()
        result = entity_status_distribution(df)
        self.assertEqual(result.loc[result["entity_status"] == "ACTIVE", "count"].iloc[0], 2)

    def test_registration_timeline(self):
        df = _sample_entities()
        result = registration_timeline(df)
        self.assertGreater(len(result), 0)
        self.assertIn("period", result.columns)

    def test_country_profile(self):
        df = _sample_entities()
        profile = country_structural_profile(df, "MY")
        self.assertEqual(profile["country"], "MY")
        self.assertEqual(profile["total_entities"], 3)
        self.assertEqual(profile["top_city"], "Kuala Lumpur")
        self.assertEqual(profile["hq_mismatch_count"], 1)

    def test_compare_countries(self):
        df_my = _sample_entities()
        df_sg = pd.DataFrame([
            {"lei": "S1", "legal_name": "SG CORP", "category": "GENERAL", "legal_form": "LTD",
             "entity_status": "ACTIVE", "registered_at": "2022-01-01", "country_legal": "SG",
             "city_legal": "SINGAPORE", "country_hq": "SG", "city_hq": "SINGAPORE"},
        ])
        result = compare_countries({"MY": df_my, "SG": df_sg})
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["country"], "MY")  # more entities = first

    def test_full_report_keys(self):
        df = _sample_entities()
        report = full_structural_report(df, "MY")
        expected_keys = {
            "discovery", "geographic_concentration", "hq_vs_legal_mismatch",
            "legal_form_distribution", "category_distribution",
            "entity_status_distribution", "registration_timeline", "profile",
        }
        self.assertEqual(set(report.keys()), expected_keys)

    def test_empty_dataframe_handling(self):
        empty = pd.DataFrame(columns=["lei", "legal_name", "category", "legal_form",
                                       "entity_status", "registered_at", "country_legal",
                                       "city_legal", "country_hq", "city_hq"])
        summary = entity_discovery_summary(empty)
        self.assertEqual(summary["total_records"], 0)

        geo = geographic_concentration(empty)
        self.assertTrue(geo.empty)

        profile = country_structural_profile(empty, "XX")
        self.assertEqual(profile["total_entities"], 0)


if __name__ == "__main__":
    unittest.main()
