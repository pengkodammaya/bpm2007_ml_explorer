from __future__ import annotations

import unittest
import pandas as pd
from src.reporting_exceptions import enrich_with_exception_flags


class ReportingExceptionTests(unittest.TestCase):

    def test_non_consolidating_flag(self):
        entities = pd.DataFrame([
            {"lei": "L1", "legal_name": "ACME SDN BHD"},
            {"lei": "L2", "legal_name": "BETA CORP"},
        ])
        exceptions = pd.DataFrame([
            {"lei": "L1", "exception_reason": "NON_CONSOLIDATING", "exception_reference": None},
        ])
        result = enrich_with_exception_flags(entities, exceptions)
        self.assertEqual(result.loc[result["lei"] == "L1", "is_non_consolidating"].iloc[0], 1)
        self.assertEqual(result.loc[result["lei"] == "L1", "has_exception_filed"].iloc[0], 1)

    def test_no_exception_filed(self):
        entities = pd.DataFrame([
            {"lei": "L1", "legal_name": "ACME SDN BHD"},
        ])
        exceptions = pd.DataFrame(columns=["lei", "exception_reason", "exception_reference"])
        result = enrich_with_exception_flags(entities, exceptions)
        self.assertEqual(result.loc[result["lei"] == "L1", "is_non_consolidating"].iloc[0], 0)
        self.assertEqual(result.loc[result["lei"] == "L1", "has_exception_filed"].iloc[0], 0)

    def test_other_exception_reason(self):
        entities = pd.DataFrame([
            {"lei": "L1", "legal_name": "ACME SDN BHD"},
        ])
        exceptions = pd.DataFrame([
            {"lei": "L1", "exception_reason": "NO_KNOWN_PERSON", "exception_reference": None},
        ])
        result = enrich_with_exception_flags(entities, exceptions)
        self.assertEqual(result.loc[result["lei"] == "L1", "is_non_consolidating"].iloc[0], 0)
        self.assertEqual(result.loc[result["lei"] == "L1", "has_exception_filed"].iloc[0], 1)

    def test_empty_inputs(self):
        entities = pd.DataFrame(columns=["lei", "legal_name"])
        exceptions = pd.DataFrame(columns=["lei", "exception_reason", "exception_reference"])
        result = enrich_with_exception_flags(entities, exceptions)
        self.assertIn("is_non_consolidating", result.columns)
        self.assertIn("has_exception_filed", result.columns)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
