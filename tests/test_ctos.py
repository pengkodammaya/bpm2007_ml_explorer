from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd
from requests.exceptions import ReadTimeout

from src.ctos import (
    fetch_ctos_company_directory,
    load_ctos_company_snapshot,
    match_entities_to_ctos,
    normalize_malaysia_company_name,
    parse_ctos_company_detail,
    parse_ctos_listing,
)


class CTOSNormalizeTests(unittest.TestCase):
    def test_normalizes_malaysia_suffix_variants(self) -> None:
        self.assertEqual(
            normalize_malaysia_company_name("Acme Sendirian Berhad"),
            "ACME SDN BHD",
        )
        self.assertEqual(
            normalize_malaysia_company_name("Acme Berhad."),
            "ACME BHD",
        )


class CTOSParseTests(unittest.TestCase):
    def test_parse_listing_extracts_company_links(self) -> None:
        html = """
        <ul>
          <li><a href="/oneoffreport_api/single-report/malaysia-company/123/ACME-SDN-BHD">ACME SDN BHD</a></li>
          <li><a href="/not-company">Ignore Me</a></li>
        </ul>
        """
        result = parse_ctos_listing(html)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ctos_name"], "ACME SDN BHD")
        self.assertIn("single-report/malaysia-company", result.iloc[0]["ctos_url"])

    def test_parse_detail_extracts_registration_fields(self) -> None:
        html = """
        <div>Company Registration No. 0862380T / 200901019283</div>
        <div>Nature of Business DORMANT - --> Date of Registration 2009-06-28 State null - --> COMPANY DESCRIPTION</div>
        """
        result = parse_ctos_company_detail(html)

        self.assertEqual(result["ctos_registration_no"], "0862380T / 200901019283")
        self.assertEqual(result["ctos_nature_of_business"], "DORMANT")
        self.assertEqual(result["ctos_registration_date"], "2009-06-28")
        self.assertIsNone(result["ctos_state"])


class CTOSFetchTests(unittest.TestCase):
    def test_directory_retries_transient_listing_timeout(self) -> None:
        response = Mock()
        response.text = """
        <a href="/oneoffreport_api/single-report/malaysia-company/123/ACME-SDN-BHD">ACME SDN BHD</a>
        """
        response.raise_for_status.return_value = None

        with patch("src.ctos.requests.get", side_effect=[ReadTimeout("slow"), response]) as get:
            result = fetch_ctos_company_directory(
                letters=["A"],
                pages_per_letter=1,
                sleep_s=0,
                max_retries=2,
                retry_sleep_s=0,
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ctos_listing_letter"], "A")
        self.assertEqual(get.call_count, 2)

    def test_directory_skips_page_after_repeated_listing_timeout(self) -> None:
        with patch("src.ctos.requests.get", side_effect=ReadTimeout("slow")) as get:
            result = fetch_ctos_company_directory(
                letters=["A"],
                pages_per_letter=1,
                sleep_s=0,
                max_retries=2,
                retry_sleep_s=0,
            )

        self.assertTrue(result.empty)
        self.assertEqual(get.call_count, 2)

    def test_directory_stops_letter_after_consecutive_failed_pages(self) -> None:
        response = Mock()
        response.text = """
        <a href="/oneoffreport_api/single-report/malaysia-company/123/BETA-SDN-BHD">BETA SDN BHD</a>
        """
        response.raise_for_status.return_value = None
        empty_response = Mock()
        empty_response.text = ""
        empty_response.raise_for_status.return_value = None

        with patch(
            "src.ctos.requests.get",
            side_effect=[ReadTimeout("a1"), ReadTimeout("a2"), response, empty_response],
        ) as get:
            result = fetch_ctos_company_directory(
                letters=["A", "B"],
                pages_per_letter=3,
                sleep_s=0,
                max_retries=1,
                retry_sleep_s=0,
                max_consecutive_failures=2,
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ctos_listing_letter"], "B")
        self.assertEqual(result.iloc[0]["ctos_listing_page"], 1)
        self.assertEqual(get.call_count, 4)


class CTOSMatchTests(unittest.TestCase):
    def test_exact_normalized_match(self) -> None:
        entities = pd.DataFrame([
            {"lei": "L1", "legal_name": "ACME SENDIRIAN BERHAD"},
        ])
        ctos = pd.DataFrame([
            {
                "ctos_name": "ACME SDN BHD",
                "ctos_url": "https://example.test/acme",
                "ctos_registration_no": "123",
            },
        ])

        matches = match_entities_to_ctos(entities, ctos)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches.iloc[0]["ctos_match_method"], "exact")
        self.assertEqual(matches.iloc[0]["ctos_registered_malaysia"], 1)

    def test_threshold_blocks_weak_match(self) -> None:
        entities = pd.DataFrame([
            {"lei": "L1", "legal_name": "ACME SDN BHD"},
        ])
        ctos = pd.DataFrame([
            {"ctos_name": "TOTALLY DIFFERENT SDN BHD", "ctos_url": "https://example.test/x"},
        ])

        matches = match_entities_to_ctos(entities, ctos, threshold=99)

        self.assertTrue(matches.empty)

    def test_generic_corporation_suffix_does_not_drive_fuzzy_match(self) -> None:
        entities = pd.DataFrame([
            {"lei": "L1", "legal_name": "MAGNUM CORPORATION SDN. BHD."},
        ])
        ctos = pd.DataFrame([
            {"ctos_name": "A & A CORPORATION SDN BHD", "ctos_url": "https://example.test/aa"},
        ])

        matches = match_entities_to_ctos(entities, ctos, threshold=95)

        self.assertTrue(matches.empty)

    def test_activity_word_overlap_does_not_drive_fuzzy_match(self) -> None:
        entities = pd.DataFrame([
            {"lei": "L1", "legal_name": "HONEYWELL ENGINEERING SDN. BHD."},
            {"lei": "L2", "legal_name": "BUNGE AGRIBUSINESS (M) SDN. BHD."},
        ])
        ctos = pd.DataFrame([
            {"ctos_name": "B & B ENGINEERING SDN. BHD.", "ctos_url": "https://example.test/bb"},
            {"ctos_name": "AGRIBUSINESS SDN BHD", "ctos_url": "https://example.test/agri"},
        ])

        matches = match_entities_to_ctos(entities, ctos, threshold=95)

        self.assertTrue(matches.empty)

    def test_strict_fuzzy_allows_minor_token_typo_with_same_leading_brand(self) -> None:
        entities = pd.DataFrame([
            {"lei": "L1", "legal_name": "ACME TECHNOLOGIES SDN. BHD."},
        ])
        ctos = pd.DataFrame([
            {"ctos_name": "ACME TECHNOLOGY SDN BHD", "ctos_url": "https://example.test/acme"},
        ])

        matches = match_entities_to_ctos(entities, ctos, threshold=90)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches.iloc[0]["ctos_match_method"], "fuzzy")

    def test_exact_only_disables_fuzzy_matching(self) -> None:
        entities = pd.DataFrame([
            {"lei": "L1", "legal_name": "ACME TECHNOLOGIES SDN. BHD."},
        ])
        ctos = pd.DataFrame([
            {"ctos_name": "ACME TECHNOLOGY SDN BHD", "ctos_url": "https://example.test/acme"},
        ])

        matches = match_entities_to_ctos(entities, ctos, threshold=90, fuzzy=False)

        self.assertTrue(matches.empty)


class CTOSLocalSnapshotTests(unittest.TestCase):
    def test_loads_user_ctos_crawl_snapshot(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/my_companies.parquet"
            pd.DataFrame([
                {"name": "ACME SDN BHD", "_dataset": "company", "_letter": "a", "_page": 1},
                {"name": "ACME SDN BHD", "_dataset": "company", "_letter": "a", "_page": 1},
            ]).to_parquet(path, index=False)

            result = load_ctos_company_snapshot(path, use_duckdb=False)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ctos_name"], "ACME SDN BHD")
        self.assertEqual(result.iloc[0]["ctos_listing_letter"], "a")
        self.assertEqual(result.iloc[0]["ctos_listing_page"], 1)
        self.assertIn("ctos_registration_no", result.columns)


if __name__ == "__main__":
    unittest.main()
