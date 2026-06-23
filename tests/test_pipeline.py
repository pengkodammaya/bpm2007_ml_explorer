from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.pipeline import (
    _foreign_parent_flags,
    _should_use_cached_file,
    build_gleif_search_benchmark,
    build_uie_assignments,
    build_uie_review_targets,
)


class PipelineTests(unittest.TestCase):
    def test_should_use_cached_file_respects_force_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.parquet"

            self.assertFalse(_should_use_cached_file(path))

            path.write_bytes(b"cached")
            self.assertTrue(_should_use_cached_file(path))
            self.assertFalse(_should_use_cached_file(path, force_refresh=True))

    def test_foreign_parent_flags_source_entities(self) -> None:
        entities = pd.DataFrame(
            [
                {"lei": "A1", "country_legal": "MY"},
                {"lei": "A2", "country_legal": "MY"},
                {"lei": "P1", "country_legal": "SG"},
                {"lei": "P2", "country_legal": "MY"},
            ]
        )
        relationships = pd.DataFrame(
            [
                {"source_lei": "A1", "target_lei": "P1", "relationship_type": "direct_parent"},
                {"source_lei": "A2", "target_lei": "P2", "relationship_type": "ultimate_parent"},
            ]
        )

        flags = _foreign_parent_flags(relationships, entities)

        self.assertEqual(flags.to_dict("records"), [{"lei": "A1", "foreign_parent": 1}])

    def test_foreign_parent_flags_handles_no_relationships(self) -> None:
        flags = _foreign_parent_flags(pd.DataFrame(), pd.DataFrame())

        self.assertTrue(flags.empty)
        self.assertEqual(list(flags.columns), ["lei", "foreign_parent"])

    def test_uie_assignment_prefers_known_ultimate_parent(self) -> None:
        domestic = pd.DataFrame([
            {"lei": "D1", "legal_name": "Domestic One", "country_legal": "MY"},
        ])
        all_entities = pd.DataFrame([
            {"lei": "D1", "legal_name": "Domestic One", "country_legal": "MY"},
            {"lei": "P1", "legal_name": "Direct Parent", "country_legal": "SG"},
            {"lei": "U1", "legal_name": "Ultimate Parent", "country_legal": "US"},
        ])
        relationships = pd.DataFrame([
            {"source_lei": "D1", "target_lei": "P1", "relationship_type": "direct_parent"},
            {"source_lei": "D1", "target_lei": "U1", "relationship_type": "ultimate_parent"},
        ])

        assigned = build_uie_assignments(domestic, relationships, all_entities, country="MY")
        row = assigned.iloc[0]

        self.assertEqual(row["uie_lei"], "U1")
        self.assertEqual(row["uie_country"], "US")
        self.assertEqual(row["uie_source"], "gleif_ultimate_parent")
        self.assertEqual(row["evidence_tier"], "A_known_ultimate_parent")

    def test_uie_assignment_climbs_upstream_sg_parent_chain(self) -> None:
        domestic = pd.DataFrame([
            {"lei": "D1", "legal_name": "Malaysia Investee", "country_legal": "MY"},
        ])
        all_entities = pd.DataFrame([
            {"lei": "D1", "legal_name": "Malaysia Investee", "country_legal": "MY"},
            {"lei": "SG1", "legal_name": "Singapore Holdco", "country_legal": "SG"},
            {"lei": "U1", "legal_name": "Global Ultimate", "country_legal": "US"},
        ])
        relationships = pd.DataFrame([
            {"source_lei": "D1", "target_lei": "SG1", "relationship_type": "direct_parent"},
        ])
        chain_relationships = pd.DataFrame([
            {"source_lei": "D1", "target_lei": "SG1", "relationship_type": "direct_parent"},
            {"source_lei": "SG1", "target_lei": "U1", "relationship_type": "ultimate_parent"},
        ])

        assigned = build_uie_assignments(
            domestic,
            relationships,
            all_entities,
            country="MY",
            chain_relationships=chain_relationships,
        )
        row = assigned.iloc[0]

        self.assertEqual(row["direct_parent_country"], "SG")
        self.assertEqual(row["chain_parent_country"], "US")
        self.assertEqual(row["uie_lei"], "U1")
        self.assertEqual(row["uie_country"], "US")
        self.assertEqual(row["uie_source"], "gleif_upstream_ultimate_parent")
        self.assertEqual(row["evidence_tier"], "A_known_upstream_ultimate_parent")
        self.assertEqual(row["is_known_uie"], 1)

    def test_uie_assignment_uses_name_match_before_model_fallback(self) -> None:
        domestic = pd.DataFrame([
            {"lei": "D1", "legal_name": "Allianz Malaysia", "country_legal": "MY"},
        ])
        all_entities = pd.DataFrame([
            {"lei": "D1", "legal_name": "Allianz Malaysia", "country_legal": "MY"},
            {"lei": "P1", "legal_name": "Allianz SE", "country_legal": "DE"},
        ])
        name_matches = pd.DataFrame([
            {
                "lei": "D1",
                "matched_parent_lei": "P1",
                "brand_token": "ALLIANZ",
                "match_score": 100,
            },
        ])
        predictions = pd.DataFrame([
            {
                "lei": "D1",
                "predicted_parent_country": "US",
                "prediction_probability": 0.8,
            },
        ])

        assigned = build_uie_assignments(
            domestic,
            pd.DataFrame(),
            all_entities,
            country="MY",
            name_matches=name_matches,
            jurisdiction_predictions=predictions,
        )
        row = assigned.iloc[0]

        self.assertEqual(row["uie_country"], "DE")
        self.assertEqual(row["uie_source"], "phase1_name_match")
        self.assertEqual(row["evidence_tier"], "C_name_match_high")

    def test_uie_assignment_uses_model_when_no_parent_evidence(self) -> None:
        domestic = pd.DataFrame([
            {"lei": "D1", "legal_name": "Domestic One", "country_legal": "MY"},
        ])
        predictions = pd.DataFrame([
            {
                "lei": "D1",
                "predicted_parent_country": "SG",
                "prediction_probability": 0.42,
            },
        ])

        assigned = build_uie_assignments(
            domestic,
            pd.DataFrame(),
            domestic,
            country="MY",
            jurisdiction_predictions=predictions,
        )
        row = assigned.iloc[0]

        self.assertEqual(row["uie_country"], "SG")
        self.assertEqual(row["uie_source"], "phase3_jurisdiction_model")
        self.assertEqual(row["evidence_tier"], "E_model_predicted_jurisdiction")

    def test_uie_assignment_prefers_manual_override_over_model(self) -> None:
        domestic = pd.DataFrame([
            {"lei": "D1", "legal_name": "Strategic Domestic", "country_legal": "MY"},
        ])
        predictions = pd.DataFrame([
            {
                "lei": "D1",
                "predicted_parent_country": "US",
                "prediction_probability": 0.9,
            },
        ])
        overrides = pd.DataFrame([
            {
                "lei": "D1",
                "uie_name": "Government of Malaysia",
                "uie_country": "MY",
                "uie_source": "manual_verified",
                "evidence_tier": "A_manual_verified_uie",
                "uie_confidence": 0.98,
                "override_reason": "Public-sector strategic entity",
                "source_url": "https://example.test/source",
            },
        ])

        assigned = build_uie_assignments(
            domestic,
            pd.DataFrame(),
            domestic,
            country="MY",
            jurisdiction_predictions=predictions,
            manual_overrides=overrides,
        )
        row = assigned.iloc[0]

        self.assertEqual(row["uie_country"], "MY")
        self.assertEqual(row["uie_source"], "manual_verified")
        self.assertEqual(row["evidence_tier"], "A_manual_verified_uie")
        self.assertEqual(row["is_known_uie"], 1)
        self.assertEqual(row["source_url"], "https://example.test/source")

    def test_uie_review_targets_prioritize_weak_evidence(self) -> None:
        scored = pd.DataFrame([
            {
                "lei": "K1",
                "legal_name": "Known Parent",
                "coverage_gap_priority_score": 0.45,
                "reason_flags": "missing_direct_parent",
            },
            {
                "lei": "M1",
                "legal_name": "Model Only",
                "coverage_gap_priority_score": 0.35,
                "reason_flags": "missing_direct_parent;missing_ultimate_parent",
            },
            {
                "lei": "L1",
                "legal_name": "Low Known",
                "coverage_gap_priority_score": 0.10,
                "reason_flags": "",
            },
        ])
        uie = pd.DataFrame([
            {
                "lei": "K1",
                "uie_country": "US",
                "uie_source": "gleif_ultimate_parent",
                "evidence_tier": "A_known_ultimate_parent",
                "uie_confidence": 1.0,
            },
            {
                "lei": "M1",
                "uie_country": "SG",
                "uie_source": "phase3_jurisdiction_model",
                "evidence_tier": "E_model_predicted_jurisdiction",
                "uie_confidence": 0.55,
                "ctos_registered_malaysia": 1,
                "ctos_name": "MODEL ONLY SDN BHD",
                "ctos_match_score": 100,
            },
            {
                "lei": "L1",
                "uie_country": "GB",
                "uie_source": "gleif_ultimate_parent",
                "evidence_tier": "A_known_ultimate_parent",
                "uie_confidence": 1.0,
            },
        ])

        targets = build_uie_review_targets(scored, uie)

        self.assertEqual(targets.iloc[0]["lei"], "M1")
        self.assertEqual(targets.iloc[0]["review_reason"], "validate_model_only_investor_economy")
        self.assertEqual(targets.iloc[0]["review_status"], "todo")
        self.assertEqual(targets.iloc[0]["ctos_registered_malaysia"], 1)
        self.assertIn("K1", set(targets["lei"]))
        self.assertNotIn("L1", set(targets["lei"]))

    def test_uie_review_targets_classify_fund_vehicle(self) -> None:
        scored = pd.DataFrame([
            {
                "lei": "F1",
                "legal_name": "AHAM Global Income Fund",
                "coverage_gap_priority_score": 0.35,
                "reason_flags": "missing_direct_parent",
            },
        ])
        uie = pd.DataFrame([
            {
                "lei": "F1",
                "uie_country": "US",
                "uie_source": "phase3_jurisdiction_model",
                "evidence_tier": "E_model_predicted_jurisdiction",
                "uie_confidence": 0.7,
            },
        ])

        targets = build_uie_review_targets(scored, uie)
        row = targets.iloc[0]

        self.assertEqual(row["entity_type"], "fund_or_product_vehicle")
        self.assertEqual(row["review_status"], "product_vehicle_rule_applied")
        self.assertEqual(row["review_reason"], "fund_product_vehicle_rule_applied")

    def test_uie_review_targets_classify_local_product_vehicle_names(self) -> None:
        scored = pd.DataFrame([
            {
                "lei": "F1",
                "legal_name": "AMANAH SAHAM MALAYSIA 2 - WAWASAN",
                "coverage_gap_priority_score": 0.35,
                "reason_flags": "missing_direct_parent",
            },
            {
                "lei": "F2",
                "legal_name": "Affin Hwang Dana Malaysia",
                "coverage_gap_priority_score": 0.35,
                "reason_flags": "missing_direct_parent",
            },
            {
                "lei": "F3",
                "legal_name": "ASN SARA (MIXED ASSET CONSERVATIVE) 1",
                "coverage_gap_priority_score": 0.35,
                "reason_flags": "missing_direct_parent",
            },
        ])
        uie = pd.DataFrame([
            {
                "lei": lei,
                "uie_country": "US",
                "uie_source": "phase3_jurisdiction_model",
                "evidence_tier": "E_model_predicted_jurisdiction",
                "uie_confidence": 0.7,
            }
            for lei in ["F1", "F2", "F3"]
        ])

        targets = build_uie_review_targets(scored, uie)

        self.assertEqual(set(targets["entity_type"]), {"fund_or_product_vehicle"})
        self.assertEqual(set(targets["review_status"]), {"product_vehicle_rule_applied"})

    def test_uie_review_targets_sort_product_vehicles_after_operating_cases(self) -> None:
        scored = pd.DataFrame([
            {
                "lei": "F1",
                "legal_name": "AHAM Global Income Fund",
                "coverage_gap_priority_score": 0.50,
                "reason_flags": "missing_direct_parent",
            },
            {
                "lei": "O1",
                "legal_name": "Example Operating Sdn Bhd",
                "coverage_gap_priority_score": 0.30,
                "reason_flags": "missing_direct_parent",
            },
        ])
        uie = pd.DataFrame([
            {
                "lei": "F1",
                "uie_country": "US",
                "uie_source": "phase3_jurisdiction_model",
                "evidence_tier": "E_model_predicted_jurisdiction",
                "uie_confidence": 0.7,
            },
            {
                "lei": "O1",
                "uie_country": "SG",
                "uie_source": "phase3_jurisdiction_model",
                "evidence_tier": "E_model_predicted_jurisdiction",
                "uie_confidence": 0.7,
            },
        ])

        targets = build_uie_review_targets(scored, uie)

        self.assertEqual(targets.iloc[0]["lei"], "O1")
        self.assertEqual(targets.iloc[1]["lei"], "F1")

    def test_uie_review_targets_accept_manual_status(self) -> None:
        scored = pd.DataFrame([
            {
                "lei": "N1",
                "legal_name": "Example Nominees Sdn Bhd",
                "coverage_gap_priority_score": 0.35,
                "reason_flags": "missing_direct_parent",
            },
        ])
        uie = pd.DataFrame([
            {
                "lei": "N1",
                "uie_country": "SG",
                "uie_source": "phase3_jurisdiction_model",
                "evidence_tier": "E_model_predicted_jurisdiction",
                "uie_confidence": 0.7,
            },
        ])
        statuses = pd.DataFrame([
            {
                "lei": "N1",
                "review_status": "not_applicable",
                "entity_type": "nominee_or_custodian",
                "review_note": "Nominee vehicle; exclude from operating-company UIE queue",
            },
        ])

        targets = build_uie_review_targets(scored, uie, review_statuses=statuses)
        row = targets.iloc[0]

        self.assertEqual(row["review_status"], "not_applicable")
        self.assertEqual(row["entity_type"], "nominee_or_custodian")
        self.assertIn("Nominee vehicle", row["review_note"])

    def test_gleif_search_benchmark_flags_self_candidate(self) -> None:
        targets = pd.DataFrame([
            {
                "lei": "L1",
                "legal_name": "ACME MALAYSIA SDN BHD",
                "review_priority_score": 0.5,
                "review_status": "todo",
                "entity_type": "operating_or_holding_company",
                "uie_country": "US",
                "uie_source": "phase3_jurisdiction_model",
            },
        ])
        search_results = {
            "ACME MALAYSIA SDN BHD": pd.DataFrame([
                {
                    "query": "ACME MALAYSIA SDN BHD",
                    "rank": 1,
                    "lei": "L1",
                    "legal_name": "ACME MALAYSIA SDN BHD",
                    "country_legal": "MY",
                    "country_hq": "MY",
                    "total_results": 1,
                },
            ]),
        }

        benchmark = build_gleif_search_benchmark(targets, search_results)

        self.assertEqual(len(benchmark), 1)
        self.assertEqual(benchmark.iloc[0]["review_flag"], "self_entity_found")
        self.assertEqual(benchmark.iloc[0]["gleif_candidate_lei"], "L1")

    def test_gleif_search_benchmark_flags_country_difference(self) -> None:
        targets = pd.DataFrame([
            {
                "lei": "L1",
                "legal_name": "ACME MALAYSIA SDN BHD",
                "review_priority_score": 0.5,
                "review_status": "todo",
                "entity_type": "operating_or_holding_company",
                "uie_country": "US",
                "uie_source": "phase3_jurisdiction_model",
            },
        ])
        search_results = {
            "ACME MALAYSIA SDN BHD": pd.DataFrame([
                {
                    "rank": 1,
                    "lei": "P1",
                    "legal_name": "ACME HOLDINGS PTE LTD",
                    "country_legal": "SG",
                    "country_hq": "SG",
                    "total_results": 1,
                },
            ]),
        }

        benchmark = build_gleif_search_benchmark(targets, search_results)

        self.assertEqual(
            benchmark.iloc[0]["review_flag"],
            "candidate_country_differs_from_current_uie",
        )

    def test_gleif_search_benchmark_handles_no_candidate(self) -> None:
        targets = pd.DataFrame([
            {
                "lei": "L1",
                "legal_name": "NO MATCH SDN BHD",
                "uie_country": "US",
                "uie_source": "phase3_jurisdiction_model",
            },
        ])

        benchmark = build_gleif_search_benchmark(targets, {})

        self.assertEqual(benchmark.iloc[0]["review_flag"], "no_candidate")


if __name__ == "__main__":
    unittest.main()
