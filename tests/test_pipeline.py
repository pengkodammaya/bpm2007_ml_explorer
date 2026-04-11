import unittest

import pandas as pd

from src.pipeline import _foreign_parent_flags


class PipelineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
