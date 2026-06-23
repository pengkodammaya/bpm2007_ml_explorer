import unittest

import networkx as nx

from src.graph_build import graph_summary


class GraphBuildTests(unittest.TestCase):
    def test_graph_summary_handles_empty_graph(self) -> None:
        summary = graph_summary(nx.DiGraph())

        self.assertTrue(summary.empty)
        self.assertIn("lei", summary.columns)
        self.assertIn("weakly_connected_component_size", summary.columns)


if __name__ == "__main__":
    unittest.main()
