"""Guards against the canned tool data reporting contradictory Python floors."""

import unittest

from agent.tools import CONCEPTS_DB, MIN_PYTHON, SETUP_GUIDES_DB, lookup_concept


class MinPythonConsistencyTest(unittest.TestCase):
    def test_every_concept_uses_a_known_floor(self):
        for concept, data in CONCEPTS_DB.items():
            package = data["package"].split()[0]
            self.assertIn(package, MIN_PYTHON, concept)
            self.assertEqual(data["min_python"], MIN_PYTHON[package], concept)

    def test_installation_guide_states_the_same_floors(self):
        self.assertIn(
            f"Minimum supported Python is {MIN_PYTHON['langchain']} for langchain/langgraph, "
            f"{MIN_PYTHON['langsmith']} for langsmith, and {MIN_PYTHON['deepagents']} for deepagents.",
            SETUP_GUIDES_DB["installation"],
        )

    def test_langgraph_lookup_reports_the_supported_floor(self):
        self.assertIn("Minimum Python: 3.10+", lookup_concept.invoke({"concept_name": "LangGraph"}))


if __name__ == "__main__":
    unittest.main()
