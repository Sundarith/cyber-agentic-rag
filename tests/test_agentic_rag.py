import unittest

from agentic_rag import AgenticRAG, AgenticRAGConfig, CorpusIndex
from agentic_rag.schema import Chunk


class AgenticRAGTests(unittest.TestCase):
    def toy_corpus(self):
        return CorpusIndex(
            [
                Chunk(
                    id="capec-CAPEC-98",
                    identifier="CAPEC-98",
                    source="CAPEC",
                    type="attack_pattern",
                    name="Phishing",
                    text="# CAPEC-98\nThis attack pattern relates to CWE-451.",
                ),
                Chunk(
                    id="cwe-451",
                    identifier="CWE-451",
                    source="CWE",
                    type="weakness",
                    name="User Interface Misrepresentation of Critical Information",
                    text="# CWE-451\nThe UI misrepresents critical information to the user.",
                    metadata={"parent_cwe_ids": ["CWE-1021"]},
                ),
                Chunk(
                    id="cwe-1021",
                    identifier="CWE-1021",
                    source="CWE",
                    type="weakness",
                    name="Improper Restriction of Rendered UI Layers",
                    text="# CWE-1021\nA parent weakness in UI rendering.",
                ),
            ],
            capec_to_cwe={"CAPEC-98": ["CWE-451"]},
        )

    def test_graph_expansion_reaches_cwe(self):
        corpus = self.toy_corpus()
        neighbors = corpus.expand("CAPEC-98")
        self.assertIn("CWE-451", [ev.identifier for ev in neighbors])

    def test_agent_collects_root_cause_evidence(self):
        agent = AgenticRAG(self.toy_corpus(), AgenticRAGConfig(max_steps=4, retrieve_k=3))
        result = agent.answer("What CWE underlies CAPEC-98?")
        ids = [ev.identifier for ev in result["evidence"]]
        self.assertIn("CAPEC-98", ids)
        self.assertIn("CWE-451", ids)
        self.assertTrue(result["verification"].supported)


if __name__ == "__main__":
    unittest.main()
