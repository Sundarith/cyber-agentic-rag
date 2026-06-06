import json
import unittest

from agentic_rag import CorpusIndex, GraniteOrchestrator
from agentic_rag.orchestrator import OrchestratorConfig
from agentic_rag.schema import Chunk


class MockLLMClient:
    """Returns scripted action strings, one per turn — no live endpoint."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = 0

    def complete(self, messages):
        out = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        return out


def act(**kw):
    return json.dumps(kw)


class OrchestratorTests(unittest.TestCase):
    def corpus(self):
        return CorpusIndex(
            [
                Chunk(id="a", identifier="T1000", source="MITRE ATT&CK", type="technique",
                      name="Synthetic Technique", text="# T1000 related to CAPEC-100."),
                Chunk(id="c", identifier="CAPEC-100", source="CAPEC", type="attack_pattern",
                      name="Cand Pattern", text="# CAPEC-100\nRelated Root Causes (CWE)\n- CWE-100"),
                Chunk(id="w", identifier="CWE-100", source="CWE", type="weakness",
                      name="The Weakness", text="# CWE-100 weakness."),
            ],
            capec_to_cwe={"CAPEC-100": ["CWE-100"]},
            attack_to_capec={"T1000": ["CAPEC-100"]},
        )

    def test_happy_path_validates(self):
        scripts = [
            act(tool="resolve", args={"name": "Synthetic Technique"}),
            act(tool="open", args={"id": "T1000"}),
            act(tool="expand", args={"id": "T1000"}),
            act(tool="expand", args={"id": "CAPEC-100"}),
            act(answer={"text": "path found", "cited_ids": ["T1000", "CAPEC-100", "CWE-100"],
                        "path": ["T1000", "CAPEC-100", "CWE-100"]}),
        ]
        orch = GraniteOrchestrator(self.corpus(), client=MockLLMClient(scripts))
        res = orch.answer("For Synthetic Technique, which CWE via its CAPEC?")
        self.assertTrue(res["llm_final"]["validation"]["supported"])
        ids = {ev.identifier for ev in res["evidence"]}
        self.assertTrue({"T1000", "CAPEC-100", "CWE-100"} <= ids)
        actions = [s["action"] for s in res["trace"]]
        self.assertEqual(actions, ["resolve", "open", "expand", "expand", "answer"])

    def test_validation_gate_rejects_unsupported_id(self):
        # The model answers with a CWE it never retrieved and an unproven edge.
        scripts = [act(answer={"text": "It is CWE-79", "cited_ids": ["CWE-79"], "path": ["T1000", "CWE-79"]})]
        orch = GraniteOrchestrator(self.corpus(), client=MockLLMClient(scripts))
        res = orch.answer("q")
        val = res["llm_final"]["validation"]
        self.assertFalse(val["supported"])
        self.assertIn("CWE-79", val["unsupported_ids"])
        self.assertIn("T1000->CWE-79", val["invalid_edges"])
        self.assertIn("validation", res["answer"])

    def test_budget_cap_forces_final_answer(self):
        # Never emits an answer; loop must terminate and force a final answer.
        scripts = [act(tool="search", args={"query": "anything"})]
        orch = GraniteOrchestrator(self.corpus(), client=MockLLMClient(scripts),
                                   config=OrchestratorConfig(max_iters=3))
        res = orch.answer("q")
        self.assertTrue(res["llm_final"]["validation"].get("forced"))
        self.assertEqual(sum(1 for s in res["trace"] if s["action"] == "search"), 3)

    def test_malformed_json_triggers_retry_then_recovers(self):
        scripts = [
            "not json at all",
            act(answer={"text": "ok", "cited_ids": [], "path": []}),
        ]
        orch = GraniteOrchestrator(self.corpus(), client=MockLLMClient(scripts))
        res = orch.answer("q")
        self.assertEqual(res["trace"][0]["action"], "parse_error")
        self.assertEqual(res["trace"][-1]["action"], "answer")

    def test_resolve_auto_opens_resolved_node(self):
        # resolve auto-opens the resolved id so the loop makes progress and the
        # model can cite it (this is what breaks the repeated-resolve loop).
        scripts = [act(tool="resolve", args={"name": "Synthetic Technique"}),
                   act(answer={"text": "done", "cited_ids": [], "path": []})]
        orch = GraniteOrchestrator(self.corpus(), client=MockLLMClient(scripts))
        res = orch.answer("q")
        self.assertEqual(res["trace"][0]["action"], "resolve")
        self.assertEqual(res["trace"][0]["new_evidence"], ["T1000"])


if __name__ == "__main__":
    unittest.main()
