import unittest

from agentic_rag import AgenticRAG, AgenticRAGConfig, CorpusIndex
from agentic_rag.planner import QueryPlanner
from agentic_rag.schema import AgentState, Chunk, Verification


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

    def test_expand_prioritizes_capec_over_high_degree_campaigns(self):
        # A technique with many campaign neighbors that sort alphabetically
        # before the CAPEC (C0001 < CAPEC-999) must still surface the CAPEC
        # within a small expansion budget.
        corpus = CorpusIndex(
            [
                Chunk(id="attack-T9000", identifier="T9000", source="MITRE ATT&CK", type="technique",
                      name="High Degree Technique", text="# T9000\nA technique."),
                Chunk(id="c-C0001", identifier="C0001", source="MITRE ATT&CK", type="campaign",
                      name="Camp One", text="# C0001\nCampaign using T9000."),
                Chunk(id="c-C0002", identifier="C0002", source="MITRE ATT&CK", type="campaign",
                      name="Camp Two", text="# C0002\nCampaign using T9000."),
                Chunk(id="capec-CAPEC-999", identifier="CAPEC-999", source="CAPEC", type="attack_pattern",
                      name="Gold Pattern", text="# CAPEC-999\nRelated Root Causes (CWE)\n- CWE-999"),
                Chunk(id="cwe-999", identifier="CWE-999", source="CWE", type="weakness",
                      name="Gold Weakness", text="# CWE-999\nWeakness."),
            ],
            capec_to_cwe={"CAPEC-999": ["CWE-999"]},
            attack_to_capec={"T9000": ["CAPEC-999"]},
        )
        # Budget of 2 would drop the CAPEC under naive alphabetical ordering.
        top = [ev.identifier for ev in corpus.expand("T9000", k=2)]
        self.assertEqual(top[0], "CAPEC-999")

    def test_agent_collects_root_cause_evidence(self):
        agent = AgenticRAG(self.toy_corpus(), AgenticRAGConfig(max_steps=4, retrieve_k=3))
        result = agent.answer("What CWE underlies CAPEC-98?")
        ids = [ev.identifier for ev in result["evidence"]]
        self.assertIn("CAPEC-98", ids)
        self.assertIn("CWE-451", ids)
        self.assertTrue(result["verification"].supported)

    def multihop_corpus(self):
        return CorpusIndex(
            [
                Chunk(
                    id="attack-T1000",
                    identifier="T1000",
                    source="MITRE ATT&CK",
                    type="technique",
                    name="Synthetic Technique",
                    text="# T1000\nThis technique is related to CAPEC-100 and CAPEC-200.",
                ),
                Chunk(
                    id="capec-CAPEC-100",
                    identifier="CAPEC-100",
                    source="CAPEC",
                    type="attack_pattern",
                    name="First Candidate",
                    text="# CAPEC-100\nRelated Root Causes (CWE)\n- CWE-100",
                ),
                Chunk(
                    id="capec-CAPEC-200",
                    identifier="CAPEC-200",
                    source="CAPEC",
                    type="attack_pattern",
                    name="Second Candidate",
                    text="# CAPEC-200\nRelated Root Causes (CWE)\n- CWE-200",
                ),
                Chunk(
                    id="cwe-100",
                    identifier="CWE-100",
                    source="CWE",
                    type="weakness",
                    name="First Weakness",
                    text="# CWE-100\nFirst weakness.",
                ),
                Chunk(
                    id="cwe-200",
                    identifier="CWE-200",
                    source="CWE",
                    type="weakness",
                    name="Second Weakness",
                    text="# CWE-200\nSecond weakness.",
                ),
            ],
            capec_to_cwe={"CAPEC-100": ["CWE-100"], "CAPEC-200": ["CWE-200"]},
            attack_to_capec={"T1000": ["CAPEC-100", "CAPEC-200"]},
        )

    def test_agent_finishes_pending_capec_expansions_after_first_cwe(self):
        agent = AgenticRAG(self.multihop_corpus(), AgenticRAGConfig(max_steps=5, retrieve_k=3, evidence_budget=5))
        result = agent.answer("For ATT&CK technique T1000, which CWE weakness is connected through CAPEC?")
        ids = [ev.identifier for ev in result["evidence"]]
        self.assertIn("T1000", ids)
        self.assertIn("CAPEC-100", ids)
        self.assertIn("CAPEC-200", ids)
        self.assertIn("CWE-100", ids)
        self.assertIn("CWE-200", ids)
        self.assertEqual(
            [step["value"] for step in result["trace"] if step["action"] == "expand"],
            ["T1000", "CAPEC-100", "CAPEC-200"],
        )

    def test_agent_expands_direct_attack_capec_frontier_before_search(self):
        agent = AgenticRAG(self.multihop_corpus(), AgenticRAGConfig(max_steps=4, retrieve_k=3, evidence_budget=5))
        result = agent.answer("For ATT&CK technique T1000, which CWE weakness is connected through CAPEC?")
        self.assertEqual(result["trace"][0]["action"], "open")
        self.assertEqual(result["trace"][1]["action"], "expand")
        self.assertEqual(result["trace"][1]["value"], "T1000")
        self.assertEqual(result["trace"][2]["action"], "expand")
        self.assertEqual(result["trace"][2]["value"], "CAPEC-100")
        self.assertEqual(result["trace"][2]["frontier_source"], "graph_direct_path")

    def test_planner_prioritizes_capec_expansion_when_cwe_missing(self):
        corpus = self.multihop_corpus()
        state = AgentState(
            question="Which CWE weakness is connected through CAPEC?",
            evidence=[corpus.open("CAPEC-200")],
        )
        actions = QueryPlanner().next_actions(
            state,
            Verification(False, 0.4, ["root_cause"], ["CAPEC-200"]),
            retrieve_k=3,
        )
        self.assertEqual(actions[0].kind.value, "expand")
        self.assertEqual(actions[0].value, "CAPEC-200")

    def test_hidden_id_question_resolves_and_recovers_path(self):
        agent = AgenticRAG(self.multihop_corpus(), AgenticRAGConfig(max_steps=5, retrieve_k=3, evidence_budget=5))
        result = agent.answer(
            "For Synthetic Technique, which CWE weakness is connected through its related CAPEC attack pattern?"
        )
        resolution = result["resolution"]
        self.assertIsNotNone(resolution)
        self.assertEqual(resolution["chosen_id"], "T1000")
        self.assertEqual(resolution["type"], "technique")
        self.assertFalse(resolution["ambiguous"])
        ids = [ev.identifier for ev in result["evidence"]]
        self.assertIn("T1000", ids)
        self.assertIn("CAPEC-100", ids)
        self.assertIn("CWE-100", ids)
        # The agent opens the resolved identifier before any search.
        self.assertEqual(result["trace"][0]["action"], "open")
        self.assertEqual(result["trace"][0]["value"], "T1000")

    def test_explicit_id_question_does_not_resolve(self):
        agent = AgenticRAG(self.multihop_corpus(), AgenticRAGConfig(max_steps=4, retrieve_k=3))
        result = agent.answer("For ATT&CK technique T1000, which CWE weakness is connected through CAPEC?")
        self.assertIsNone(result["resolution"])

    def test_planner_routes_forward_and_reverse(self):
        planner = QueryPlanner()
        fwd = planner.route("For Process Discovery, which CWE is reached via its CAPEC?")
        self.assertEqual(fwd.direction, "forward")
        rev = planner.route("Which attack patterns reference Exposure of Sensitive Information?")
        self.assertEqual(rev.direction, "reverse")
        self.assertEqual(rev.target_type, "attack_pattern")
        self.assertEqual(rev.prefer_types[0], "weakness")

    def test_reverse_question_returns_full_capec_set(self):
        agent = AgenticRAG(self.multihop_corpus(), AgenticRAGConfig(max_steps=5, retrieve_k=3, evidence_budget=5))
        # multihop_corpus: CAPEC-100 and CAPEC-200 both reference distinct CWEs;
        # build a reverse question on a CWE shared by one CAPEC.
        result = agent.answer("Which attack patterns reference Second Weakness?")
        self.assertEqual(result["route"]["direction"], "reverse")
        self.assertEqual(result["resolution"]["chosen_id"], "CWE-200")
        # CWE-200 is referenced by CAPEC-200 in the toy graph.
        self.assertIn("CAPEC-200", result["reverse_answer"])
        self.assertNotIn("CAPEC-100", result["reverse_answer"])

    def test_planner_builds_direct_relation_frontier_after_attack_expand(self):
        corpus = self.multihop_corpus()
        attack_evidence = corpus.expand("T1000")
        state = AgentState(
            question="For ATT&CK technique T1000, which CWE weakness is connected through CAPEC?",
            evidence=attack_evidence,
            opened_ids={"T1000"},
        )
        actions = QueryPlanner().priority_actions_after_step(
            state,
            result_action("expand", "T1000"),
            attack_evidence,
            Verification(False, 0.7, ["root_cause"], ["T1000", "CAPEC-100", "CAPEC-200"]),
            retrieve_k=3,
        )
        self.assertEqual([action.value for action in actions], ["CAPEC-100", "CAPEC-200"])
        self.assertIn("direct ATT&CK relation frontier", actions[0].rationale)


def result_action(kind: str, value: str):
    from agentic_rag.schema import Action, ActionType

    return Action(ActionType(kind), value, "test action")


if __name__ == "__main__":
    unittest.main()
