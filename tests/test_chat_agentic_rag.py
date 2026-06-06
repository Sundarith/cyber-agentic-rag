import unittest

from agentic_rag.schema import Chunk, Evidence, Verification
from agentic_rag.synthesizer import StructuredSynthesisResult, StructuredSynthesisValidation
from chat_agentic_rag import build_final_answer, build_path_answer, build_user_facing_answer, format_evidence, format_result, handle_command


class ChatAgenticRAGTests(unittest.TestCase):
    def test_format_result_includes_answer_trace_and_evidence(self):
        chunk = Chunk(
            id="cwe-79",
            identifier="CWE-79",
            source="CWE",
            type="weakness",
            name="Cross-site Scripting",
            text="CWE-79 covers XSS.",
        )
        result = {
            "question": "What is CWE-79?",
            "answer": "CWE-79 is Cross-site Scripting.",
            "verification": Verification(True, 0.9, [], ["CWE-79"]),
            "trace": [
                {
                    "step": 1,
                    "action": "open",
                    "value": "CWE-79",
                    "new_evidence": ["CWE-79"],
                    "confidence": 0.9,
                }
            ],
            "evidence": [Evidence(chunk, 1.0, "open")],
        }
        text = format_result(result, "lexical")
        self.assertIn("Final Answer", text)
        self.assertIn("Retrieval Draft", text)
        self.assertIn("Trace", text)
        self.assertIn("Evidence", text)
        self.assertIn("CWE-79", text)
        self.assertLess(text.index("Retrieval Draft"), text.index("Final Answer"))
        self.assertTrue(text.rstrip().endswith("CWE-79 - Cross-site Scripting"))

    def test_format_evidence_includes_backend(self):
        chunk = Chunk(id="capec-1", identifier="CAPEC-1", source="CAPEC", name="Access Control", text="")
        text = format_evidence(1, Evidence(chunk, 0.75, "search"), "lexical")
        self.assertIn("CAPEC-1", text)
        self.assertIn("backend=lexical", text)

    def test_handle_command_toggles_state(self):
        state = {"show_trace": True, "show_evidence": True, "json_output": False}
        self.assertTrue(handle_command("/trace", state))
        self.assertFalse(state["show_trace"])
        self.assertTrue(handle_command("/json", state))
        self.assertTrue(state["json_output"])
        self.assertFalse(handle_command("/quit", state))

    def test_build_final_answer_prefers_complete_trace_path(self):
        result = {
            "question": "Which CWE weakness is connected to T1000 through CAPEC?",
            "answer": "",
            "trace": [
                {
                    "action": "expand",
                    "expanded_from": "T1000",
                    "new_evidence": ["CAPEC-100"],
                },
                {
                    "action": "expand",
                    "expanded_from": "CAPEC-100",
                    "new_evidence": ["CWE-100"],
                },
            ],
            "evidence": [],
        }
        self.assertIn("CWE-100", build_final_answer(result))
        self.assertNotIn("T1000 -> CAPEC-100 -> CWE-100", build_final_answer(result))
        self.assertEqual(build_path_answer(result), "T1000 -> CAPEC-100 -> CWE-100")

    def test_build_final_answer_uses_polished_path_with_names(self):
        attack = Chunk(id="attack-T1057", identifier="T1057", source="MITRE ATT&CK", name="Process Discovery", text="")
        capec = Chunk(id="capec-573", identifier="CAPEC-573", source="CAPEC", name="Process Footprinting", text="")
        cwe = Chunk(
            id="cwe-200",
            identifier="CWE-200",
            source="CWE",
            name="Exposure of Sensitive Information to an Unauthorized Actor",
            text="",
        )
        result = {
            "question": "Which CWE weakness is connected to T1057 through CAPEC?",
            "answer": "",
            "trace": [
                {"action": "expand", "expanded_from": "T1057", "new_evidence": ["CAPEC-573"]},
                {"action": "expand", "expanded_from": "CAPEC-573", "new_evidence": ["CWE-200"]},
            ],
            "evidence": [Evidence(attack, 1.0, "open"), Evidence(capec, 1.0, "expand"), Evidence(cwe, 1.0, "expand")],
        }
        answer = build_final_answer(result)
        self.assertIn("CWE-200 (Exposure of Sensitive Information to an Unauthorized Actor)", answer)
        self.assertIn("ATT&CK T1057 (Process Discovery)", answer)
        self.assertNotIn("->", answer)
        self.assertEqual(build_path_answer(result), "T1057 -> CAPEC-573 -> CWE-200")

    def test_format_result_puts_path_before_final_answer_at_end(self):
        attack = Chunk(id="attack-T1057", identifier="T1057", source="MITRE ATT&CK", name="Process Discovery", text="")
        capec = Chunk(id="capec-573", identifier="CAPEC-573", source="CAPEC", name="Process Footprinting", text="")
        cwe = Chunk(id="cwe-200", identifier="CWE-200", source="CWE", name="Exposure", text="")
        result = {
            "question": "Which CWE weakness is connected to T1057 through CAPEC?",
            "answer": "debug draft",
            "final_answer": "CWE-200 is the connected CWE weakness.",
            "verification": Verification(True, 0.9, [], ["T1057", "CAPEC-573", "CWE-200"]),
            "trace": [
                {"action": "expand", "expanded_from": "T1057", "new_evidence": ["CAPEC-573"]},
                {"action": "expand", "expanded_from": "CAPEC-573", "new_evidence": ["CWE-200"]},
            ],
            "evidence": [Evidence(attack, 1.0, "open"), Evidence(capec, 1.0, "expand"), Evidence(cwe, 1.0, "expand")],
        }
        text = format_result(result, "lexical", show_trace=False, show_evidence=False)
        self.assertLess(text.index("Retrieval Draft"), text.index("ATT&CK -> CAPEC -> CWE Path"))
        self.assertLess(text.index("ATT&CK -> CAPEC -> CWE Path"), text.index("Final Answer"))
        self.assertTrue(text.rstrip().endswith("CWE-200 is the connected CWE weakness."))

    def test_build_final_answer_reports_multiple_cwe_candidates(self):
        cwe_100 = Chunk(id="cwe-100", identifier="CWE-100", source="CWE", name="First", text="")
        cwe_200 = Chunk(id="cwe-200", identifier="CWE-200", source="CWE", name="Second", text="")
        result = {
            "question": "Which CWE weakness is present in the evidence?",
            "answer": "",
            "trace": [],
            "evidence": [Evidence(cwe_100, 1.0, "expand"), Evidence(cwe_200, 1.0, "expand")],
        }
        answer = build_final_answer(result)
        self.assertIn("Multiple CWE candidates", answer)
        self.assertIn("CWE-100", answer)
        self.assertIn("CWE-200", answer)

    def test_build_final_answer_hides_candidates_when_relation_path_missing(self):
        cwe_100 = Chunk(id="cwe-100", identifier="CWE-100", source="CWE", name="First", text="")
        cwe_200 = Chunk(id="cwe-200", identifier="CWE-200", source="CWE", name="Second", text="")
        result = {
            "question": "Which CWE weakness is connected through CAPEC?",
            "answer": "",
            "trace": [],
            "evidence": [Evidence(cwe_100, 1.0, "expand"), Evidence(cwe_200, 1.0, "expand")],
        }
        answer = build_final_answer(result)
        self.assertIn("not have enough validated evidence", answer)
        self.assertNotIn("CWE-100", answer)

    def test_build_user_facing_answer_falls_back_when_composer_rejects(self):
        class RejectingComposer:
            def compose_grounded_answer(self, question, evidence, selected_path):
                raise ValueError("composed answer mentioned unsupported IDs: CAPEC-647")

        result = {
            "question": "Which CWE weakness is connected to T1057 through CAPEC?",
            "answer": "",
            "trace": [
                {"action": "expand", "expanded_from": "T1057", "new_evidence": ["CAPEC-573"]},
                {"action": "expand", "expanded_from": "CAPEC-573", "new_evidence": ["CWE-200"]},
            ],
            "evidence": [
                Evidence(Chunk(id="attack-T1057", identifier="T1057", source="MITRE ATT&CK", name="Process Discovery", text=""), 1.0, "open"),
                Evidence(Chunk(id="capec-573", identifier="CAPEC-573", source="CAPEC", name="Process Footprinting", text=""), 1.0, "expand"),
                Evidence(Chunk(id="cwe-200", identifier="CWE-200", source="CWE", name="Exposure", text=""), 1.0, "expand"),
            ],
            "structured_answer": StructuredSynthesisResult(
                raw="{}",
                parsed={},
                answerable=True,
                selected_cwe="CWE-200",
                selected_path=["T1057", "CAPEC-573", "CWE-200"],
                cited_evidence_ids=["T1057", "CAPEC-573", "CWE-200"],
                confidence="high",
                reason="supported",
                validation=StructuredSynthesisValidation(
                    supported=True,
                    selected_cwe_valid=True,
                    selected_path_valid=True,
                    unsupported_ids=[],
                    unsupported_citations=[],
                    warnings=[],
                    candidate_paths=[["T1057", "CAPEC-573", "CWE-200"]],
                    allowed_ids=["T1057", "CAPEC-573", "CWE-200"],
                ),
            ),
        }
        answer, source = build_user_facing_answer(result, RejectingComposer())
        self.assertEqual(source, "deterministic_fallback")
        self.assertIn("CWE-200", answer)
        self.assertNotIn("CAPEC-647", answer)
        self.assertIn("unsupported IDs", result["final_answer_warning"])


if __name__ == "__main__":
    unittest.main()
