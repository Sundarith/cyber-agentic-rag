import unittest

from agentic_rag.schema import Chunk, Evidence, Verification
from agentic_rag.synthesizer import (
    DEFAULT_GRANITE_MODEL,
    GraniteGroundedSynthesizer,
    GraniteStructuredSynthesizer,
    parse_json_object,
    validate_composed_answer,
    validate_structured_payload,
)


class FakeChatClient:
    def __init__(self):
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return "The evidence cites CWE-79 as the weakness.\nCWE-79"


class FakeComposeClient:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return self.response


class GraniteSynthesizerTests(unittest.TestCase):
    def test_granite_prompt_uses_evidence_and_cwe_final_line_instruction(self):
        client = FakeChatClient()
        synthesizer = GraniteGroundedSynthesizer(client=client)
        evidence = [
            Evidence(
                chunk=Chunk(
                    id="cwe-79",
                    identifier="CWE-79",
                    source="CWE",
                    type="weakness",
                    name="Cross-site Scripting",
                    text="CWE-79 is improper neutralization during web page generation.",
                ),
                score=1.0,
                tool="search",
            )
        ]
        answer = synthesizer.synthesize(
            "Map this CVE to CWE.",
            evidence,
            Verification(supported=True, confidence=0.9, missing=[], cited_ids=["CWE-79"]),
        )

        self.assertEqual(answer.splitlines()[-1], "CWE-79")
        self.assertEqual(DEFAULT_GRANITE_MODEL, "ibm-granite/granite-4.1-8b")
        self.assertIsNotNone(client.messages)
        prompt = client.messages[1]["content"]
        self.assertIn("CWE-79", prompt)
        self.assertIn("final line must contain only one CWE ID", prompt)

    def test_parse_json_object_accepts_plain_json(self):
        parsed = parse_json_object('{"answerable": true, "selected_cwe": "CWE-79"}')
        self.assertTrue(parsed["answerable"])
        self.assertEqual(parsed["selected_cwe"], "CWE-79")

    def test_parse_json_object_accepts_markdown_wrapped_json(self):
        parsed = parse_json_object(
            """```json
{"answerable": false, "selected_path": []}
```"""
        )
        self.assertFalse(parsed["answerable"])
        self.assertEqual(parsed["selected_path"], [])

    def test_parse_json_object_rejects_malformed_output(self):
        with self.assertRaises(ValueError):
            parse_json_object("not json")

    def test_validator_accepts_supported_path(self):
        evidence = [
            Evidence(Chunk(id="t1057", identifier="T1057", source="MITRE ATT&CK", name="Process Discovery", text=""), 1.0, "open"),
            Evidence(Chunk(id="capec-646", identifier="CAPEC-646", source="CAPEC", name="Peripheral Footprinting", text=""), 1.0, "expand"),
            Evidence(Chunk(id="cwe-200", identifier="CWE-200", source="CWE", name="Exposure", text=""), 1.0, "expand"),
        ]
        validation = validate_structured_payload(
            {
                "answerable": True,
                "selected_cwe": "CWE-200",
                "selected_path": ["T1057", "CAPEC-646", "CWE-200"],
                "cited_evidence_ids": ["T1057", "CAPEC-646", "CWE-200"],
            },
            evidence,
            [["T1057", "CAPEC-646", "CWE-200"]],
        )
        self.assertTrue(validation.supported)
        self.assertEqual(validation.warnings, [])

    def test_validator_flags_hallucinated_capec_647(self):
        evidence = [
            Evidence(Chunk(id="t1057", identifier="T1057", source="MITRE ATT&CK", name="Process Discovery", text=""), 1.0, "open"),
            Evidence(Chunk(id="capec-646", identifier="CAPEC-646", source="CAPEC", name="Peripheral Footprinting", text=""), 1.0, "expand"),
            Evidence(Chunk(id="cwe-200", identifier="CWE-200", source="CWE", name="Exposure", text=""), 1.0, "expand"),
        ]
        validation = validate_structured_payload(
            {
                "answerable": True,
                "selected_cwe": "CWE-200",
                "selected_path": ["T1057", "CAPEC-647", "CWE-200"],
                "cited_evidence_ids": ["T1057", "CAPEC-647", "CWE-200"],
            },
            evidence,
            [["T1057", "CAPEC-646", "CWE-200"]],
        )
        self.assertFalse(validation.supported)
        self.assertIn("CAPEC-647", validation.unsupported_ids)
        self.assertIn("CAPEC-647", validation.unsupported_citations)
        self.assertIn("selected_path_not_in_trace", validation.warnings)

    def test_composer_returns_grounded_answer_for_valid_path(self):
        client = FakeComposeClient(
            "CWE-200 is the connected weakness for ATT&CK T1057 through CAPEC-573."
        )
        synthesizer = GraniteStructuredSynthesizer(client=client)
        evidence = [
            Evidence(Chunk(id="t1057", identifier="T1057", source="MITRE ATT&CK", name="Process Discovery", text=""), 1.0, "open"),
            Evidence(Chunk(id="capec-573", identifier="CAPEC-573", source="CAPEC", name="Process Footprinting", text=""), 1.0, "expand"),
            Evidence(Chunk(id="cwe-200", identifier="CWE-200", source="CWE", name="Exposure", text=""), 1.0, "expand"),
        ]
        answer = synthesizer.compose_grounded_answer(
            "Which CWE?",
            evidence,
            ["T1057", "CAPEC-573", "CWE-200"],
        )
        self.assertIn("CWE-200", answer)
        self.assertIn("CAPEC-573", answer)
        self.assertIn("T1057", client.messages[1]["content"])
        self.assertIn("Do not include the ATT&CK -> CAPEC -> CWE path", client.messages[1]["content"])

    def test_composer_rejects_unsupported_ids(self):
        with self.assertRaises(ValueError):
            validate_composed_answer(
                "The path is T1057 -> CAPEC-647 -> CWE-200.",
                ["T1057", "CAPEC-573", "CWE-200"],
            )


if __name__ == "__main__":
    unittest.main()
