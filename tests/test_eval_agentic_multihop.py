import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag import CorpusIndex
from agentic_rag.schema import Chunk, Evidence
from eval_agentic_multihop import (
    annotate_gold_hops,
    build_diagnostic_items,
    build_parser,
    classify_failure,
    heuristic_select,
    run_eval,
    score_path,
    score_resolution,
)


class AgenticMultihopEvalTests(unittest.TestCase):
    def toy_corpus(self):
        return CorpusIndex(
            [
                Chunk(
                    id="attack-T1000",
                    identifier="T1000",
                    source="MITRE ATT&CK",
                    type="technique",
                    name="Password Guessing",
                    text="# T1000\nTechnique related to CAPEC-100.",
                ),
                Chunk(
                    id="capec-CAPEC-100",
                    identifier="CAPEC-100",
                    source="CAPEC",
                    type="attack_pattern",
                    name="Guessing Passwords",
                    text="# CAPEC-100\nRelated Root Causes (CWE)\n- CWE-100",
                ),
                Chunk(
                    id="cwe-100",
                    identifier="CWE-100",
                    source="CWE",
                    type="weakness",
                    name="Weak Password Requirements",
                    text="# CWE-100\nWeak password requirements.",
                ),
                Chunk(
                    id="cwe-101",
                    identifier="CWE-101",
                    source="CWE",
                    type="weakness",
                    name="Distractor One",
                    text="# CWE-101\nDistractor.",
                ),
                Chunk(
                    id="cwe-102",
                    identifier="CWE-102",
                    source="CWE",
                    type="weakness",
                    name="Distractor Two",
                    text="# CWE-102\nDistractor.",
                ),
                Chunk(
                    id="cwe-103",
                    identifier="CWE-103",
                    source="CWE",
                    type="weakness",
                    name="Distractor Three",
                    text="# CWE-103\nDistractor.",
                ),
            ],
            capec_to_cwe={"CAPEC-100": ["CWE-100"]},
            attack_to_capec={"T1000": ["CAPEC-100"]},
        )

    def write_relations(self, root: Path):
        processed = root / "data" / "processed"
        processed.mkdir(parents=True)
        attack_rel = processed / "capec_attack_relations.json"
        capec_rel = processed / "capec_cwe_relations.json"
        attack_rel.write_text(json.dumps({"tech_to_capec": {"T1000": ["CAPEC-100"]}}), encoding="utf-8")
        capec_rel.write_text(json.dumps({"capec_to_cwe": {"CAPEC-100": ["CWE-100"]}}), encoding="utf-8")
        return attack_rel, capec_rel

    def write_chunks(self, root: Path):
        processed = root / "data" / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        rows = self.toy_corpus().chunks
        for name, source in [
            ("attack_chunks.jsonl", "MITRE ATT&CK"),
            ("capec_chunks.jsonl", "CAPEC"),
            ("cwe_chunks.jsonl", "CWE"),
        ]:
            with (processed / name).open("w", encoding="utf-8") as f:
                for chunk in rows:
                    if chunk.source == source:
                        f.write(
                            json.dumps(
                                {
                                    "id": chunk.id,
                                    "identifier": chunk.identifier,
                                    "source": chunk.source,
                                    "type": chunk.type,
                                    "name": chunk.name,
                                    "text": chunk.text,
                                }
                            )
                            + "\n"
                        )

    def test_build_diagnostic_items_from_relations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attack_rel, capec_rel = self.write_relations(root)
            items = build_diagnostic_items(self.toy_corpus(), attack_rel, capec_rel, seed=7)
            self.assertEqual(len(items), 1)
            item = items[0]
            self.assertEqual(item["gold_path"], ["T1000", "CAPEC-100", "CWE-100"])
            self.assertEqual(item["required_hops"], ["attack_to_capec", "capec_to_cwe"])
            self.assertEqual(len(item["choices"]), 4)
            self.assertIn(item["gold_answer"], "ABCD")

    def test_score_path_full_and_recovery(self):
        item = {"gold_path": ["T1000", "CAPEC-100", "CWE-100"]}
        evidence = [
            Evidence(self.toy_corpus().by_identifier["T1000"], 1.0, "open"),
            Evidence(self.toy_corpus().by_identifier["CAPEC-100"], 1.0, "expand"),
            Evidence(self.toy_corpus().by_identifier["CWE-100"], 1.0, "expand"),
        ]
        trace = [
            {"step": 1, "new_evidence": ["T1000"]},
            {"step": 2, "new_evidence": ["CAPEC-100"]},
            {"step": 3, "new_evidence": ["CWE-100"]},
        ]
        scores = score_path(item, evidence, trace)
        self.assertTrue(scores["full_path_retrieved"])
        self.assertEqual(scores["gold_first_seen_step"], 3)
        self.assertTrue(scores["recovery_after_initial_miss"])

    def test_annotate_gold_hops_marks_completed_steps(self):
        item = {"gold_path": ["T1000", "CAPEC-100", "CWE-100"]}
        trace = [
            {"step": 1, "new_evidence": ["T1000"]},
            {"step": 2, "new_evidence": ["CAPEC-100"]},
            {"step": 3, "new_evidence": ["CWE-100"]},
        ]
        annotated = annotate_gold_hops(item, trace)
        self.assertEqual(annotated[1]["gold_hop_completed"], ["attack_to_capec"])
        self.assertEqual(annotated[2]["gold_hop_completed"], ["capec_to_cwe"])

    def test_score_path_partial_second_hop_missing(self):
        item = {"gold_path": ["T1000", "CAPEC-100", "CWE-100"]}
        evidence = [
            Evidence(self.toy_corpus().by_identifier["T1000"], 1.0, "open"),
            Evidence(self.toy_corpus().by_identifier["CAPEC-100"], 1.0, "expand"),
        ]
        scores = score_path(item, evidence, [{"step": 1, "new_evidence": ["T1000", "CAPEC-100"]}])
        self.assertFalse(scores["full_path_retrieved"])
        self.assertTrue(scores["first_hop_success"])
        self.assertFalse(scores["second_hop_success"])

    def test_heuristic_select_from_evidence(self):
        item = {
            "choices": [
                "A: CWE-101 - no",
                "B: CWE-100 - yes",
                "C: CWE-102 - no",
                "D: CWE-103 - no",
            ]
        }
        evidence = [Evidence(self.toy_corpus().by_identifier["CWE-100"], 1.0, "expand")]
        selected, debug = heuristic_select(item, "", evidence)
        self.assertEqual(selected, "B")
        self.assertEqual(debug["source"], "evidence_order")

    def test_classify_failure_cases(self):
        self.assertEqual(
            classify_failure(True, {"full_path_retrieved": False, "first_hop_success": False, "second_hop_success": False}, "A", []),
            "answer_correct_without_gold_path",
        )
        self.assertEqual(
            classify_failure(False, {"full_path_retrieved": True, "first_hop_success": True, "second_hop_success": True}, "A", []),
            "gold_path_found_answer_wrong",
        )

    def test_build_diagnostic_items_hidden_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attack_rel, capec_rel = self.write_relations(root)
            items = build_diagnostic_items(self.toy_corpus(), attack_rel, capec_rel, seed=7, hidden_id=True)
            item = items[0]
            self.assertTrue(item["hidden_id"])
            self.assertEqual(item["gold_entity_name"], "Password Guessing")
            # Name-only phrasing: the technique ID must not leak into the question.
            self.assertIn("Password Guessing", item["question"])
            self.assertNotIn("T1000", item["question"])
            self.assertEqual(item["gold_path"], ["T1000", "CAPEC-100", "CWE-100"])

    def test_score_resolution_and_failure(self):
        item = {"gold_path": ["T1000", "CAPEC-100", "CWE-100"], "hidden_id": True}
        correct = score_resolution(item, {"chosen_id": "T1000", "ambiguous": False, "mode": "exact"})
        self.assertTrue(correct["resolution_evaluated"])
        self.assertTrue(correct["resolution_correct"])

        wrong = score_resolution(item, {"chosen_id": "T9999", "ambiguous": False, "mode": "fallback"})
        self.assertTrue(wrong["resolution_evaluated"])
        self.assertFalse(wrong["resolution_correct"])
        reason = classify_failure(
            False,
            {"full_path_retrieved": False, "first_hop_success": False, "second_hop_success": False},
            "A",
            [],
            wrong,
        )
        self.assertEqual(reason, "resolution_failure")

    def test_score_resolution_not_evaluated_without_hidden_id(self):
        item = {"gold_path": ["T1000", "CAPEC-100", "CWE-100"], "hidden_id": False}
        scores = score_resolution(item, {"chosen_id": "T1000"})
        self.assertFalse(scores["resolution_evaluated"])

    def test_build_diagnostic_items_capec_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attack_rel, capec_rel = self.write_relations(root)
            items = build_diagnostic_items(self.toy_corpus(), attack_rel, capec_rel, seed=7, hidden_id=True, entry="capec")
            item = items[0]
            self.assertEqual(item["direction"], "forward")
            self.assertEqual(item["entry_id"], "CAPEC-100")
            self.assertEqual(item["gold_path"], ["CAPEC-100", "CWE-100"])
            self.assertIn("Guessing Passwords", item["question"])
            self.assertNotIn("CAPEC-100", item["question"])  # hidden -> name only

    def test_build_diagnostic_items_cwe_reverse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attack_rel, capec_rel = self.write_relations(root)
            items = build_diagnostic_items(self.toy_corpus(), attack_rel, capec_rel, seed=7, entry="cwe")
            item = items[0]
            self.assertEqual(item["direction"], "reverse")
            self.assertEqual(item["entry_id"], "CWE-100")
            self.assertEqual(item["gold_set"], ["CAPEC-100"])
            self.assertNotIn("choices", item)

    def test_score_path_two_node_capec_entry(self):
        item = {"gold_path": ["CAPEC-100", "CWE-100"]}
        evidence = [
            Evidence(self.toy_corpus().by_identifier["CAPEC-100"], 1.0, "open"),
            Evidence(self.toy_corpus().by_identifier["CWE-100"], 1.0, "expand"),
        ]
        scores = score_path(item, evidence, [{"step": 1, "new_evidence": ["CAPEC-100", "CWE-100"]}])
        self.assertTrue(scores["full_path_retrieved"])
        self.assertTrue(scores["second_hop_success"])

    def test_run_eval_reverse_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_relations(root)
            self.write_chunks(root)
            trace_log = root / "logs" / "trace.jsonl"
            args = build_parser().parse_args(
                ["1", "--root", str(root), "--trace-log", str(trace_log),
                 "--modes", "agentic_lexical", "--max-steps", "6", "--entry", "cwe"]
            )
            results, summary = run_eval(args)
            row = results[0]
            self.assertEqual(row["direction"], "reverse")
            self.assertEqual(row["predicted_set"], ["CAPEC-100"])
            self.assertEqual(row["set_recall"], 1.0)
            self.assertTrue(row["correct"])
            self.assertEqual(summary["reverse_total"], 1)
            self.assertEqual(summary["reverse_exact_set_match"], 1)

    def test_run_eval_hidden_id_reports_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_relations(root)
            self.write_chunks(root)
            trace_log = root / "logs" / "trace.jsonl"
            args = build_parser().parse_args(
                ["1", "--root", str(root), "--trace-log", str(trace_log),
                 "--modes", "agentic_lexical", "--max-steps", "6", "--hidden-id"]
            )
            results, summary = run_eval(args)
            self.assertTrue(results[0]["hidden_id"])
            self.assertTrue(results[0]["resolution_evaluated"])
            self.assertEqual(results[0]["resolution_chosen_id"], "T1000")
            self.assertTrue(results[0]["resolution_correct"])
            self.assertEqual(summary["resolution_evaluated"], 1)
            self.assertEqual(summary["entity_resolution_accuracy"], 1.0)

    def test_run_eval_agentic_granite_mode(self):
        import eval_agentic_multihop as evalmod

        scripts = [
            json.dumps({"tool": "open", "args": {"id": "T1000"}}),
            json.dumps({"tool": "expand", "args": {"id": "T1000"}}),
            json.dumps({"tool": "expand", "args": {"id": "CAPEC-100"}}),
            json.dumps({"answer": {"text": "CWE-100", "cited_ids": ["T1000", "CAPEC-100", "CWE-100"],
                                   "path": ["T1000", "CAPEC-100", "CWE-100"]}}),
        ]

        class MockClient:
            def __init__(self, *a, **k):
                self.calls = 0

            def complete(self, messages):
                out = scripts[min(self.calls, len(scripts) - 1)]
                self.calls += 1
                return out

        original = evalmod.OpenAIChatClient
        evalmod.OpenAIChatClient = MockClient
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.write_relations(root)
                self.write_chunks(root)
                trace_log = root / "logs" / "trace.jsonl"
                args = build_parser().parse_args(
                    ["1", "--root", str(root), "--trace-log", str(trace_log), "--modes", "agentic_granite"]
                )
                results, summary = run_eval(args)
                row = results[0]
                self.assertEqual(row["mode"], "agentic_granite")
                # The LLM-driven loop recovered the full gold path deterministically.
                self.assertTrue(row["full_path_retrieved"])
        finally:
            evalmod.OpenAIChatClient = original

    def test_run_eval_writes_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_relations(root)
            self.write_chunks(root)
            trace_log = root / "logs" / "trace.jsonl"
            args = build_parser().parse_args(
                [
                    "1",
                    "--root",
                    str(root),
                    "--trace-log",
                    str(trace_log),
                    "--modes",
                    "agentic_lexical",
                    "--max-steps",
                    "6",
                ]
            )
            results, summary = run_eval(args)
            self.assertEqual(len(results), 1)
            self.assertTrue(trace_log.exists())
            self.assertEqual(summary["total"], 1)
            self.assertIn("full_path_retrieved", results[0])


if __name__ == "__main__":
    unittest.main()
