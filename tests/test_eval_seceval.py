import json
import tempfile
import unittest
from pathlib import Path

from eval_seceval import (
    build_parser,
    extract_answer_letters,
    filter_topics,
    load_rows,
    normalize_answer,
    run_eval,
    summarize,
)


class SecEvalTests(unittest.TestCase):
    def test_normalize_answer_dedupes_and_sorts(self):
        self.assertEqual(normalize_answer("ca"), "AC")
        self.assertEqual(normalize_answer("AA"), "A")
        self.assertEqual(normalize_answer(""), "")

    def test_extract_answer_letters_uses_final_answer_like_line(self):
        self.assertEqual(extract_answer_letters("Reasoning says A, but final:\nBD"), "BD")

    def test_load_rows_filters_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "questions.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "ok",
                            "question": "q",
                            "choices": ["A: one", "B: two", "C: three", "D: four"],
                            "answer": "AC",
                            "topics": ["WebSecurity"],
                        },
                        {
                            "id": "bad",
                            "question": "q",
                            "choices": ["A: one", "B: two", "C: three", "D: four"],
                            "answer": "",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            rows = load_rows(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["answer"], "AC")

    def test_filter_topics(self):
        rows = [{"topics": ["WebSecurity"]}, {"topics": ["Cryptography"]}]
        self.assertEqual(filter_topics(rows, ["websecurity"]), [rows[0]])

    def test_summarize(self):
        summary = summarize(
            [
                {"correct": True, "duration_s": 1.0, "evidence_count": 2, "tool_calls": 1, "topics": ["A"]},
                {
                    "correct": False,
                    "duration_s": 3.0,
                    "evidence_count": 4,
                    "tool_calls": 2,
                    "topics": ["A", "B"],
                    "failure_reason": "wrong_choice",
                },
            ]
        )
        self.assertEqual(summary["correct"], 1)
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["failure_reasons"], {"wrong_choice": 1})
        self.assertAlmostEqual(summary["topic_accuracy"]["A"]["accuracy"], 0.5)

    def test_run_eval_writes_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            dataset = root / "questions.json"
            trace_log = root / "logs" / "trace.jsonl"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "id": "q1",
                            "source": "unit",
                            "question": "Which issue is cross-site scripting?",
                            "choices": [
                                "A: SQL injection",
                                "B: Cross-site scripting",
                                "C: Buffer overflow",
                                "D: Weak password storage",
                            ],
                            "answer": "B",
                            "topics": ["WebSecurity"],
                            "keyword": "XSS",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (processed / "cwe_chunks.jsonl").write_text(
                json.dumps(
                    {
                        "id": "cwe-79",
                        "identifier": "CWE-79",
                        "source": "CWE",
                        "type": "weakness",
                        "name": "Cross-site Scripting",
                        "text": "CWE-79 covers cross-site scripting in web pages.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                [
                    "1",
                    "--root",
                    str(root),
                    "--dataset",
                    str(dataset),
                    "--trace-log",
                    str(trace_log),
                    "--modes",
                    "lexical_baseline",
                ]
            )
            results, summary = run_eval(args)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["gold_answer"], "B")
            self.assertIn("selected_answer", results[0])
            self.assertTrue(trace_log.exists())
            self.assertEqual(summary["total"], 1)


if __name__ == "__main__":
    unittest.main()
