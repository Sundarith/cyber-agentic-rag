import csv
import json
import tempfile
import unittest
from pathlib import Path

from eval_agentic_rcm import (
    answer_contains_cwe,
    build_parser,
    extract_cwe_ids,
    extract_cve_id,
    filter_rows,
    load_cve_cwe_index,
    load_rows,
    retrieval_backend_for_mode,
    resolve_data_path,
    run_eval,
    sample_rows,
    summarize,
    summarize_by_mode,
)


class AgenticRCMEvalTests(unittest.TestCase):
    def test_extract_cve_id(self):
        self.assertEqual(extract_cve_id("https://x/CVE-2021-44228"), "CVE-2021-44228")
        self.assertEqual(extract_cve_id("none"), "")

    def test_answer_contains_cwe_uses_identifier_boundary(self):
        self.assertTrue(answer_contains_cwe("Final answer: CWE-451", "CWE-451"))
        self.assertFalse(answer_contains_cwe("Related: CWE-4510", "CWE-451"))

    def test_extract_cwe_ids_dedupes_in_order(self):
        self.assertEqual(extract_cwe_ids("CWE-79, cwe-89, CWE-79"), ["CWE-79", "CWE-89"])

    def test_retrieval_backend_for_mode(self):
        self.assertEqual(retrieval_backend_for_mode("agentic_lexical"), "lexical")
        self.assertEqual(retrieval_backend_for_mode("agentic_hybrid_no_graph"), "legacy_hybrid")

    def test_load_rows_and_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cti-rcm.tsv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["URL", "GT", "Prompt"], delimiter="\t")
                writer.writeheader()
                writer.writerow({"URL": "https://nvd/CVE-1-1", "GT": "CWE-79", "Prompt": "prompt one"})
                writer.writerow({"URL": "https://nvd/CVE-2024-1234", "GT": "CWE-89", "Prompt": "prompt two"})

            rows = load_rows(path)
            mapped = filter_rows(rows, {"CVE-2024-1234": ["CWE-89"]}, nvd_mapped=True)
            unmapped = filter_rows(rows, {"CVE-2024-1234": ["CWE-89"]}, nvd_unmapped=True)
            self.assertEqual(len(mapped), 1)
            self.assertEqual(mapped[0]["GT"], "CWE-89")
            self.assertEqual(len(unmapped), 1)

    def test_load_cve_cwe_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            path.write_text(json.dumps({"cve-2024-1234": ["cwe-89"]}), encoding="utf-8")
            self.assertEqual(load_cve_cwe_index(path), {"CVE-2024-1234": ["CWE-89"]})

    def test_sample_rows_is_seeded(self):
        rows = [{"i": str(i)} for i in range(10)]
        self.assertEqual(sample_rows(rows, 3, 7), sample_rows(rows, 3, 7))

    def test_summarize(self):
        summary = summarize(
            [
                {
                    "passed": True,
                    "nvd_mapped": True,
                    "duration_s": 1.0,
                    "tool_calls": 2,
                    "evidence_count": 3,
                    "gold_in_evidence": True,
                    "verifier_rejections": 1,
                    "graph_expansions": 1,
                    "failure_reason": "",
                },
                {
                    "passed": False,
                    "nvd_mapped": False,
                    "duration_s": 3.0,
                    "tool_calls": 4,
                    "evidence_count": 1,
                    "gold_in_evidence": False,
                    "verifier_rejections": 2,
                    "graph_expansions": 0,
                    "failure_reason": "gold_not_in_evidence",
                },
            ]
        )
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["mapped_total"], 1)
        self.assertEqual(summary["unmapped_total"], 1)
        self.assertAlmostEqual(summary["avg_duration_s"], 2.0)
        self.assertAlmostEqual(summary["avg_tool_calls"], 3.0)
        self.assertAlmostEqual(summary["avg_evidence_count"], 2.0)
        self.assertAlmostEqual(summary["gold_evidence_recall"], 0.5)
        self.assertEqual(summary["failure_reasons"], {"gold_not_in_evidence": 1})

    def test_summarize_by_mode(self):
        summary = summarize_by_mode(
            [
                {"mode": "a", "passed": True, "nvd_mapped": False},
                {"mode": "b", "passed": False, "nvd_mapped": False, "failure_reason": "no_evidence"},
            ]
        )
        self.assertEqual(summary["a"]["passed"], 1)
        self.assertEqual(summary["b"]["failure_reasons"], {"no_evidence": 1})

    def test_run_eval_writes_mode_aware_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            dataset = root / "cti-rcm.tsv"
            trace_log = root / "logs" / "trace.jsonl"

            with dataset.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["URL", "GT", "Prompt"], delimiter="\t")
                writer.writeheader()
                writer.writerow(
                    {
                        "URL": "https://nvd.nist.gov/vuln/detail/CVE-2024-1234",
                        "GT": "CWE-79",
                        "Prompt": "What CWE root cause applies to CVE-2024-1234?",
                    }
                )

            (processed / "cve_cwe_index.json").write_text(
                json.dumps({"CVE-2024-1234": ["CWE-79"]}),
                encoding="utf-8",
            )
            (processed / "cve_chunks.jsonl").write_text(
                json.dumps(
                    {
                        "id": "cve-CVE-2024-1234",
                        "identifier": "CVE-2024-1234",
                        "source": "CVE",
                        "type": "vulnerability",
                        "name": "Synthetic XSS CVE",
                        "text": "CVE-2024-1234 is a reflected script injection issue associated with CWE-79.",
                    }
                )
                + "\n",
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
                        "text": "CWE-79 is improper neutralization of input during web page generation.",
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
                    "agentic_lexical",
                    "--max-steps",
                    "4",
                ]
            )
            results, summary = run_eval(args)

            self.assertEqual(len(results), 2)
            self.assertEqual({row["mode"] for row in results}, {"lexical_baseline", "agentic_lexical"})
            self.assertTrue(all(row["gold_in_evidence"] for row in results))
            self.assertIn("agentic_lexical", summary)
            self.assertTrue(trace_log.exists())
            first_trace_row = json.loads(trace_log.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("trace_actions", first_trace_row)

    def test_resolve_data_path_uses_root_for_default(self):
        root = Path("/tmp/example")
        self.assertEqual(
            resolve_data_path(root, Path("data/processed/cve_cwe_index.json"), Path("data/processed/cve_cwe_index.json")),
            root / "data/processed/cve_cwe_index.json",
        )
        self.assertEqual(
            resolve_data_path(root, Path("custom.json"), Path("data/processed/cve_cwe_index.json")),
            Path("custom.json"),
        )


if __name__ == "__main__":
    unittest.main()
