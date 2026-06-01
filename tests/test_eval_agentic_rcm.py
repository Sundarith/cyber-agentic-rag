import csv
import json
import tempfile
import unittest
from pathlib import Path

from eval_agentic_rcm import (
    answer_contains_cwe,
    extract_cve_id,
    filter_rows,
    load_cve_cwe_index,
    load_rows,
    resolve_data_path,
    sample_rows,
    summarize,
)


class AgenticRCMEvalTests(unittest.TestCase):
    def test_extract_cve_id(self):
        self.assertEqual(extract_cve_id("https://x/CVE-2021-44228"), "CVE-2021-44228")
        self.assertEqual(extract_cve_id("none"), "")

    def test_answer_contains_cwe_uses_identifier_boundary(self):
        self.assertTrue(answer_contains_cwe("Final answer: CWE-451", "CWE-451"))
        self.assertFalse(answer_contains_cwe("Related: CWE-4510", "CWE-451"))

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
                {"passed": True, "nvd_mapped": True, "duration_s": 1.0, "tool_calls": 2},
                {"passed": False, "nvd_mapped": False, "duration_s": 3.0, "tool_calls": 4},
            ]
        )
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["mapped_total"], 1)
        self.assertEqual(summary["unmapped_total"], 1)
        self.assertAlmostEqual(summary["avg_duration_s"], 2.0)
        self.assertAlmostEqual(summary["avg_tool_calls"], 3.0)

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
