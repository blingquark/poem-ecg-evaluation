import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURE = REPOSITORY / "examples" / "synthetic_predictions.csv"
HEADER = "group,true_label,score_N,score_S,score_V\n"


class EvaluationCLITests(unittest.TestCase):
    def run_cli(self, csv_path):
        return subprocess.run(
            [sys.executable, "-B", "-m", "poem_eval", str(csv_path)],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
        )

    def run_csv(self, content):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "predictions.csv"
            csv_path.write_text(content, encoding="utf-8")
            return self.run_cli(csv_path)

    def read_report(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_prediction_csv_to_evaluation_report(self):
        report = self.read_report(self.run_cli(FIXTURE))
        self.assertEqual(report["class_order"], ["N", "S", "V"])
        self.assertEqual(report["num_beats"], 12)
        self.assertEqual(report["num_groups"], 4)
        self.assertEqual(report["confusion_matrix"], [[4, 1, 0], [2, 1, 0], [0, 0, 4]])
        self.assertAlmostEqual(report["accuracy"], 9 / 12)
        self.assertAlmostEqual(report["per_class"]["S"]["f1"], 2 / 5)
        self.assertAlmostEqual(report["macro_f1"], (8 / 11 + 2 / 5 + 1) / 3)
        self.assertAlmostEqual(report["s_average_precision"], 11 / 15)
        self.assertAlmostEqual(report["group_s_f1"], 1 / 3)
        self.assertAlmostEqual(report["no_s_fp_per_1000"], 250)

    def test_reordering_rows_with_tied_scores_preserves_report(self):
        header, *rows = FIXTURE.read_text(encoding="utf-8").splitlines()
        reordered = "\n".join([header, *reversed(rows)]) + "\n"
        original = self.read_report(self.run_cli(FIXTURE))
        self.assertEqual(self.read_report(self.run_csv(reordered)), original)

    def test_absent_support_is_json_null(self):
        report = self.read_report(self.run_csv(HEADER + "only,N,1,0,0\n"))
        self.assertIsNone(report["macro_f1"])
        self.assertIsNone(report["s_average_precision"])
        self.assertIsNone(report["group_s_f1"])
        self.assertEqual(report["no_s_fp_per_1000"], 0)

    def test_argmax_ties_use_first_class_and_absent_groups_are_null(self):
        report = self.read_report(self.run_csv(HEADER + "only,S,0.5,0.5,0\n"))
        self.assertEqual(report["confusion_matrix"], [[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        self.assertIsNone(report["s_average_precision"])
        self.assertEqual(report["group_s_f1"], 0)
        self.assertIsNone(report["no_s_fp_per_1000"])

    def test_malformed_csv_exits_without_partial_report(self):
        cases = (
            ("group,true_label,score_N\n", "Expected columns"),
            (HEADER, "No prediction rows"),
            (HEADER + "alpha,X,1,0,0\n", "Row 2"),
            (HEADER + "alpha,N,0.8,0.8,0\n", "Row 2"),
            (HEADER + "alpha,N,nan,0,1\n", "Row 2"),
            (HEADER + "alpha,N,0.8,0.2\n", "Row 2"),
        )
        for content, expected_error in cases:
            with self.subTest(content=content):
                result = self.run_csv(content)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn(expected_error, result.stderr)
