import unittest

from poem_eval.metrics import confusion_matrix, summarize_confusion


class ConfusionMatrixTests(unittest.TestCase):
    def test_rows_are_true_labels(self):
        true_labels = ["N", "N", "S", "V", "V"]
        predicted_labels = ["N", "S", "S", "S", "V"]
        counts = confusion_matrix(true_labels, predicted_labels)
        self.assertEqual(counts, ((1, 1, 0), (0, 1, 0), (0, 1, 1)))

    def test_rejects_unequal_lengths(self):
        with self.assertRaises(ValueError):
            confusion_matrix(["N", "S"], ["N"])

    def test_rejects_unknown_labels(self):
        with self.assertRaises(ValueError):
            confusion_matrix(["N"], ["X"])


class ClassificationMetricTests(unittest.TestCase):
    def test_scores_match_hand_calculation(self):
        scores = summarize_confusion(((8, 2, 0), (1, 2, 1), (0, 3, 7)))
        supraventricular = scores["per_class"]["S"]
        self.assertAlmostEqual(supraventricular["precision"], 2 / 7)
        self.assertAlmostEqual(supraventricular["recall"], 2 / 4)
        self.assertAlmostEqual(supraventricular["f1"], 4 / 11)
        self.assertAlmostEqual(scores["macro_f1"], (16 / 19 + 4 / 11 + 14 / 18) / 3)
        self.assertAlmostEqual(scores["accuracy"], 17 / 24)

    def test_supported_class_with_no_predictions(self):
        scores = summarize_confusion(((10, 0, 0), (3, 0, 0), (0, 0, 2)))
        self.assertEqual(scores["per_class"]["S"], {"precision": None, "recall": 0, "f1": 0})

    def test_false_positives_for_an_unobserved_class(self):
        scores = summarize_confusion(((9, 1, 0), (0, 0, 0), (0, 1, 2)))
        self.assertEqual(scores["per_class"]["S"], {"precision": 0, "recall": None, "f1": 0})

    def test_class_absent_from_truth_and_predictions(self):
        scores = summarize_confusion(((10, 0, 0), (0, 0, 0), (0, 0, 2)))
        self.assertEqual(scores["per_class"]["S"], {"precision": None, "recall": None, "f1": None})
        self.assertIsNone(scores["macro_f1"])
