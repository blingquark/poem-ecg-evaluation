# POEM ECG Evaluation

Selected evaluation utilities and tests for the POEM ECG beat-classification study. The example starts from synthetic N/S/V predictions and runs through the same metric definitions used by the companion tests. Its JSON report contains beat-level and group-level scores for the example.

## Quick Start

From the repository root, use Python 3.10 or newer. Both commands run on CPU using the Python standard library.

```bash
python -m poem_eval examples/synthetic_predictions.csv
python -m unittest discover -s tests -v
```

The example contains 12 beats in four synthetic groups. It produces an accuracy of 0.75, an S-F1 of 0.40, an S-AP of approximately 0.7333, a Group-S-F1 of approximately 0.3333, and 250 false-positive S predictions per 1,000 beats in groups with no true S beats.

## Repository Layout

- `poem_eval/` contains the metric functions, CSV evaluation, and command-line entry point.
- `examples/` contains synthetic prediction data for the runnable example.
- `tests/` contains metric unit tests and end-to-end command-line tests with hand-computed expected results.

## Input and Metrics

Each CSV row contains `group,true_label,score_N,score_S,score_V`. The group is a nonempty identifier, the label is `N`, `S`, or `V`, and the three probability scores must be between 0 and 1 and sum to 1 within `1e-6`. Predictions use the highest score and break ties in N/S/V order.

- The confusion matrix uses true labels as rows and predictions as columns, both in N/S/V order. Per-class precision, recall, F1, macro-F1, and accuracy are reported as fractions.
- S-AP is non-interpolated average precision for S versus the other labels, with equal scores processed together.
- Group-S-F1 is the unweighted mean of S-F1 over groups with at least one true S beat.
- The no-S false-positive rate is the unweighted mean of false-positive S predictions per 1,000 beats over groups with zero true S beats.

Undefined metrics appear as JSON `null`. Macro-F1 requires three defined class F1 scores, S-AP requires both positive and negative examples, and each group metric requires at least one eligible group. All example values are derived from the synthetic fixture.
