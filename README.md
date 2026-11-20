# ECG evaluation tests

This repository contains selected evaluation unit tests associated with an ECG beat-classification study. The tests use hand-constructed labels for three classes (N, S, and V) and check confusion counts, per-class precision, recall and F1, macro-F1, and overall accuracy.

## Run the tests

Use Python 3.10 or newer. The tests run on CPU with the Python standard library.

```bash
python -m unittest discover -s tests -v
```

## Scope

A passing run verifies the metric calculations on synthetic examples. True labels are the rows of the confusion matrix, and predicted labels are the columns, both in N/S/V order. Undefined scores are represented by `None`. Macro-F1 averages the three class F1 scores when all three are defined.
