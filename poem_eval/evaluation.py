from collections import defaultdict
import csv
import math

from .metrics import (
    CLASSES,
    confusion_matrix,
    s_average_precision,
    summarize_confusion,
    summarize_groups,
)


COLUMNS = ["group", "true_label", *(f"score_{label}" for label in CLASSES)]


def read_predictions(path):
    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file, strict=True)
        if next(reader, None) != COLUMNS:
            raise ValueError(f"Expected columns in order: {', '.join(COLUMNS)}")

        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(COLUMNS):
                raise ValueError(f"Row {row_number}: expected {len(COLUMNS)} fields")
            group, true_label, *fields = (field.strip() for field in row)
            if not group or true_label not in CLASSES:
                raise ValueError(f"Row {row_number}: expected a group and a true label in N/S/V")
            try:
                scores = tuple(map(float, fields))
            except ValueError as error:
                raise ValueError(f"Row {row_number}: scores must be numbers") from error
            if not all(0 <= score <= 1 for score in scores) or not math.isclose(
                sum(scores), 1.0, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(f"Row {row_number}: probabilities must be in [0, 1] and sum to 1")
            yield group, true_label, scores


def evaluate_csv(path):
    true_labels = []
    predicted_labels = []
    s_scores = []
    group_labels = defaultdict(list)
    for group, true_label, scores in read_predictions(path):
        prediction = CLASSES[max(range(len(CLASSES)), key=scores.__getitem__)]
        true_labels.append(true_label)
        predicted_labels.append(prediction)
        s_scores.append(scores[1])
        group_labels[group].append((true_label, prediction))

    if not true_labels:
        raise ValueError("No prediction rows")

    counts = confusion_matrix(true_labels, predicted_labels)
    group_confusions = (confusion_matrix(*zip(*pairs)) for pairs in group_labels.values())
    return {
        "num_beats": len(true_labels),
        "num_groups": len(group_labels),
        "class_order": CLASSES,
        "confusion_matrix": counts,
        **summarize_confusion(counts),
        "s_average_precision": s_average_precision(true_labels, s_scores),
        **summarize_groups(group_confusions),
    }
