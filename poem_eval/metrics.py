from itertools import groupby
from statistics import fmean


CLASSES = ("N", "S", "V")


def confusion_matrix(true_labels, predicted_labels):
    """Count label pairs in a matrix with true labels as rows."""
    class_index = {label: index for index, label in enumerate(CLASSES)}
    counts = [[0 for _ in CLASSES] for _ in CLASSES]
    for true_label, predicted_label in zip(true_labels, predicted_labels, strict=True):
        if true_label not in class_index or predicted_label not in class_index:
            raise ValueError("Labels must be N, S, or V")
        counts[class_index[true_label]][class_index[predicted_label]] += 1
    return tuple(tuple(row) for row in counts)


def summarize_confusion(counts):
    """Compute classification scores from a 3-by-3 confusion matrix in N/S/V order."""
    support = tuple(map(sum, counts))
    predicted = tuple(map(sum, zip(*counts)))
    scores = {}
    for index, label in enumerate(CLASSES):
        true_positive = counts[index][index]
        actual_count = support[index]
        predicted_count = predicted[index]
        f1_denominator = actual_count + predicted_count
        scores[label] = {
            "precision": true_positive / predicted_count if predicted_count else None,
            "recall": true_positive / actual_count if actual_count else None,
            "f1": 2 * true_positive / f1_denominator if f1_denominator else None,
        }

    f1_scores = [scores[label]["f1"] for label in CLASSES]
    total = sum(support)
    correct = sum(counts[index][index] for index in range(len(CLASSES)))
    return {
        "per_class": scores,
        "macro_f1": None if None in f1_scores else sum(f1_scores) / len(CLASSES),
        "accuracy": correct / total if total else None,
    }


def s_average_precision(true_labels, s_scores):
    """Compute non-interpolated average precision over distinct S scores."""
    positive_count = true_labels.count("S")
    if positive_count == 0 or positive_count == len(true_labels):
        return None

    ranked = sorted(zip(s_scores, true_labels), key=lambda pair: pair[0], reverse=True)
    seen = 0
    true_positives = 0
    weighted_precision = 0.0
    for _, tied_rows in groupby(ranked, key=lambda pair: pair[0]):
        newly_positive = 0
        for _, label in tied_rows:
            seen += 1
            newly_positive += label == "S"
        true_positives += newly_positive
        weighted_precision += newly_positive * (true_positives / seen)
    return weighted_precision / positive_count


def summarize_groups(group_confusions):
    """Give equal weight to each eligible group within each metric."""
    s_f1_scores = []
    no_s_fp_rates = []
    for counts in group_confusions:
        s_row = counts[1]
        if sum(s_row):
            s_f1_scores.append(summarize_confusion(counts)["per_class"]["S"]["f1"])
        else:
            false_positives = sum(row[1] for row in counts)
            no_s_fp_rates.append(1000 * false_positives / sum(map(sum, counts)))

    return {
        "group_s_f1": fmean(s_f1_scores) if s_f1_scores else None,
        "no_s_fp_per_1000": fmean(no_s_fp_rates) if no_s_fp_rates else None,
    }
