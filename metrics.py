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
