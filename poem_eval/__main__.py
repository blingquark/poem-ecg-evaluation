import argparse
import csv
import json
from pathlib import Path

from .evaluation import evaluate_csv


def main():
    parser = argparse.ArgumentParser(description="Evaluate N/S/V beat predictions from a CSV file.")
    parser.add_argument("csv", type=Path, help="Prediction CSV with group, label, and probabilities")
    args = parser.parse_args()
    try:
        report = evaluate_csv(args.csv)
    except (OSError, csv.Error, ValueError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
