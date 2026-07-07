"""Held-out NER backtest: comparator-space accuracy on unseen labels.

Reproduces ml/train.py's exact train/test split (sklearn train_test_split,
test_size=0.2, random_state=42, over ml/training_data/dataset.jsonl in file
order) and replays the HELD-OUT rows' stored OCR text through the ONNX NER
extractor, scoring each field with the shared comparators against ground
truth.

Why this exists: the training report's entity-span F1 is not the product
metric. This script answers the product question — "how often would the
NER value have been correct?" — in the same comparator space used for
every other accuracy number in the project.

Scoring notes:
- NER accuracy/coverage is computed on all held-out rows.
- The selected-value head-to-head EXCLUDES rows imported by
  ml/import_datasets.py (their stored field values ARE ground truth, so
  'selected accuracy' would be meaningless for them). Both n's are shown.
- These held-out rows were unseen by the model, but they are in-domain
  (same carriers/label styles as training). Fresh shadow-scan data via
  eval_candidates.py remains the final arbiter.

Usage (from the project root):
  venv/bin/python ner_backtest.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

import config
from comparators import compare_field, has_ground_truth
from ner_extractor import NerExtractor
import storage

DATASET_PATH = Path("ml/training_data/dataset.jsonl")

FIELDS = (
    "recipient_name",
    "street_address",
    "city",
    "state",
    "zip_code",
    "tracking_number",
    "carrier",
)


def held_out_ids():
    """Reproduce ml/train.py's split and return the held-out example ids."""
    examples = []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if len(examples) < 25:
        raise SystemExit("dataset too small for a held-out split (train.py trains on all)")
    _, test = train_test_split(examples, test_size=0.2, random_state=42)
    return [example["id"] for example in test]


def main():
    if not (Path(config.NER_MODEL_DIR) / "label_reader.onnx").exists():
        print("No ONNX model artifact — run ml/train.py first.")
        return 1

    ids = held_out_ids()
    extractor = NerExtractor(config.NER_MODEL_DIR)
    print(f"Held-out examples: {len(ids)} (seed-42 split of {DATASET_PATH})")
    print(f"Model: {extractor.version}\n")

    per_field = defaultdict(lambda: {
        "n": 0, "ner_covered": 0, "ner_correct": 0,
        "n_scans": 0, "ner_correct_scans": 0, "selected_correct_scans": 0,
    })
    missing_rows = 0

    for label_id in ids:
        row = storage.get_label(label_id)
        if row is None or not row.get("ground_truth") or not row.get("ocr_text"):
            missing_rows += 1
            continue
        gt = row["ground_truth"]
        predictions = extractor.predict_fields(row["ocr_text"])

        for field in FIELDS:
            if field not in gt or not has_ground_truth(gt, field):
                continue
            stats = per_field[field]
            stats["n"] += 1
            prediction = predictions.get(field)
            ner_value = (prediction or {}).get("value") or ""
            ner_correct = bool(
                ner_value and compare_field({field: ner_value}, gt, field)
            )
            if ner_value:
                stats["ner_covered"] += 1
            if ner_correct:
                stats["ner_correct"] += 1

            # Head-to-head vs the pipeline's selected value — real scans only
            # (dataset-imported rows store ground truth in the value columns).
            if (row.get(f"{field}_source") or "") != "dataset":
                stats["n_scans"] += 1
                if ner_correct:
                    stats["ner_correct_scans"] += 1
                if compare_field(row, gt, field):
                    stats["selected_correct_scans"] += 1

    if missing_rows:
        print(f"Skipped {missing_rows} held-out ids without a usable DB row\n")

    header = (f"{'Field':<18} {'n':>4} {'Coverage':>9} {'NER acc':>8} "
              f"| {'real-scan n':>11} {'NER acc':>8} {'Selected':>9}")
    print(header)
    print("-" * len(header))
    for field in FIELDS:
        stats = per_field[field]
        n, n_scans = stats["n"], stats["n_scans"]
        coverage = f"{stats['ner_covered'] / n * 100:.0f}%" if n else "—"
        ner_acc = f"{stats['ner_correct'] / n * 100:.1f}%" if n else "—"
        ner_scan = (f"{stats['ner_correct_scans'] / n_scans * 100:.1f}%"
                    if n_scans else "—")
        sel_scan = (f"{stats['selected_correct_scans'] / n_scans * 100:.1f}%"
                    if n_scans else "—")
        print(f"{field:<18} {n:>4} {coverage:>9} {ner_acc:>8} "
              f"| {n_scans:>11} {ner_scan:>8} {sel_scan:>9}")

    print(
        "\nLeft block: all held-out rows. Right block: head-to-head on rows"
        "\nthat came from real scans (selected value comparable). Small n —"
        "\ntreat as directional; fresh shadow data (eval_candidates.py) decides."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
