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
from scoring import normalize_comparison_value
from selection import plausible_recipient_name
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
    name_rows = []  # per-row recipient_name data for the policy simulation
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
                selected_correct = compare_field(row, gt, field)
                if ner_correct:
                    stats["ner_correct_scans"] += 1
                if selected_correct:
                    stats["selected_correct_scans"] += 1
                if field == "recipient_name":
                    name_rows.append({
                        "base_value": row.get(field) or "",
                        "base_source": row.get(f"{field}_source") or "",
                        "base_correct": selected_correct,
                        "ner_value": ner_value,
                        "ner_conf": (prediction or {}).get("confidence", 0.0),
                        "ner_correct": ner_correct,
                    })

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

    simulate_name_policy(name_rows)
    return 0


# ---------------------------------------------------------------------------
# Gated recipient_name policy simulation (offline evidence for the
# NerNamePolicy gate — see docs; recipient_name ONLY, by construction).
# ---------------------------------------------------------------------------

FILL_THRESHOLD = 0.50  # blank baseline is ~0% accurate; filling is nearly free


def _gate_fires(row, fill_thr, override_thr):
    """Mirror of the NerNamePolicy gate, applied to one backtest row.

    Returns the reason string when the gate would replace the selected value
    with the NER value, else None. llm-sourced selections are never touched
    (calibration: recipient_name|llm = 95.4%, above anything NER offers).
    """
    if row["base_source"] == "llm":
        return None
    if not row["ner_value"]:
        return None
    if normalize_comparison_value(row["ner_value"]) == normalize_comparison_value(
        row["base_value"]
    ):
        return None  # same value — keep legacy provenance, correctness unchanged
    if not plausible_recipient_name(row["ner_value"]):
        return None
    if not row["base_value"]:
        return "fill" if row["ner_conf"] >= fill_thr else None
    return "override" if row["ner_conf"] >= override_thr else None


def simulate_name_policy(name_rows):
    n = len(name_rows)
    if not n:
        print("\nNo real-scan recipient_name rows — policy simulation skipped.")
        return

    print(f"\nGated-policy simulation — recipient_name only (n={n} real-scan rows)")
    print("Gate: llm-sourced selections are never overridden; NER fills a blank")
    print(f"(conf >= {FILL_THRESHOLD}) or overrides a non-llm value "
          "(conf >= override_thr) when plausible_recipient_name passes.")

    # Cross-tab: where does the base pipeline stand, and how good is NER there?
    by_source = defaultdict(lambda: {"n": 0, "base_correct": 0,
                                     "covered": 0, "ner_correct_covered": 0})
    for row in name_rows:
        entry = by_source[row["base_source"] or "?"]
        entry["n"] += 1
        entry["base_correct"] += row["base_correct"]
        if row["ner_value"]:
            entry["covered"] += 1
            entry["ner_correct_covered"] += row["ner_correct"]

    print(f"\n{'Base source':<14} {'n':>4} {'base acc':>9} {'NER cov':>8} "
          f"{'NER acc(cov)':>13}")
    for source in sorted(by_source):
        entry = by_source[source]
        base_acc = f"{entry['base_correct'] / entry['n'] * 100:.1f}%"
        cov = f"{entry['covered'] / entry['n'] * 100:.0f}%"
        ner_acc = (f"{entry['ner_correct_covered'] / entry['covered'] * 100:.1f}%"
                   if entry["covered"] else "—")
        print(f"{source:<14} {entry['n']:>4} {base_acc:>9} {cov:>8} {ner_acc:>13}")

    base_correct = sum(row["base_correct"] for row in name_rows)
    print(f"\nThreshold sweep (fill_thr={FILL_THRESHOLD}; "
          f"base accuracy {base_correct / n * 100:.1f}%):")
    header = (f"{'override_thr':>12} {'fills':>6} {'overrides':>10} {'wins':>5} "
              f"{'losses':>7} {'policy acc':>11}")
    print(header)
    print("-" * len(header))
    for override_thr in [x / 100 for x in range(50, 100, 5)]:
        fills = overrides = wins = losses = correct = 0
        for row in name_rows:
            fired = _gate_fires(row, FILL_THRESHOLD, override_thr)
            outcome = row["ner_correct"] if fired else row["base_correct"]
            correct += outcome
            if fired == "fill":
                fills += 1
            elif fired == "override":
                overrides += 1
            if fired and row["ner_correct"] and not row["base_correct"]:
                wins += 1
            if fired and row["base_correct"] and not row["ner_correct"]:
                losses += 1
        print(f"{override_thr:>12.2f} {fills:>6} {overrides:>10} {wins:>5} "
              f"{losses:>7} {correct / n * 100:>10.1f}%")

    print(
        "\nwins = gate fired, NER right where selected was wrong;"
        "\nlosses = gate fired, NER wrong where selected was right."
        "\nHeld-out but in-domain, small n — thresholds chosen here are"
        "\nprovisional until eval_candidates.py accumulates ner rows."
    )


if __name__ == "__main__":
    sys.exit(main())
