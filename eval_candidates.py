"""Candidate-level accuracy report from persisted selection provenance.

Scores every persisted candidate (winners AND losers — including shadow
sources like NER that the Selector ignores) against ground-truth
annotations using the shared comparators. This is the out-of-sample
evidence stream for deciding whether a shadow source should be allowed to
influence selection: unlike selected-output accuracy, it measures what each
source WOULD have contributed.

Requires rows that have both a ground_truth annotation and the candidates
provenance column (persisted for every scan since the calibration sprint).
Rows without provenance are reported and skipped.

Usage (from the project root):
  venv/bin/python eval_candidates.py
"""

import sys
from collections import defaultdict

from comparators import compare_field, has_ground_truth
import storage

FIELDS = (
    "recipient_name",
    "street_address",
    "city",
    "state",
    "zip_code",
    "tracking_number",
    "carrier",
)


def dedupe_latest(labels):
    seen = set()
    deduped = []
    for label in labels:
        tracking = label.get("tracking_number") or ""
        if tracking and tracking in seen:
            continue
        if tracking:
            seen.add(tracking)
        deduped.append(label)
    return deduped


def main():
    labels = dedupe_latest(storage.get_annotated_labels())
    with_provenance = [
        label for label in labels
        if label.get("candidates") and label.get("ground_truth")
    ]
    print(f"Annotated labels (deduped): {len(labels)}")
    print(f"  with candidate provenance: {len(with_provenance)}")
    if not with_provenance:
        print(
            "\nNo annotated rows carry candidate provenance yet — provenance is"
            "\nrecorded for every scan since the calibration sprint, so this"
            "\nreport fills in as newly scanned labels get annotated."
        )
        return 0

    # stats[(source, field)] = {n, correct, matches_selected}
    stats = defaultdict(lambda: {"n": 0, "correct": 0, "beat_selected": 0})
    scorable_by_field = defaultdict(int)

    for label in with_provenance:
        gt = label["ground_truth"]
        candidates_by_field = label["candidates"]
        for field in FIELDS:
            if field not in gt or not has_ground_truth(gt, field):
                continue
            source = label.get(f"{field}_source") or ""
            if source == "dataset":
                continue
            scorable_by_field[field] += 1
            selected_correct = compare_field(label, gt, field)
            for candidate in candidates_by_field.get(field, []):
                value = candidate.get("value") or ""
                if not value:
                    continue
                entry = stats[(candidate.get("source", "?"), field)]
                entry["n"] += 1
                candidate_correct = compare_field({field: value}, gt, field)
                if candidate_correct:
                    entry["correct"] += 1
                if candidate_correct and not selected_correct:
                    entry["beat_selected"] += 1

    header = (f"{'Source':<8} {'Field':<18} {'n':>5} {'Coverage':>9} "
              f"{'Accuracy':>9} {'BeatSelected':>13}")
    print()
    print("Coverage = candidate produced a non-empty value for a scorable field.")
    print("BeatSelected = candidate was correct where the selected value was wrong.")
    print()
    print(header)
    print("-" * len(header))
    for (source, field) in sorted(stats):
        entry = stats[(source, field)]
        scorable = scorable_by_field[field]
        coverage = f"{entry['n'] / scorable * 100:.0f}%" if scorable else "—"
        accuracy = f"{entry['correct'] / entry['n'] * 100:.1f}%" if entry["n"] else "—"
        print(f"{source:<8} {field:<18} {entry['n']:>5} {coverage:>9} "
              f"{accuracy:>9} {entry['beat_selected']:>13}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
