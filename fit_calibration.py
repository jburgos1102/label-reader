"""Fit the confidence calibration table from annotated labels.

Reads annotated rows from label_storage.db, labels correctness with the
shared comparators (comparators.py — same definition as the regression
suite), buckets by (field, source, validation signature) with hierarchical
fallback, and writes the Laplace-smoothed table to
calibration/confidence_table.json along with a reliability report comparing
legacy stored confidences against the fitted values.

Validation features are recomputed from stored columns so historical rows
participate:
  - tracking_number: carrier checksum (tracking.validate_tracking_checksum)
  - llm-sourced fields: value found in stored OCR text (same normalization
    as scoring.score_llm_result)

Usage (from the project root):
  venv/bin/python fit_calibration.py            # fit + report + write artifact
  venv/bin/python fit_calibration.py --dry-run  # fit + report, no write
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

from calibration import (
    CALIBRATION_TABLE_PATH,
    MIN_BUCKET_N,
    bucket_keys,
    lookup,
    validation_signature,
)
from comparators import compare_field, has_ground_truth
from scoring import normalize_comparison_value
import storage
from tracking import validate_tracking_checksum

FIELDS = (
    "recipient_name",
    "street_address",
    "city",
    "state",
    "zip_code",
    "tracking_number",
    "carrier",
)


def wilson_interval(correct, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    phat = correct / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def dedupe_latest(labels):
    """Latest scan per tracking number (rows are newest-first)."""
    seen = set()
    deduped = []
    for label in labels:
        tn = label.get("tracking_number") or ""
        if tn and tn in seen:
            continue
        if tn:
            seen.add(tn)
        deduped.append(label)
    return deduped


def recomputed_validations(label, field, source):
    """Validation features recoverable from stored columns (see module doc)."""
    validations = {}
    if field == "tracking_number":
        checksum = validate_tracking_checksum(
            label.get("tracking_number") or "", label.get("carrier") or ""
        )
        if checksum is not None:
            validations["checksum_valid"] = checksum
    if source == "llm":
        value = label.get(field) or ""
        ocr_text = label.get("ocr_text") or ""
        if value and ocr_text:
            value_norm = normalize_comparison_value(value)
            if value_norm:
                validations["found_in_ocr"] = (
                    value_norm in normalize_comparison_value(ocr_text)
                )
    return validations


def collect_samples():
    """One sample per scorable (label, field): correctness + bucket features."""
    labels = dedupe_latest(storage.get_annotated_labels())
    samples = []
    for label in labels:
        gt = label["ground_truth"]
        for field in FIELDS:
            if field not in gt:
                continue
            source = label.get(f"{field}_source") or ""
            if source == "dataset":
                continue  # extracted columns hold ground truth — unscorable
            if not has_ground_truth(gt, field):
                continue
            samples.append({
                "field": field,
                "source": source or "unknown",
                "validations": recomputed_validations(label, field, source),
                "correct": bool(compare_field(label, gt, field)),
                "legacy_confidence": label.get(f"{field}_confidence"),
            })
    return len(labels), samples


def fit(samples, min_n):
    """Aggregate counts at every bucket level and compute smoothed rates."""
    counts = {}
    for sample in samples:
        signature = validation_signature(sample["validations"])
        for key in bucket_keys(sample["field"], sample["source"], signature):
            entry = counts.setdefault(key, {"n": 0, "correct": 0})
            entry["n"] += 1
            entry["correct"] += 1 if sample["correct"] else 0

    buckets = {}
    for key, entry in counts.items():
        n, correct = entry["n"], entry["correct"]
        low, high = wilson_interval(correct, n)
        buckets[key] = {
            "n": n,
            "correct": correct,
            "accuracy": round(correct / n, 4) if n else None,
            "p": round((correct + 1) / (n + 2), 4),  # Laplace-smoothed
            "wilson_95": [round(low, 4), round(high, 4)],
            "usable": n >= min_n,
        }
    return buckets


def expected_calibration_error(pairs, bins=10):
    """ECE over (predicted_confidence, correct) pairs; skips None predictions."""
    pairs = [(p, c) for p, c in pairs if p is not None]
    if not pairs:
        return None, 0
    totals = [0] * bins
    corrects = [0] * bins
    conf_sums = [0.0] * bins
    for p, correct in pairs:
        b = min(int(p * bins), bins - 1)
        totals[b] += 1
        corrects[b] += 1 if correct else 0
        conf_sums[b] += p
    ece = sum(
        totals[b] * abs(corrects[b] / totals[b] - conf_sums[b] / totals[b])
        for b in range(bins) if totals[b]
    ) / len(pairs)
    return ece, len(pairs)


def report(samples, table, min_n):
    print(f"\nSamples: {len(samples)} scorable (label, field) pairs")
    print(f"Bucket usable threshold: n >= {min_n} (thinner buckets fall back)\n")

    header = (f"{'Bucket (field|source[|validations])':<58} {'n':>5} "
              f"{'acc':>7} {'p_cal':>7} {'wilson95':>17} {'legacy_avg':>11}")
    print(header)
    print("-" * len(header))

    legacy_by_bucket = {}
    for sample in samples:
        signature = validation_signature(sample["validations"])
        for key in bucket_keys(sample["field"], sample["source"], signature):
            if sample["legacy_confidence"] is not None:
                legacy_by_bucket.setdefault(key, []).append(sample["legacy_confidence"])

    buckets = table["buckets"]
    for key in sorted(buckets, key=lambda k: (k.count("|") == 0, k)):
        b = buckets[key]
        legacy_vals = legacy_by_bucket.get(key, [])
        legacy_avg = f"{sum(legacy_vals) / len(legacy_vals):.2f}" if legacy_vals else "—"
        marker = "" if b["usable"] else "  (thin — falls back)"
        print(f"{key:<58} {b['n']:>5} {b['accuracy']:>7.3f} {b['p']:>7.3f} "
              f"[{b['wilson_95'][0]:.3f}, {b['wilson_95'][1]:.3f}] {legacy_avg:>11}"
              f"{marker}")

    # Reliability: legacy stored confidence vs calibrated lookup (in-sample)
    legacy_pairs = [(s["legacy_confidence"], s["correct"]) for s in samples]
    calibrated_pairs = [
        (lookup(table, s["field"], s["source"], s["validations"], min_n=min_n)[0],
         s["correct"])
        for s in samples
    ]
    legacy_ece, n_legacy = expected_calibration_error(legacy_pairs)
    cal_ece, n_cal = expected_calibration_error(calibrated_pairs)
    print("\nReliability (expected calibration error, 10 bins, in-sample):")
    print(f"  legacy stored confidence : ECE = {legacy_ece:.4f}  (n={n_legacy})")
    if cal_ece is not None:
        print(f"  calibrated table lookup  : ECE = {cal_ece:.4f}  (n={n_cal}, "
              f"{len(samples) - n_cal} without a usable bucket)")


def main():
    parser = argparse.ArgumentParser(description="Fit the confidence calibration table.")
    parser.add_argument("--min-n", type=int, default=MIN_BUCKET_N,
                        help="minimum samples for a bucket to be usable")
    parser.add_argument("--out", default=CALIBRATION_TABLE_PATH)
    parser.add_argument("--dry-run", action="store_true", help="report only, no write")
    args = parser.parse_args()

    label_count, samples = collect_samples()
    if not samples:
        print("No scorable annotated samples found.")
        return 1

    buckets = fit(samples, args.min_n)
    table = {
        "metadata": {
            "fitted_at": datetime.now(timezone.utc).isoformat(),
            "annotated_labels": label_count,
            "samples": len(samples),
            "min_bucket_n": args.min_n,
            "correctness_definition": "comparators.compare_field (shared fuzzy comparators)",
            "dedupe": "latest scan per tracking number",
            "excluded": "source='dataset' fields; empty ground truth",
        },
        "buckets": buckets,
    }

    report(samples, table, args.min_n)

    if args.dry_run:
        print("\n--dry-run: artifact not written")
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table, f, indent=2, sort_keys=True)
    print(f"\nArtifact written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
