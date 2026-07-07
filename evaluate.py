import argparse
import json
import os
from pathlib import Path

import storage


EVALUATE_LLM = os.getenv("EVALUATE_LLM", "").strip().lower() == "true"

if not EVALUATE_LLM:
    os.environ["OPENAI_API_KEY"] = ""

# extract_label_data is re-exported: regression_test.py imports it from here.
from label_reader import extract_label_data

DATASETS_DIR = Path("datasets")
FIELDS_TO_COMPARE = [
    "recipient_name",
    "street_address",
    "city",
    "state",
    "zip_code",
    "tracking_number",
    "carrier",
]


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# Comparators moved to comparators.py — the single definition of "correct"
# shared by this report, regression_test.py (which imports compare_field /
# has_ground_truth from here), and confidence calibration.
from comparators import (  # noqa: E402,F401  (re-exported)
    compare_field,
    has_ground_truth,
    normalize_street_address,
    normalize_value,
)
from calibration import load_table as load_calibration_table  # noqa: E402

def find_image_for_expected(expected_path):
    carrier_dir = expected_path.parent.parent
    images_dir = carrier_dir / "images"
    image_stem = expected_path.stem

    for extension in IMAGE_EXTENSIONS:
        image_path = images_dir / f"{image_stem}{extension}"

        if image_path.exists():
            return image_path

    return None


def load_expected_json(expected_path):
    with expected_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    parser = argparse.ArgumentParser(description="Evaluate extraction accuracy against annotated labels.")
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Among labels sharing the same tracking number, evaluate only the most recent scan.",
    )
    args = parser.parse_args()

    labels = storage.get_annotated_labels()  # newest-first (ORDER BY processed_at DESC)

    if args.latest_only:
        seen: set[str] = set()
        deduped = []
        for label in labels:
            tn = label.get("tracking_number") or ""
            if tn and tn in seen:
                continue
            if tn:
                seen.add(tn)
            deduped.append(label)
        print(
            f"Deduplication: kept {len(deduped)} of {len(labels)} annotated labels"
            f" (latest scan per tracking number)"
        )
        labels = deduped

    n = len(labels)

    if n == 0:
        print("No annotated labels found in the database.")
        return

    if n < 5:
        print(
            f"Warning: only {n} annotated label{'s' if n != 1 else ''}"
            " — results may not be representative"
        )

    print(f"\nAnnotated labels evaluated: {n}")

    # Staleness check: warn when the annotation set has grown well past what
    # the committed calibration table was fitted on.
    calibration_table = load_calibration_table()
    if calibration_table:
        fitted_on = calibration_table.get("metadata", {}).get("annotated_labels") or 0
        seen_tn: set[str] = set()
        deduped_n = 0
        for label in labels:
            tn = label.get("tracking_number") or ""
            if tn and tn in seen_tn:
                continue
            if tn:
                seen_tn.add(tn)
            deduped_n += 1
        if fitted_on and deduped_n > fitted_on * 1.25:
            print(
                f"Note: calibration table was fitted on {fitted_on} labels; "
                f"the DB now has {deduped_n} (deduped) — re-run fit_calibration.py"
            )

    field_correct = {f: 0 for f in FIELDS_TO_COMPARE}
    field_strict_correct = {f: 0 for f in FIELDS_TO_COMPARE}
    field_total = {f: 0 for f in FIELDS_TO_COMPARE}
    field_conf_sum = {f: 0.0 for f in FIELDS_TO_COMPARE}
    field_conf_count = {f: 0 for f in FIELDS_TO_COMPARE}
    gt_empty_skipped = 0

    source_correct: dict[str, int] = {}
    source_total: dict[str, int] = {}

    # Buckets ordered high-to-low; first match wins.
    conf_buckets = [
        (0.90, "0.90 – 1.00"),
        (0.70, "0.70 – 0.89"),
        (0.50, "0.50 – 0.69"),
        (0.00, "0.00 – 0.49"),
    ]
    bucket_correct = {lbl: 0 for _, lbl in conf_buckets}
    bucket_total = {lbl: 0 for _, lbl in conf_buckets}

    failures = []
    dataset_fields_excluded = 0
    dataset_rows = set()

    for label in labels:
        gt = label["ground_truth"]  # already a dict
        label_id = label["id"]

        for field in FIELDS_TO_COMPARE:
            if field not in gt:
                continue

            extracted_val = label.get(field)
            gt_val = gt[field]
            confidence = label.get(f"{field}_confidence")
            source = label.get(f"{field}_source") or ""

            # Rows imported by ml/import_datasets.py store the ground truth in
            # the extracted columns (source='dataset'), so scoring them would
            # count as correct by construction. Exclude them from accuracy.
            if source == "dataset":
                dataset_fields_excluded += 1
                dataset_rows.add(label_id)
                continue

            # Same gate as the regression suite: a field is only scorable
            # when the annotation actually contains a value for it.
            if not has_ground_truth(gt, field):
                gt_empty_skipped += 1
                continue

            # Primary metric: shared fuzzy comparators — the same definition
            # of "correct" the regression suite uses (tolerates case,
            # punctuation, suffix abbreviations, name-order differences).
            is_correct = compare_field(label, gt, field)
            # Legacy strict metric (exact lowercase match), reported alongside
            # for comparison with historical numbers.
            is_strict = (
                (extracted_val or "").strip().lower()
                == (gt_val or "").strip().lower()
            )

            field_total[field] += 1
            if is_correct:
                field_correct[field] += 1
            if is_strict:
                field_strict_correct[field] += 1

            if confidence is not None:
                field_conf_sum[field] += confidence
                field_conf_count[field] += 1

            src = source or "unknown"
            source_total[src] = source_total.get(src, 0) + 1
            if is_correct:
                source_correct[src] = source_correct.get(src, 0) + 1

            if confidence is not None:
                for lo, lbl in conf_buckets:
                    if confidence >= lo:
                        bucket_total[lbl] += 1
                        if is_correct:
                            bucket_correct[lbl] += 1
                        break

            if not is_correct:
                failures.append(
                    {
                        "label_id": label_id,
                        "field": field,
                        "extracted": extracted_val,
                        "ground_truth": gt_val,
                        "confidence": confidence,
                        "source": source,
                    }
                )

    if dataset_fields_excluded:
        print(
            f"Excluded {dataset_fields_excluded} field values across "
            f"{len(dataset_rows)} dataset-imported labels (source='dataset': "
            f"extracted columns hold ground truth, so they cannot be scored)"
        )

    if gt_empty_skipped:
        print(
            f"Skipped {gt_empty_skipped} field values with empty ground truth "
            f"(not scorable; same gate as the regression suite)"
        )

    # ── Per-field accuracy table ──────────────────────────────────────────────
    print()
    print("Accuracy = shared comparators (comparators.py, same definition as the")
    print("regression suite). Strict = legacy exact lowercase match, for comparison.")
    COL = (20, 9, 7, 10, 8, 15)
    header = (
        f"{'Field':<{COL[0]}} {'Correct':>{COL[1]}} {'Total':>{COL[2]}}"
        f" {'Accuracy':>{COL[3]}} {'Strict':>{COL[4]}} {'Avg Confidence':>{COL[5]}}"
    )
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)

    overall_correct = overall_strict = overall_total = overall_conf_count = 0
    overall_conf_sum = 0.0

    for field in FIELDS_TO_COMPARE:
        correct = field_correct[field]
        strict = field_strict_correct[field]
        total = field_total[field]
        pct = f"{correct / total * 100:.1f}%" if total else "—"
        strict_pct = f"{strict / total * 100:.1f}%" if total else "—"
        avg_conf = (
            f"{field_conf_sum[field] / field_conf_count[field]:.2f}"
            if field_conf_count[field]
            else "—"
        )
        print(
            f"{field:<{COL[0]}} {correct:>{COL[1]}} {total:>{COL[2]}}"
            f" {pct:>{COL[3]}} {strict_pct:>{COL[4]}} {avg_conf:>{COL[5]}}"
        )
        overall_correct += correct
        overall_strict += strict
        overall_total += total
        overall_conf_sum += field_conf_sum[field]
        overall_conf_count += field_conf_count[field]

    print(sep)
    overall_pct = f"{overall_correct / overall_total * 100:.1f}%" if overall_total else "—"
    overall_strict_pct = (
        f"{overall_strict / overall_total * 100:.1f}%" if overall_total else "—"
    )
    overall_avg = (
        f"{overall_conf_sum / overall_conf_count:.2f}" if overall_conf_count else "—"
    )
    print(
        f"{'OVERALL':<{COL[0]}} {overall_correct:>{COL[1]}} {overall_total:>{COL[2]}}"
        f" {overall_pct:>{COL[3]}} {overall_strict_pct:>{COL[4]}} {overall_avg:>{COL[5]}}"
    )

    # ── Per-source accuracy ───────────────────────────────────────────────────
    print()
    print("Per-source accuracy:")
    src_h = f"  {'Source':<14} {'Correct':>9} {'Total':>7} {'Accuracy':>9}"
    print(src_h)
    print("  " + "-" * (len(src_h) - 2))
    for src in sorted(source_total):
        correct = source_correct.get(src, 0)
        total = source_total[src]
        pct = f"{correct / total * 100:.1f}%" if total else "—"
        print(f"  {src:<14} {correct:>9} {total:>7} {pct:>9}")

    # ── Confidence calibration ────────────────────────────────────────────────
    print()
    print("Confidence calibration:")
    cal_h = f"  {'Group':<14} {'Correct':>9} {'Total':>7} {'Accuracy':>9}"
    print(cal_h)
    print("  " + "-" * (len(cal_h) - 2))
    for _, lbl in conf_buckets:
        correct = bucket_correct[lbl]
        total = bucket_total[lbl]
        pct = f"{correct / total * 100:.1f}%" if total else "—"
        print(f"  {lbl:<14} {correct:>9} {total:>7} {pct:>9}")

    # ── Failed labels ─────────────────────────────────────────────────────────
    if failures:
        print()
        print("Failed labels:")
        fail_h = (
            f"  {'label_id':<36}  {'field':<18}  {'extracted':<22}"
            f"  {'ground_truth':<22}  {'conf':>6}  source"
        )
        print(fail_h)
        print("  " + "-" * (len(fail_h) - 2))
        for fail in failures:
            conf = (
                f"{fail['confidence']:.2f}" if fail["confidence"] is not None else "—"
            )
            print(
                f"  {fail['label_id']:<36}  {fail['field']:<18}"
                f"  {repr(fail['extracted'] or '')[:20]:<22}"
                f"  {repr(fail['ground_truth'] or '')[:20]:<22}"
                f"  {conf:>6}  {fail['source']}"
            )
    else:
        print()
        print("No failures — all annotated fields matched extracted values.")


if __name__ == "__main__":
    main()
