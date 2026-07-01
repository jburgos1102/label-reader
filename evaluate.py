import json
import os
import re
from pathlib import Path

import storage


EVALUATE_LLM = os.getenv("EVALUATE_LLM", "").strip().lower() == "true"
OCR_DIAGNOSTICS = os.getenv("OCR_DIAGNOSTICS", "").strip().lower() == "true"

if not EVALUATE_LLM:
    os.environ["OPENAI_API_KEY"] = ""

from label_reader import extract_label_data
from ocr import get_last_ocr_diagnostics
from pipeline import build_extraction_result

DATASETS_DIR = Path("datasets")
GOLD_SET_PATH = DATASETS_DIR / "gold_set.txt"
FIELDS_TO_COMPARE = [
    "recipient_name",
    "street_address",
    "city",
    "state",
    "zip_code",
    "tracking_number",
    "carrier",
]
OCR_DIAGNOSTIC_FIELDS = (
    "recipient_name",
    "street_address",
    "city",
    "state",
    "zip_code",
)


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

CHARACTER_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)

STREET_SUFFIXES = {
    "STREET": "ST",
    "ST": "ST",
    "AVENUE": "AVE",
    "AVE": "AVE",
    "ROAD": "RD",
    "RD": "RD",
    "BUILDING": "BLDG",
    "BLDG": "BLDG",
}

SENDER_MARKERS = (
    "BILL SENDER",
    "RETURN ADDRESS",
    "SHIP FROM",
    "SENDER ADDRESS",
)

GOLD_FIELD_LABELS = {
    "zip_code": "ZIP",
    "tracking_number": "Tracking",
}


def normalize_image_path(image_path):
    normalized = os.path.normpath(str(image_path).strip())
    return Path(normalized).as_posix().casefold()


def load_gold_set():
    if not GOLD_SET_PATH.exists():
        return set()

    gold_paths = set()
    with GOLD_SET_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            candidate = line.strip()
            if candidate and not candidate.startswith("#"):
                gold_paths.add(normalize_image_path(candidate))

    return gold_paths


def print_gold_metrics(field_totals, field_correct):
    for field in FIELDS_TO_COMPARE:
        total = field_totals[field]
        correct = field_correct[field]
        accuracy = (correct / total * 100) if total else 0
        readable_name = GOLD_FIELD_LABELS.get(
            field,
            field.replace("_", " ").title(),
        )
        print(f"{readable_name} Accuracy: {accuracy:.1f}% ({correct}/{total})")


def print_failure_analysis(failures):
    failures_by_field = {field: [] for field in FIELDS_TO_COMPARE}

    for failure in failures:
        field = failure.get("field")
        if field in failures_by_field:
            failures_by_field[field].append(failure)

    ranked_fields = sorted(
        FIELDS_TO_COMPARE,
        key=lambda field: (
            -len(failures_by_field[field]),
            FIELDS_TO_COMPARE.index(field),
        ),
    )

    print("\n=================================")
    print("FAILURE ANALYSIS")
    print("=================================")
    print("Failure Counts By Field:")

    for field in ranked_fields:
        print(f"{field}: {len(failures_by_field[field])}")

    print("\nTop Examples:")

    for field in ranked_fields:
        field_failures = failures_by_field[field]
        if not field_failures:
            continue

        print(f"\n{field}:")
        for failure in field_failures[:5]:
            expected = json.dumps(failure.get("expected", ""), ensure_ascii=False)
            actual = json.dumps(failure.get("actual", ""), ensure_ascii=False)
            print(f"- {failure['label']}")
            print(f"  expected: {expected}")
            print(f"  actual: {actual}")


def normalize_value(value):
    if value is None:
        return ""

    normalized = str(value).translate(CHARACTER_TRANSLATION).upper().strip()
    normalized = normalized.replace(".", "")
    normalized = normalized.replace(",", " ")
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    return " ".join(normalized.split())


def normalize_alphanumeric(value):
    return re.sub(r"[^A-Z0-9]", "", normalize_value(value))


def normalize_street_address(value):
    normalized = normalize_value(value)

    for suffix, abbreviation in STREET_SUFFIXES.items():
        normalized = re.sub(rf"\b{suffix}\b", abbreviation, normalized)

    normalized = re.sub(r"\bHUB(?:\s+BOX)?\s*#?\s*\d+\b", " ", normalized)
    normalized = re.sub(r"\bPO\s+BOX\s+\d+\b", " ", normalized)
    normalized = re.sub(r"\bDICKINSON\s+COLLEGE\b", " ", normalized)
    normalized = re.sub(r"\b(?:DEPT|DEPARTMENT)(?:\s+OF)?\b.*$", " ", normalized)
    normalized = " ".join(normalized.split())

    street_start = re.search(
        r"(?<![A-Z0-9])(?:\d+[A-Z]*(?:-[A-Z0-9]+)?|[A-Z]\d+[A-Z0-9-]*)(?=\s|$)",
        normalized,
    )
    if not street_start:
        return ""

    return normalized[street_start.start() :]


def normalize_name_tokens(value):
    if value is None:
        return []

    name = str(value).translate(CHARACTER_TRANSLATION).upper().strip()
    name = name.replace(".", " ")

    if name.count(",") == 1:
        last_name, remaining_name = name.split(",", 1)
        if last_name.strip() and remaining_name.strip():
            name = f"{remaining_name} {last_name}"

    name = name.replace(",", " ")
    return re.findall(r"[A-Z0-9]+(?:['-][A-Z0-9]+)*", name)


def compare_street_address(actual_value, expected_value):
    raw_actual = normalize_value(actual_value)
    raw_expected = normalize_value(expected_value)

    if any(
        marker in raw_actual and marker not in raw_expected
        for marker in SENDER_MARKERS
    ):
        return False

    actual = normalize_street_address(actual_value)
    expected = normalize_street_address(expected_value)

    if not actual or not expected:
        return False

    actual_numbers = re.findall(r"\d+", actual)
    expected_numbers = re.findall(r"\d+", expected)
    if actual_numbers != expected_numbers:
        return False

    return actual == expected


def compare_recipient_name(actual_value, expected_value):
    actual_tokens = normalize_name_tokens(actual_value)
    expected_tokens = normalize_name_tokens(expected_value)

    if not actual_tokens or not expected_tokens:
        return False

    if actual_tokens == expected_tokens:
        return True

    if len(actual_tokens) == 1 and len(expected_tokens) >= 2:
        actual_joined = actual_tokens[0]
        expected_first_last = expected_tokens[0] + expected_tokens[-1]
        expected_last_first = expected_tokens[-1] + expected_tokens[0]
        return actual_joined in (expected_first_last, expected_last_first)

    if len(expected_tokens) == 1 and len(actual_tokens) >= 2:
        expected_joined = expected_tokens[0]
        actual_first_last = actual_tokens[0] + actual_tokens[-1]
        actual_last_first = actual_tokens[-1] + actual_tokens[0]
        return expected_joined in (actual_first_last, actual_last_first)

    if len(actual_tokens) < 2 or len(expected_tokens) < 2:
        return False

    if actual_tokens[0] != expected_tokens[0]:
        return False

    if actual_tokens[-1] != expected_tokens[-1]:
        return False

    actual_middle = actual_tokens[1:-1]
    expected_middle = expected_tokens[1:-1]

    for expected_part in expected_middle:
        if not any(
            actual_part == expected_part
            or (len(expected_part) == 1 and actual_part.startswith(expected_part))
            for actual_part in actual_middle
        ):
            return False

    return True


def compare_zip_code(actual_value, expected_value):
    actual = re.sub(r"\D", "", str(actual_value or ""))
    expected = re.sub(r"\D", "", str(expected_value or ""))
    return bool(actual and expected and actual == expected)


def compare_tracking_number(actual_value, expected_value):
    actual = normalize_alphanumeric(actual_value)
    expected = normalize_alphanumeric(expected_value)
    return bool(actual and expected and actual == expected)


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


def has_ground_truth(expected_data, field_name):
    value = expected_data.get(field_name)
    if value is None or not str(value).strip():
        return False

    if field_name == "street_address":
        return bool(normalize_street_address(value))

    return True


def compare_field(actual_data, expected_data, field_name):
    actual_value = actual_data.get(field_name, "")
    expected_value = expected_data.get(field_name, "")

    if field_name == "recipient_name":
        return compare_recipient_name(actual_value, expected_value)

    if field_name == "street_address":
        return compare_street_address(actual_value, expected_value)

    if field_name == "zip_code":
        return compare_zip_code(actual_value, expected_value)

    if field_name == "tracking_number":
        return compare_tracking_number(actual_value, expected_value)

    return normalize_value(actual_value) == normalize_value(expected_value)


def expected_value_in_ocr(expected_value, ocr_text, field_name):
    if field_name == "zip_code":
        expected = re.sub(r"\D", "", str(expected_value or ""))
        text = re.sub(r"\D", "", str(ocr_text or ""))
        return bool(expected and expected in text)

    expected = normalize_value(expected_value)
    text = normalize_value(ocr_text)
    if not expected:
        return False

    pattern = rf"(?<![A-Z0-9]){re.escape(expected)}(?![A-Z0-9])"
    return bool(re.search(pattern, text))


def build_ocr_failure_diagnostic(label, field, expected_value):
    ocr_diagnostics = get_last_ocr_diagnostics()
    selected_text = ocr_diagnostics.get("selected_text", "")
    rotation_texts = ocr_diagnostics.get("rotations", {})
    return {
        "label": label,
        "field": field,
        "expected": expected_value,
        "selected": expected_value_in_ocr(expected_value, selected_text, field),
        "rotations": {
            degrees: expected_value_in_ocr(
                expected_value,
                rotation_texts.get(degrees, ""),
                field,
            )
            for degrees in (0, 90, 180, 270)
        },
    }


def print_ocr_failure_diagnostics(diagnostics):
    print("\n=================================")
    print("OCR FAILURE DIAGNOSTICS")
    print("=================================")
    print("Showing up to 5 failed labels per field.")

    shown_by_field = {field: 0 for field in OCR_DIAGNOSTIC_FIELDS}
    shown_any = False

    for diagnostic in diagnostics:
        field = diagnostic["field"]
        if shown_by_field[field] >= 5:
            continue

        shown_by_field[field] += 1
        shown_any = True
        print(f"\n{diagnostic['label']}")
        print(f"{field} expected: {diagnostic['expected']}")
        print(
            "selected OCR contained expected: "
            f"{str(diagnostic['selected']).lower()}"
        )
        for degrees in (0, 90, 180, 270):
            contained = diagnostic["rotations"][degrees]
            print(
                f"rotation {degrees} contained expected: "
                f"{str(contained).lower()}"
            )

    if not shown_any:
        print("No eligible OCR field failures were found.")


def main():
    labels = storage.get_annotated_labels()
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

    field_correct = {f: 0 for f in FIELDS_TO_COMPARE}
    field_total = {f: 0 for f in FIELDS_TO_COMPARE}
    field_conf_sum = {f: 0.0 for f in FIELDS_TO_COMPARE}
    field_conf_count = {f: 0 for f in FIELDS_TO_COMPARE}

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

            is_correct = (
                (extracted_val or "").strip().lower()
                == (gt_val or "").strip().lower()
            )

            field_total[field] += 1
            if is_correct:
                field_correct[field] += 1

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

    # ── Per-field accuracy table ──────────────────────────────────────────────
    COL = (20, 9, 7, 10, 15)
    header = (
        f"{'Field':<{COL[0]}} {'Correct':>{COL[1]}} {'Total':>{COL[2]}}"
        f" {'Accuracy':>{COL[3]}} {'Avg Confidence':>{COL[4]}}"
    )
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)

    overall_correct = overall_total = overall_conf_count = 0
    overall_conf_sum = 0.0

    for field in FIELDS_TO_COMPARE:
        correct = field_correct[field]
        total = field_total[field]
        pct = f"{correct / total * 100:.1f}%" if total else "—"
        avg_conf = (
            f"{field_conf_sum[field] / field_conf_count[field]:.2f}"
            if field_conf_count[field]
            else "—"
        )
        print(
            f"{field:<{COL[0]}} {correct:>{COL[1]}} {total:>{COL[2]}}"
            f" {pct:>{COL[3]}} {avg_conf:>{COL[4]}}"
        )
        overall_correct += correct
        overall_total += total
        overall_conf_sum += field_conf_sum[field]
        overall_conf_count += field_conf_count[field]

    print(sep)
    overall_pct = f"{overall_correct / overall_total * 100:.1f}%" if overall_total else "—"
    overall_avg = (
        f"{overall_conf_sum / overall_conf_count:.2f}" if overall_conf_count else "—"
    )
    print(
        f"{'OVERALL':<{COL[0]}} {overall_correct:>{COL[1]}} {overall_total:>{COL[2]}}"
        f" {overall_pct:>{COL[3]}} {overall_avg:>{COL[4]}}"
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
