import json
import os
import re
from pathlib import Path


EVALUATE_LLM = os.getenv("EVALUATE_LLM", "").strip().lower() == "true"
OCR_DIAGNOSTICS = os.getenv("OCR_DIAGNOSTICS", "").strip().lower() == "true"

if not EVALUATE_LLM:
    os.environ["OPENAI_API_KEY"] = ""

from label_reader import extract_label_data
from ocr import get_last_ocr_diagnostics

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
    expected_files = sorted(DATASETS_DIR.glob("*/expected/*.json"))
    gold_set_configured = GOLD_SET_PATH.exists()
    gold_set = load_gold_set()

    if not expected_files:
        print("No expected JSON files found.")
        return

    field_totals = {field: 0 for field in FIELDS_TO_COMPARE}
    field_passes = {field: 0 for field in FIELDS_TO_COMPARE}
    llm_field_totals = {field: 0 for field in FIELDS_TO_COMPARE}
    llm_field_passes = {field: 0 for field in FIELDS_TO_COMPARE}
    gold_rule_totals = {field: 0 for field in FIELDS_TO_COMPARE}
    gold_rule_correct = {field: 0 for field in FIELDS_TO_COMPARE}
    gold_openai_totals = {field: 0 for field in FIELDS_TO_COMPARE}
    gold_openai_correct = {field: 0 for field in FIELDS_TO_COMPARE}
    hybrid_counts = {
        field: {
            "rule_only": 0,
            "openai_only": 0,
            "both_passed": 0,
            "both_failed": 0,
        }
        for field in FIELDS_TO_COMPARE
    }
    labels_tested = 0
    llm_labels_scored = 0
    llm_labels_skipped = 0
    failures = []
    llm_failures = []
    ocr_failure_diagnostics = []

    for expected_path in expected_files:
        image_path = find_image_for_expected(expected_path)

        if image_path is None:
            failures.append(
                {
                    "label": str(expected_path),
                    "field": "image",
                    "expected": "matching image file",
                    "actual": "missing",
                }
            )
            continue

        expected_data = load_expected_json(expected_path)
        actual_data = extract_label_data(str(image_path))
        labels_tested += 1
        rule_field_results = {}
        is_gold_label = normalize_image_path(image_path) in gold_set

        print(f"\nTesting: {image_path}")

        for field in FIELDS_TO_COMPARE:
            if not has_ground_truth(expected_data, field):
                continue

            field_totals[field] += 1
            passed = compare_field(actual_data, expected_data, field)
            rule_field_results[field] = passed

            if is_gold_label:
                gold_rule_totals[field] += 1
                if passed:
                    gold_rule_correct[field] += 1

            if passed:
                field_passes[field] += 1
            else:
                failure = {
                    "label": str(image_path),
                    "field": field,
                    "expected": expected_data.get(field, ""),
                    "actual": actual_data.get(field, ""),
                }
                failures.append(failure)

                if OCR_DIAGNOSTICS and field in OCR_DIAGNOSTIC_FIELDS:
                    ocr_failure_diagnostics.append(
                        build_ocr_failure_diagnostic(
                            str(image_path),
                            field,
                            expected_data.get(field, ""),
                        )
                    )

                print(
                    f"  FAIL {field}: "
                    f"expected={expected_data.get(field, '')!r} "
                    f"actual={actual_data.get(field, '')!r}"
                )

        if not EVALUATE_LLM:
            continue

        llm_result = actual_data.get("llm_result")
        if not isinstance(llm_result, dict) or not llm_result.get("llm_enabled"):
            llm_labels_skipped += 1
            continue

        llm_labels_scored += 1

        for field in FIELDS_TO_COMPARE:
            if not has_ground_truth(expected_data, field):
                continue

            llm_field_totals[field] += 1
            passed = compare_field(llm_result, expected_data, field)
            rule_passed = rule_field_results[field]

            if is_gold_label:
                gold_openai_totals[field] += 1
                if passed:
                    gold_openai_correct[field] += 1

            if rule_passed and passed:
                hybrid_counts[field]["both_passed"] += 1
            elif rule_passed:
                hybrid_counts[field]["rule_only"] += 1
            elif passed:
                hybrid_counts[field]["openai_only"] += 1
            else:
                hybrid_counts[field]["both_failed"] += 1

            if passed:
                llm_field_passes[field] += 1
            else:
                llm_failures.append(
                    {
                        "label": str(image_path),
                        "field": field,
                        "expected": expected_data.get(field, ""),
                        "actual": llm_result.get(field, ""),
                    }
                )

    print("\n=================================")
    print("RULE-BASED RESULTS")
    print("=================================")
    print(f"Labels tested: {labels_tested}")

    for field in FIELDS_TO_COMPARE:
        total = field_totals[field]
        passed = field_passes[field]
        accuracy = (passed / total * 100) if total else 0

        readable_name = field.replace("_", " ").title()
        print(f"{readable_name} Accuracy: {accuracy:.1f}% ({passed}/{total})")

    if failures:
        print("\nFailures:")

        for failure in failures:
            print(
                f"- {failure['label']} | {failure['field']} | "
                f"expected={failure['expected']!r} | actual={failure['actual']!r}"
            )
    else:
        print("\nAll fields matched expected values.")

    print_failure_analysis(failures)

    if OCR_DIAGNOSTICS:
        print_ocr_failure_diagnostics(ocr_failure_diagnostics)

    print("\n=================================")
    print("OPENAI RESULTS")
    print("=================================")

    if not EVALUATE_LLM:
        print("OpenAI scoring skipped. Set EVALUATE_LLM=true to enable it.")
    else:
        print(f"Labels scored: {llm_labels_scored}")
        print(f"Labels skipped: {llm_labels_skipped}")

        for field in FIELDS_TO_COMPARE:
            total = llm_field_totals[field]
            passed = llm_field_passes[field]
            accuracy = (passed / total * 100) if total else 0

            readable_name = field.replace("_", " ").title()
            print(f"{readable_name} Accuracy: {accuracy:.1f}% ({passed}/{total})")

        if llm_failures:
            print("\nFailures:")

            for failure in llm_failures:
                print(
                    f"- {failure['label']} | {failure['field']} | "
                    f"expected={failure['expected']!r} | "
                    f"actual={failure['actual']!r}"
                )
        elif llm_labels_scored:
            print("\nAll fields matched expected values.")
        else:
            print("\nNo enabled OpenAI results were available to score.")

    print("\n=================================")
    print("GOLD SET RESULTS")
    print("=================================")

    if not gold_set_configured:
        print("Gold Set: not configured")
    else:
        print("RULE-BASED")
        print_gold_metrics(gold_rule_totals, gold_rule_correct)
        print("\nOPENAI")

        if EVALUATE_LLM:
            print_gold_metrics(gold_openai_totals, gold_openai_correct)
        else:
            print("OpenAI scoring skipped. Set EVALUATE_LLM=true to enable it.")

    if EVALUATE_LLM:
        print("\n=================================")
        print("HYBRID FIELD COMPARISON")
        print("=================================")

        for field in FIELDS_TO_COMPARE:
            counts = hybrid_counts[field]

            if counts["openai_only"] > counts["rule_only"]:
                suggested_source = "OpenAI"
            elif counts["rule_only"] > counts["openai_only"]:
                suggested_source = "Rule-Based"
            else:
                suggested_source = "Tie / Needs Review"

            readable_name = field.replace("_", " ").title()
            print(f"\n{readable_name}:")
            print(f"  Rule only: {counts['rule_only']}")
            print(f"  OpenAI only: {counts['openai_only']}")
            print(f"  Both passed: {counts['both_passed']}")
            print(f"  Both failed: {counts['both_failed']}")
            print(f"  Suggested source: {suggested_source}")


if __name__ == "__main__":
    main()
