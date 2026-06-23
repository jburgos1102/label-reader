import json
import os
import re
from pathlib import Path


EVALUATE_LLM = os.getenv("EVALUATE_LLM", "").strip().lower() == "true"

if not EVALUATE_LLM:
    os.environ["OPENAI_API_KEY"] = ""

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

    return " ".join(normalized.split())


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
    actual = normalize_street_address(actual_value)
    expected = normalize_street_address(expected_value)

    if not actual or not expected:
        return False

    if any(marker in actual and marker not in expected for marker in SENDER_MARKERS):
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
    return value is not None and bool(str(value).strip())


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


def main():
    expected_files = sorted(DATASETS_DIR.glob("*/expected/*.json"))

    if not expected_files:
        print("No expected JSON files found.")
        return

    field_totals = {field: 0 for field in FIELDS_TO_COMPARE}
    field_passes = {field: 0 for field in FIELDS_TO_COMPARE}
    llm_field_totals = {field: 0 for field in FIELDS_TO_COMPARE}
    llm_field_passes = {field: 0 for field in FIELDS_TO_COMPARE}
    labels_tested = 0
    llm_labels_scored = 0
    llm_labels_skipped = 0
    failures = []
    llm_failures = []

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

        print(f"\nTesting: {image_path}")

        for field in FIELDS_TO_COMPARE:
            if not has_ground_truth(expected_data, field):
                continue

            field_totals[field] += 1
            passed = compare_field(actual_data, expected_data, field)

            if passed:
                field_passes[field] += 1
            else:
                failures.append(
                    {
                        "label": str(image_path),
                        "field": field,
                        "expected": expected_data.get(field, ""),
                        "actual": actual_data.get(field, ""),
                    }
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

    print("\n=================================")
    print("OPENAI RESULTS")
    print("=================================")

    if not EVALUATE_LLM:
        print("OpenAI scoring skipped. Set EVALUATE_LLM=true to enable it.")
        return

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
                f"expected={failure['expected']!r} | actual={failure['actual']!r}"
            )
    elif llm_labels_scored:
        print("\nAll fields matched expected values.")
    else:
        print("\nNo enabled OpenAI results were available to score.")


if __name__ == "__main__":
    main()
