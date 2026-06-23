import json
import os
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


def normalize_value(value):
    if value is None:
        return ""

    return str(value).strip().lower()


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


def compare_field(actual_data, expected_data, field_name):
    actual_value = normalize_value(actual_data.get(field_name, ""))
    expected_value = normalize_value(expected_data.get(field_name, ""))

    return actual_value == expected_value


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
