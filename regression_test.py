import os
import sys


# Regression checks must stay rule-based only and must never call OpenAI.
os.environ["EVALUATE_LLM"] = ""
os.environ["OPENAI_API_KEY"] = ""

from evaluate import (  # noqa: E402
    DATASETS_DIR,
    FIELDS_TO_COMPARE,
    compare_field,
    extract_label_data,
    find_image_for_expected,
    has_ground_truth,
    load_expected_json,
)


BASELINE_METRICS = {
    "recipient_name": 54.2,
    "street_address": 78.3,
    "city": 80.8,
    "state": 84.6,
    "zip_code": 80.8,
    "tracking_number": 73.1,
    "carrier": 100.0,
}


def calculate_rule_based_metrics():
    expected_files = sorted(DATASETS_DIR.glob("*/expected/*.json"))
    field_totals = {field: 0 for field in FIELDS_TO_COMPARE}
    field_passes = {field: 0 for field in FIELDS_TO_COMPARE}
    labels_tested = 0

    for expected_path in expected_files:
        image_path = find_image_for_expected(expected_path)
        if image_path is None:
            continue

        expected_data = load_expected_json(expected_path)
        actual_data = extract_label_data(str(image_path))
        labels_tested += 1

        for field in FIELDS_TO_COMPARE:
            if not has_ground_truth(expected_data, field):
                continue

            field_totals[field] += 1
            if compare_field(actual_data, expected_data, field):
                field_passes[field] += 1

    metrics = {}
    for field in FIELDS_TO_COMPARE:
        total = field_totals[field]
        passed = field_passes[field]
        metrics[field] = (passed / total * 100) if total else 0.0

    return labels_tested, field_passes, field_totals, metrics


def main():
    labels_tested, field_passes, field_totals, metrics = calculate_rule_based_metrics()
    regressions = []

    for field, baseline in BASELINE_METRICS.items():
        actual = round(metrics[field], 1)
        if actual < baseline:
            regressions.append((field, baseline, actual))

    if regressions:
        print("REGRESSION TEST FAILED")
        print(f"Labels tested: {labels_tested}")

        for field, baseline, actual in regressions:
            print(f"\n{field}:")
            print(f"  baseline: {baseline:.1f}%")
            print(f"  actual: {actual:.1f}%")

        return 1

    print("REGRESSION TEST PASSED")
    print(f"Labels tested: {labels_tested}")

    for field in FIELDS_TO_COMPARE:
        actual = round(metrics[field], 1)
        passed = field_passes[field]
        total = field_totals[field]
        baseline = BASELINE_METRICS[field]
        print(
            f"{field}: {actual:.1f}% ({passed}/{total}) "
            f"baseline {baseline:.1f}%"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
