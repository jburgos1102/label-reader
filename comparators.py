"""Shared field comparators — the project's single definition of "correct".

Moved verbatim from evaluate.py so that the dataset regression suite, the
DB evaluation report, and confidence calibration all score correctness the
same way. If a comparator changes, every accuracy number and every
calibrated confidence changes with it — treat edits here like schema
changes (regenerate calibration artifacts, re-baseline reports).
"""

import re

CHARACTER_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
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
