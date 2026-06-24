import re


OCR_LETTER_TO_DIGIT = str.maketrans(
    {
        "O": "0",
        "I": "1",
        "L": "1",
        "S": "5",
        "B": "8",
        "G": "6",
    }
)


# --- IDENTIFY CARRIER ---
def identify_carrier(tracking_number):
    tracking_number = tracking_number.upper()

    if tracking_number.startswith("SPX"):
        return "SpeedX"

    if tracking_number.startswith("TBA"):
        return "Amazon"

    if tracking_number.startswith("YWNJC"):
        return "YunExpress"

    if tracking_number.startswith("UUS"):
        return "UniUni"

    if tracking_number.startswith("DDIYS"):
        return "TikTok"

    if tracking_number.startswith("1LSD"):
        return "TikTok"

    if (
        tracking_number.isdigit()
        and tracking_number.startswith(("91", "92", "93", "94", "95"))
        and len(tracking_number) >= 20
    ):
        return "USPS"

    if (
        tracking_number.isdigit()
        and tracking_number.startswith("56")
        and len(tracking_number) >= 20
    ):
        return "USPS"

    if re.fullmatch(r"1Z[A-Z0-9]{16}", tracking_number):
        return "UPS"

    if tracking_number.isdigit() and len(tracking_number) in (12, 15, 20, 22):
        return "FEDEX"

    return "Unknown"


def identify_carrier_with_context(tracking_number, ocr_text):
    """Resolve carrier using tracking evidence plus explicit service context."""
    carrier = identify_carrier(tracking_number)
    upper_text = str(ocr_text or "").upper()

    has_ups_final_mile_service = bool(
        re.search(r"\b(?:SUREPOST|GROUND\s+SAVER)\b", upper_text)
    )
    has_ups_tracking_context = bool(
        re.search(r"\b1Z(?:[\s_-]*[A-Z0-9]){16}\b", upper_text)
    )
    has_standalone_ups_marker = bool(
        re.search(r"(?:^|\n)\s*UPS\s*(?:\n|$)", upper_text)
    )
    if (
        carrier == "USPS"
        and has_ups_final_mile_service
        and has_ups_tracking_context
        and has_standalone_ups_marker
    ):
        return "UPS"

    has_usps_context = bool(
        re.search(r"\bUSPS\b", upper_text)
        or re.search(r"\bU\.?\s*S\.?\s+POSTAGE\s+PAID\b", upper_text)
        or re.search(r"\bUS\s+POSTAGE\s+PAID\b", upper_text)
    )
    has_explicit_fedex = bool(re.search(r"\bFED\s*EX\b", upper_text))

    if carrier == "FEDEX" and has_usps_context and not has_explicit_fedex:
        return "USPS"

    if carrier != "Unknown":
        return carrier

    if re.search(r"\b(?:SPEEDX|SPX[A-Z0-9]*)\b", upper_text):
        return "SpeedX"

    if has_usps_context:
        return "USPS"

    if re.search(r"\bUPS\b", upper_text) and has_ups_tracking_context:
        return "UPS"

    if has_explicit_fedex:
        return "FEDEX"

    return carrier


# --- TRACKING HELPER FUNCTIONS ---


def clean_tracking_candidate(candidate):
    candidate = candidate.upper()
    candidate = re.sub(r"[^A-Z0-9]", "", candidate)

    if candidate.startswith("1BA"):
        candidate = "TBA" + candidate[3:]

    if candidate.startswith("1LSDO"):
        candidate = candidate.replace("O", "0", 1)

    return candidate


def is_valid_ups_check_digit(candidate):
    if not re.fullmatch(r"1Z[A-Z0-9]{16}", candidate):
        return False

    body = candidate[2:-1]
    check_digit = candidate[-1]
    if not check_digit.isdigit():
        return False

    values = []
    for character in body:
        if character.isdigit():
            values.append(int(character))
        else:
            values.append((ord(character) - 63) % 10)

    weighted_sum = sum(
        value * (1 if index % 2 == 0 else 2)
        for index, value in enumerate(values)
    )
    expected_check_digit = (10 - weighted_sum % 10) % 10
    return expected_check_digit == int(check_digit)


def clean_ups_ocr_candidate(candidate):
    candidate = clean_tracking_candidate(candidate)
    if not re.fullmatch(r"1Z[A-Z0-9]{16}", candidate):
        return candidate

    if is_valid_ups_check_digit(candidate):
        return candidate

    corrected = candidate[:2] + candidate[2:].translate(OCR_LETTER_TO_DIGIT)
    if corrected != candidate and is_valid_ups_check_digit(corrected):
        return corrected

    return candidate


def clean_usps_ocr_candidate(candidate):
    candidate = clean_tracking_candidate(candidate)
    corrected = candidate.translate(OCR_LETTER_TO_DIGIT)

    if (
        corrected.isdigit()
        and corrected.startswith(("91", "92", "93", "94", "95", "56"))
    ):
        return corrected

    return candidate


def is_valid_tracking_candidate(candidate):
    if not candidate:
        return False

    candidate = clean_tracking_candidate(candidate)

    if len(candidate) < 12:
        return False

    if candidate.startswith(("HTTP", "AMZNTO", "WWW")):
        return False

    if candidate.startswith(("TBA", "YWNJC", "UUS", "DDIYS", "1LSD")):
        return True

    if re.fullmatch(r"1Z[A-Z0-9]{16}", candidate):
        return True

    if (
        candidate.isdigit()
        and candidate.startswith(("91", "92", "93", "94", "95"))
        and len(candidate) >= 20
    ):
        return True

    if candidate.isdigit() and candidate.startswith("56") and len(candidate) >= 20:
        return True

    if candidate.isdigit() and len(candidate) in (12, 15, 20, 22):
        return True

    return False


def is_valid_usps_tracking_candidate(candidate):
    candidate = clean_tracking_candidate(candidate)

    if (
        candidate.isdigit()
        and candidate.startswith(("91", "92", "93", "94", "95"))
        and len(candidate) >= 20
    ):
        return True

    if candidate.isdigit() and candidate.startswith("56") and len(candidate) >= 20:
        return True

    return False


def is_valid_usps_ocr_tracking_candidate(candidate):
    candidate = clean_tracking_candidate(candidate)

    if (
        candidate.isdigit()
        and candidate.startswith(("91", "92", "93", "94", "95"))
        and len(candidate) in (22, 26, 34)
    ):
        return True

    if (
        candidate.isdigit()
        and candidate.startswith("56")
        and len(candidate) in (22, 26, 34)
    ):
        return True

    return False


def extract_usps_tracking_candidates_from_text(text):
    candidates = []

    for numeric_match in re.findall(
        r"(?:[0-9OILSBG][\s_\-]*){18,34}",
        text,
        flags=re.IGNORECASE,
    ):
        candidate = clean_usps_ocr_candidate(numeric_match)

        if is_valid_usps_ocr_tracking_candidate(candidate):
            candidates.append(candidate)

    return candidates


def extract_tracking_from_ocr_lines(lines):
    for line_index, line in enumerate(lines):
        upper_line = line.upper()

        if "USPS" not in upper_line or "TRACKING" not in upper_line:
            continue

        nearby_lines = lines[line_index + 1 : min(line_index + 6, len(lines))]

        # Prefer a valid USPS number on one OCR line before combining nearby
        # lines, so trailing noise like "0 0000000" is not appended.
        for nearby_line in nearby_lines:
            for candidate in extract_usps_tracking_candidates_from_text(nearby_line):
                return candidate

        candidate_text = line

        for nearby_line in nearby_lines:
            candidate_text += " " + nearby_line

            for candidate in extract_usps_tracking_candidates_from_text(candidate_text):
                return candidate

    for line_index, line in enumerate(lines):
        upper_line = line.upper()

        if "TRACKING" not in upper_line:
            continue

        candidate_text = line

        if line_index + 1 < len(lines):
            candidate_text += " " + lines[line_index + 1]

        if line_index + 2 < len(lines):
            candidate_text += " " + lines[line_index + 2]

        ups_match = re.search(
            r"1Z(?:[\s_-]*[A-Z0-9]){16}",
            candidate_text,
            re.IGNORECASE,
        )

        if ups_match:
            candidate = clean_ups_ocr_candidate(ups_match.group())

            if is_valid_tracking_candidate(candidate):
                return candidate

        numeric_matches = re.findall(
            r"(?:[0-9OILSBG][\s_\-]*){18,34}",
            candidate_text,
            flags=re.IGNORECASE,
        )

        for numeric_match in numeric_matches:
            candidate = clean_usps_ocr_candidate(numeric_match)

            if (
                is_valid_tracking_candidate(candidate)
                and identify_carrier(candidate) != "FEDEX"
            ):
                return candidate

    for line in lines:
        candidate = clean_tracking_candidate(line)

        if candidate.startswith("1Z"):
            candidate = clean_ups_ocr_candidate(candidate)
        else:
            candidate = clean_usps_ocr_candidate(candidate)

        if (
            is_valid_tracking_candidate(candidate)
            and identify_carrier(candidate) != "FEDEX"
        ):
            return candidate

    return ""
