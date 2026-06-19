import re


# --- IDENTIFY CARRIER ---
def identify_carrier(tracking_number):
    tracking_number = tracking_number.upper()

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


# --- TRACKING HELPER FUNCTIONS ---


def clean_tracking_candidate(candidate):
    candidate = candidate.upper()
    candidate = re.sub(r"[^A-Z0-9]", "", candidate)

    if candidate.startswith("1BA"):
        candidate = "TBA" + candidate[3:]

    if candidate.startswith("1LSDO"):
        candidate = candidate.replace("O", "0", 1)

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

    for numeric_match in re.findall(r"(?:\d[\s_\-]*){18,34}", text):
        candidate = clean_tracking_candidate(numeric_match)

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
            candidate = clean_tracking_candidate(ups_match.group())

            if is_valid_tracking_candidate(candidate):
                return candidate

        numeric_matches = re.findall(r"(?:\d[\s_\-]*){18,34}", candidate_text)

        for numeric_match in numeric_matches:
            candidate = clean_tracking_candidate(numeric_match)

            if (
                is_valid_tracking_candidate(candidate)
                and identify_carrier(candidate) != "FEDEX"
            ):
                return candidate

    for line in lines:
        candidate = clean_tracking_candidate(line)

        if (
            is_valid_tracking_candidate(candidate)
            and identify_carrier(candidate) != "FEDEX"
        ):
            return candidate

    return ""
