import pytesseract
from pytesseract import Output
import re

from logger import log


_last_ocr_diagnostics = {
    "selected_text": "",
    "rotations": {},
}


def get_last_ocr_diagnostics():
    """Return a copy of the rotation texts from the most recent OCR call."""
    return {
        "selected_text": _last_ocr_diagnostics["selected_text"],
        "rotations": dict(_last_ocr_diagnostics["rotations"]),
    }


def score_ocr_text(text):
    """Score OCR text using generic shipping-label structure and readability."""
    if not text or not text.strip():
        return -20

    upper_text = text.upper()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    score = 0

    marker_weights = (
        (r"\bUSPS\s+DELIVER\s+TO\b", 3),
        (r"\bDELIVER\s+TO\b", 7),
        (r"\bSHIP\s+TO\b", 5),
        (r"\bSHIP\b", 2),
        (r"(?:^|\n)\s*TO\s*:", 4),
        (r"\bUSPS\s+TRACKING\b", 5),
        (r"\bUPS\s+TRACKING\b", 5),
        (r"\bTRACKING\b", 3),
        (r"\bUSPS\b", 2),
        (r"\bUPS\b", 2),
        (r"\bFEDEX\b", 2),
        (r"\bPRIORITY\s+MAIL\b", 2),
        (r"\bPARCEL\s+SELECT\b", 2),
        (r"\bGROUND\b", 1),
        (r"\bPOSTAGE\b", 1),
        (r"\bUNIUNI\b", 1),
    )
    for pattern, weight in marker_weights:
        if re.search(pattern, upper_text, flags=re.MULTILINE):
            score += weight

    state_zip_matches = re.findall(
        r"\b[A-Z]{2}\s+\d{5}(?:[-–— ]?\d{4})?\b",
        upper_text,
    )
    score += min(len(state_zip_matches), 2) * 5

    city_state_zip_matches = re.findall(
        r"(?:^|\n)\s*[A-Z][A-Z .'-]+,?\s+[A-Z]{2}\s+\d{5}",
        upper_text,
    )
    score += min(len(city_state_zip_matches), 2) * 3

    zip_plus_four_matches = re.findall(r"\b\d{5}[-–— ]\d{4}\b", upper_text)
    score += min(len(zip_plus_four_matches), 2) * 2

    address_like_lines = 0
    for line in lines:
        if re.search(r"\d", line) and re.search(
            r"\b(?:STREET|ST|ROAD|RD|AVENUE|AVE|BOULEVARD|BLVD|DRIVE|DR|"
            r"LANE|LN|COURT|CT|WAY|BUILDING|BLDG|CENTER|HUB)\b",
            line,
            flags=re.IGNORECASE,
        ):
            address_like_lines += 1
        elif re.search(r"\bP\.?\s*O\.?\s+BOX\s+\d+\b", line, re.IGNORECASE):
            address_like_lines += 1
    score += min(address_like_lines, 3) * 4

    readable_lines = 0
    single_character_lines = 0
    for line in lines:
        visible_characters = [
            character for character in line if not character.isspace()
        ]
        alphabetic_characters = sum(character.isalpha() for character in line)
        if len(visible_characters) == 1:
            single_character_lines += 1
        if (
            len(visible_characters) >= 3
            and alphabetic_characters >= 3
            and alphabetic_characters / len(visible_characters) >= 0.5
        ):
            readable_lines += 1
    score += min(readable_lines, 6)

    visible_text = [character for character in text if not character.isspace()]
    alphabetic_count = sum(character.isalpha() for character in visible_text)
    symbol_count = sum(not character.isalnum() for character in visible_text)
    alphabetic_ratio = alphabetic_count / len(visible_text) if visible_text else 0
    symbol_ratio = symbol_count / len(visible_text) if visible_text else 1

    if alphabetic_ratio < 0.15:
        score -= 8
    elif alphabetic_ratio < 0.3:
        score -= 4

    if symbol_ratio > 0.35:
        score -= 5
    elif symbol_ratio > 0.2:
        score -= 2

    if lines:
        single_character_ratio = single_character_lines / len(lines)
        if single_character_ratio > 0.4:
            score -= 5
        elif single_character_ratio > 0.25:
            score -= 2

    reversed_fragments = re.findall(
        r"\b(?:S4LVIS|G3LINN|CIVD|3DVLSOD|NNGWNN|ONIYOVYL|SDNS)\b",
        upper_text,
    )
    score -= min(len(reversed_fragments) * 2, 8)

    return score


def get_best_ocr_text(image):
    """Return (best_text, ocr_confidence) for the best rotation of image.

    ocr_confidence is Tesseract's mean word-level confidence (0–100).
    """
    global _last_ocr_diagnostics

    rotations = [0, 90, 180, 270]

    best_text = ""
    best_score = -1
    best_degrees = 0
    rotation_texts = {}

    for degrees in rotations:
        rotated_image = image.rotate(degrees, expand=True)
        text = pytesseract.image_to_string(rotated_image)
        rotation_texts[degrees] = text

        score = score_ocr_text(text)

        log.debug("OCR rotation %s score: %s", degrees, score)
        log.debug("OCR rotation %s preview: %s", degrees, text[:200])

        if score > best_score:
            best_score = score
            best_text = text
            best_degrees = degrees

    # Compute Tesseract word-level confidence for the winning rotation only
    best_rotated = image.rotate(best_degrees, expand=True)
    data = pytesseract.image_to_data(best_rotated, output_type=Output.DICT)
    conf_values = [c for c in data["conf"] if c != -1]
    ocr_confidence = sum(conf_values) / len(conf_values) if conf_values else 0.0

    log.debug("OCR best rotation %s confidence: %.1f", best_degrees, ocr_confidence)

    _last_ocr_diagnostics = {
        "selected_text": best_text,
        "rotations": rotation_texts,
    }

    return best_text, ocr_confidence
