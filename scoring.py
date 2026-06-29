import re

import config


SCORED_FIELDS = (
    "recipient_name",
    "street_address",
    "city",
    "state",
    "zip_code",
    "tracking_number",
    "carrier",
)


def normalize_comparison_value(value):
    normalized = str(value or "").upper().strip()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return " ".join(normalized.split())


def score_label_data(label_data):
    """Compute per-field rule-based confidence and attach warnings to label_data."""
    confidence = {
        "recipient_name": 0.0,
        "street_address": 0.0,
        "city": 0.0,
        "state": 0.0,
        "zip_code": 0.0,
        "tracking_number": 0.0,
        "overall": 0.0,
    }
    warnings = []

    tracking_number = label_data.get("tracking_number", "")
    recipient_name = label_data.get("recipient_name", "")
    street_address = label_data.get("street_address", "")
    city = label_data.get("city", "")
    state = label_data.get("state", "")
    zip_code = label_data.get("zip_code", "")

    if tracking_number:
        if len(tracking_number) >= config.TRACKING_MIN_LENGTH:
            confidence["tracking_number"] = config.CONFIDENCE_TRACKING_HIGH
        else:
            confidence["tracking_number"] = config.CONFIDENCE_TRACKING_LOW
            warnings.append("tracking_number_short")
    else:
        warnings.append("tracking_number_missing")

    if recipient_name:
        if re.fullmatch(r"[A-Za-z][A-Za-z .'-]+", recipient_name):
            confidence["recipient_name"] = config.CONFIDENCE_NAME_HIGH
        else:
            confidence["recipient_name"] = config.CONFIDENCE_NAME_LOW
            warnings.append("recipient_name_low_confidence")
    else:
        warnings.append("recipient_name_missing")

    if street_address:
        if re.search(r"\d", street_address):
            confidence["street_address"] = config.CONFIDENCE_STREET_HIGH
        else:
            confidence["street_address"] = config.CONFIDENCE_STREET_LOW
            warnings.append("street_address_missing_number")
    else:
        warnings.append("street_address_missing")

    if city:
        if re.fullmatch(r"[A-Za-z .'-]+", city):
            confidence["city"] = config.CONFIDENCE_CITY_HIGH
        else:
            confidence["city"] = config.CONFIDENCE_CITY_LOW
            warnings.append("city_low_confidence")
    else:
        warnings.append("city_missing")

    if re.fullmatch(r"[A-Z]{2}", state):
        confidence["state"] = config.CONFIDENCE_STATE_VALID
    else:
        warnings.append("state_missing_or_invalid")

    if re.fullmatch(r"\d{5}(-\d{4})?", zip_code):
        confidence["zip_code"] = config.CONFIDENCE_ZIP_OCR
    else:
        warnings.append("zip_code_missing_or_invalid")

    field_scores = [
        confidence["recipient_name"],
        confidence["street_address"],
        confidence["city"],
        confidence["state"],
        confidence["zip_code"],
        confidence["tracking_number"],
    ]
    confidence["overall"] = round(sum(field_scores) / len(field_scores), 2)

    label_data["confidence"] = confidence
    label_data["warnings"] = warnings

    return label_data


def score_llm_result(llm_result, ocr_text):
    """Cross-validate each LLM field value against the raw OCR text.

    Returns per-field confidence floats, or None for each field when the LLM
    was not enabled.  A value found verbatim in the OCR is treated as high
    confidence; a value absent from the OCR may be a hallucination.
    """
    if not isinstance(llm_result, dict) or not llm_result.get("llm_enabled"):
        return {field: None for field in SCORED_FIELDS}

    ocr_normalized = normalize_comparison_value(ocr_text)
    scores = {}

    for field in SCORED_FIELDS:
        value = llm_result.get(field, "")
        if not value:
            scores[field] = 0.0
            continue

        value_normalized = normalize_comparison_value(value)
        if not value_normalized:
            scores[field] = 0.0
            continue

        if value_normalized in ocr_normalized:
            scores[field] = config.CONFIDENCE_LLM_OCR_MATCH
        else:
            scores[field] = config.CONFIDENCE_LLM_OCR_MISMATCH

    return scores
