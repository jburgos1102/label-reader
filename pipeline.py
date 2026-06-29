from PIL import Image

from address import normalize_extracted_fields, parse_address_from_lines
from barcodes import extract_tracking_number
from llm_extractor import extract_fields_with_llm
from logger import log
from ocr import get_best_ocr_text
from scoring import normalize_comparison_value, score_label_data
from tracking import (
    extract_tracking_from_ocr_lines,
    identify_carrier,
    identify_carrier_with_context,
)


def run(image_path):
    """Full label extraction pipeline. Returns the internal result dict."""
    image = Image.open(image_path)
    image = image.convert("RGB")

    tracking_number = extract_tracking_number(image)
    label_data = {
        "tracking_number": tracking_number,
        "carrier": identify_carrier(tracking_number),
    }

    text = get_best_ocr_text(image)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    log.debug("Raw OCR text:\n%s", text)
    for line_index, line in enumerate(lines):
        log.debug("OCR line %s: %r", line_index, line)

    ocr_tracking_candidate = extract_tracking_from_ocr_lines(lines)

    if ocr_tracking_candidate:
        log.debug("OCR tracking candidate: %r", ocr_tracking_candidate)

        has_usps_tracking_label = any(
            "USPS" in line.upper() and "TRACKING" in line.upper() for line in lines
        )
        keep_usps_barcode_tracking = (
            has_usps_tracking_label
            and identify_carrier(label_data["tracking_number"]) == "USPS"
            and identify_carrier(ocr_tracking_candidate) == "UPS"
        )

        if not keep_usps_barcode_tracking and (
            not label_data["tracking_number"]
            or identify_carrier(label_data["tracking_number"]) == "Unknown"
            or "TRACKING" in text.upper()
        ):
            label_data["tracking_number"] = ocr_tracking_candidate
            label_data["carrier"] = identify_carrier(ocr_tracking_candidate)

    label_data.update(parse_address_from_lines(lines))

    label_data = normalize_extracted_fields(label_data)
    label_data["carrier"] = identify_carrier_with_context(
        label_data["tracking_number"],
        text,
    )
    label_data = score_label_data(label_data)

    try:
        llm_result = extract_fields_with_llm(text, label_data)
    except Exception:
        log.exception("OpenAI extraction failed; using the rule-based result")
        llm_result = {
            "recipient_name": label_data.get("recipient_name", ""),
            "street_address": label_data.get("street_address", ""),
            "city": label_data.get("city", ""),
            "state": label_data.get("state", ""),
            "zip_code": label_data.get("zip_code", ""),
            "tracking_number": label_data.get("tracking_number", ""),
            "carrier": label_data.get("carrier", ""),
            "llm_enabled": False,
            "llm_provider": "openai",
            "llm_notes": "OpenAI extraction failed; using the rule-based result.",
        }

    label_data["llm_result"] = llm_result

    selected_fields = (
        "recipient_name",
        "street_address",
        "city",
        "state",
        "zip_code",
        "tracking_number",
        "carrier",
    )
    llm_available = isinstance(llm_result, dict) and bool(
        llm_result.get("llm_enabled")
    )
    selected_result = {field: label_data.get(field, "") for field in selected_fields}
    selected_sources = {}

    for field in selected_fields:
        rule_based_value = label_data.get(field, "")
        openai_value = llm_result.get(field, "") if llm_available else ""
        values_agree = llm_available and normalize_comparison_value(
            rule_based_value
        ) == normalize_comparison_value(openai_value)
        selected_sources[field] = "agreement" if values_agree else "rule_based"

    selected_result["source"] = selected_sources
    label_data["selected_result"] = selected_result

    comparison = {}

    for field in selected_fields:
        rule_based_value = label_data.get(field, "")
        openai_value = llm_result.get(field, "") if llm_available else ""
        selected_value = selected_result.get(field, "")
        comparison[field] = {
            "rule_based": rule_based_value,
            "openai": openai_value,
            "selected": selected_value,
            "agreement": llm_available
            and normalize_comparison_value(rule_based_value)
            == normalize_comparison_value(openai_value),
        }

    label_data["comparison"] = comparison

    return label_data
