import time

from PIL import Image

from address import normalize_extracted_fields, parse_address_from_lines
from barcodes import extract_tracking_number
from llm_extractor import extract_fields_with_llm
from logger import log
from models import EXTRACTION_FIELDS, ExtractionResult, FieldValue
from ocr import get_best_ocr_text
from scoring import normalize_comparison_value, score_label_data
from tracking import (
    extract_tracking_from_ocr_lines,
    identify_carrier,
    identify_carrier_with_context,
)


def run(image_path):
    """Full label extraction pipeline. Returns the internal result dict."""
    _start = time.monotonic()

    image = Image.open(image_path)
    image = image.convert("RGB")

    barcode_tracking = extract_tracking_number(image)
    label_data = {
        "tracking_number": barcode_tracking,
        "carrier": identify_carrier(barcode_tracking),
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

    final_tracking = label_data["tracking_number"]
    if final_tracking and final_tracking == barcode_tracking:
        tracking_source = "barcode"
    elif final_tracking and final_tracking == ocr_tracking_candidate:
        tracking_source = "ocr"
    elif final_tracking:
        tracking_source = "rule"
    else:
        tracking_source = "blank"

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

    label_data["_ocr_text"] = text
    label_data["_tracking_source"] = tracking_source
    label_data["_processing_ms"] = round((time.monotonic() - _start) * 1000)

    return label_data


def build_extraction_result(internal, label_id):
    """Convert a pipeline.run() internal dict to a typed ExtractionResult."""
    confidence = internal.get("confidence", {})
    tracking_source = internal.get("_tracking_source", "rule")
    selected_sources = internal.get("selected_result", {}).get("source", {})
    llm_result = internal.get("llm_result", {})
    llm_called = bool(isinstance(llm_result, dict) and llm_result.get("llm_enabled"))
    processing_ms = internal.get("_processing_ms", 0)

    conflicts = []
    if llm_called:
        for field, cmp in internal.get("comparison", {}).items():
            if field in EXTRACTION_FIELDS and not cmp.get("agreement") and cmp.get("openai"):
                conflicts.append(field)

    tracking_confidence = confidence.get("tracking_number", 0.0)

    def _fv(name):
        value = internal.get(name, "")
        conf = confidence.get(name, 0.0)
        if name == "carrier":
            conf = tracking_confidence
        if not value:
            source = "blank"
        elif name == "tracking_number":
            source = tracking_source
        else:
            raw = selected_sources.get(name, "rule_based")
            source = "agreement" if raw == "agreement" else "rule"
        return FieldValue(value=value, confidence=conf, source=source)

    return ExtractionResult(
        label_id=label_id,
        recipient_name=_fv("recipient_name"),
        street_address=_fv("street_address"),
        city=_fv("city"),
        state=_fv("state"),
        zip_code=_fv("zip_code"),
        tracking_number=_fv("tracking_number"),
        carrier=_fv("carrier"),
        llm_called=llm_called,
        conflicts=conflicts,
        processing_ms=processing_ms,
    )
