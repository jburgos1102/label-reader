import io
import time

from PIL import Image

import config
from address import normalize_extracted_fields, parse_address_from_lines
from barcodes import extract_tracking_number, get_last_raw_barcodes
from llm_extractor import extract_fields_with_llm, fallback_result
from logger import log
from models import EXTRACTION_FIELDS, ExtractionResult, FieldValue
from ocr import get_best_ocr_text
from scoring import normalize_comparison_value, score_label_data, score_llm_result
from tracking import (
    extract_tracking_from_ocr_lines,
    identify_carrier,
    identify_carrier_with_context,
    validate_tracking_checksum,
)


LLM_POLICIES = ("off", "auto", "force_vision")


def _resolve_llm_policy(skip_llm, llm_policy):
    """Normalize the legacy skip_llm flag and llm_policy into one policy.

    skip_llm=True is a strict alias for llm_policy="off"; combining it with a
    policy that allows LLM calls is a contradiction and raises rather than
    silently picking a winner.
    """
    if llm_policy is not None and llm_policy not in LLM_POLICIES:
        raise ValueError(
            f"llm_policy must be one of {LLM_POLICIES}, got {llm_policy!r}"
        )
    if skip_llm and llm_policy in ("auto", "force_vision"):
        raise ValueError(
            f"skip_llm=True contradicts llm_policy={llm_policy!r}; "
            "use llm_policy='off' (or drop skip_llm)"
        )
    if llm_policy is not None:
        return llm_policy
    return "off" if skip_llm else "auto"


def run(image_path, skip_llm=False, llm_policy=None):
    """Full label extraction pipeline. Returns the internal result dict.

    llm_policy: "off" (no LLM call of any kind), "auto" (text LLM plus
    vision when triggers fire — the historical default), or "force_vision"
    (one vision call regardless of triggers). skip_llm=True is a strict
    alias for "off".
    """
    llm_policy = _resolve_llm_policy(skip_llm, llm_policy)
    _start = time.monotonic()

    image = Image.open(image_path)
    image = image.convert("RGB")

    # Retain JPEG bytes for optional vision LLM calls
    _buf = io.BytesIO()
    image.save(_buf, format="JPEG", quality=92)
    image_bytes = _buf.getvalue()

    barcode_tracking = extract_tracking_number(image)
    barcode_raw = "|".join(get_last_raw_barcodes())
    label_data = {
        "tracking_number": barcode_tracking,
        "carrier": identify_carrier(barcode_tracking),
    }

    ocr_text, ocr_confidence, ocr_rotations_tried = get_best_ocr_text(image)
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

    log.debug("Raw OCR text:\n%s", ocr_text)
    log.debug("OCR confidence: %.1f", ocr_confidence)
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
            or "TRACKING" in ocr_text.upper()
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
        ocr_text,
    )
    label_data = score_label_data(label_data)

    # Checksum validation — lower confidence and warn if format is known but check fails
    _checksum_valid = validate_tracking_checksum(
        label_data.get("tracking_number", ""),
        label_data.get("carrier", ""),
    )
    if _checksum_valid is False:
        label_data["confidence"]["tracking_number"] = config.CONFIDENCE_TRACKING_CHECKSUM_FAIL
        label_data.setdefault("warnings", []).append("tracking_checksum_failed")
    log.debug(
        "Tracking checksum: %s (carrier=%s valid=%s)",
        label_data.get("tracking_number", ""),
        label_data.get("carrier", ""),
        _checksum_valid,
    )

    # Vision-trigger evaluation. The reason names are telemetry: they are
    # surfaced under metadata.llm.trigger_reasons. Computed for every policy
    # (including "off") so the decision is always visible in logs/telemetry.
    _address_fields = ("city", "state", "street_address", "zip_code")
    _scored_conf = label_data.get("confidence", {})
    blank_address_fields = sum(
        1 for f in _address_fields
        if not label_data.get(f) or _scored_conf.get(f, 0.0) == 0.0
    )
    trigger_checks = (
        ("ocr_confidence_low",
         ocr_confidence < config.OCR_CONFIDENCE_VISION_THRESHOLD),
        ("ocr_text_short",
         len(ocr_text.strip()) < config.OCR_TEXT_LENGTH_VISION_THRESHOLD),
        ("blank_address_fields",
         blank_address_fields >= config.VISION_TRIGGER_BLANK_FIELDS),
        ("street_rejected", label_data.get("_street_rejected", False)),
        ("name_looks_like_street", label_data.get("_name_looks_like_street", False)),
        ("name_is_sender_artifact", label_data.get("_name_is_sender_artifact", False)),
        ("name_single_word", label_data.get("_name_single_word", False)),
        ("name_looks_like_org", label_data.get("_name_looks_like_org", False)),
    )
    trigger_reasons = [name for name, fired in trigger_checks if fired]
    use_vision = bool(trigger_reasons)
    log.debug(
        "Vision trigger: policy=%s ocr_conf=%.1f text_len=%d reasons=%s",
        llm_policy, ocr_confidence, len(ocr_text.strip()), trigger_reasons,
    )

    selected_fields = (
        "recipient_name",
        "street_address",
        "city",
        "state",
        "zip_code",
        "tracking_number",
        "carrier",
    )

    llm_mode = "none"

    def _skip_stub(notes):
        return fallback_result(label_data, enabled=False, notes=notes)

    if llm_policy == "off":
        # Authoritative switch: no LLM call of any kind, even when the vision
        # trigger fired.
        llm_result = _skip_stub("LLM skipped by policy (off).")
    elif llm_policy == "force_vision" or use_vision:
        try:
            llm_result = extract_fields_with_llm(ocr_text, label_data, image=image_bytes)
            llm_mode = "vision"
        except Exception:
            log.exception("Vision LLM extraction failed; using rule-based result")
            llm_result = _skip_stub("Vision LLM extraction failed; using rule-based result.")
    else:
        try:
            llm_result = extract_fields_with_llm(ocr_text, label_data)
            llm_mode = "text"
        except Exception:
            log.exception("LLM extraction failed; using rule-based result")
            llm_result = _skip_stub("LLM extraction failed; using rule-based result.")

        # Secondary trigger: if text LLM ran but street is still empty/rejected,
        # try vision LLM to recover fields that OCR garbled
        if llm_mode == "text" and llm_result.get("llm_enabled"):
            rule_street_empty = (
                not label_data.get("street_address")
                or label_data.get("_street_rejected")
            )
            llm_street_empty = not llm_result.get("street_address", "")
            if rule_street_empty and llm_street_empty:
                try:
                    vision_result = extract_fields_with_llm(
                        ocr_text, label_data, image=image_bytes
                    )
                    for f in selected_fields:
                        if not llm_result.get(f) and vision_result.get(f):
                            llm_result[f] = vision_result[f]
                    llm_mode = "vision_fallback"
                except Exception:
                    log.exception("Vision fallback LLM failed")

    label_data["llm_result"] = llm_result

    llm_available = isinstance(llm_result, dict) and bool(
        llm_result.get("llm_enabled")
    )
    selected_result = {}
    selected_sources = {}

    for field in selected_fields:
        rule_based_value = label_data.get(field, "")
        openai_value = llm_result.get(field, "") if llm_available else ""
        values_agree = llm_available and normalize_comparison_value(
            rule_based_value
        ) == normalize_comparison_value(openai_value)

        if not rule_based_value and llm_available and openai_value:
            # Rule produced nothing; LLM fills the gap.
            selected_result[field] = openai_value
            selected_sources[field] = "llm"
        elif (
            llm_mode == "vision"
            and llm_available
            and openai_value
            and not values_agree
        ):
            # Vision was triggered because rule-based extraction is unreliable;
            # defer to LLM on any conflict rather than letting rule win.
            selected_result[field] = openai_value
            selected_sources[field] = "llm"
        else:
            selected_result[field] = rule_based_value
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

    label_data["_ocr_text"] = ocr_text
    label_data["_ocr_confidence"] = ocr_confidence
    label_data["_ocr_rotations_tried"] = ocr_rotations_tried
    label_data["_barcode_raw"] = barcode_raw
    label_data["_llm_mode"] = llm_mode
    label_data["_llm_requested_mode"] = llm_policy
    label_data["_llm_trigger_reasons"] = trigger_reasons
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
    llm_mode = internal.get("_llm_mode", "none")
    ocr_rotations_tried = internal.get("_ocr_rotations_tried", 4)
    processing_ms = internal.get("_processing_ms", 0)

    # Only flag a conflict when both rule and LLM produced real, differing values.
    # Empty rule or "Unknown" carrier = rule failed; LLM filling it is a fallback, not a conflict.
    conflicts = []
    if llm_called:
        for field, cmp in internal.get("comparison", {}).items():
            rule_val = cmp.get("rule_based", "")
            if (
                field in EXTRACTION_FIELDS
                and not cmp.get("agreement")
                and cmp.get("openai")
                and rule_val
                and rule_val != "Unknown"
            ):
                conflicts.append(field)

    tracking_confidence = confidence.get("tracking_number", 0.0)

    # Additive telemetry (metadata.llm). "fields_from_llm" records where
    # selection chose the LLM value — a change marker, not proof of
    # improvement; improvement is computed offline against ground truth.
    _llm_dict = llm_result if isinstance(llm_result, dict) else {}
    llm_telemetry = {
        "requested_mode": internal.get("_llm_requested_mode", "auto"),
        "called": llm_called,
        "mode": llm_mode,
        "provider": _llm_dict.get("llm_provider", "none"),
        "model": _llm_dict.get("llm_model", ""),
        "latency_ms": _llm_dict.get("llm_latency_ms"),
        "trigger_reasons": internal.get("_llm_trigger_reasons", []),
        "fields_from_llm": sorted(
            field for field, source in selected_sources.items() if source == "llm"
        ),
    }

    # BUG 1 fix: cross-validate LLM values against OCR text so LLM-sourced fields
    # get 0.85 (found in OCR) or 0.75 (plausible from image) instead of rule-based 0.0.
    llm_scores = score_llm_result(llm_result, internal.get("_ocr_text", ""))

    def _fv(name):
        rule_value = internal.get(name, "")
        sel_value = internal.get("selected_result", {}).get(name, "")
        raw = selected_sources.get(name, "rule_based")
        # When the selection loop chose LLM (e.g. vision conflict override),
        # prefer the selected LLM value; otherwise prefer the rule value.
        value = (sel_value or rule_value) if raw == "llm" else (rule_value or sel_value)
        conf = confidence.get(name, 0.0)
        if name == "carrier":
            # Carrier has no scored confidence of its own; it is inferred from
            # the tracking number, so it inherits tracking confidence. This
            # under-reports carrier confidence when carrier came from OCR
            # context with no tracking number (conf stays 0.0 even when the
            # carrier is right) — revisit with calibrated confidence.
            conf = tracking_confidence
        if not value:
            if name == "street_address" and internal.get("_street_rejected"):
                source = "rejected_rule"
            else:
                source = "blank"
        elif name == "tracking_number":
            source = tracking_source
        else:
            if raw == "agreement":
                source = "agreement"
            elif raw == "llm":
                source = "llm"
                llm_conf = llm_scores.get(name)
                if llm_conf is not None:
                    conf = llm_conf
            else:
                source = "rule"
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
        llm_mode=llm_mode,
        ocr_rotations_tried=ocr_rotations_tried,
        llm_telemetry=llm_telemetry,
    )
