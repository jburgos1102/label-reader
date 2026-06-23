"""OpenAI-backed field extraction for OCR text from shipping labels."""

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


FIELD_NAMES = (
    "recipient_name",
    "street_address",
    "city",
    "state",
    "zip_code",
    "tracking_number",
    "carrier",
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {field: {"type": "string"} for field in FIELD_NAMES},
    "required": list(FIELD_NAMES),
    "additionalProperties": False,
}


def _safe_result(
    rule_result: dict[str, Any] | None,
    *,
    enabled: bool,
    notes: str,
) -> dict[str, Any]:
    """Return the rule result in the stable LLM interface shape."""
    rule_result = rule_result if isinstance(rule_result, dict) else {}
    result = {
        field: str(rule_result.get(field) or "").strip() for field in FIELD_NAMES
    }
    result.update(
        {
            "llm_enabled": enabled,
            "llm_provider": "openai",
            "llm_notes": notes,
        }
    )
    return result


def extract_fields_with_llm(text: str, rule_result: dict[str, Any]) -> dict[str, Any]:
    """Extract shipping-label fields from OCR text using OpenAI."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _safe_result(
            rule_result,
            enabled=False,
            notes="OpenAI extraction is disabled because OPENAI_API_KEY is not configured.",
        )

    ocr_text = str(text or "").strip()
    if not ocr_text:
        return _safe_result(
            rule_result,
            enabled=True,
            notes="OpenAI extraction was skipped because the OCR text is empty.",
        )

    try:
        client = OpenAI(api_key=api_key, timeout=30.0)
        response = client.responses.create(
            model="gpt-5",
            instructions=(
                "Extract the recipient shipping fields from the supplied OCR text. "
                "Use only evidence present in the OCR text. Do not infer missing values. "
                "Return an empty string for every field that cannot be determined. "
                "Return strict JSON matching the provided schema and no other content."
            ),
            input=ocr_text,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "shipping_label_fields",
                    "schema": OUTPUT_SCHEMA,
                    "strict": True,
                }
            },
        )
        extracted = json.loads(response.output_text)
        if not isinstance(extracted, dict):
            raise ValueError("OpenAI response was not a JSON object")
    except Exception:
        return _safe_result(
            rule_result,
            enabled=True,
            notes="OpenAI extraction failed; using the rule-based result.",
        )

    result = _safe_result(rule_result, enabled=True, notes="")
    for field in FIELD_NAMES:
        value = extracted.get(field)
        if isinstance(value, str) and value.strip():
            result[field] = value.strip()

    return result
