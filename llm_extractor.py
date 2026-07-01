"""LLM-backed field extraction for OCR text from shipping labels.

Provider priority: Groq (if GROQ_API_KEY set) → OpenAI (if OPENAI_API_KEY set).
Both providers are called via the OpenAI SDK using chat completions.
When an image (bytes) is passed the vision model is used; otherwise the text model.
"""

import base64
import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

import config


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

_SYSTEM_PROMPT = (
    "Extract recipient shipping fields from the supplied OCR text, correcting "
    "only obvious OCR spacing and label-noise errors. OCR may drop the first "
    "letter of a label marker: for example, treat HIP as SHIP only when its "
    "position and context show that it is a marker, and never include SHIP or "
    "HIP in the recipient name. Remove label markers and promotional text such "
    "as SHIP, TO, TRACKING, UPS GROUND, USPS TRACKING, DO NOT MISS OUT, and "
    "amzn.to/social from extracted field values. Normalize obvious street "
    "spacing, such as 28N COLLEGE ST to 28 N COLLEGE ST. Remove trailing OCR "
    "junk from tracking numbers, such as SS when it is not part of a valid UPS "
    "tracking number. When preferred_tracking_number is non-empty, treat it as "
    "trusted barcode/rule-based evidence and use it instead of a conflicting "
    "OCR tracking candidate. "
    "In the TO: section, extract the individual person's full name as "
    "recipient_name. A person's name is 2–3 recognizable personal name words "
    "(e.g., KRISTINA SCHUMACHER, JOHN WALKER, PAUL BUNJEVIC SR.). Company and "
    "organization names contain words like SYSTEMS, USA, INC, CORP, GROUP, "
    "ASSOCIATES, INDUSTRIES, SOLUTIONS, etc. — do not use these as "
    "recipient_name. If both a person name and a company name appear in the TO: "
    "block, extract only the person name regardless of which line it appears on. "
    "If there is no individual person named and the recipient is purely an "
    "organization or department, use the organization name. "
    "FedEx labels have two address blocks: the ORIGIN / FROM / ORIGIN ID section "
    "(top-left) contains the sender's name and address — ignore this section "
    "entirely. The TO: section (center or right) contains the recipient's name "
    "and delivery address — extract only from this section. "
    "'BILL SENDER' is a FedEx billing instruction, not a person or organization "
    "name. If you see 'BILL SENDER' anywhere on the label, do not use it as "
    "recipient_name. "
    "When the TO: section contains multiple lines before the street address, "
    "apply this logic: if one line is a recognizable person's full name "
    "(first + last) and other lines are a company or org, extract only the "
    "person's name. If ALL non-address lines describe an organization (no person "
    "name present), combine them into a single recipient_name string. The street "
    "address line (starts with a number) is never part of recipient_name. "
    "Do not invent missing values; return an empty "
    "string for every field that cannot be determined. "
    f"Return a JSON object with exactly these keys: {', '.join(FIELD_NAMES)}. "
    "No other keys. No markdown. No explanation."
)


def _detect_provider() -> dict[str, Any] | None:
    """Return a provider config dict or None if no API key is configured."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        return {
            "name": "groq",
            "api_key": groq_key,
            "base_url": config.GROQ_BASE_URL,
            "text_model": config.GROQ_TEXT_MODEL,
            "vision_model": config.GROQ_VISION_MODEL,
            "timeout": config.GROQ_TIMEOUT,
        }
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return {
            "name": "openai",
            "api_key": openai_key,
            "base_url": None,
            "text_model": config.OPENAI_MODEL,
            "vision_model": config.OPENAI_MODEL,
            "timeout": config.OPENAI_TIMEOUT,
        }
    return None


def _safe_result(
    rule_result: dict[str, Any] | None,
    *,
    enabled: bool,
    notes: str,
    provider: str = "none",
) -> dict[str, Any]:
    rule_result = rule_result if isinstance(rule_result, dict) else {}
    result = {field: str(rule_result.get(field) or "").strip() for field in FIELD_NAMES}
    result.update({"llm_enabled": enabled, "llm_provider": provider, "llm_notes": notes})
    return result


def extract_fields_with_llm(
    text: str,
    rule_result: dict[str, Any],
    image: bytes | None = None,
) -> dict[str, Any]:
    """Extract shipping-label fields using the configured LLM provider.

    Pass `image` (JPEG bytes) to use the vision model, which reads the label
    directly and can recover values that OCR garbled.  Omit it to use the
    faster text model with the OCR output only.
    """
    provider = _detect_provider()
    if not provider:
        return _safe_result(
            rule_result,
            enabled=False,
            notes="LLM extraction disabled: set GROQ_API_KEY or OPENAI_API_KEY to enable.",
        )

    use_vision = image is not None
    model = provider["vision_model"] if use_vision else provider["text_model"]
    provider_name = provider["name"]

    ocr_text = str(text or "").strip()
    if not ocr_text and not use_vision:
        return _safe_result(
            rule_result,
            enabled=True,
            notes="LLM extraction skipped: OCR text is empty.",
            provider=provider_name,
        )

    client_kwargs: dict[str, Any] = {
        "api_key": provider["api_key"],
        "timeout": provider["timeout"],
    }
    if provider["base_url"]:
        client_kwargs["base_url"] = provider["base_url"]

    payload = json.dumps(
        {
            "ocr_text": ocr_text,
            "preferred_tracking_number": str(
                (rule_result or {}).get("tracking_number") or ""
            ).strip(),
        },
        ensure_ascii=False,
    )

    if use_vision:
        b64 = base64.b64encode(image).decode("utf-8")
        user_content: Any = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": payload},
        ]
    else:
        user_content = payload

    try:
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        extracted = json.loads(response.choices[0].message.content)
        if not isinstance(extracted, dict):
            raise ValueError("LLM response was not a JSON object")
    except Exception:
        return _safe_result(
            rule_result,
            enabled=True,
            notes=f"{provider_name} extraction failed; falling back to rule-based result.",
            provider=provider_name,
        )

    result = _safe_result(rule_result, enabled=True, notes="", provider=provider_name)
    for field in FIELD_NAMES:
        value = extracted.get(field)
        if isinstance(value, str) and value.strip():
            result[field] = value.strip()

    return result
