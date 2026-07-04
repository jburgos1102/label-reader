"""LLM-backed field extraction for OCR text from shipping labels.

Provider priority: Groq (if GROQ_API_KEY set) → OpenAI (if OPENAI_API_KEY set).
Both providers are called via the OpenAI SDK using chat completions.
When an image (bytes) is passed the vision model is used; otherwise the text model.
"""

import base64
import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

import config
from logger import log


load_dotenv()


# Clients are cached per (provider, key, url, timeout) so each request does not
# pay client construction; the OpenAI client is thread-safe for concurrent use.
_client_cache: dict[tuple, OpenAI] = {}


def _get_client(provider: dict[str, Any]) -> OpenAI:
    cache_key = (
        provider["name"],
        provider["api_key"],
        provider["base_url"],
        provider["timeout"],
    )
    client = _client_cache.get(cache_key)
    if client is None:
        client_kwargs: dict[str, Any] = {
            "api_key": provider["api_key"],
            "timeout": provider["timeout"],
        }
        if provider["base_url"]:
            client_kwargs["base_url"] = provider["base_url"]
        client = OpenAI(**client_kwargs)
        _client_cache[cache_key] = client
    return client


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
    "OCR tracking candidate. If a name or word is printed twice in a row with "
    "no other word between (e.g. 'KRISTINA SCHUMACHER SCHUMACHER'), collapse "
    "it to a single occurrence ('KRISTINA SCHUMACHER') — this is a label "
    "printing artifact, not part of the name. "
    "In the TO: section, extract the individual person's full name as "
    "recipient_name. A person's name is 2–3 recognizable personal name words "
    "(e.g., KRISTINA SCHUMACHER, JOHN WALKER, PAUL BUNJEVIC SR.). Company and "
    "organization names contain words like SYSTEMS, USA, INC, CORP, CORPORATION, "
    "GROUP, ASSOCIATES, INDUSTRIES, SOLUTIONS, etc. Use these words only to "
    "recognize that a line is a company/org line rather than a person's name — "
    "if both a person name and a company name appear in the TO: block, extract "
    "only the person name regardless of which line it appears on. "
    "If there is no individual person named and the recipient is purely an "
    "organization or department, use the organization name — in that case a "
    "company-indicator word such as CORP, CORPORATION, or GROUP does NOT mean "
    "the line should be dropped; it stays part of recipient_name like any other "
    "organization line. "
    "FedEx labels have two address blocks: the ORIGIN / FROM / ORIGIN ID section "
    "(top-left) contains the sender's name and address — ignore this section "
    "entirely. The TO: section (center or right) contains the recipient's name "
    "and delivery address — extract only from this section. "
    "'BILL SENDER' is a FedEx billing instruction, not a person or organization "
    "name. If you see 'BILL SENDER' anywhere on the label, do not use it as "
    "recipient_name. "
    "When the TO: section contains multiple lines before the street address, "
    "apply this two-step logic in order: "
    "Step 1 — scan every non-address line for a recognizable person's full name "
    "(first + last, e.g. BRETT SIEFERT, JOHN WALKER). If ANY line is a person's "
    "name, recipient_name is that person's name ONLY. Ignore every other line "
    "in the TO: block — department lines, company lines, everything — even if "
    "you would normally combine them. For example, TO: lines 'BRETT SIEFERT' / "
    "'LINCOLN COUNTY HEALTH DEPARTMENT' / '#5 HEALTH DEPARTMENT DRIVE' / 'TROY "
    "MO 63379' must produce recipient_name='BRETT SIEFERT' — do NOT append "
    "'LINCOLN COUNTY HEALTH DEPARTMENT' to it. "
    "Step 2 — only if NO line in the TO: block is a person's name (every "
    "non-address line describes a department/org/company), combine EVERY one "
    "of those lines into a single recipient_name string, in the order they "
    "appear. Do this even when a line is a short single word that looks like "
    "it could be a place name — a short word directly under a DEPT line and "
    "above the street address is a division or client identifier, not a "
    "location, and must still be appended to recipient_name, never dropped. "
    "For example, TO: lines 'TRANSFER DEPT' / 'PUTNAM' / '30 DAN RD' / 'CANTON "
    "MA 02021' (no person name anywhere) must produce "
    "recipient_name='TRANSFER DEPT PUTNAM' (not just 'TRANSFER DEPT') and "
    "city='CANTON' (not 'PUTNAM'). Keep a DEPT line's second word even when it "
    "also happens to reappear inside the street address line below (e.g. TO: "
    "lines 'TRANSFER DEPT' / 'PERSHING' / 'ONE PERSHING PLAZA' / 'JERSEY CITY "
    "NJ 07399', no person name anywhere, must produce "
    "recipient_name='TRANSFER DEPT PERSHING', keeping 'PERSHING' even though "
    "it also appears in the street address — the two lines are separate "
    "fields and neither one is redundant. The street address line — starts "
    "with a house number, either digits (e.g. '30 DAN RD') or a spelled-out "
    "number word (e.g. 'ONE PERSHING PLAZA'), or a '#' unit marker (e.g. '#5 "
    "HEALTH DEPARTMENT DRIVE') — is never part of recipient_name. "
    "The city is read ONLY from the CITY STATE ZIP line, which is the line "
    "directly below the street address (format: CITY STATE ZIPCODE, e.g. "
    "'CANTON MA 02021'). Never take the city from a line inside the TO: block "
    "that appears above the street address, even if that line happens to look "
    "like a real city or town name — a line in that position is part of the "
    "organization/department name, not the city. "
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


def fallback_result(
    rule_result: dict[str, Any] | None,
    *,
    enabled: bool,
    notes: str,
    provider: str = "none",
) -> dict[str, Any]:
    """Build an llm_result-shaped dict from rule-based values.

    Used both here (LLM disabled/failed) and by pipeline.py (LLM skipped),
    so the fallback shape is defined in exactly one place.
    """
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
        return fallback_result(
            rule_result,
            enabled=False,
            notes="LLM extraction disabled: set GROQ_API_KEY or OPENAI_API_KEY to enable.",
        )

    use_vision = image is not None
    model = provider["vision_model"] if use_vision else provider["text_model"]
    provider_name = provider["name"]

    ocr_text = str(text or "").strip()
    if not ocr_text and not use_vision:
        return fallback_result(
            rule_result,
            enabled=True,
            notes="LLM extraction skipped: OCR text is empty.",
            provider=provider_name,
        )

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

    mode = "vision" if use_vision else "text"
    started = time.monotonic()

    try:
        client = _get_client(provider)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception:
        latency_ms = round((time.monotonic() - started) * 1000)
        log.exception(
            "LLM API call failed (provider=%s model=%s mode=%s latency_ms=%d)",
            provider_name, model, mode, latency_ms,
        )
        return fallback_result(
            rule_result,
            enabled=True,
            notes=f"{provider_name} API call failed; falling back to rule-based result.",
            provider=provider_name,
        )

    latency_ms = round((time.monotonic() - started) * 1000)

    try:
        extracted = json.loads(response.choices[0].message.content)
        if not isinstance(extracted, dict):
            raise ValueError("LLM response was not a JSON object")
    except Exception:
        log.exception(
            "LLM response was not valid JSON (provider=%s model=%s mode=%s latency_ms=%d)",
            provider_name, model, mode, latency_ms,
        )
        return fallback_result(
            rule_result,
            enabled=True,
            notes=f"{provider_name} returned unparseable output; falling back to rule-based result.",
            provider=provider_name,
        )

    log.info(
        "LLM extraction ok (provider=%s model=%s mode=%s latency_ms=%d)",
        provider_name, model, mode, latency_ms,
    )

    result = fallback_result(rule_result, enabled=True, notes="", provider=provider_name)
    for field in FIELD_NAMES:
        value = extracted.get(field)
        if isinstance(value, str) and value.strip():
            result[field] = value.strip()

    # Internal telemetry only — not part of the public API response shape.
    result["llm_model"] = model
    result["llm_latency_ms"] = latency_ms

    return result
