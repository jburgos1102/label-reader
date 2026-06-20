def extract_fields_with_llm(ocr_text, rule_based_result=None):
    return {
        "recipient_name": "",
        "street_address": "",
        "city": "",
        "state": "",
        "zip_code": "",
        "tracking_number": "",
        "carrier": "",
        "llm_enabled": False,
        "llm_provider": "stub",
        "llm_notes": (
            "LLM extraction not connected yet. Stub result returned for "
            "comparison pipeline."
        ),
    }
