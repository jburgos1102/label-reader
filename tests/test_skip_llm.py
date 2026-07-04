"""skip_llm must be authoritative: no LLM call of any kind when it is set.

Run from the project root:  venv/bin/python tests/test_skip_llm.py

Plain asserts (no pytest dependency). Uses a real dataset image so the full
pipeline runs, but stubs out the LLM function and forces the vision trigger
so the test would catch a regression where vision bypasses skip_llm.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure no accidental real provider calls even if the stub is bypassed.
os.environ["GROQ_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

import config
import pipeline


class LlmSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, text, rule_result, image=None):
        self.calls.append("vision" if image is not None else "text")
        result = {field: "" for field in pipeline.EXTRACTION_FIELDS}
        result.update(
            {"llm_enabled": True, "llm_provider": "spy", "llm_notes": ""}
        )
        return result


def main():
    image_path = sorted(glob.glob("datasets/usps/images/*"))[0]

    # Force the vision trigger for every label: Tesseract confidence is 0-100,
    # so any scan falls below a threshold of 101.
    original_threshold = config.OCR_CONFIDENCE_VISION_THRESHOLD
    original_llm = pipeline.extract_fields_with_llm
    spy = LlmSpy()
    config.OCR_CONFIDENCE_VISION_THRESHOLD = 101
    pipeline.extract_fields_with_llm = spy

    try:
        # skip_llm=True: no LLM call at all, even with the vision trigger firing
        result = pipeline.run(image_path, skip_llm=True)
        assert spy.calls == [], f"LLM was called despite skip_llm=True: {spy.calls}"
        assert result["_llm_mode"] == "none", result["_llm_mode"]
        assert result["llm_result"]["llm_enabled"] is False

        # skip_llm=False with the vision trigger: vision LLM must still run
        result = pipeline.run(image_path, skip_llm=False)
        assert spy.calls == ["vision"], spy.calls
        assert result["_llm_mode"] == "vision", result["_llm_mode"]
    finally:
        config.OCR_CONFIDENCE_VISION_THRESHOLD = original_threshold
        pipeline.extract_fields_with_llm = original_llm

    print("test_skip_llm OK")


if __name__ == "__main__":
    main()
