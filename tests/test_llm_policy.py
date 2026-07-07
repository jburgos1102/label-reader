"""Tests for the llm_policy thin slice (off / auto / force_vision).

Run from the project root:  venv/bin/python tests/test_llm_policy.py

Plain asserts (no pytest dependency). The LLM function is stubbed with a spy;
the vision trigger is forced via the OCR-confidence threshold so tests are
deterministic. API-level tests use Flask's test client and clean up the rows
and files they create.
"""

import glob
import io
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Never hit a real provider even if a stub is bypassed.
os.environ["GROQ_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

import config
import pipeline

TEST_FILENAME = "llmpolicy_test.jpg"


class LlmSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, text, rule_result, image=None):
        self.calls.append("vision" if image is not None else "text")
        result = {field: "" for field in pipeline.EXTRACTION_FIELDS}
        result.update(
            {"llm_enabled": True, "llm_provider": "spy", "llm_notes": "",
             "llm_model": "spy-model", "llm_latency_ms": 1}
        )
        return result


def _cleanup_api_rows():
    conn = sqlite3.connect(config.STORAGE_DB_PATH)
    rows = conn.execute(
        "SELECT id, image_path FROM labels WHERE original_filename = ?",
        (TEST_FILENAME,),
    ).fetchall()
    for _, image_path in rows:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
    conn.execute("DELETE FROM labels WHERE original_filename = ?", (TEST_FILENAME,))
    conn.commit()
    conn.close()


def pipeline_tests(image_path):
    spy = LlmSpy()
    original_llm = pipeline.extract_fields_with_llm
    original_threshold = config.OCR_CONFIDENCE_VISION_THRESHOLD
    pipeline.extract_fields_with_llm = spy
    config.OCR_CONFIDENCE_VISION_THRESHOLD = 101  # trigger always fires

    try:
        # off: zero calls even with the trigger firing; telemetry recorded
        result = pipeline.run(image_path, llm_policy="off")
        assert spy.calls == [], spy.calls
        assert result["_llm_requested_mode"] == "off"
        assert "ocr_confidence_low" in result["_llm_trigger_reasons"]

        # skip_llm=True is a strict alias for off
        result = pipeline.run(image_path, skip_llm=True)
        assert spy.calls == [], spy.calls
        assert result["_llm_requested_mode"] == "off"

        # force_vision: exactly one vision call
        result = pipeline.run(image_path, llm_policy="force_vision")
        assert spy.calls == ["vision"], spy.calls
        assert result["_llm_mode"] == "vision"

        # force_vision must call vision even when NO trigger fires
        spy.calls.clear()
        config.OCR_CONFIDENCE_VISION_THRESHOLD = original_threshold
        saved = {
            name: getattr(config, name)
            for name in ("OCR_TEXT_LENGTH_VISION_THRESHOLD", "VISION_TRIGGER_BLANK_FIELDS")
        }
        config.OCR_TEXT_LENGTH_VISION_THRESHOLD = 0
        config.VISION_TRIGGER_BLANK_FIELDS = 99
        try:
            result = pipeline.run(image_path, llm_policy="force_vision")
            assert spy.calls == ["vision"], spy.calls
        finally:
            for name, value in saved.items():
                setattr(config, name, value)
        config.OCR_CONFIDENCE_VISION_THRESHOLD = 101

        # contradictions and invalid values raise ValueError
        for bad_kwargs in (
            {"skip_llm": True, "llm_policy": "auto"},
            {"skip_llm": True, "llm_policy": "force_vision"},
            {"llm_policy": "banana"},
        ):
            try:
                pipeline.run(image_path, **bad_kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected ValueError for {bad_kwargs}")

        # metadata.llm is additive and populated
        spy.calls.clear()
        internal = pipeline.run(image_path, llm_policy="auto")
        payload = pipeline.build_extraction_result(internal, "test-id").to_dict()
        meta = payload["metadata"]
        for key in ("llm_called", "conflicts", "processing_ms", "llm_mode",
                    "ocr_rotations_tried"):
            assert key in meta, key  # existing fields untouched
        llm_meta = meta["llm"]
        assert llm_meta["requested_mode"] == "auto"
        assert llm_meta["called"] is True
        assert llm_meta["provider"] == "spy"
        assert llm_meta["model"] == "spy-model"
        assert llm_meta["latency_ms"] == 1
        assert "ocr_confidence_low" in llm_meta["trigger_reasons"]
        assert isinstance(llm_meta["fields_from_llm"], list)
    finally:
        pipeline.extract_fields_with_llm = original_llm
        config.OCR_CONFIDENCE_VISION_THRESHOLD = original_threshold

    print("pipeline llm_policy tests OK")


def api_tests(image_path):
    from app import app

    spy = LlmSpy()
    original_llm = pipeline.extract_fields_with_llm
    original_allowed = set(config.API_LLM_MODES_ALLOWED)
    pipeline.extract_fields_with_llm = spy
    client = app.test_client()
    raw = open(image_path, "rb").read()
    uploads_before = set(glob.glob("uploads/*"))

    def post(**params):
        data = {"label_image": (io.BytesIO(raw), TEST_FILENAME)}
        data.update(params)
        return client.post("/api/scan", data=data, content_type="multipart/form-data")

    try:
        # default is off: 200, zero LLM calls, telemetry says off
        r = post()
        assert r.status_code == 200, (r.status_code, r.data[:200])
        assert spy.calls == [], spy.calls
        body = json.loads(r.data)
        assert body["metadata"]["llm"]["requested_mode"] == "off"
        assert body["metadata"]["llm"]["called"] is False

        # valid but not allowed by the kill switch: 400. Set the allowlist
        # explicitly — the config default is deployment-dependent (it was
        # widened for the demo) and this test is about the 400 behavior.
        config.API_LLM_MODES_ALLOWED = {"off"}
        r = post(llm="auto")
        assert r.status_code == 400 and b"not enabled" in r.data, (r.status_code, r.data)
        assert spy.calls == []

        # invalid value: 400
        r = post(llm="banana")
        assert r.status_code == 400 and b"must be one of" in r.data, (r.status_code, r.data)

        # allowed after widening the kill switch: 200 and the LLM runs
        config.API_LLM_MODES_ALLOWED = {"off", "auto"}
        r = post(llm="auto")
        assert r.status_code == 200, (r.status_code, r.data[:200])
        assert len(spy.calls) >= 1, spy.calls
        body = json.loads(r.data)
        assert body["metadata"]["llm"]["requested_mode"] == "auto"
        assert body["metadata"]["llm"]["called"] is True

        # force_vision still rejected (not in the widened allowlist)
        r = post(llm="force_vision")
        assert r.status_code == 400 and b"not enabled" in r.data
    finally:
        pipeline.extract_fields_with_llm = original_llm
        config.API_LLM_MODES_ALLOWED = original_allowed
        _cleanup_api_rows()
        for path in set(glob.glob("uploads/*")) - uploads_before:
            os.remove(path)

    print("api llm_policy tests OK")


def main():
    image_path = sorted(glob.glob("datasets/usps/images/*"))[0]
    pipeline_tests(image_path)
    api_tests(image_path)


if __name__ == "__main__":
    main()
