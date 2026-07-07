"""Tests for the ONNX NER extractor: flag gating, graceful degradation,
and (when the model artifact is present) a real inference smoke test.

Run from the project root:  venv/bin/python tests/test_ner_extractor.py
"""

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["GROQ_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

SAMPLE_OCR = """SHIP TO:
JOHN SMITH
28 N COLLEGE ST
CARLISLE, PA 17013
USPS TRACKING # 9400 1000 0000 0000 0000 00
"""


def flag_off_never_imports_onnxruntime():
    # Fresh interpreter: with the flag off, importing and calling the module
    # must not pull in onnxruntime at all.
    code = (
        "import os, sys\n"
        "os.environ.pop('NER_ENABLED', None)\n"
        "import ner_extractor\n"
        "assert ner_extractor.get_extractor() is None\n"
        "assert ner_extractor.ner_field_predictions('SOME TEXT') == {}\n"
        "assert 'onnxruntime' not in sys.modules, 'onnxruntime imported with flag off'\n"
        "print('flag-off OK')\n"
    )
    env = dict(os.environ)
    env.pop("NER_ENABLED", None)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, result.stderr
    assert "flag-off OK" in result.stdout


def missing_model_degrades_gracefully():
    import config
    import ner_extractor

    original_enabled = config.NER_ENABLED
    original_dir = config.NER_MODEL_DIR
    config.NER_ENABLED = True
    config.NER_MODEL_DIR = "/nonexistent/model/dir"
    ner_extractor.reset_for_tests()
    try:
        assert ner_extractor.get_extractor() is None
        assert ner_extractor._state["failed"] is True  # latched
        assert ner_extractor.get_extractor() is None   # no retry storm
        assert ner_extractor.ner_field_predictions(SAMPLE_OCR) == {}
    finally:
        config.NER_ENABLED = original_enabled
        config.NER_MODEL_DIR = original_dir
        ner_extractor.reset_for_tests()
    print("missing-model degradation OK")


def inference_error_returns_empty():
    import config
    import ner_extractor

    class Boom:
        def predict_fields(self, text):
            raise RuntimeError("boom")

    original_enabled = config.NER_ENABLED
    config.NER_ENABLED = True
    ner_extractor.reset_for_tests()
    ner_extractor._state["extractor"] = Boom()
    try:
        assert ner_extractor.ner_field_predictions(SAMPLE_OCR) == {}
    finally:
        config.NER_ENABLED = original_enabled
        ner_extractor.reset_for_tests()
    print("inference-error path OK")


def real_model_smoke():
    import config
    import ner_extractor

    onnx_path = os.path.join(config.NER_MODEL_DIR, "label_reader.onnx")
    if not os.path.exists(onnx_path):
        print("real-model smoke SKIPPED (no model artifact)")
        return

    original_enabled = config.NER_ENABLED
    config.NER_ENABLED = True
    ner_extractor.reset_for_tests()
    try:
        extractor = ner_extractor.get_extractor()
        assert extractor is not None

        assert extractor.predict_fields("") == {}

        started = time.monotonic()
        predictions = extractor.predict_fields(SAMPLE_OCR)
        first_ms = (time.monotonic() - started) * 1000
        started = time.monotonic()
        extractor.predict_fields(SAMPLE_OCR)
        second_ms = (time.monotonic() - started) * 1000

        assert isinstance(predictions, dict)
        for field, prediction in predictions.items():
            assert set(prediction) == {"value", "confidence", "alternates"}, prediction
            assert prediction["value"], prediction
            assert 0.0 <= prediction["confidence"] <= 1.0
        print(f"real-model smoke OK: fields={sorted(predictions)} "
              f"latency first={first_ms:.0f}ms warm={second_ms:.0f}ms "
              f"version={extractor.version}")
    finally:
        config.NER_ENABLED = original_enabled
        ner_extractor.reset_for_tests()


def main():
    flag_off_never_imports_onnxruntime()
    missing_model_degrades_gracefully()
    inference_error_returns_empty()
    real_model_smoke()
    print("test_ner_extractor OK")


if __name__ == "__main__":
    main()
