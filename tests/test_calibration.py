"""Unit tests for calibration helpers and the ConfidenceModel seam.

Run from the project root:  venv/bin/python tests/test_calibration.py
Plain asserts, no images/OCR required.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from calibration import (
    CalibratedConfidence,
    LegacyConfidence,
    bucket_keys,
    get_confidence_model,
    lookup,
    reset_confidence_model,
    validation_signature,
)
from selection import llm_candidates, rule_candidates

SYNTHETIC_TABLE = {
    "metadata": {"min_bucket_n": 25},
    "buckets": {
        "tracking_number|rule|checksum_valid=True": {"n": 50, "correct": 49, "p": 0.96},
        "tracking_number|rule": {"n": 80, "correct": 60, "p": 0.75},
        "city|llm": {"n": 30, "correct": 27, "p": 0.90},
        "city|llm|found_in_ocr=True": {"n": 10, "correct": 10, "p": 0.92},  # thin
        "llm": {"n": 120, "correct": 100, "p": 0.83},
        "rule": {"n": 200, "correct": 160, "p": 0.80},
    },
}


def main():
    # --- signature / keys ----------------------------------------------------
    assert validation_signature({}) == ""
    assert validation_signature(None) == ""
    assert validation_signature({"b": True, "a": False}) == "a=False,b=True"
    assert validation_signature({"x": None, "y": True}) == "y=True"  # None omitted
    assert validation_signature({"x": "notabool"}) == ""
    assert bucket_keys("city", "rule", "a=True") == [
        "city|rule|a=True", "city|rule", "rule"]
    assert bucket_keys("city", "rule") == ["city|rule", "rule"]

    # --- lookup fallback chain -------------------------------------------------
    p, key = lookup(SYNTHETIC_TABLE, "tracking_number", "rule", {"checksum_valid": True})
    assert (p, key) == (0.96, "tracking_number|rule|checksum_valid=True")
    p, key = lookup(SYNTHETIC_TABLE, "tracking_number", "rule", {"checksum_valid": False})
    assert (p, key) == (0.75, "tracking_number|rule")  # no False bucket -> level 2
    p, key = lookup(SYNTHETIC_TABLE, "city", "llm", {"found_in_ocr": True})
    assert (p, key) == (0.90, "city|llm")  # thin signature bucket skipped
    p, key = lookup(SYNTHETIC_TABLE, "state", "llm")
    assert (p, key) == (0.83, "llm")  # falls to source level
    p, key = lookup(SYNTHETIC_TABLE, "state", "barcode")
    assert (p, key) == (None, None)  # no bucket anywhere
    assert lookup(None, "city", "rule") == (None, None)  # no table

    # --- models ----------------------------------------------------------------
    legacy = LegacyConfidence()
    assert legacy.candidate_confidence("city", "rule", 0.85, {}) == 0.85

    calibrated = CalibratedConfidence(table=SYNTHETIC_TABLE)
    assert calibrated.candidate_confidence(
        "tracking_number", "rule", 0.4, {"checksum_valid": True}) == 0.96
    assert calibrated.candidate_confidence("state", "barcode", 0.44, {}) == 0.44  # falls back to base
    # Missing artifact: model degrades to passthrough
    no_artifact = CalibratedConfidence(path="/nonexistent/confidence_table.json")
    assert no_artifact.candidate_confidence("city", "rule", 0.85) == 0.85
    # table=None with no path loads the committed artifact (real table)
    real = CalibratedConfidence(table=None)
    real_city_rule = real.candidate_confidence("city", "rule", 0.85)
    assert 0.0 < real_city_rule < 1.0 and real_city_rule != 0.85, real_city_rule

    # --- get_confidence_model respects config ----------------------------------
    original_mode = config.CONFIDENCE_MODE
    try:
        reset_confidence_model()
        config.CONFIDENCE_MODE = "legacy"
        assert isinstance(get_confidence_model(), LegacyConfidence)
        config.CONFIDENCE_MODE = "calibrated"
        reset_confidence_model()
        assert isinstance(get_confidence_model(), CalibratedConfidence)
        config.CONFIDENCE_MODE = "bogus"
        reset_confidence_model()
        try:
            get_confidence_model()
        except ValueError:
            pass
        else:
            raise AssertionError("bogus mode should raise")
    finally:
        config.CONFIDENCE_MODE = original_mode
        reset_confidence_model()

    # --- builders under legacy mode are pure passthrough (byte parity) ---------
    assert config.CONFIDENCE_MODE == "legacy"
    fields = ("city", "tracking_number")
    rc = rule_candidates(
        {"city": "Troy", "tracking_number": "1Z999", "parser_used": "p"},
        fields, {"city": 0.85, "tracking_number": 0.4},
        tracking_checksum_valid=True,
    )
    assert rc["city"].confidence == 0.85
    assert rc["tracking_number"].confidence == 0.4
    lc = llm_candidates(
        {"llm_enabled": True, "llm_provider": "x", "city": "TROY", "tracking_number": ""},
        fields, {"city": 0.85, "tracking_number": 0.0},
    )
    assert lc["city"].confidence == 0.85
    assert lc["tracking_number"].confidence == 0.0

    print("test_calibration OK")


if __name__ == "__main__":
    main()
