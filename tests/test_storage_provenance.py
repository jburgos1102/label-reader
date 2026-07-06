"""Selection provenance must round-trip through storage.

Run from the project root:  venv/bin/python tests/test_storage_provenance.py
Uses a temporary database — the real label_storage.db is never touched.
"""

import glob
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["GROQ_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

import pipeline
import storage
from selection import selection_provenance


def stub_llm(text, rule_result, image=None):
    result = {field: "STUBVAL" for field in pipeline.EXTRACTION_FIELDS}
    result.update({
        "llm_enabled": True, "llm_provider": "stub", "llm_notes": "",
        "llm_model": "stub-model", "llm_latency_ms": 7,
    })
    return result


def main():
    image_path = sorted(glob.glob("datasets/usps/images/*"))[0]

    original_llm = pipeline.extract_fields_with_llm
    original_db = storage._DB_PATH
    pipeline.extract_fields_with_llm = stub_llm

    with tempfile.TemporaryDirectory() as tmp:
        storage._DB_PATH = Path(tmp) / "test.db"
        try:
            storage.init_db()

            internal = pipeline.run(image_path, llm_policy="auto")
            result = pipeline.build_extraction_result(internal, "prov-test")
            reasons, candidates = selection_provenance(internal["_selections"])
            telemetry = result.llm_telemetry

            storage.store(
                result,
                ocr_text=internal["_ocr_text"],
                parser_used=internal.get("parser_used", ""),
                selection_reasons=reasons,
                candidates=candidates,
                llm_requested_mode=telemetry["requested_mode"],
                llm_model=telemetry["model"],
                llm_latency_ms=telemetry["latency_ms"],
                llm_trigger_reasons=telemetry["trigger_reasons"],
            )

            row = storage.get_label("prov-test")
            assert row is not None

            # JSON columns parsed back into Python objects
            assert isinstance(row["selection_reasons"], dict)
            assert set(row["selection_reasons"]) == set(pipeline.EXTRACTION_FIELDS)
            assert all(r for r in row["selection_reasons"].values()), row["selection_reasons"]

            assert isinstance(row["candidates"], dict)
            city_cands = row["candidates"]["city"]
            assert {c["source"] for c in city_cands} == {"rule", "llm"}
            for cand in city_cands:
                assert set(cand) >= {"field", "value", "source", "confidence",
                                     "validations", "reason"}, cand
            tracking_rule = next(
                c for c in row["candidates"]["tracking_number"] if c["source"] == "rule"
            )
            assert "checksum_valid" in tracking_rule["validations"]

            assert row["llm_requested_mode"] == "auto"
            assert row["llm_model"] == "stub-model"
            assert row["llm_latency_ms"] == 7
            assert isinstance(row["llm_trigger_reasons"], list)

            # Rows stored WITHOUT provenance (legacy path) read back as None
            storage.store(result)
            legacy = storage.get_label("prov-test")  # same id, INSERT OR REPLACE
            assert legacy["selection_reasons"] is None
            assert legacy["candidates"] is None
        finally:
            storage._DB_PATH = original_db
            pipeline.extract_fields_with_llm = original_llm

    print("test_storage_provenance OK")


if __name__ == "__main__":
    main()
