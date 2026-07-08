"""End-to-end guarantee for NER recipient_name selection (NerNamePolicy).

Runs the real pipeline (OCR included) with ner_field_predictions stubbed to
offer a high-confidence, plausible NER value for EVERY field, then proves:

  1. Flags off (any combination short of both) -> public payload
     byte-identical to the no-NER baseline.
  2. Both flags on -> recipient_name is the NER value with source "ner",
     reason recorded in provenance, metadata.ner.fields_from_ner set — and
     every other extracted field byte-identical to the baseline even though
     NER offered 0.99-confidence candidates for all of them.
  3. Below-threshold / implausible NER values change nothing at all.
  4. Storage round-trip: recipient_name_source persists as "ner".

No model artifact required (predictions are stubbed).
Run from the project root:  venv/bin/python tests/test_ner_name_selection.py
"""

import copy
import glob
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["GROQ_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

import config
import ner_extractor
import pipeline
import storage
from selection import selection_provenance

NER_NAME = "Forced Ner Name"

ALL_FIELD_PREDICTIONS = {
    "recipient_name": {"value": NER_NAME, "confidence": 0.99, "alternates": 0},
    "street_address": {"value": "999 Ner Injection Way", "confidence": 0.99,
                       "alternates": 0},
    "city": {"value": "Nerville", "confidence": 0.99, "alternates": 0},
    "state": {"value": "NV", "confidence": 0.99, "alternates": 0},
    "zip_code": {"value": "89001", "confidence": 0.99, "alternates": 0},
    "tracking_number": {"value": "1Z999AA10123456784", "confidence": 0.99,
                        "alternates": 0},
    "carrier": {"value": "UPS", "confidence": 0.99, "alternates": 0},
}

OTHER_FIELDS = ("street_address", "city", "state", "zip_code",
                "tracking_number", "carrier")


def run_with(image_path, ner_enabled, name_enabled, predictions):
    config.NER_ENABLED = ner_enabled
    config.NER_NAME_SELECTION_ENABLED = name_enabled
    ner_extractor.ner_field_predictions = lambda text: (
        copy.deepcopy(predictions) if ner_enabled else {}
    )
    internal = pipeline.run(image_path, llm_policy="off")
    payload = pipeline.build_extraction_result(internal, "ner-name-test").to_dict()
    payload["metadata"]["processing_ms"] = 0
    return internal, payload


def main():
    images = sorted(glob.glob("datasets/usps/images/*"))[:1] + sorted(
        glob.glob("datasets/ups/images/*")
    )[:1]
    assert images, "no dataset images found"

    original_predict = ner_extractor.ner_field_predictions
    original_flags = (config.NER_ENABLED, config.NER_NAME_SELECTION_ENABLED)
    try:
        for image_path in images:
            name = os.path.basename(image_path)

            _, baseline = run_with(image_path, False, False, {})

            # 1. Shadow mode (selection flag off): byte-identical payload
            _, shadow = run_with(image_path, True, False, ALL_FIELD_PREDICTIONS)
            assert shadow == baseline, (name, "shadow changed output")

            # selection flag without candidates flag: also identical
            _, half = run_with(image_path, False, True, ALL_FIELD_PREDICTIONS)
            assert half == baseline, (name, "name flag alone changed output")

            # 2. Both flags on: recipient_name from NER, everything else frozen
            internal_on, active = run_with(
                image_path, True, True, ALL_FIELD_PREDICTIONS
            )
            extracted = active["extracted"]
            assert extracted["recipient_name"]["value"] == NER_NAME, extracted
            assert extracted["recipient_name"]["source"] == "ner", extracted
            assert extracted["recipient_name"]["confidence"] == 0.99, extracted
            for field in OTHER_FIELDS:
                assert extracted[field] == baseline["extracted"][field], (
                    name, field, extracted[field], baseline["extracted"][field])
            assert active["metadata"]["ner"] == {
                "fields_from_ner": ["recipient_name"]
            }, active["metadata"]
            reason = internal_on["_selections"]["recipient_name"].reason
            assert reason in ("ner_name_fill", "ner_name_override"), reason
            # non-name selections must not even carry an "ner" source
            for field in OTHER_FIELDS:
                assert internal_on["_selections"][field].source != "ner", field

            # 3a. Below-threshold NER: nothing changes (incl. no metadata.ner)
            weak = copy.deepcopy(ALL_FIELD_PREDICTIONS)
            weak["recipient_name"]["confidence"] = (
                config.NER_NAME_FILL_MIN_CONFIDENCE - 0.01
            )
            _, weak_payload = run_with(image_path, True, True, weak)
            assert weak_payload == baseline, (name, "weak NER changed output")

            # 3b. Implausible NER value (digits): nothing changes
            implausible = copy.deepcopy(ALL_FIELD_PREDICTIONS)
            implausible["recipient_name"]["value"] = "28 N College St"
            _, imp_payload = run_with(image_path, True, True, implausible)
            assert imp_payload == baseline, (name, "implausible NER changed output")

            # 4. Storage round-trip: source column persists as "ner"
            reasons, candidates = selection_provenance(internal_on["_selections"])
            with tempfile.TemporaryDirectory() as tmp:
                original_db = storage._DB_PATH
                storage._DB_PATH = Path(tmp) / "t.db"
                try:
                    storage.init_db()
                    result = pipeline.build_extraction_result(
                        internal_on, "ner-name-test"
                    )
                    storage.store(
                        result, selection_reasons=reasons, candidates=candidates
                    )
                    row = storage.get_label("ner-name-test")
                finally:
                    storage._DB_PATH = original_db
            assert row["recipient_name"] == NER_NAME, row["recipient_name"]
            assert row["recipient_name_source"] == "ner", row["recipient_name_source"]
            for field in OTHER_FIELDS:
                assert row[f"{field}_source"] != "ner", field

            print(f"  {name}: only recipient_name moved; "
                  f"{len(OTHER_FIELDS)} other fields byte-identical across "
                  f"4 flag/gate variants")
    finally:
        ner_extractor.ner_field_predictions = original_predict
        config.NER_ENABLED, config.NER_NAME_SELECTION_ENABLED = original_flags

    print("test_ner_name_selection OK")


if __name__ == "__main__":
    main()
