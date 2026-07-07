"""NER shadow-candidate integration: candidates appear and persist, and the
selected output is provably identical with the flag on vs off.

Run from the project root:  venv/bin/python tests/test_ner_shadow.py
Skips (loudly) when the model artifact is absent.
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


def normalized_payload(internal):
    payload = pipeline.build_extraction_result(internal, "shadow-test").to_dict()
    payload = copy.deepcopy(payload)
    payload["metadata"]["processing_ms"] = 0
    return payload


def main():
    if not os.path.exists(os.path.join(config.NER_MODEL_DIR, "label_reader.onnx")):
        print("test_ner_shadow SKIPPED (no model artifact)")
        return

    image_paths = sorted(glob.glob("datasets/usps/images/*"))[:2] + sorted(
        glob.glob("datasets/ups/images/*")
    )[:1]

    original_enabled = config.NER_ENABLED
    ner_seen_anywhere = False
    try:
        for image_path in image_paths:
            config.NER_ENABLED = False
            ner_extractor.reset_for_tests()
            internal_off = pipeline.run(image_path, llm_policy="off")
            payload_off = normalized_payload(internal_off)

            config.NER_ENABLED = True
            ner_extractor.reset_for_tests()
            internal_on = pipeline.run(image_path, llm_policy="off")
            payload_on = normalized_payload(internal_on)

            # 1. Public output byte-identical with the flag on
            assert payload_on == payload_off, (image_path, payload_on, payload_off)
            # 2. selected_result identical too (internal shape)
            assert internal_on["selected_result"] == internal_off["selected_result"]

            # 3. NER candidates ride along in the selections when predicted
            ner_fields = [
                field for field, s in internal_on["_selections"].items()
                if any(c.source == "ner" for c in s.candidates)
            ]
            if ner_fields:
                ner_seen_anywhere = True
                sample = next(
                    c for c in internal_on["_selections"][ner_fields[0]].candidates
                    if c.source == "ner"
                )
                assert sample.reason.startswith("ner:"), sample.reason
                assert 0.0 <= sample.confidence <= 1.0
                # 4. and none of them was selected (shadow means shadow)
                for field in ner_fields:
                    assert internal_on["_selections"][field].source != "ner"

            # 5. loser persistence: provenance -> storage round-trip
            reasons, candidates = selection_provenance(internal_on["_selections"])
            with tempfile.TemporaryDirectory() as tmp:
                original_db = storage._DB_PATH
                storage._DB_PATH = Path(tmp) / "t.db"
                try:
                    storage.init_db()
                    result = pipeline.build_extraction_result(internal_on, "shadow-test")
                    storage.store(result, selection_reasons=reasons, candidates=candidates)
                    row = storage.get_label("shadow-test")
                finally:
                    storage._DB_PATH = original_db
            stored_ner = [
                c for cands in (row["candidates"] or {}).values()
                for c in cands if c["source"] == "ner"
            ]
            assert len(stored_ner) == len(ner_fields), (len(stored_ner), ner_fields)
            for cand in stored_ner:
                assert set(cand) >= {"field", "value", "source", "confidence",
                                     "validations", "reason"}

            print(f"  {os.path.basename(image_path)}: output identical; "
                  f"ner candidates on {len(ner_fields)} fields, all persisted as losers")
    finally:
        config.NER_ENABLED = original_enabled
        ner_extractor.reset_for_tests()

    assert ner_seen_anywhere, "NER produced no candidates on any test image"
    print("test_ner_shadow OK")


if __name__ == "__main__":
    main()
