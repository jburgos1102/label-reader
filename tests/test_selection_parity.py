"""Golden parity harness for the Candidate/Selector migration.

Captures the full public output (ExtractionResult.to_dict()) of the pipeline
over a carrier-mixed image set under three LLM configurations, and compares
against a committed golden file. The golden file was generated from the
pre-Selector pipeline, so a passing run proves the migration is
behavior-preserving for every branch the matrix exercises:

  - "off":          llm_policy="off"  (no LLM candidates at all)
  - "auto":         llm_policy="auto" with a deterministic LLM stub
                    (text mode; vision when the image's real triggers fire)
  - "force_vision": llm_policy="force_vision" with the same stub
                    (exercises the vision conflict-override branch)

The stub is a pure function of the rule result: recipient_name always
conflicts, blank rule fields get deterministic fill values (exercising the
blank-fill branch, including tracking), everything else agrees.

Run from the project root:
  venv/bin/python tests/test_selection_parity.py           # compare
  venv/bin/python tests/test_selection_parity.py --regen   # rewrite golden
"""

import copy
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["GROQ_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

import pipeline

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "golden", "selection_parity.json")

IMAGE_STEMS = [
    "datasets/usps/images/usps_001",
    "datasets/usps/images/usps_005",
    "datasets/usps/images/usps_010",
    "datasets/usps/images/usps_015",
    "datasets/usps/images/usps_021",
    "datasets/ups/images/ups_001",
    "datasets/ups/images/ups_002",
    "datasets/fedex/images/fedex_001",
    "datasets/speedx/images/speedx_001",
]

FILL_VALUES = {
    "recipient_name": "LLM FILLED NAME",
    "street_address": "123 LLM FILL ST",
    "city": "LLMVILLE",
    "state": "ZZ",
    "zip_code": "99999",
    "tracking_number": "1Z999AA10123456784",
    "carrier": "UPS",
}


def stub_llm(text, rule_result, image=None):
    """Deterministic LLM stand-in: pure function of the rule result."""
    result = {}
    for field_name in pipeline.EXTRACTION_FIELDS:
        rule_value = str((rule_result or {}).get(field_name) or "")
        if not rule_value:
            result[field_name] = FILL_VALUES[field_name]      # blank-fill branch
        elif field_name == "recipient_name":
            result[field_name] = "LLM CONFLICT NAME"          # conflict branch
        else:
            result[field_name] = rule_value                   # agreement branch
    result.update({
        "llm_enabled": True,
        "llm_provider": "stub",
        "llm_notes": "",
        "llm_model": "stub-model",
        "llm_latency_ms": 1,
    })
    return result


def normalize(payload):
    """Strip run-to-run volatility (timing) from a to_dict() payload."""
    payload = copy.deepcopy(payload)
    payload["metadata"]["processing_ms"] = 0
    return payload


def collect():
    results = {}
    original_llm = pipeline.extract_fields_with_llm
    try:
        for stem in IMAGE_STEMS:
            matches = sorted(glob.glob(stem + ".*"))
            assert matches, f"no image found for {stem}"
            image_path = matches[0]
            name = os.path.basename(stem)
            results[name] = {}

            for config_name, llm_policy, llm_fn in (
                ("off", "off", original_llm),
                ("auto", "auto", stub_llm),
                ("force_vision", "force_vision", stub_llm),
            ):
                pipeline.extract_fields_with_llm = llm_fn
                internal = pipeline.run(image_path, llm_policy=llm_policy)
                payload = pipeline.build_extraction_result(internal, "golden").to_dict()
                results[name][config_name] = normalize(payload)
    finally:
        pipeline.extract_fields_with_llm = original_llm
    return results


def main():
    actual = collect()

    if "--regen" in sys.argv:
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, "w") as f:
            json.dump(actual, f, indent=2, sort_keys=True)
        print(f"golden written: {GOLDEN_PATH} "
              f"({len(actual)} images x {len(next(iter(actual.values())))} configs)")
        return 0

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    failures = []
    for name in sorted(set(golden) | set(actual)):
        for config_name in sorted(set(golden.get(name, {})) | set(actual.get(name, {}))):
            expected = golden.get(name, {}).get(config_name)
            got = actual.get(name, {}).get(config_name)
            if expected != got:
                failures.append((name, config_name))
                print(f"MISMATCH: {name} [{config_name}]")
                print("  expected:", json.dumps(expected, sort_keys=True)[:400])
                print("  actual:  ", json.dumps(got, sort_keys=True)[:400])

    if failures:
        print(f"\nPARITY FAILED: {len(failures)} mismatching cases")
        return 1

    total = sum(len(v) for v in actual.values())
    print(f"PARITY OK: {total} cases ({len(actual)} images x 3 configs) match golden")
    return 0


if __name__ == "__main__":
    sys.exit(main())
