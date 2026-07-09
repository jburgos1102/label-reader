"""Tests for eval_candidates.py: the per-model-version NER breakdown and its
"scans where NER emitted anything" denominator, plus the anchored
NER_MODEL_DIR (loading must not depend on the process working directory).

Run from the project root:  venv/bin/python tests/test_eval_candidates.py
"""

import contextlib
import io
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["GROQ_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

OLD_VERSION = "ner:distilbert@2026-07-07"
NEW_VERSION = "ner:distilbert@2026-07-08"


def _candidate(field, value, source, reason=""):
    return {"field": field, "value": value, "source": source,
            "confidence": 0.9, "validations": {}, "reason": reason}


def _fixture_labels():
    # Scan A: old model, NER wrong, selected (rule) value correct.
    scan_a = {
        "tracking_number": "TRACK-A",
        "recipient_name": "JOHN SMITH",
        "recipient_name_source": "rule_based",
        "ground_truth": {"recipient_name": "JOHN SMITH"},
        "candidates": {
            "recipient_name": [
                _candidate("recipient_name", "JOHN SMITH", "rule"),
                _candidate("recipient_name", "MAIN ST", "ner", OLD_VERSION),
            ],
        },
    }
    # Scan B: new model, NER correct where the selected value was wrong
    # (BeatSelected).
    scan_b = {
        "tracking_number": "TRACK-B",
        "recipient_name": "WRONG VALUE",
        "recipient_name_source": "rule_based",
        "ground_truth": {"recipient_name": "JANE DOE"},
        "candidates": {
            "recipient_name": [
                _candidate("recipient_name", "WRONG VALUE", "rule"),
                _candidate("recipient_name", "JANE DOE", "ner", NEW_VERSION),
            ],
        },
    }
    # Scan C: NER emitted nothing (disabled / model missing / nothing found).
    scan_c = {
        "tracking_number": "TRACK-C",
        "recipient_name": "ANA LIMA",
        "recipient_name_source": "rule_based",
        "ground_truth": {"recipient_name": "ANA LIMA"},
        "candidates": {
            "recipient_name": [
                _candidate("recipient_name", "ANA LIMA", "rule"),
            ],
        },
    }
    return [scan_a, scan_b, scan_c]


def _run_report(labels):
    import eval_candidates
    import storage

    original = storage.get_annotated_labels
    storage.get_annotated_labels = lambda: labels
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            eval_candidates.main()
    finally:
        storage.get_annotated_labels = original
    return output.getvalue()


def _table_row(output, *cells):
    """Return the first table line containing every cell, split into columns."""
    for line in output.splitlines():
        columns = line.split()
        if all(cell in columns for cell in cells):
            return columns
    raise AssertionError(f"no row with {cells!r} in:\n{output}")


def ner_breakdown_by_version():
    output = _run_report(_fixture_labels())

    assert "2 of 3 scans have NER candidates" in output, output
    assert "1 have" in output and "none" in output, output

    # Old model: 1 scan, 1 name candidate, wrong.
    row = _table_row(output, OLD_VERSION, "recipient_name")
    assert row[2:] == ["1", "1", "100%", "0.0%", "0"], row

    # New model: 1 scan, 1 name candidate, correct, beat the selected value.
    row = _table_row(output, NEW_VERSION, "recipient_name")
    assert row[2:] == ["1", "1", "100%", "100.0%", "1"], row

    # Aggregate table unchanged: ner n=2 over 3 scorable fields.
    row = _table_row(output, "ner", "recipient_name")
    assert row[2:] == ["2", "67%", "50.0%", "1"], row
    row = _table_row(output, "rule", "recipient_name")
    assert row[2:] == ["3", "100%", "66.7%", "0"], row
    print("ner-breakdown-by-version OK")


def no_ner_scans_still_reported():
    labels = [label for label in _fixture_labels()
              if label["tracking_number"] == "TRACK-C"]
    output = _run_report(labels)
    assert "0 of 1 scans have NER candidates" in output, output
    assert OLD_VERSION not in output and NEW_VERSION not in output, output
    print("no-ner-scans OK")


def missing_reason_falls_back_to_ner():
    labels = _fixture_labels()[:1]
    labels[0]["candidates"]["recipient_name"][1]["reason"] = ""
    output = _run_report(labels)
    row = _table_row(output, "ner", "recipient_name", "0.0%")
    assert row[0] == "ner", row  # bare fallback version, not a crash
    print("missing-reason fallback OK")


def model_dir_is_anchored():
    import config

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.isabs(config.NER_MODEL_DIR), config.NER_MODEL_DIR
    assert config.NER_MODEL_DIR == os.path.join(project_root, "ml", "models")

    # The anchor must hold from any working directory: import config from /
    # and check the path resolves identically.
    code = (
        "import sys\n"
        f"sys.path.insert(0, {project_root!r})\n"
        "import config\n"
        "print(config.NER_MODEL_DIR)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd="/",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == config.NER_MODEL_DIR, result.stdout
    print("model-dir anchoring OK")


def main():
    ner_breakdown_by_version()
    no_ner_scans_still_reported()
    missing_reason_falls_back_to_ner()
    model_dir_is_anchored()
    print("test_eval_candidates OK")


if __name__ == "__main__":
    main()
