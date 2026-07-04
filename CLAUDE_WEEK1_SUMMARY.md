# Week 1 Correctness Fixes — Summary

Branch: `claude-week-1-fixes` (4 commits on top of `main`/c8e86fa).
Scope: correctness only — no selector engine, no restructure, no public JSON shape changes.

## Files changed

```
app.py           | 16 ++++++++++++-
barcodes.py      | 18 +++++++++------
evaluate.py      | 17 ++++++++++++++
llm_extractor.py | 70 ++++++++++++++++++++++++++++++++++++++++++++++++--------
ocr.py           | 22 +++++++++++-------
```

## What was fixed

### 1. Thread-safety (`barcodes.py`, `ocr.py`) — commit 1f147d8
`_last_raw_barcodes` and `_last_ocr_diagnostics` were module-level mutable globals,
so concurrent Flask requests could persist one label's barcode/OCR diagnostics onto
another label's row. Both now live in `threading.local()`. Function signatures and
single-request behavior are unchanged.
Verified: `py_compile` + a two-thread isolation test (worker thread cannot see the
main thread's scan state).

### 2. Evaluation contamination (`evaluate.py`) — commit 2ee673f
Rows imported by `ml/import_datasets.py` store ground truth in the *extracted*
columns (`source='dataset'`, confidence 1.0), so `evaluate.py` scored them 100%
correct by construction. `main()` now skips `source='dataset'` fields and prints the
exclusion count.
Measured effect on the current DB: **161 self-graded field scores across 23 labels
removed**. Honest numbers now (197 annotated labels): recipient_name 44.8%,
street_address 78.7%, city 93.7%, state 94.3%, zip 97.1%, tracking 98.9%,
carrier 100%, overall 86.8%.
`import_datasets.py` itself was deliberately left unchanged: its `tracking_number`
column keeps re-runs idempotent, and `export_training_data.py` only reads
`ocr_text` + `ground_truth`, so training is unaffected.

### 3. LLM logging / telemetry / client reuse (`llm_extractor.py`) — commit c4fee1c
- API failures and JSON-parse failures were silently swallowed into the fallback with
  one generic note; they are now logged (`log.exception`) with provider/model/mode/
  latency and get distinct `llm_notes` messages.
- Successful calls log provider/model/mode/latency at INFO.
- `llm_model` and `llm_latency_ms` are added to the internal `llm_result` dict only
  (not part of `ExtractionResult.to_dict()`, so the public JSON is unchanged).
- The OpenAI client is cached per provider config instead of constructed per call.
Verified: no-key stub path, fake-key 401 path (exception logged, graceful fallback).

### 4. Upload safety (`app.py`) — commit 6946503
- Uploads are stored as `uploads/<uuid>.<ext>` — two concurrent uploads with the same
  original filename previously overwrote each other mid-processing. The original
  filename is still recorded in the DB row.
- `MAX_CONTENT_LENGTH = 20 MB` + a 413 handler (JSON for `/api/*`, error template
  otherwise).
Verified with Flask test client: 413 on a 21 MB body; two same-named uploads landed
in two distinct files and both processed end-to-end (200). Test rows/files were
removed from the dev DB, `uploads/`, and `captures/` afterwards.

## What was NOT fixed (identified only, per instructions)

Dead code / dead config — left in place, deletions deferred:
- `config.LLM_CONFIDENCE_THRESHOLD` — referenced nowhere; the "conditional LLM" gate
  it implies was never built (text LLM still runs on every non-camera upload).
- `config.CONFIDENCE_ZIP_VALID` ("reserved"), `config.VISION_TRIGGER_NAME_LOOKS_LIKE_STREET`
  (flag consulted nowhere; pipeline uses the runtime flag directly).
- `config.CAMERA_*` block — `camera_label_detector.py` hardcodes 0.10 / 2.0 / 10 / 3
  instead of reading these.
- `evaluate.py`: `load_gold_set`, `print_gold_metrics`, `print_failure_analysis`,
  `print_ocr_failure_diagnostics`, `build_ocr_failure_diagnostic`, `expected_value_in_ocr`
  — never called by `main()` or `regression_test.py`.
- `openai_test.py` — one-off smoke script.
- `tracking.py` — `_validate_ups_checksum` and `is_valid_ups_check_digit` are duplicate
  implementations of the same algorithm (mathematically equivalent; consolidation is a
  small refactor, deferred).
- `address.py:754` — leftover `if "18974" in line:` debug branch; `pipeline.py` zip-first
  block defines an unused `separator_match`.

Other known issues intentionally out of scope this pass: vision trigger overriding
`skip_llm` on `/api/scan` (TODO: design decision — is vision allowed on the camera
path?); parser precedence in `parse_address_from_lines`; double Tesseract pass per
rotation; unauthenticated `/labels` endpoints.

## Risks

- **Thread-local diagnostics**: `get_last_raw_barcodes()` / `get_last_ocr_diagnostics()`
  now only see scans made on the *same thread*. All current callers (pipeline within a
  request; evaluate scripts, single-threaded) satisfy this. Any future executor-based
  parallelism inside one request would need diagnostics returned as values instead.
- **Evaluation numbers dropped** (that's the fix, not a regression): re-baseline any
  dashboards/notes that quoted the old inflated figures, especially recipient_name.
- **20 MB cap**: largest images seen in `uploads/` are well under this, but raise the
  constant if scanners produce bigger files.
- **Security note (not touched, per instructions):** `.env` contains live AWS access
  keys, Groq/OpenAI keys, and a Brynka API token in plaintext. It is gitignored and was
  never committed, but the AWS keys should be rotated and replaced with an AWS profile.

## Commands run

- `git status` / `git log` / `git diff --stat main...HEAD`
- `venv/bin/python -m py_compile *.py ml/*.py` — all modules compile
- Two-thread isolation test for `barcodes`/`ocr` thread-locals
- `venv/bin/python evaluate.py` — runs clean, exclusion reporting verified against real DB
- `llm_extractor` fallback tests (no key; invalid key → logged 401 + graceful fallback)
- Flask test client: `/` 200, 413 on oversized body, duplicate-filename uploads → unique files
- `venv/bin/python regression_test.py` — result recorded below

## Regression test result

`venv/bin/python regression_test.py` (LLM disabled, rule-based only, 26 labels):

```
REGRESSION TEST PASSED
recipient_name: 58.3% (14/24) baseline 54.2%
street_address: 78.3% (18/23) baseline 78.3%
city: 80.8% (21/26) baseline 80.8%
state: 84.6% (22/26) baseline 84.6%
zip_code: 80.8% (21/26) baseline 80.8%
tracking_number: 73.1% (19/26) baseline 73.1%
carrier: 100.0% (26/26) baseline 100.0%
```

All fields at or above baseline — extraction behavior preserved by these changes.

## Recommended next steps

1. Decide the `skip_llm` vs vision-trigger question (pipeline.py:161) — one-line fix
   once decided.
2. Delete the dead code/config listed above (trivial follow-up commit).
3. Persist `parser_used` / `llm_mode` / `llm_latency_ms` to SQLite (columns exist for
   neither) — needed for the Week 2 calibration work.
4. Single-pass OCR (`image_to_data` only) — biggest latency win, ~1 day.
5. Start Week 2: candidate/selector engine per docs/confidence-selection.md.
