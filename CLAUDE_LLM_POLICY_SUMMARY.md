# LLM Policy (Thin Slice) — Summary

Branch: `feature/llm-policy` (on top of `fix-strict-skip-llm`).
Scope: the thin-slice design only — no Candidate/Selector, no Mistral OCR,
no provider A/B, no UI provider selection, no storage schema changes.

## What was built

### 1. `llm_policy` — three explicit modes (`pipeline.py`)

```
run(image_path, skip_llm=False, llm_policy=None)
```

| Policy | Behavior |
|---|---|
| `off` | Strict: no LLM call of any kind (text, vision, or vision fallback). |
| `auto` | Historical default: text LLM always, vision when a trigger fires. |
| `force_vision` | Exactly one vision call, regardless of triggers. |

- `skip_llm=True` remains a **strict alias** for `llm_policy="off"` — the
  existing `tests/test_skip_llm.py` passes unchanged.
- Contradictions raise: `run(..., skip_llm=True, llm_policy="auto")` and
  `..."force_vision"` are `ValueError`s, as is any unknown policy string.
  Loud beats guessing.

### 2. `/api/scan` API shape (`app.py`)

```
POST /api/scan
  label_image: <file>
  llm: off | auto | force_vision     # optional; DEFAULT IS "off"
```

- Default is `off` — AI never runs on a camera scan unless explicitly
  requested. `/upload` (browser form) keeps its historical `auto` behavior.
- Unknown value → `400 "llm must be one of: off, auto, force_vision."`
- Valid but not enabled → `400 "llm mode 'X' is not enabled on this server."`
- Param validation happens **before** the upload is saved, so rejected
  requests write nothing to disk.

### 3. Kill switch (`config.py`)

`API_LLM_MODES_ALLOWED = {"off"}` — the server-side allowlist of modes
callers may request. Default is strict because `/api/scan` is currently
unauthenticated; widening it to `{"off", "auto", "force_vision"}` is a
deliberate operator action.

### 4. Camera detector consistency (`camera_label_detector.py`, `label_reader.py`)

The desktop camera tool previously called `run()` with the default —
meaning it had FULL LLM behavior while the web camera path was strict-off.
It now passes `llm_policy="off"` explicitly; `extract_label_data()` gained a
pass-through `llm_policy` parameter (default `None` → `auto`, so
`evaluate.py`/`regression_test.py` behavior is unchanged).

### 5. Telemetry — additive `metadata.llm` (`pipeline.py`, `models.py`)

The eight loose trigger booleans were replaced by a named list; every scan
(including `off`) records why vision would have fired:

```json
"metadata": {
  "llm_called": false,            // existing fields all unchanged
  "llm_mode": "none",
  "llm": {                        // NEW, additive
    "requested_mode": "off",
    "called": false,
    "mode": "none",
    "provider": "none",
    "model": "",
    "latency_ms": null,
    "trigger_reasons": ["ocr_confidence_low", "name_looks_like_org"],
    "fields_from_llm": []
  }
}
```

`fields_from_llm` = fields where selection chose the LLM value. It is a
*change* marker, not an *improvement* claim — improvement is computed offline
by joining with ground-truth annotations.

## Files changed

`pipeline.py` (policy resolution, trigger_reasons, telemetry), `app.py`
(param + 400s), `config.py` (`API_LLM_MODES_ALLOWED`), `models.py`
(`llm_telemetry` → `metadata.llm`), `label_reader.py` (pass-through),
`camera_label_detector.py` (explicit off), `tests/test_llm_policy.py` (new).

## Tests

- **`tests/test_llm_policy.py`** (new, passing):
  - `llm_policy="off"` → zero LLM calls with the vision trigger forced;
    telemetry still records `requested_mode`/`trigger_reasons`
  - `skip_llm=True` alias → zero calls
  - `force_vision` → exactly one vision call, with triggers firing AND with
    all triggers suppressed
  - contradictions + invalid policy → `ValueError`
  - `metadata.llm` populated; all pre-existing metadata keys untouched
  - API: default POST is `off` (200, zero calls); `llm=auto` under default
    config → 400 "not enabled"; `llm=banana` → 400 "must be one of";
    `llm=auto` after widening the allowlist → 200 with LLM called;
    `force_vision` still 400 when not allowlisted
  - self-cleaning: test rows, captures, and uploaded files are removed
- **`tests/test_skip_llm.py`** — passes unchanged (alias guarantee).
- **`regression_test.py`** — PASSED, identical to baseline (26 labels;
  recipient_name 58.3%, street 78.3%, city 80.8%, state 84.6%, zip 80.8%,
  tracking 73.1%, carrier 100.0%).
- **`evaluate.py`** — output identical to pre-change (197 labels, 86.8%
  overall, 161 dataset-sourced values excluded).
- `py_compile` clean across all touched modules.

## Risks

- `/upload` still defaults to `auto` (unchanged historical behavior) — the
  kill switch governs only `/api/scan`. Unifying `/upload` under the
  allowlist is a product decision, deferred.
- The kill switch is config-file-only; changing it requires a deploy/restart.
  Fine for now, revisit when config moves to env-based settings.
- `metadata.llm` is additive but clients that iterate metadata keys will see
  a new key — flagged, considered acceptable per "additive only".
- Telemetry is response-only; it is NOT yet persisted to SQLite (deliberately
  out of scope per the thin-slice instructions). Until the columns land,
  trigger-reason accuracy joins aren't possible offline — recommended as the
  next small follow-up (via the existing `_MIGRATED_COLUMNS` mechanism).

## Deferred (by design)

Storage persistence of `metadata.llm`; per-field improvement attribution;
provider A/B and per-request provider selection; Mistral OCR-4 (belongs in
the OCR-engine bake-off, not this abstraction); camera UI toggle.

## Next milestone

Week 2 Candidate/Selector — awaiting explicit approval, per instructions.
