# Week 2 — Candidate / Selector Architecture

Branch: `feature/candidate-selector` (on top of `feature/llm-policy`).
Scope: architecture only. No new sources, no scoring/calibration changes,
no accuracy changes — by design and by proof (below).

## What was built

Extraction sources now produce `Candidate` objects and a `Selector`
arbitrates one `Selection` per field. Full description + mermaid diagram:
`docs/architecture-candidate-selector.md`.

```
rule engine ──> rule_candidates() ──┐
LLM (policy-gated) ─> llm_candidates() ─┤──> Selector.select(field, candidates, ctx)
future NER / OCR engines / providers ──┘         │
                                          Selection(value, source, confidence,
                                                    reason, candidates)
                                                  │
                                    build_extraction_result → FieldValue → JSON
```

- **`Candidate`** — field, value, source, confidence, `validations`
  (checksum result on the rule tracking candidate; found-in-OCR on LLM
  candidates), optional reason. Empty values are legitimate candidates.
- **`Selection`** — the decision: value, legacy source label, confidence, a
  human-readable reason (`rule_blank_llm_fill`, `vision_conflict_override`,
  `rule_llm_agree`, `rule_default`), and the full candidate list.
- **`SelectionContext`** — per-scan signals (currently `llm_mode`).
- **`Selector`** — legacy-parity policy: rule-blank → LLM fill; vision-mode
  conflict → LLM override; otherwise rule wins with `agreement` when
  normalized values match. Unknown sources are deliberately ignored so a new
  source cannot change behavior before its policy exists.

Design goals met: a future NER model / OCR engine / LLM provider is a new
candidate builder plus one append in `pipeline.run()` — the Selector,
`build_extraction_result`, storage, and JSON are all source-agnostic. New
validation rules land in `Candidate.validations`.

## Commit story

| Commit | Stage |
|---|---|
| `48aed6e` | `selection.py` (Candidate/Selection/SelectionContext/Selector + builders) + 14 unit-test scenario groups. No pipeline changes. |
| `751f142` | Golden parity harness: 27 cases (9 carrier-mixed images × off / stubbed-auto / stubbed-force_vision) captured from the **pre-Selector** pipeline. |
| `760b650` | `run()` routes selection through the Selector; inline loop deleted; `_selections`/`_llm_scores` ride the internal dict. Parity 27/27, regression identical. |
| `12d1ae7` | `build_extraction_result` consumes Selections; vestigial value or-logic and duplicate `score_llm_result` call removed; legacy special cases (carrier/tracking/blank) kept verbatim and documented. Parity 27/27, regression identical. |
| (this)   | Architecture doc + summary. |

## Proof of behavior preservation

- **Golden parity**: `tests/test_selection_parity.py` — 27 full
  `ExtractionResult.to_dict()` payloads captured before the migration; every
  stage kept all 27 byte-identical. The deterministic LLM stub forces the
  blank-fill (including tracking), conflict, vision-override, agreement, and
  no-LLM branches; the goldens contain every selection source label
  (`rule`, `llm`, `agreement`, `blank`, `barcode`, `ocr`).
- **Regression suite**: PASSED after each stage, identical to baseline
  (recipient 58.3%, street 78.3%, city 80.8%, state 84.6%, zip 80.8%,
  tracking 73.1%, carrier 100.0%).
- **Existing suites**: `test_skip_llm`, `test_llm_policy` pass unchanged —
  llm_policy gating and telemetry are untouched.
- **Public JSON**: unchanged (guaranteed by the goldens, which are full
  payloads including `metadata.llm`).
- `evaluate.py` output unchanged (DB-based; result below under Tests run).

## Files changed

- `selection.py` — new (the architecture).
- `pipeline.py` — selection loop → Selector; `_fv` reads Selections.
- `tests/test_selection.py` — new (policy unit tests).
- `tests/test_selection_parity.py`, `tests/golden/selection_parity.json` — new.
- `docs/architecture-candidate-selector.md` — new (diagram + extension contract).

## Deliberately preserved legacy quirks (now documented inline)

These are behavior-parity decisions, not endorsements — they are the first
targets for the calibrated-confidence sprint:

1. Tracking keeps rule/checksum confidence even when the value was LLM-filled.
2. Carrier inherits tracking confidence (0.0 when carrier came from OCR
   context without a tracking number).
3. Both-empty rule+LLM values are labeled `agreement`.
4. `vision_fallback` mode does not get the vision conflict override.
5. LLM confidence is the uncalibrated 0.85/0.75 found-in-OCR constant.

## Risks

- The golden file encodes today's behavior including OCR output; a Tesseract
  version change would fail parity for environmental (not code) reasons —
  regenerate with `--regen` and review the diff if that happens.
- `Selection.reason` is internal-only (debug logs + objects); it is not yet
  persisted or exposed, keeping JSON unchanged. Persisting reasons is part of
  the telemetry follow-up.
- `pipeline._selector` is a module-level default; swapping policies is a
  one-line change but not yet configurable.

## Tests run (final state)

- `tests/test_selection.py` — OK (14 scenario groups)
- `tests/test_selection_parity.py` — PARITY OK 27/27
- `tests/test_skip_llm.py`, `tests/test_llm_policy.py` — OK
- `regression_test.py` — PASSED, identical to baseline
- `evaluate.py` — unchanged (197 labels, 86.8% overall, 161 dataset-sourced
  values excluded)
- `py_compile` — clean

## What this unblocks (awaiting approval — NOT started)

1. **Calibrated confidence** — fit `P(correct | source, validations)` from
   the annotated DB; swap the legacy policy for a confidence-ranked one.
2. **NER integration** — `ner_candidates()` from the DistilBERT model.
3. **OCR engine bake-off** (PaddleOCR / Textract / Mistral OCR) — candidates
   from a second engine.
4. **Additional LLM providers** — provider identity already travels on the
   candidate.
