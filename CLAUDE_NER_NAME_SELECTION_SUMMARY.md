# NER recipient_name Selection — Design, Evidence, Rollout

**Branch:** `feature/ner-name-selection` · **Date:** 2026-07-08

NER candidates (shadow-only since the NER shadow sprint) may now win
selection for **recipient_name only**, behind a new default-off flag.
Every other field — street_address, city, state, zip_code,
tracking_number, carrier — is structurally untouched: `NerNamePolicy`
returns the legacy Selector's decision for any non-name field before NER
is even consulted. OCR, barcode, LLM, calibration, and address parsing
are unchanged.

## Why

`evaluate.py --latest-only` (253 labels, 2026-07-08): recipient_name is
the weakest field at **41.3%** (95/230). The calibration table splits
that by selected source:

| Selected source | n | Accuracy |
|---|---|---|
| llm | 65 | 95.4% |
| rule | 134 | 23.9% |
| agreement | 25 | 4.0% |
| blank | 6 | 0% |

The LLM is nearly always right when it runs; everything else is nearly
always wrong. The NER backtest (33 held-out real-scan rows) puts NER at
72.7% vs 51.5% selected. So the policy protects llm-sourced selections
and rescues the rest.

## The gate (`selection.NerNamePolicy`)

For recipient_name only, after computing the legacy decision:

1. No NER candidate, or legacy selection is **llm-sourced** → keep legacy.
2. NER value normalizes equal to the legacy value → keep legacy provenance.
3. NER value must pass `plausible_recipient_name` (no digits, 2+ tokens —
   targets NER's address-bleed failure mode; recorded as the
   `name_plausible` validation on every ner candidate for calibration).
4. Legacy value blank → fill at `NER_NAME_FILL_MIN_CONFIDENCE = 0.50`
   (reason `ner_name_fill`).
5. Legacy value non-empty → override at
   `NER_NAME_OVERRIDE_MIN_CONFIDENCE = 0.80` (reason `ner_name_override`).

## Threshold choice (ner_backtest.py sweep, n=33)

| override_thr | overrides | wins | losses | policy acc |
|---|---|---|---|---|
| 0.50 | 15 | 13 | 1 | 87.9% |
| 0.70 | 12 | 10 | 1 | 78.8% |
| **0.80** | **10** | **10** | **0** | **81.8%** |
| 0.90 | 8 | 9 | 0 | 78.8% |

**0.80 chosen as the zero-loss point**: +30.3pt over the 51.5% base with
no observed regression. 0.50 scores higher (87.9%) at the cost of one
regression — revisit once `eval_candidates.py` has fresh ner rows
(currently it has **zero**: all provenance-bearing annotations predate
NER enablement, so every number above is in-domain held-out data, n=33,
Wilson 95% CI on NER accuracy roughly [56%, 85%]).

## Expected improvement (evaluate.py space, n=230)

recipient_name: **41.3% → ~64–81%, central ≈ 73%.** All other fields
byte-identical (enforced by tests, not just expected). Per-source table
gains an `ner` row (~73–83%); the `llm` row must stay ~95%.

## Evaluation method

- **Before (done):** backtest sweep above; baselines recorded
  (evaluate.py 41.3%, backtest head-to-head 72.7 vs 51.5).
- **After enabling:** annotate newly scanned labels as usual, then
  `evaluate.py --latest-only` (expect the recipient_name climb and the new
  `ner` source row) and `eval_candidates.py` (ner `BeatSelected` should
  shrink toward 0 as wins are absorbed). Once `recipient_name|ner` has
  25+ annotated samples, re-run `fit_calibration.py` to give ner a
  calibrated bucket (confidence currently passes through as the raw span
  probability under `CONFIDENCE_MODE="legacy"`).
- **Rollback trigger:** post-enablement recipient_name accuracy below
  ~60% after 25+ fresh annotations, or any movement in non-name fields
  (which would indicate a bug, not a tuning problem).

## Enablement & rollback

Both env vars must be set for NER to influence anything:

```
NER_ENABLED=true                  # ner candidates exist (model loads)
NER_NAME_SELECTION_ENABLED=true   # NerNamePolicy may select them (name only)
```

Rollback, graduated:
1. Unset `NER_NAME_SELECTION_ENABLED` → exact shadow-mode behavior,
   candidates still persisted and measured. Immediate (flags are checked
   per pipeline run, no restart needed for code paths; env changes need a
   process restart as usual).
2. Unset `NER_ENABLED` → no NER inference at all, onnxruntime never loads.
3. `git revert` of the selection/pipeline commits; the eval commit
   (backtest sweep) is harmless and can stay.

## Tests

- `tests/test_selection_ner_name.py` — unit: flag matrix for
  `get_selector`, every gate branch, and a structural sweep proving a
  0.99-confidence NER candidate cannot move any non-name field.
- `tests/test_ner_name_selection.py` — end-to-end: real pipeline with
  stubbed NER predictions offering high-confidence values for EVERY
  field; asserts only recipient_name moves, payloads are byte-identical
  under every other flag/gate combination, `metadata.ner.fields_from_ner`
  telemetry, and the `recipient_name_source="ner"` storage round-trip.
- `tests/test_selection_parity.py` (existing golden) — still passes:
  default-flag output is byte-identical to the pre-NER pipeline.
- `tests/test_ner_shadow.py` (existing) — still passes: shadow mode with
  the selection flag off remains provably inert.
