# Confidence Calibration Sprint — Commits 0–3 Summary

Branch: `feature/confidence-calibration` (on top of `main`).
Scope: calibration foundation only. **Runtime confidence values and public
JSON are unchanged** — `CONFIDENCE_MODE` defaults to `"legacy"`, golden
parity stayed 27/27 through every commit, and the regression suite passed
identical to baseline. Commits 4+ (flipping the default, CalibratedSelector)
were explicitly out of scope and are NOT included.

## Commit story

| Commit | Content |
|---|---|
| `62e4305` | **Provenance persistence** — every stored scan now records `parser_used`, per-field selection reasons, the full candidate lists (winners *and losers*), and LLM telemetry (mode/model/latency/trigger_reasons), via the existing column-migration mechanism. Closes the "calibration data lost on every scan" gap and, over time, the selection-bias gap (candidate accuracy vs selected-value accuracy). |
| `897b6d7` | **Shared comparators** — `comparators.py` is the single definition of "correct" (moved verbatim; `evaluate.py` re-exports so `regression_test.py` is untouched). `evaluate.py` now scores with the fuzzy comparators, gated by `has_ground_truth` exactly like the regression suite, with the legacy strict metric printed alongside. |
| `c80f2de` | **Fit script + committed artifact + reliability report** — `fit_calibration.py` fits Laplace-smoothed empirical P(correct) per (field, source, validation-signature) bucket with hierarchical fallback (min n=25); `calibration/confidence_table.json` is committed so every confidence change is a reviewable diff; `evaluate.py` warns when annotations outgrow the fitted table by >25%. |
| (commit 3) | **ConfidenceModel seam** — `LegacyConfidence` (passthrough, default) and `CalibratedConfidence` (table lookup, per-lookup fallback to the legacy value) behind `config.CONFIDENCE_MODE`; candidate builders route through it. With the default, this is provably inert (goldens byte-identical). |

## Fitting methodology

- **Samples**: 941 scorable (label, field) pairs from annotated rows, deduped
  to the latest scan per tracking number; `source='dataset'` fields and
  empty-ground-truth fields excluded.
- **Correctness**: `comparators.compare_field` — the same definition the
  regression suite uses.
- **Estimator**: Laplace-smoothed `(correct+1)/(n+2)` per bucket, with
  hierarchical fallback `(field|source|validations)` → `(field|source)` →
  `(source)` → caller's legacy value. Buckets under n=25 are marked
  unusable and fall through. Every bucket carries a Wilson 95% interval.
- **Validation features** (recomputed from stored columns so historical rows
  participate): tracking checksum validity; LLM value found-in-OCR.

## Key findings (from calibration/confidence_table.json)

Reliability: **legacy stored confidence ECE = 0.211; calibrated table
ECE = 0.010** (10-bin, in-sample). The legacy constants are badly
miscalibrated in both directions:

| Bucket | n | Accuracy | Legacy avg conf | Calibrated p |
|---|---|---|---|---|
| tracking_number\|barcode | 131 | 100% | **0.44** | 0.993 |
| carrier\|agreement | 89 | 100% | **0.43** | 0.989 |
| recipient_name\|llm | 65 | 95.4% | 0.85 | 0.940 |
| recipient_name\|rule | 43 | 30.2% | **0.84** | 0.311 |
| recipient_name\|agreement | 25 | **4.0%** | **0.85** | 0.074 |
| street_address\|llm | 47 | 100% | 0.84 | 0.980 |
| street_address\|agreement | 41 | 63.4% | 0.69 | 0.628 |
| zip/city/state\|rule | 45 ea. | 91.1% | 0.78–0.95 | 0.894 |

The `recipient_name|agreement` result deserves emphasis: when the rule
engine and the LLM *agree* on a recipient name, they are wrong 96% of the
time in this dataset — agreement usually means the LLM echoed the rule's
garbled value. Meanwhile the legacy heuristic reports 0.85 for exactly
those fields. This is the strongest single argument that calibration was
the right sprint.

## Files changed

`storage.py`, `app.py`, `selection.py` (provenance + seam), `comparators.py`
(new), `evaluate.py`, `calibration.py` (new), `fit_calibration.py` (new),
`calibration/confidence_table.json` (new artifact), `config.py`
(`CONFIDENCE_MODE="legacy"`), tests: `test_storage_provenance.py`,
`test_calibration.py` (new), plus existing suites.

## Tests run (final state)

- `tests/test_calibration.py` — signature/keys/lookup-fallback/models/config
  seam, and legacy-mode passthrough parity — OK
- `tests/test_selection.py`, `test_storage_provenance.py`, `test_skip_llm.py`,
  `test_llm_policy.py` — OK
- `tests/test_selection_parity.py` — **27/27 byte-identical after every commit**
- `regression_test.py` — PASSED identical to baseline
- `evaluate.py` — runs clean; new fuzzy/strict dual reporting
  (e.g. street_address 78.7% fuzzy vs 78.1% strict on 169 scorable values)
- `py_compile` — clean

## Does the data support flipping to calibrated confidence?

**Yes for confidence *reporting*, with three caveats — and no for selection
*decisions* yet.**

Supporting evidence:
- Every high-volume bucket has n ≥ 25 with tight Wilson intervals; ECE
  improves 0.211 → 0.010; per-lookup fallback to legacy values means
  uncovered cases (e.g. `tracking|ocr`, `blank`) degrade gracefully.

Caveats to address in the flip branch:
1. **In-sample ECE.** Run a k-fold (or temporal split) ECE check before
   flipping; at n=941 the smoothed estimates should hold up, but verify.
2. **Temporal blend.** Annotated rows span many parser/prompt versions;
   P(correct|source) is a blend of old and new pipeline behavior
   (`recipient_name|agreement` at 4% may partly reflect older bugs).
   Re-fit after a batch of fresh scans and compare before trusting the most
   damning buckets.
3. **Golden regen + downstream review.** The flip changes reported
   confidence values (values/sources unchanged) — regenerate goldens with a
   reviewed diff and split the parity assertion (structure/values strict;
   confidence against new golden). Anyone consuming `confidence` downstream
   should be told the scale now means P(correct).

**Not supported yet: CalibratedSelector.** The fitted probabilities are
P(correct | *selected by the legacy policy*) — the accuracy of losing
candidates is unobserved in historical data. Commit 0 now persists full
candidate lists, so after enough fresh annotated scans accumulate, candidate-
level calibration (the input a re-ranking Selector actually needs) becomes
fittable. Until then, re-ranking from these numbers would extrapolate beyond
the data.

## Recommended follow-up order

1. Flip branch: k-fold ECE check → `CONFIDENCE_MODE="calibrated"` → golden
   regen with reviewed diff (values/sources must be unchanged).
2. Annotation push (fresh scans through the current pipeline) → re-fit →
   compare tables; only then evaluate a CalibratedSelector on the
   candidate-level data commit 0 is now collecting.
