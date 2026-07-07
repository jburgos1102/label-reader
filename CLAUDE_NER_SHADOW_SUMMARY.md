# NER Shadow Candidate Source — Sprint Summary

Branch: `feature/ner-shadow-candidates` (on top of `main`, which now includes
the merged calibration sprint). Scope: NER as a **measured shadow source
only** — it emits candidates, they are persisted, and the Selector ignores
them. Selected output is provably unchanged.

## Commit story

| Commit | Content |
|---|---|
| `e24e332` | (on main, pre-branch) test fix: stop assuming the config-default LLM allowlist |
| `7f29290` | Shared word tokenizer + BIO span decoder (`ner_decode.py`); training export now imports it — train/serve tokenization cannot drift |
| `50887c6` | ONNX Runtime extractor (`ner_extractor.py`): lazy singleton, graceful degradation, `NER_ENABLED` flag |
| `7442243` | Pipeline integration: `ner_candidates()` shadow source; parity proven flag-on AND flag-off |
| `b1922d8` | `eval_candidates.py`: per-source candidate accuracy from persisted provenance |
| `9a0d6f9` | `ner_backtest.py`: held-out comparator-space accuracy + results |

## How it works

- `NER_ENABLED` (env var, default **off**). Off = `onnxruntime` never
  imported (asserted in a fresh interpreter). On = DistilBERT ONNX inference
  per scan; predictions become `Candidate(source="ner")` with the
  uncalibrated span probability as confidence, format/checksum validations,
  and the model version (`ner:distilbert@2026-07-07`) in `reason`.
- The legacy Selector **ignores unknown sources by design**, so NER cannot
  influence selection. All NER candidates persist as losers in the
  `candidates` provenance column for measurement.
- **Missing/corrupt model**: one warning, source latches off for the process,
  scans continue normally. Per-scan inference errors return no candidates.
- **ONNX over PyTorch**: no torch/transformers in the serving path.
  Serving deps added: `onnxruntime==1.27.0`, `tokenizers==0.22.2`.

## Measured performance (this machine)

Session load 0.13–0.42s (lazy, once per process); warm inference ~9ms per
scan (~12–22ms first call after load). Against 1–4s of Tesseract per scan:
negligible.

## Held-out backtest (the go/no-go data)

39 held-out rows (seed-42 split, unseen by the model), scored with the
shared comparators. Right block = head-to-head on the 33 rows from real
scans (dataset-imported rows excluded — their stored values are ground truth):

| Field | n | Coverage | NER acc | real-scan n | NER acc | Selected acc |
|---|---|---|---|---|---|---|
| recipient_name | 38 | 89% | 65.8% | 33 | **72.7%** | **51.5%** |
| street_address | 37 | 86% | 51.4% | 33 | 59.4% | 87.5% |
| city | 39 | 74% | 66.7% | 33 | 75.8% | 100.0% |
| state | 39 | 95% | 92.3% | 33 | 97.0% | 100.0% |
| zip_code | 39 | 92% | 92.3% | 33 | 100.0% | 100.0% |
| tracking_number | 39 | 0% | 0.0% | 33 | 0.0% | 100.0% |
| carrier | 39 | 13% | 12.8% | 33 | 9.1% | 100.0% |

**Reading:** NER decisively beats the pipeline on **recipient_name**
(+21 points on the project's weakest field) and loses everywhere else it
matters; it essentially never emits tracking spans (barcode owns tracking
anyway) and rarely emits carrier. The first Selector-policy proposal writes
itself: *NER for recipient_name only*, gated on fresh shadow data
confirming the backtest.

Caveats: n=33 is directional; held-out rows are in-domain (same
carriers/label styles as training); entity-F1 ≠ these numbers (street F1
0.48 → 59% comparator accuracy, name F1 0.64 → 73% — conversion is not 1:1
in either direction).

## Evidence pipeline going forward

1. Enable shadow collection where scans happen: `NER_ENABLED=true`.
2. Annotate scans as usual (`/labels/<id>/correct`).
3. `venv/bin/python eval_candidates.py` — per-source coverage, accuracy,
   and beat-selected counts from the persisted candidates (out-of-sample by
   construction). Currently reports 0 provenance-bearing annotated rows;
   fills in as new scans get annotated.
4. When the recipient_name shadow numbers confirm the backtest, propose the
   Selector policy change (separate approval, per standing instructions).

## Verification (all green)

- `test_ner_decode`, `test_ner_extractor` (flag-off import isolation,
  missing-model latch, inference-error path, real-model smoke),
  `test_ner_shadow` (per-image payload identical flag-on vs flag-off; no
  selection ever has source="ner"; losers round-trip through storage),
  `test_selection`, `test_calibration`, `test_storage_provenance`,
  `test_skip_llm`, `test_llm_policy` — all pass.
- Golden parity 27/27 byte-identical with `NER_ENABLED=true` **and** false.
- `regression_test.py` PASSED with flag off and on — outputs identical.
- `evaluate.py` runs clean; `py_compile` clean across the repo.

## Notes / follow-ups (out of scope, not started)

- **Calibration table is stale**: `evaluate.py` now warns — fitted on 158
  labels, DB has 253 deduped (annotation has continued). Re-run
  `fit_calibration.py` when convenient; NER buckets will appear in the
  table once provenance-bearing annotated rows exist.
- Model artifact distribution (`ml/models/` is gitignored) is a deploy
  concern; graceful degradation covers hosts without the file.
- int8 quantization (~67MB, faster CPU) once NER earns a selection role.
- CoreML untouched, per instructions.
