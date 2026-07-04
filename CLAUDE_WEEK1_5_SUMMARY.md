# Week 1.5 — Strict skip_llm Fix + Cleanup Summary

Branch: `fix-strict-skip-llm` (on top of the Week 1 commits — see
`CLAUDE_WEEK1_SUMMARY.md`). Scope: one correctness fix + cleanup only.
No Candidate/Selector work, no folder moves, no new dependencies, no public
JSON changes, no extraction-strategy changes.

## Commits created

| Commit | Change |
|---|---|
| `6bedc99` | fix: make skip_llm authoritative — no vision LLM on camera scans (+ `tests/test_skip_llm.py`) |
| `8452f0c` | chore: remove dead config constants, wire camera detector to config |
| `5346947` | refactor: consolidate duplicate UPS check-digit implementations |
| `00a3159` | chore: remove dead helpers and one-off script |
| `9e63586` | refactor: single fallback_result builder for LLM-skipped/failed results |
| (final)  | docs: carrier-confidence comment + this summary |

## Files changed

- `pipeline.py` — skip_llm checked before the vision trigger (the secondary
  vision fallback already sat inside the non-skip branch, so one reorder covers
  every LLM entry point); `_skip_stub` now delegates to
  `llm_extractor.fallback_result`; comment on the carrier-confidence coupling.
- `tests/test_skip_llm.py` — new. Stubs the LLM function, forces the vision
  trigger (threshold 101 > any Tesseract confidence), asserts zero LLM calls
  with `skip_llm=True` and exactly one vision call with `skip_llm=False`.
- `config.py` — removed `LLM_CONFIDENCE_THRESHOLD`, `CONFIDENCE_ZIP_VALID`,
  `VISION_TRIGGER_NAME_LOOKS_LIKE_STREET`.
- `camera_label_detector.py` — now reads `CAMERA_MIN_AREA_RATIO`,
  `CAMERA_STABLE_SECONDS`, `CAMERA_HISTORY_FRAMES`, `CAMERA_HISTORY_THRESHOLD`
  from config (previously hardcoded the identical values — no behavior change).
- `tracking.py` — `_validate_ups_checksum` delegates to
  `is_valid_ups_check_digit`; duplicate check-digit math removed.
- `evaluate.py` — dead gold-set/OCR-diagnostics block removed (~170 lines).
- `address.py` — `18974` debug leftover and unused `separator_match` removed
  (TODO left: the ZIP+4 separator token is not actually validated; fixing that
  would change parser behavior, deferred).
- `llm_extractor.py` — `_safe_result` made public as `fallback_result`.
- `openai_test.py` — deleted.

## Dead-code justifications (why dead / how verified / why safe)

Every deletion was verified with a repo-wide grep (`--include='*.py'`,
excluding `venv/`) plus an import/smoke run; there is no dynamic
`getattr(config, ...)` or reflection anywhere in the project.

1. **`config.LLM_CONFIDENCE_THRESHOLD`** — only occurrence was its definition.
   The conditional-LLM gate it describes was never implemented, so the constant
   actively misrepresented system behavior. Safe: nothing imports it.
2. **`config.CONFIDENCE_ZIP_VALID`** — definition only; its own comment says
   "reserved (future)". Safe: trivially re-added when barcode-sourced ZIP exists.
3. **`config.VISION_TRIGGER_NAME_LOOKS_LIKE_STREET`** — definition only;
   `pipeline.py` consults the runtime `_name_looks_like_street` flag directly,
   so this toggle never toggled anything.
4. **`CAMERA_*` constants** — were defined-only, but they document real tuning
   knobs, so instead of deleting them the camera detector was wired to read them
   (it hardcoded the identical values 0.10 / 2.0 / 10 / 3 → no behavior change).
5. **UPS check-digit duplicate** — `_validate_ups_checksum` and
   `is_valid_ups_check_digit` implement the same algorithm; the only difference
   (letter value `ord-63` vs `(ord-63) % 10`) is congruent mod 10, so results
   are identical. **Verified empirically: 0 mismatches over 200,000 random
   1Z numbers** before consolidating. The `None`-for-wrong-format contract
   (means "not checksum-validated", no confidence penalty) is preserved.
6. **`evaluate.py` gold-set/OCR-diagnostics block** — `load_gold_set`,
   `print_gold_metrics`, `print_failure_analysis`, `expected_value_in_ocr`,
   `build_ocr_failure_diagnostic`, `print_ocr_failure_diagnostics`,
   `normalize_image_path`, `GOLD_SET_PATH`, `GOLD_FIELD_LABELS`,
   `OCR_DIAGNOSTIC_FIELDS`, the `OCR_DIAGNOSTICS` env flag, and the
   `get_last_ocr_diagnostics` / `build_extraction_result` imports form a closed
   subgraph referenced only by each other. `main()` never calls into it, and
   `regression_test.py` imports only `DATASETS_DIR`, `FIELDS_TO_COMPARE`,
   `compare_field`, `extract_label_data`, `find_image_for_expected`,
   `has_ground_truth`, `load_expected_json` — all untouched. Safe: verified by
   grep and by importing `regression_test` + running `evaluate.py` after
   deletion. `datasets/gold_set.txt` (data file) was kept.
7. **`openai_test.py`** — standalone smoke script, imported/referenced nowhere.
8. **`address.py` leftovers** — the `if "18974" in line:` branch only emitted a
   debug log for one hardcoded ZIP; `separator_match` was assigned and never
   read (`re.fullmatch` is pure — removing the assignment cannot change
   behavior). A TODO records that the separator token was never actually
   validated, since *adding* that validation would change parser behavior.
9. **`_skip_stub` duplication** — rebuilt `llm_extractor._safe_result`'s shape
   field-for-field. Now delegates to the (renamed, public) `fallback_result`.
   The only textual difference — `.strip()` / `None`-coercion — is a no-op on
   the already-normalized pipeline values.

## Tests run

- `tests/test_skip_llm.py` — passes (zero LLM calls with `skip_llm=True` even
  with the vision trigger forced; vision call still made with `skip_llm=False`).
- `regression_test.py` — **PASSED** after the skip_llm fix and again after all
  cleanup, identical to baseline: recipient_name 58.3%, street_address 78.3%,
  city 80.8%, state 84.6%, zip 80.8%, tracking 73.1%, carrier 100.0%.
- `evaluate.py` — runs clean; output identical before/after cleanup
  (197 annotated labels, 161 dataset-sourced field values excluded,
  overall 86.8%).
- `python -m py_compile` over every module in the repo (root, `ml/`, `tests/`)
  — all compile.
- UPS checksum equivalence: 200k randomized inputs pre-change, 100k
  post-change, plus format-edge cases (`None` vs `False` contract).

## Risks

- **skip_llm fix is a deliberate behavior change on `/api/scan`**: camera scans
  that previously got a vision-LLM assist (low OCR confidence / name-shape
  triggers) now return pure rule-based results. That is the requested
  semantics, but expect camera-path field quality to drop on hard labels until
  the caller opts in to LLM or Week 2 gating lands.
- `fallback_result` is now a public cross-module function; changes to its shape
  affect both `llm_extractor` and `pipeline` (that's the point, but it's now a
  contract).
- The deleted gold-set/OCR-diagnostics tooling is recoverable from git history
  (`git show 00a3159^ -- evaluate.py`) if the Week 2+ eval rework wants to
  resurrect the rotation-level OCR diagnostics idea.

## Remaining technical debt (unchanged from the architecture review)

1. Overfit rule engine (`address.py` site-specific parsers) and overfit LLM
   prompt; benchmark comparators still strip `DICKINSON COLLEGE`.
2. No conditional gating for the *text* LLM on `/upload` — it still runs on
   every upload; the gate design in `docs/confidence-selection.md` is the
   Week 2 Candidate/Selector work (not started, per instructions).
3. Untyped internal dict in `pipeline.run()` / dual result representations.
4. Two inconsistent accuracy definitions (regression fuzzy comparators vs
   evaluate.py strict equality).
5. Double Tesseract pass per rotation; 4× barcode decode; sequential stages.
6. `parser_used` / `llm_mode` / `llm_latency_ms` still not persisted to SQLite.
7. Unauthenticated `/labels` endpoints (PII); `.env` still holds live AWS keys
   (flagged in Week 1 — rotate them).
8. ML train/serve `max_length` mismatch (256 vs 64); NER model not integrated.

## Recommendations / next steps

1. Merge this branch after review (`git log main..fix-strict-skip-llm`).
2. Next milestone (awaiting your explicit approval, per your instruction):
   **Candidate/Selector architecture** from the review — it also subsumes debt
   items 2–4 above.
3. Independent quick wins that don't collide with Week 2: persist
   `parser_used`/`llm_mode` columns; single-pass OCR via `image_to_data`.
