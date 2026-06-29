# Confidence-Based Selection Engine Design

## 1. Purpose

The selection engine exists to decide the final production-facing label fields after multiple extraction sources have produced candidates.

The project currently has strong rule-based extraction for many fields, an OpenAI-backed `llm_result`, a `comparison` section, and a `selected_result` section. Today, `selected_result` is intentionally conservative: it copies rule-based values unless rule-based and OpenAI agree, in which case the source is marked as `agreement`.

The future confidence-based selection engine should make field-by-field choices using evidence, not blanket trust in either source. Its job is to improve final output quality while protecting high-performing fields from unnecessary LLM overrides.

Current rule-based metrics:

| Field | Accuracy |
| --- | ---: |
| recipient_name | 58.3% |
| street_address | 78.3% |
| city | 80.8% |
| state | 84.6% |
| zip_code | 80.8% |
| tracking_number | 73.1% |
| carrier | 100.0% |

## 2. Current Inputs

The selection engine should use the existing extraction response shape.

### Rule-based top-level fields

These are the current primary parser outputs:

- `recipient_name`
- `street_address`
- `city`
- `state`
- `zip_code`
- `tracking_number`
- `carrier`
- `parser_used`
- `parser_matches`

### `confidence`

The rule-based parser currently adds a confidence object with field-level scores and an overall score. These scores are useful but should be treated carefully because they are mostly format-based.

### `warnings`

Warnings identify suspicious or missing rule-based fields, such as:

- missing recipient
- short or missing tracking number
- invalid ZIP
- low-confidence city/name patterns

Warnings should become selection signals, especially for deciding whether a field is eligible for LLM fallback.

### `llm_result`

The OpenAI extraction result contains the same core fields plus metadata:

- `llm_enabled`
- `llm_provider`
- `llm_notes`

The selection engine should only consider LLM values when `llm_result` is present, enabled, and field values are non-empty.

### `comparison`

The comparison object shows rule-based value, OpenAI value, selected value, and whether normalized rule/OpenAI values agree for each field.

Agreement is a strong signal. Disagreement is not automatically a reason to use OpenAI.

### `selected_result`

This is the final system output shape. Future selection logic should continue writing:

- selected field values
- selected source per field

Future versions should also add a reason per field.

### `tracking_source`

Tracking selection needs explicit provenance. A future extraction result should distinguish whether a tracking candidate came from:

- `barcode`
- `ocr`
- `llm`

Barcode-derived tracking should be strongly protected. A valid barcode-derived value should not be overridden by OpenAI.

### `barcode_confidence`

Barcode confidence should be treated separately from generic rule confidence. Barcode-derived tracking is usually stronger evidence than OCR text because it is machine-readable and less vulnerable to spacing, rotation, and OCR confusion.

If tracking provenance is `barcode`, the selection engine should treat the tracking value as protected unless there is clear evidence the barcode was not a shipment tracking barcode.

## 3. Field-by-Field Selection Strategy

### `recipient_name`

Likely preferred source: rule-based unless blank, noisy, or low confidence.

Rationale:

- Current recipient accuracy is the weakest core field at 58.3%.
- Recipient names are often affected by OCR noise and label routing text.
- OpenAI may be useful as a fallback when the rule-based parser returns blank, routing text, or organization-like noise.
- OpenAI should not freely override a clean rule-based person name.

Possible LLM fallback cases:

- rule-based recipient is blank
- rule-based recipient contains label markers such as `SHIP`, `TO`, or `HUB`
- rule-based recipient looks like an organization or routing block
- rule confidence is low and LLM gives a clean person-name candidate

LLM recipient values must pass the same clean-person-name validation used by the rule parser. In implementation terms, an OpenAI recipient fallback should satisfy logic equivalent to `_is_clean_person_name_candidate()` before it can be selected.

### `street_address`

Likely preferred source: rule-based for high-confidence specialized parsers.

Rationale:

- Current street accuracy is 78.3%.
- Rule-based extraction contains specialized logic for Dickinson, Penn State, HUB routing, PO Box handling, and damaged OCR recovery.
- LLM output may normalize addresses nicely, but can also omit mailroom-specific delivery details.

Use LLM only when:

- rule-based street is blank or clearly garbage
- LLM street is non-empty and address-like
- field is not a known specialized parser success case

### `city`

Likely preferred source: rule-based.

Rationale:

- Current city accuracy is 80.8%.
- Rule-based city extraction is tied to address block parsing and ZIP/state validation.
- OpenAI should not override a city extracted from a high-confidence city/state/ZIP parser.

LLM fallback may be considered only when rule-based city is blank or invalid.

### `state`

Likely preferred source: rule-based.

Rationale:

- Current state accuracy is 84.6%.
- State values are highly structured and easy to validate.
- A valid two-letter rule-based state should usually win.

LLM fallback may be considered only when rule-based state is blank or invalid and LLM state is a valid two-letter state.

### `zip_code`

Likely preferred source: rule-based.

Rationale:

- Current ZIP accuracy is 80.8%.
- ZIP codes are structured and often recoverable through rule-based city/state/ZIP patterns.
- LLM may normalize ZIP+4 formatting, but should not override validated rule values unless rule value is blank or invalid.

LLM fallback may be considered only when:

- rule ZIP is blank or invalid
- LLM ZIP is valid `#####` or `#####-####`

### `tracking_number`

Likely preferred source: barcode/rule-based.

Rationale:

- Tracking is high-risk because wrong tracking values are operationally harmful.
- Barcode-derived tracking should be treated as stronger evidence than OCR or LLM text.
- LLM should never override valid barcode/rule tracking.

LLM fallback should be limited to cases where:

- rule tracking is blank
- no valid barcode/rule candidate exists
- LLM value matches known carrier tracking patterns

Before trusting an LLM tracking number:

- it must match the expected carrier pattern
- it should appear verbatim or near-verbatim in OCR text
- it must not contradict barcode-derived tracking
- it must not override a valid barcode-derived value

### `carrier`

Likely preferred source: rule-based.

Rationale:

- Current carrier accuracy is 100.0%.
- Carrier is strongly inferred from tracking prefixes and explicit shipping context.
- LLM should not override carrier while rule-based carrier remains this reliable.

## 4. Confidence Concepts

### `rule_confidence`

Field-level confidence from rule-based extraction. This currently reflects format and completeness more than proven accuracy.

### `llm_confidence`

A future derived score for LLM values. This should not rely only on the LLM saying it is confident. It should include validation checks:

- field is non-empty
- field has expected format
- field does not contain label markers
- field agrees with OCR text where possible
- field does not conflict with high-confidence rule evidence

### `agreement`

Whether rule-based and OpenAI values match after normalization. Agreement should be treated as a strong confidence boost.

### `parser_used`

The primary parser that produced the rule-based result. Some parsers should be trusted more for specific fields, such as specialized deliver-to or college mailroom parsers.

### `parser_matches`

All parser patterns that matched during extraction. Multiple matching parsers can increase confidence when they support the same field values.

### `warnings`

Parser warnings should reduce rule confidence or make a field eligible for fallback.

### `tracking_source`

The provenance of the selected or candidate tracking number. Proposed values:

- `barcode`
- `ocr`
- `llm`

This should be used before generic confidence scoring. A barcode-derived tracking number should be protected more strongly than OCR- or LLM-derived tracking.

### `barcode_confidence`

A tracking-specific confidence signal for barcode-derived candidates. This should be high when the barcode value passes known tracking validation and carrier inference.

### `selected_source`

The source selected for each field. Proposed values:

- `agreement`
- `rule_based`
- `openai`
- `blank`

Avoid ambiguous values such as `fallback`. Fallback semantics should be expressed through `selected_reason`, not `selected_source`.

### `selected_reason`

A human-readable reason for the decision. Example:

- `rule_high_confidence`
- `rule_medium_confidence`
- `rule_openai_agree`
- `rule_blank_openai_valid`
- `rule_low_confidence_openai_valid`
- `barcode_tracking_preferred`
- `carrier_rule_protected`

## 5. Proposed Constants

These constants are design placeholders only. Do not implement them in this branch.

```python
LLM_FALLBACK_THRESHOLD = 0.60
HIGH_CONFIDENCE_THRESHOLD = 0.85
AGREEMENT_CONFIDENCE = 0.95
LOW_CONFIDENCE_THRESHOLD = 0.50  # below this, eligible fields may consider LLM fallback
MEDIUM_CONFIDENCE_THRESHOLD = 0.70
TRACKING_PROTECTION_THRESHOLD = 0.90
CARRIER_RULE_CONFIDENCE = 1.00
```

## 6. Initial Conservative Selection Rules

Proposed v1 behavior:

Decision order matters. Agreement should be checked before rule-confidence logic.

1. Agreement:
   - if rule-based and OpenAI agree after normalization:
     - select rule-based value
     - source: `agreement`
     - reason: `rule_openai_agree`

2. Rule high confidence:
   - if rule confidence is at or above `HIGH_CONFIDENCE_THRESHOLD`:
     - select rule-based value
     - source: `rule_based`
     - reason: `rule_high_confidence`

3. Rule medium confidence:
   - if rule confidence is medium and there are no serious warnings:
     - select rule-based value
     - source: `rule_based`
     - reason: `rule_medium_confidence`

4. Rule low confidence with eligible LLM fallback:
   - if rule confidence is below `LOW_CONFIDENCE_THRESHOLD` and LLM is present:
     - allow OpenAI only for eligible fields:
       - `recipient_name`
       - possibly `street_address`
       - possibly `city`, `state`, `zip_code` when rule values are blank or invalid
     - source: `openai`
     - reason: `rule_low_confidence_openai_valid`

5. Blank rule with eligible LLM fallback:
   - if rule value is blank and OpenAI has a valid field value:
     - source: `openai`
     - reason: `rule_blank_openai_valid`

6. Blank or no usable value:
   - if neither rule nor LLM has a usable value:
     - select blank
     - source: `blank`
     - reason: `no_valid_candidate`

7. Tracking protection:
   - never let LLM override tracking if barcode/rule tracking is valid
   - require LLM tracking to match carrier pattern and appear verbatim or near-verbatim in OCR text
   - reject LLM tracking if it contradicts barcode-derived tracking

8. Carrier protection:
   - never let LLM override carrier while rule carrier accuracy remains 100%

9. Always log selected source and reason.

## 7. Logging Plan

Use the shared structured logger.

Log at debug level for normal decisions:

- selected field
- rule value presence
- LLM value presence
- selected source
- selected reason
- confidence values
- tracking source
- barcode confidence when applicable

Log at info level for LLM override decisions:

- field overridden
- rule value
- LLM value
- reason

Log at warning level for unusual cases:

- low-confidence fields
- invalid LLM value
- conflicting high-confidence sources
- missing `selected_result`

Suggested future fields:

```json
{
  "selected_sources": {
    "recipient_name": "openai"
  },
  "selected_reasons": {
    "recipient_name": "rule_blank_openai_valid"
  }
}
```

Suggested future decision trace:

```json
{
  "selection_metadata": {
    "recipient_name": {
      "selected_source": "rule_based",
      "selected_reason": "explicit_to_line",
      "rule_confidence": 0.85,
      "llm_confidence": 0.78,
      "agreement": false
    }
  }
}
```

`selection_metadata` is for observability and debugging. It should be logged at debug level and may be included in JSON output only if the UI/API can tolerate the additional metadata.

## 8. Evaluation Plan

`evaluate.py` should continue measuring:

- rule-based top-level output
- OpenAI `llm_result`
- final `selected_result`
- Gold Set rule-based output
- Gold Set OpenAI output
- Gold Set selected output
- hybrid rule-vs-OpenAI comparison

`regression_test.py` should remain a rule-based baseline gate for now. A later branch can add selected-result regression thresholds once the selection engine is intentionally allowed to diverge from rule-based output.

Future selected-result evaluation should answer:

- Did selected output beat rule-based output?
- Which fields benefited from LLM fallback?
- Which fields regressed?
- Which parser warnings predict successful LLM fallback?

## 9. Risks

- Current confidence is mostly format-based, not accuracy-predictive.
- LLM may hallucinate names, addresses, or tracking numbers.
- Gold Set may overrepresent cases that have already been tuned.
- OCR missing text cannot be recovered by selection alone.
- LLM may normalize away mailroom-specific delivery details that are operationally important.
- Overly broad fallback rules could hide parser regressions.

## 10. Implementation Plan

Recommended future branches:

1. `feature/selection-reasons`
   - Add `selected_reasons` without changing selected values.

2. `feature/selected-result-regression-test`
   - Add selected-result thresholds before the full confidence engine is allowed to change final output.

3. `feature/confidence-threshold-constants`
   - Centralize threshold constants and field eligibility lists.

4. `feature/llm-recipient-fallback`
   - Allow OpenAI fallback for recipient names only when rule value is blank/noisy/low confidence.

5. `feature/confidence-selection-engine-v1`
   - Implement field-by-field selection using confidence, warnings, agreement, and source protections.

## 11. Non-Goals

- Do not refactor parser modules yet.
- Do not add AWS.
- Do not change OpenAI prompts in this branch.
- Do not change extraction behavior in this branch.
- Do not change current selection behavior in this branch.
- Do not change evaluation logic in this branch.
