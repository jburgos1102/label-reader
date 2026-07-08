"""Candidate/Selector architecture for per-field value selection.

Every extraction source (rule engine, LLM, and future NER / additional OCR
engines / additional LLM providers) produces Candidate objects; the Selector
arbitrates one Selection per field. Sources never encode selection policy,
and the policy never knows how a value was produced — adding a new source
means emitting more candidates, not changing the selector.

Extension contract for a future source (e.g. NER):
    1. Produce ``list[Candidate]`` with a new ``source`` name and a calibrated
       ``confidence``; attach any format checks to ``validations``.
    2. Append them to the per-field candidate lists in ``pipeline.run()``.
    3. Extend the Selector policy (or add a new policy class) to rank the new
       source. The legacy policy below deliberately ignores unknown sources so
       that adding a source cannot silently change behavior before its policy
       is written.

This module is behavior-parity first: ``Selector`` reproduces the historical
selection rules exactly (guarded by tests/test_selection_parity.py). Smarter,
calibrated policies come later and slot in behind the same interface.
"""

import re
from dataclasses import asdict, dataclass, field as dataclass_field

import config
from calibration import get_confidence_model
from scoring import normalize_comparison_value
from tracking import identify_carrier, validate_tracking_checksum


@dataclass
class Candidate:
    """One source's proposed value for one field.

    An empty ``value`` is a legitimate candidate ("this source found
    nothing") — the legacy agreement rule depends on it: when both the rule
    engine and the LLM produce empty values, the field is selected as empty
    with source "agreement".
    """

    field: str
    value: str
    source: str                 # "rule" | "llm" | future: "ner", "ocr4", ...
    confidence: float           # 0.0–1.0 (source-specific until calibration)
    validations: dict = dataclass_field(default_factory=dict)  # name -> bool/None
    reason: str = ""            # how the source produced the value (optional)


@dataclass
class Selection:
    """The selector's decision for one field."""

    field: str
    value: str
    source: str                 # legacy labels: "rule_based" | "llm" | "agreement"
    confidence: float
    reason: str
    candidates: list = dataclass_field(default_factory=list)


@dataclass
class SelectionContext:
    """Per-scan context the policy may consult (not tied to any one source)."""

    llm_mode: str = "none"      # "none" | "text" | "vision" | "vision_fallback"


class Selector:
    """Legacy-parity selection policy.

    Ported verbatim from the historical pipeline.py loop:
      1. Rule value blank and LLM produced a value  -> LLM
         (reason: rule_blank_llm_fill)
      2. Vision mode, LLM produced a value, and the two disagree -> LLM.
         Vision was triggered because rule-based extraction is unreliable,
         so the LLM wins conflicts rather than letting the rule win.
         (reason: vision_conflict_override; note: "vision" only — the
         opportunistic "vision_fallback" mode does not get this override)
      3. Otherwise the rule value wins; the source is "agreement" when the
         normalized values match (including both empty), "rule_based" when
         they differ or no LLM candidate exists.

    Candidates from sources other than "rule"/"llm" are ignored by this
    policy (see module docstring).
    """

    def select(self, field_name, candidates, context):
        rule = next((c for c in candidates if c.source == "rule"), None)
        llm = next((c for c in candidates if c.source == "llm"), None)

        rule_value = rule.value if rule else ""
        rule_confidence = rule.confidence if rule else 0.0
        llm_value = llm.value if llm else ""

        values_agree = llm is not None and (
            normalize_comparison_value(rule_value)
            == normalize_comparison_value(llm_value)
        )

        if not rule_value and llm is not None and llm_value:
            return Selection(
                field=field_name,
                value=llm_value,
                source="llm",
                confidence=llm.confidence,
                reason="rule_blank_llm_fill",
                candidates=list(candidates),
            )

        if context.llm_mode == "vision" and llm is not None and llm_value and not values_agree:
            return Selection(
                field=field_name,
                value=llm_value,
                source="llm",
                confidence=llm.confidence,
                reason="vision_conflict_override",
                candidates=list(candidates),
            )

        return Selection(
            field=field_name,
            value=rule_value,
            source="agreement" if values_agree else "rule_based",
            confidence=rule_confidence,
            reason="rule_llm_agree" if values_agree else "rule_default",
            candidates=list(candidates),
        )


class NerNamePolicy(Selector):
    """Legacy policy plus a gated NER override for recipient_name ONLY.

    Every field first gets the legacy Selector decision; any field other
    than recipient_name returns it before NER is even looked at, so this
    policy is structurally incapable of changing street_address, city,
    state, zip_code, tracking_number, or carrier.

    The recipient_name gate (evidence: calibration table + ner_backtest.py
    sweep, both as of 2026-07-08):
      - An llm-sourced selection is never overridden (measured 95.4%
        accurate vs NER's ~73-83%).
      - When the NER value normalizes to the legacy value, the legacy
        selection is kept (same value, cheaper provenance).
      - The NER value must pass plausible_recipient_name.
      - A blank legacy selection is filled at
        config.NER_NAME_FILL_MIN_CONFIDENCE (blank baseline ~0% accurate);
        a non-empty one is overridden only at
        config.NER_NAME_OVERRIDE_MIN_CONFIDENCE (the sweep's zero-loss
        point).
    """

    def select(self, field_name, candidates, context):
        base = super().select(field_name, candidates, context)
        if field_name != "recipient_name":
            return base

        ner = next((c for c in candidates if c.source == "ner"), None)
        if ner is None or not ner.value:
            return base
        if base.source == "llm":
            return base
        if normalize_comparison_value(ner.value) == normalize_comparison_value(
            base.value
        ):
            return base
        if not plausible_recipient_name(ner.value):
            return base

        if not base.value:
            threshold = config.NER_NAME_FILL_MIN_CONFIDENCE
            reason = "ner_name_fill"
        else:
            threshold = config.NER_NAME_OVERRIDE_MIN_CONFIDENCE
            reason = "ner_name_override"
        if ner.confidence < threshold:
            return base

        return Selection(
            field=field_name,
            value=ner.value,
            source="ner",
            confidence=ner.confidence,
            reason=reason,
            candidates=list(candidates),
        )


def get_selector():
    """Selection policy for the current config: NerNamePolicy only when BOTH
    NER flags are set (candidates exist AND may influence recipient_name),
    otherwise the legacy-parity Selector. Checked per call so flag changes
    (tests, rollback) take effect without a restart."""
    if config.NER_ENABLED and config.NER_NAME_SELECTION_ENABLED:
        return NerNamePolicy()
    return Selector()


# ---------------------------------------------------------------------------
# Candidate builders for the existing sources
# ---------------------------------------------------------------------------


def selection_provenance(selections):
    """JSON-serializable provenance for storage: per-field selection reasons
    and the full candidate lists (including the losers — needed later to
    calibrate candidate accuracy, not just selected-value accuracy)."""
    reasons = {name: s.reason for name, s in selections.items()}
    candidates = {
        name: [asdict(c) for c in s.candidates]
        for name, s in selections.items()
    }
    return reasons, candidates


def rule_candidates(label_data, fields, confidence, tracking_checksum_valid=None):
    """Build one rule-engine candidate per field from the scored label_data.

    Confidence routes through the configured ConfidenceModel: "legacy"
    (default) passes the heuristic value through unchanged; "calibrated"
    replaces it with measured P(correct) from the fitted table.
    """
    model = get_confidence_model()
    candidates = {}
    for field_name in fields:
        validations = {}
        if field_name == "tracking_number":
            # True/False when the carrier format is checksum-validatable,
            # None otherwise — observability only, no selection effect yet.
            validations["checksum_valid"] = tracking_checksum_valid
        candidates[field_name] = Candidate(
            field=field_name,
            value=label_data.get(field_name, ""),
            source="rule",
            confidence=model.candidate_confidence(
                field_name, "rule", confidence.get(field_name, 0.0), validations
            ),
            validations=validations,
            reason=label_data.get("parser_used", "") or "",
        )
    return candidates


def llm_candidates(llm_result, fields, llm_scores):
    """Build one LLM candidate per field, or none when the LLM was not called.

    Empty values still yield candidates when the LLM ran — required for the
    legacy empty/empty "agreement" behavior (see Candidate docstring).
    """
    if not isinstance(llm_result, dict) or not llm_result.get("llm_enabled"):
        return {}

    model = get_confidence_model()
    candidates = {}
    for field_name in fields:
        score = llm_scores.get(field_name)
        confidence = score if score is not None else 0.0
        validations = {
            "found_in_ocr": confidence == config.CONFIDENCE_LLM_OCR_MATCH,
        }
        candidates[field_name] = Candidate(
            field=field_name,
            value=llm_result.get(field_name, ""),
            source="llm",
            confidence=model.candidate_confidence(
                field_name, "llm", confidence, validations
            ),
            validations=validations,
            reason=llm_result.get("llm_provider", "") or "",
        )
    return candidates


def plausible_recipient_name(value):
    """Cheap plausibility check for a proposed recipient name.

    Targets NER's known failure mode of bleeding into address lines: street
    and tracking spans contain digits, and single-word names are already a
    rule-engine suspect signal (_name_single_word vision trigger) — require
    2+ tokens with no digits. Shared by the backtest simulation
    (ner_backtest.py) and the name-selection policy so the offline numbers
    and the runtime gate can never diverge.
    """
    if not value or any(ch.isdigit() for ch in value):
        return False
    return len(value.split()) >= 2


def _ner_validations(field_name, value):
    """Cheap format/checksum validations for an NER prediction — recorded on
    the candidate as future calibration features."""
    validations = {}
    if field_name == "recipient_name":
        validations["name_plausible"] = plausible_recipient_name(value)
    elif field_name == "state":
        validations["format_valid"] = bool(re.fullmatch(r"[A-Z]{2}", value))
    elif field_name == "zip_code":
        validations["format_valid"] = bool(re.fullmatch(r"\d{5}(-\d{4})?", value))
    elif field_name == "tracking_number":
        validations["checksum_valid"] = validate_tracking_checksum(
            value, identify_carrier(value)
        )
    return validations


def ner_candidates(ner_predictions, fields, model_version="ner"):
    """Build candidates from NER predictions.

    The legacy Selector IGNORES source="ner" candidates (shadow mode);
    NerNamePolicy may select them for recipient_name only. Confidence is the
    model's uncalibrated span probability, routed through the configured
    ConfidenceModel like every other source (the calibrated table has no
    ner buckets yet, so it passes through unchanged). Fields where the
    model found nothing emit no candidate.
    """
    model = get_confidence_model()
    candidates = {}
    for field_name in fields:
        prediction = ner_predictions.get(field_name)
        if not prediction or not prediction.get("value"):
            continue
        validations = _ner_validations(field_name, prediction["value"])
        # Non-boolean entry: excluded from calibration signatures, kept for
        # offline analysis (how often the model saw competing spans).
        validations["alternates"] = prediction.get("alternates", 0)
        candidates[field_name] = Candidate(
            field=field_name,
            value=prediction["value"],
            source="ner",
            confidence=model.candidate_confidence(
                field_name, "ner", prediction.get("confidence", 0.0), validations
            ),
            validations=validations,
            reason=model_version,
        )
    return candidates
