"""Unit tests for NerNamePolicy (gated NER recipient_name selection).

Run from the project root:  venv/bin/python tests/test_selection_ner_name.py
Plain asserts, no pytest dependency, no images/OCR/model required.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from selection import (
    Candidate,
    NerNamePolicy,
    SelectionContext,
    Selector,
    get_selector,
    plausible_recipient_name,
)

FILL = config.NER_NAME_FILL_MIN_CONFIDENCE
OVERRIDE = config.NER_NAME_OVERRIDE_MIN_CONFIDENCE

OTHER_FIELDS = (
    "street_address",
    "city",
    "state",
    "zip_code",
    "tracking_number",
    "carrier",
)


def c(field, value, source, confidence=0.5):
    return Candidate(field=field, value=value, source=source, confidence=confidence)


def select(field, candidates, llm_mode="none"):
    return NerNamePolicy().select(field, candidates, SelectionContext(llm_mode=llm_mode))


def main():
    # -- plausible_recipient_name ------------------------------------------
    assert plausible_recipient_name("Caroline Wilder")
    assert plausible_recipient_name("Mary Jo Van Der Berg")
    assert not plausible_recipient_name("")            # empty
    assert not plausible_recipient_name("Wilder")      # single token
    assert not plausible_recipient_name("28 N College St")  # digits = address bleed
    assert not plausible_recipient_name("Apt 4B")

    # -- get_selector flag matrix ------------------------------------------
    orig = (config.NER_ENABLED, config.NER_NAME_SELECTION_ENABLED)
    try:
        for ner_on, name_on, expected in (
            (False, False, Selector),
            (True, False, Selector),
            (False, True, Selector),
            (True, True, NerNamePolicy),
        ):
            config.NER_ENABLED = ner_on
            config.NER_NAME_SELECTION_ENABLED = name_on
            assert type(get_selector()) is expected, (ner_on, name_on)
    finally:
        config.NER_ENABLED, config.NER_NAME_SELECTION_ENABLED = orig

    # -- structural field gate: no other field can ever change --------------
    # Even a maximally attractive NER candidate (confidence 0.99, plausible
    # value, weak/blank rule) must leave every non-name field identical to
    # the legacy Selector's decision.
    for field in OTHER_FIELDS:
        for rule_value in ("", "SOMEVALUE"):
            for llm_mode, llm_cand in (
                ("none", None),
                ("text", c(field, "LLMVALUE", "llm", 0.85)),
                ("vision", c(field, "LLMVALUE", "llm", 0.75)),
            ):
                cands = [c(field, rule_value, "rule", 0.45)]
                if llm_cand:
                    cands.append(llm_cand)
                without_ner = list(cands)
                with_ner = cands + [c(field, "Totally Plausible", "ner", 0.99)]
                legacy = Selector().select(
                    field, without_ner, SelectionContext(llm_mode=llm_mode)
                )
                policy = select(field, with_ner, llm_mode=llm_mode)
                assert (policy.value, policy.source, policy.reason) == (
                    legacy.value, legacy.source, legacy.reason), (field, policy)
                assert policy.source != "ner", (field, policy)

    # -- fill: blank selection, no LLM, confident plausible NER -------------
    s = select("recipient_name", [
        c("recipient_name", "", "rule", 0.0),
        c("recipient_name", "Ian Wolf", "ner", FILL + 0.05),
    ])
    assert (s.value, s.source, s.reason) == ("Ian Wolf", "ner", "ner_name_fill"), s
    assert s.confidence == FILL + 0.05

    # -- fill blocked below the fill threshold ------------------------------
    s = select("recipient_name", [
        c("recipient_name", "", "rule", 0.0),
        c("recipient_name", "Ian Wolf", "ner", FILL - 0.05),
    ])
    assert (s.value, s.source) == ("", "rule_based"), s

    # -- override: rule value beaten only above the override threshold ------
    s = select("recipient_name", [
        c("recipient_name", "Wilder", "rule", 0.85),
        c("recipient_name", "Caroline Wilder", "ner", OVERRIDE + 0.05),
    ])
    assert (s.value, s.source, s.reason) == (
        "Caroline Wilder", "ner", "ner_name_override"), s

    s = select("recipient_name", [
        c("recipient_name", "Wilder", "rule", 0.85),
        c("recipient_name", "Caroline Wilder", "ner", OVERRIDE - 0.05),
    ])
    assert (s.value, s.source, s.reason) == ("Wilder", "rule_based", "rule_default"), s

    # -- fill threshold does NOT unlock overrides (blank-only) ---------------
    assert FILL < OVERRIDE
    s = select("recipient_name", [
        c("recipient_name", "Wilder", "rule", 0.85),
        c("recipient_name", "Caroline Wilder", "ner", (FILL + OVERRIDE) / 2),
    ])
    assert s.source == "rule_based", s

    # -- llm-sourced selections are never overridden ------------------------
    # rule_blank_llm_fill
    s = select("recipient_name", [
        c("recipient_name", "", "rule", 0.0),
        c("recipient_name", "Ana Cruz", "llm", 0.85),
        c("recipient_name", "Bob Jones", "ner", 0.99),
    ], llm_mode="text")
    assert (s.value, s.source, s.reason) == ("Ana Cruz", "llm", "rule_blank_llm_fill"), s
    # vision_conflict_override
    s = select("recipient_name", [
        c("recipient_name", "Wilder", "rule", 0.85),
        c("recipient_name", "Ana Cruz", "llm", 0.75),
        c("recipient_name", "Bob Jones", "ner", 0.99),
    ], llm_mode="vision")
    assert (s.value, s.source, s.reason) == (
        "Ana Cruz", "llm", "vision_conflict_override"), s

    # -- NER agreeing with the legacy value keeps legacy provenance ---------
    s = select("recipient_name", [
        c("recipient_name", "Caroline Wilder", "rule", 0.85),
        c("recipient_name", "CAROLINE WILDER.", "ner", 0.99),
    ])
    assert (s.value, s.source, s.reason) == (
        "Caroline Wilder", "rule_based", "rule_default"), s

    # -- implausible NER values never fire, regardless of confidence --------
    for bad in ("Wilder", "28 N College St"):
        s = select("recipient_name", [
            c("recipient_name", "", "rule", 0.0),
            c("recipient_name", bad, "ner", 0.99),
        ])
        assert s.source != "ner", (bad, s)

    # -- rule/llm empty-empty agreement is fillable (4%-accurate bucket) ----
    s = select("recipient_name", [
        c("recipient_name", "", "rule", 0.0),
        c("recipient_name", "", "llm", 0.0),
        c("recipient_name", "Ian Wolf", "ner", FILL + 0.05),
    ], llm_mode="text")
    assert (s.value, s.source, s.reason) == ("Ian Wolf", "ner", "ner_name_fill"), s

    # -- no NER candidate: byte-identical to legacy --------------------------
    cands = [c("recipient_name", "Wilder", "rule", 0.85)]
    legacy = Selector().select("recipient_name", cands, SelectionContext())
    policy = select("recipient_name", list(cands))
    assert (policy.value, policy.source, policy.reason) == (
        legacy.value, legacy.source, legacy.reason)

    # -- Selection carries the full candidate list ---------------------------
    cands = [
        c("recipient_name", "", "rule", 0.0),
        c("recipient_name", "Ian Wolf", "ner", FILL + 0.05),
    ]
    assert select("recipient_name", cands).candidates == cands

    print("test_selection_ner_name OK")


if __name__ == "__main__":
    main()
