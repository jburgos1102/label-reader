"""Unit tests for the Candidate/Selector policy (legacy-parity rules).

Run from the project root:  venv/bin/python tests/test_selection.py
Plain asserts, no pytest dependency, no images/OCR required.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selection import (
    Candidate,
    SelectionContext,
    Selector,
    llm_candidates,
    rule_candidates,
)


def c(value, source, confidence=0.5):
    return Candidate(field="city", value=value, source=source, confidence=confidence)


def select(candidates, llm_mode="text"):
    return Selector().select("city", candidates, SelectionContext(llm_mode=llm_mode))


def main():
    # 1. No LLM candidate: rule wins with source rule_based
    s = select([c("TROY", "rule", 0.85)])
    assert (s.value, s.source, s.confidence, s.reason) == (
        "TROY", "rule_based", 0.85, "rule_default"), s

    # 2. Rule blank + LLM value: LLM fills the gap
    s = select([c("", "rule", 0.0), c("TROY", "llm", 0.75)])
    assert (s.value, s.source, s.confidence, s.reason) == (
        "TROY", "llm", 0.75, "rule_blank_llm_fill"), s

    # 3. Text-mode conflict: rule wins
    s = select([c("TROY", "rule", 0.85), c("CANTON", "llm", 0.85)], llm_mode="text")
    assert (s.value, s.source, s.reason) == ("TROY", "rule_based", "rule_default"), s

    # 4. Vision-mode conflict: LLM overrides
    s = select([c("TROY", "rule", 0.85), c("CANTON", "llm", 0.75)], llm_mode="vision")
    assert (s.value, s.source, s.confidence, s.reason) == (
        "CANTON", "llm", 0.75, "vision_conflict_override"), s

    # 5. vision_fallback mode does NOT get the override (only "vision")
    s = select([c("TROY", "rule", 0.85), c("CANTON", "llm", 0.75)],
               llm_mode="vision_fallback")
    assert (s.value, s.source) == ("TROY", "rule_based"), s

    # 6. Agreement after normalization (case/punctuation differences)
    s = select([c("Troy.", "rule", 0.85), c("TROY", "llm", 0.85)], llm_mode="vision")
    assert (s.value, s.source, s.confidence, s.reason) == (
        "Troy.", "agreement", 0.85, "rule_llm_agree"), s

    # 7. Both empty with LLM present: legacy empty/empty agreement
    s = select([c("", "rule", 0.0), c("", "llm", 0.0)])
    assert (s.value, s.source, s.reason) == ("", "agreement", "rule_llm_agree"), s

    # 8. Both empty with NO LLM: rule_based, not agreement
    s = select([c("", "rule", 0.0)])
    assert (s.value, s.source) == ("", "rule_based"), s

    # 9. Unknown sources are ignored by the legacy policy
    s_with = select([c("TROY", "rule", 0.85), c("CANTON", "llm", 0.8),
                     c("BOSTON", "ner", 0.99)], llm_mode="text")
    s_without = select([c("TROY", "rule", 0.85), c("CANTON", "llm", 0.8)],
                       llm_mode="text")
    assert (s_with.value, s_with.source) == (s_without.value, s_without.source), s_with

    # 10. No rule candidate at all behaves as blank rule
    s = select([c("TROY", "llm", 0.75)])
    assert (s.value, s.source, s.reason) == ("TROY", "llm", "rule_blank_llm_fill"), s

    # 11. Selection carries the full candidate list for observability
    cands = [c("TROY", "rule", 0.85), c("CANTON", "llm", 0.8)]
    assert select(cands).candidates == cands

    # --- builders -----------------------------------------------------------

    fields = ("city", "tracking_number")
    label_data = {"city": "Troy", "tracking_number": "1Z999", "parser_used": "deliver_to"}
    confidence = {"city": 0.85, "tracking_number": 0.4}
    rc = rule_candidates(label_data, fields, confidence, tracking_checksum_valid=False)
    assert rc["city"].value == "Troy" and rc["city"].confidence == 0.85
    assert rc["city"].source == "rule" and rc["city"].reason == "deliver_to"
    assert rc["tracking_number"].validations == {"checksum_valid": False}

    # LLM builder: no candidates when the LLM did not run
    assert llm_candidates({"llm_enabled": False}, fields, {}) == {}
    assert llm_candidates("not a dict", fields, {}) == {}

    llm_result = {"llm_enabled": True, "llm_provider": "groq",
                  "city": "TROY", "tracking_number": ""}
    scores = {"city": 0.85, "tracking_number": 0.0}
    lc = llm_candidates(llm_result, fields, scores)
    assert lc["city"].confidence == 0.85 and lc["city"].validations == {"found_in_ocr": True}
    assert lc["tracking_number"].value == "" and lc["tracking_number"].confidence == 0.0
    assert lc["tracking_number"].validations == {"found_in_ocr": False}
    assert lc["city"].reason == "groq"

    print("test_selection OK (14 scenario groups)")


if __name__ == "__main__":
    main()
