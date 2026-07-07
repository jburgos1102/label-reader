"""Unit tests for shared NER tokenization and BIO span decoding.

Run from the project root:  venv/bin/python tests/test_ner_decode.py
Pure functions — no model artifacts, no heavy imports.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ner_decode import (
    Span,
    best_span,
    decode_bio_spans,
    field_value_from_span,
    tokenize,
)


def spans_for(labels, probs=None):
    probs = probs or [0.9] * len(labels)
    return decode_bio_spans(labels, probs)


def main():
    # --- tokenize parity behavior -------------------------------------------
    assert tokenize("JOHN SMITH, 28 N. COLLEGE ST") == [
        "JOHN", "SMITH", ",", "28", "N.", "COLLEGE", "ST"]
    assert tokenize("1Z999AA10123456784") == ["1Z999AA10123456784"]
    assert tokenize("") == []

    # Training-export parity: the same tokenize must be importable there
    import ml.export_training_data as etd
    assert etd.tokenize is tokenize

    # --- basic span decoding -------------------------------------------------
    labels = ["B-NAME", "I-NAME", "O", "B-CITY"]
    result = spans_for(labels, [0.8, 0.6, 0.9, 0.7])
    assert result["recipient_name"] == [Span("recipient_name", 0, 2, 0.7)]
    assert result["city"] == [Span("city", 3, 4, 0.7)]

    # --- multiple spans for one field ---------------------------------------
    labels = ["B-CITY", "O", "B-CITY", "I-CITY"]
    result = spans_for(labels, [0.5, 0.9, 0.9, 0.7])
    assert len(result["city"]) == 2
    chosen = best_span(result["city"])
    assert (chosen.start, chosen.end) == (2, 4)  # mean 0.8 beats 0.5

    # best_span tie -> earliest span
    tie = [Span("city", 3, 4, 0.8), Span("city", 0, 1, 0.8)]
    assert best_span(tie).start == 0
    assert best_span([]) is None

    # --- lenient decoding: stray I- starts a span ----------------------------
    result = spans_for(["O", "I-STREET", "I-STREET"])
    assert result["street_address"][0].start == 1
    assert result["street_address"][0].end == 3

    # entity switch without O closes the previous span
    result = spans_for(["B-NAME", "I-STREET"])
    assert result["recipient_name"][0].end == 1
    assert result["street_address"][0].start == 1

    # B- immediately after B- of same entity starts a NEW span
    result = spans_for(["B-ZIP", "B-ZIP"])
    assert len(result["zip_code"]) == 2

    # span at the very end of the sequence is closed
    result = spans_for(["O", "B-CARRIER"])
    assert result["carrier"][0].end == 2

    # unknown entity types and empty labels are ignored
    result = spans_for(["B-BOGUS", "I-BOGUS", "", "O"])
    assert result == {}

    # no spans at all
    assert spans_for(["O", "O"]) == {}

    # --- field value construction --------------------------------------------
    words = ["1Z", "999", "AA1", "TROY", "mo", "JOHN", "SMITH"]
    assert field_value_from_span(
        "tracking_number", words, Span("tracking_number", 0, 3, 0.9)) == "1Z999AA1"
    assert field_value_from_span("state", words, Span("state", 4, 5, 0.9)) == "MO"
    assert field_value_from_span(
        "recipient_name", words, Span("recipient_name", 5, 7, 0.9)) == "JOHN SMITH"

    print("test_ner_decode OK")


if __name__ == "__main__":
    main()
