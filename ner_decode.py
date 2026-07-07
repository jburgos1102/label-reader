"""Shared word tokenization and BIO span decoding for the NER model.

This module is the single source of truth for how OCR text is split into
word tokens — training export (ml/export_training_data.py) and runtime
inference (ner_extractor.py) both import it, so the tokenization used to
build training data can never drift from the tokenization used at
inference time.

Everything here is pure and dependency-free (stdlib only): no model, no
onnxruntime, no tokenizers import — unit-testable without any artifacts.
"""

import re
from dataclasses import dataclass

# Word-level tokenizer (moved verbatim from ml/export_training_data.py).
TOKEN_RE = re.compile(r"[A-Za-z0-9#&'.-]+|[^\w\s]")

# Entity types <-> pipeline field names (matches ml/train.py LABELS schema).
FIELD_TO_ENTITY = {
    "recipient_name":  "NAME",
    "street_address":  "STREET",
    "city":            "CITY",
    "state":           "STATE",
    "zip_code":        "ZIP",
    "tracking_number": "TRACKING",
    "carrier":         "CARRIER",
}
ENTITY_TO_FIELD = {entity: field for field, entity in FIELD_TO_ENTITY.items()}


def tokenize(text):
    """Split OCR text into word-level tokens."""
    return TOKEN_RE.findall(text)


@dataclass
class Span:
    """One decoded entity span over word tokens ([start, end) exclusive)."""

    field: str
    start: int
    end: int
    mean_prob: float


def decode_bio_spans(labels, probs):
    """Decode word-level BIO labels into entity spans grouped by field.

    labels: one BIO tag per word ("O", "B-NAME", "I-NAME", ...).
    probs:  the model probability for each word's chosen tag.

    Lenient decoding: an I-X with no open X span starts a new span (stray
    continuations happen with imperfect models); a tag for an entity type
    not in the schema is treated as O.

    Returns {field: [Span, ...]} with spans in text order.
    """
    spans = {}
    open_entity = None
    open_start = None
    open_probs = []

    def close(end_index):
        nonlocal open_entity, open_start, open_probs
        if open_entity is not None:
            field = ENTITY_TO_FIELD.get(open_entity)
            if field is not None:
                spans.setdefault(field, []).append(Span(
                    field=field,
                    start=open_start,
                    end=end_index,
                    mean_prob=sum(open_probs) / len(open_probs),
                ))
        open_entity = None
        open_start = None
        open_probs = []

    for index, (label, prob) in enumerate(zip(labels, probs)):
        if not label or label == "O":
            close(index)
            continue
        prefix, _, entity = label.partition("-")
        if entity not in ENTITY_TO_FIELD:
            close(index)
            continue
        if prefix == "B" or open_entity != entity:
            close(index)
            open_entity = entity
            open_start = index
            open_probs = [prob]
        else:  # I- continuing the open span
            open_probs.append(prob)

    close(len(labels))
    return spans


def best_span(spans):
    """Highest mean-probability span (first wins ties), or None."""
    if not spans:
        return None
    return max(spans, key=lambda s: (s.mean_prob, -s.start))


def field_value_from_span(field, words, span):
    """Turn a span back into a field value string.

    Light, field-aware joining only — comparator-level normalization
    (case, punctuation) is deliberately NOT done here; candidates should
    carry what the model actually found.
    """
    text = " ".join(words[span.start:span.end])
    if field == "tracking_number":
        text = re.sub(r"\s+", "", text)
    elif field == "state":
        text = text.upper()
    return text.strip()
