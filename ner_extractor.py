"""ONNX Runtime inference for the shipping-label NER model.

Shadow candidate source: predictions become Candidates with source="ner"
that the legacy Selector ignores, so this module can never affect selected
output (see docs/architecture-candidate-selector.md).

Design constraints:
- Heavy imports (onnxruntime, tokenizers, numpy) happen inside the class,
  so importing this module is cheap and a disabled flag never loads them.
- The session is a lazy singleton (double-checked under a lock: Flask
  serves concurrent requests).
- A missing or unloadable model artifact logs ONE warning and disables the
  source for the process lifetime — a scan must never fail because a model
  file was not deployed.
- Tokenization comes from ner_decode.tokenize, the same function used to
  build the training data, and subword alignment takes the first-subword
  prediction per word — mirroring the training label alignment.
"""

import json
import os
import threading
import time

import config
from logger import log
from ner_decode import best_span, decode_bio_spans, field_value_from_span, tokenize

_lock = threading.Lock()
_state = {"extractor": None, "failed": False}


class NerExtractor:
    def __init__(self, model_dir):
        import numpy
        import onnxruntime
        from tokenizers import Tokenizer

        self._np = numpy
        onnx_path = os.path.join(model_dir, "label_reader.onnx")
        tokenizer_path = os.path.join(model_dir, "label_reader", "tokenizer.json")
        config_path = os.path.join(model_dir, "label_reader", "config.json")

        self.session = onnxruntime.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=config.NER_MAX_TOKENS)
        with open(config_path) as f:
            self.id2label = {
                int(label_id): label
                for label_id, label in json.load(f)["id2label"].items()
            }
        # Model identity for Candidate.reason — lets per-model-version accuracy
        # be computed from persisted candidates after retraining.
        self.version = "distilbert@" + time.strftime(
            "%Y-%m-%d", time.localtime(os.path.getmtime(onnx_path))
        )

    def predict_fields(self, ocr_text):
        """Run NER over OCR text -> {field: {value, confidence, alternates}}.

        confidence is the mean token probability of the winning span —
        an UNCALIBRATED model probability, recorded for future calibration.
        alternates counts additional spans the model found for the field.
        """
        words = tokenize(ocr_text or "")
        if not words:
            return {}

        np = self._np
        encoding = self.tokenizer.encode(words, is_pretokenized=True)
        logits = self.session.run(None, {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        })[0][0]

        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        predicted_ids = probs.argmax(axis=-1)

        # First-subword prediction per word (training alignment); words lost
        # to truncation keep "O".
        labels = ["O"] * len(words)
        word_probs = [0.0] * len(words)
        seen_words = set()
        for position, word_index in enumerate(encoding.word_ids):
            if word_index is None or word_index in seen_words:
                continue
            seen_words.add(word_index)
            label_id = int(predicted_ids[position])
            labels[word_index] = self.id2label.get(label_id, "O")
            word_probs[word_index] = float(probs[position, label_id])

        if len(seen_words) < len(words):
            log.debug(
                "NER input truncated: %d of %d words covered (max_tokens=%d)",
                len(seen_words), len(words), config.NER_MAX_TOKENS,
            )

        predictions = {}
        for field, field_spans in decode_bio_spans(labels, word_probs).items():
            span = best_span(field_spans)
            value = field_value_from_span(field, words, span)
            if value:
                predictions[field] = {
                    "value": value,
                    "confidence": round(span.mean_prob, 4),
                    "alternates": len(field_spans) - 1,
                }
        return predictions


def get_extractor():
    """Return the extractor singleton, or None (flag off / artifact unusable)."""
    if not config.NER_ENABLED or _state["failed"]:
        return None
    if _state["extractor"] is None:
        with _lock:
            if _state["extractor"] is None and not _state["failed"]:
                started = time.monotonic()
                try:
                    _state["extractor"] = NerExtractor(config.NER_MODEL_DIR)
                    log.info(
                        "NER model loaded in %.2fs (%s)",
                        time.monotonic() - started,
                        _state["extractor"].version,
                    )
                except Exception:
                    log.exception(
                        "NER model unavailable (dir=%s); NER candidates "
                        "disabled for this process",
                        config.NER_MODEL_DIR,
                    )
                    _state["failed"] = True
                    return None
    return _state["extractor"]


def ner_field_predictions(ocr_text):
    """Safe entry point for the pipeline: {} on any failure, never raises."""
    extractor = get_extractor()
    if extractor is None:
        return {}
    try:
        return extractor.predict_fields(ocr_text)
    except Exception:
        log.exception("NER inference failed; continuing without NER candidates")
        return {}


def reset_for_tests():
    with _lock:
        _state["extractor"] = None
        _state["failed"] = False
