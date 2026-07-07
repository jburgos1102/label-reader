"""Confidence calibration: shared helpers for the fitted confidence table.

The table maps (field, source, validation signature) buckets to measured
P(correct) — Laplace-smoothed empirical rates fitted from annotated labels by
fit_calibration.py, using the shared comparators as the definition of
"correct". Lookup is hierarchical: the most specific bucket with enough
samples wins, falling back to (field, source), then (source), then the
caller's legacy value.

The committed artifact (calibration/confidence_table.json) is versioned in
git so every confidence change is a reviewable diff.
"""

import json
import os

# Repo-relative artifact location (scripts run from the project root).
CALIBRATION_TABLE_PATH = os.path.join("calibration", "confidence_table.json")

# Buckets with fewer samples than this fall through to the next level.
MIN_BUCKET_N = 25


def validation_signature(validations):
    """Canonical signature for a candidate's validation results.

    Only boolean outcomes participate ("checksum_valid=True"); None means
    "not validatable" and is omitted, so those rows land in the plain
    (field, source) bucket. Keys are sorted for stability.
    """
    if not validations:
        return ""
    parts = [
        f"{name}={value}"
        for name, value in sorted(validations.items())
        if isinstance(value, bool)
    ]
    return ",".join(parts)


def bucket_keys(field, source, signature=""):
    """Lookup keys from most to least specific."""
    keys = []
    if signature:
        keys.append(f"{field}|{source}|{signature}")
    keys.append(f"{field}|{source}")
    keys.append(source)
    return keys


def load_table(path=None):
    """Load the fitted table, or None when no artifact exists."""
    path = path or CALIBRATION_TABLE_PATH
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def lookup(table, field, source, validations=None, min_n=None):
    """Return (probability, bucket_key) from the most specific bucket with
    n >= min_n, or (None, None) when no bucket qualifies."""
    if not table:
        return None, None
    buckets = table.get("buckets", {})
    if min_n is None:
        min_n = table.get("metadata", {}).get("min_bucket_n", MIN_BUCKET_N)
    signature = validation_signature(validations or {})
    for key in bucket_keys(field, source, signature):
        bucket = buckets.get(key)
        if bucket and bucket["n"] >= min_n:
            return bucket["p"], key
    return None, None


# ---------------------------------------------------------------------------
# Confidence models — the seam candidate builders route confidence through.
# config.CONFIDENCE_MODE selects the model; "legacy" (the default) is a pure
# passthrough, so runtime behavior is unchanged until the flip is approved.
# ---------------------------------------------------------------------------


class LegacyConfidence:
    """Passthrough: candidates keep the historical heuristic confidence."""

    name = "legacy"

    def candidate_confidence(self, field, source, base_confidence, validations=None):
        return base_confidence


class CalibratedConfidence:
    """Measured P(correct) from the fitted table, hierarchical fallback.

    Falls back to the caller's legacy value when no bucket has enough
    samples (or no artifact exists), so enabling this model can never
    produce a field with no confidence at all.
    """

    name = "calibrated"

    def __init__(self, table=None, path=None):
        self._table = table if table is not None else load_table(path)

    def candidate_confidence(self, field, source, base_confidence, validations=None):
        probability, _ = lookup(self._table, field, source, validations)
        return probability if probability is not None else base_confidence


_model_cache = {}


def get_confidence_model():
    """Model selected by config.CONFIDENCE_MODE ("legacy" | "calibrated")."""
    import config  # late import: config must stay dependency-free

    mode = getattr(config, "CONFIDENCE_MODE", "legacy")
    model = _model_cache.get(mode)
    if model is None:
        if mode == "calibrated":
            model = CalibratedConfidence()
        elif mode == "legacy":
            model = LegacyConfidence()
        else:
            raise ValueError(f"CONFIDENCE_MODE must be 'legacy' or 'calibrated', got {mode!r}")
        _model_cache[mode] = model
    return model


def reset_confidence_model():
    """Drop cached models (tests / after config or artifact changes)."""
    _model_cache.clear()
