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
