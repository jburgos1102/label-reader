"""
export_training_data.py
-----------------------
Reads annotated labels from label_storage.db and converts them to a
HuggingFace-compatible BIO-labeled dataset for NER training.

Ground truth fields → BIO entity types:
  recipient_name  → NAME
  street_address  → STREET
  city            → CITY
  state           → STATE
  zip_code        → ZIP
  tracking_number → TRACKING
  carrier         → CARRIER

Output: ml/training_data/dataset.json  (one JSON object per line)

Usage:
  python ml/export_training_data.py
  python ml/export_training_data.py --db /path/to/label_storage.db
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent.parent / "label_storage.db"
OUTPUT_DIR = Path(__file__).parent / "training_data"

FIELD_TO_ENTITY = {
    "recipient_name":  "NAME",
    "street_address":  "STREET",
    "city":            "CITY",
    "state":           "STATE",
    "zip_code":        "ZIP",
    "tracking_number": "TRACKING",
    "carrier":         "CARRIER",
}

# Fields where an I- continuation tag exists
HAS_CONTINUATION = {"NAME", "STREET", "CITY", "TRACKING"}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def load_annotated_labels(db_path: Path) -> list[dict]:
    """
    Load all annotated labels, deduplicated by tracking_number
    (most recent scan per label wins — same logic as evaluate.py --latest-only).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM labels WHERE ground_truth IS NOT NULL ORDER BY processed_at DESC"
    ).fetchall()
    conn.close()

    seen: dict[str, dict] = {}
    for row in rows:
        tn = row["tracking_number"] or row["id"]
        if tn not in seen:
            seen[tn] = dict(row)

    return list(seen.values())


# ---------------------------------------------------------------------------
# Tokenizer (word-level, preserving punctuation as separate tokens)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9#&'.-]+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    """Split OCR text into word-level tokens."""
    return _TOKEN_RE.findall(text)


# ---------------------------------------------------------------------------
# Span alignment: find where a ground-truth value sits in the token list
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Lowercase and strip non-alphanumeric for fuzzy comparison."""
    return re.sub(r"[^\w]", "", s).lower()


def find_token_span(tokens: list[str], value: str) -> tuple[int, int] | None:
    """
    Return (start, end) token indices (exclusive end) where `value` appears
    in `tokens`.  Tries exact match first, then case-insensitive, then
    punctuation-stripped fuzzy match.
    Returns None if not found.
    """
    value = value.strip()
    if not value:
        return None

    value_tokens = tokenize(value)
    n = len(value_tokens)
    if n == 0:
        return None

    # Pass 1: exact
    for i in range(len(tokens) - n + 1):
        if tokens[i : i + n] == value_tokens:
            return (i, i + n)

    # Pass 2: case-insensitive
    tokens_up = [t.upper() for t in tokens]
    value_up  = [t.upper() for t in value_tokens]
    for i in range(len(tokens_up) - n + 1):
        if tokens_up[i : i + n] == value_up:
            return (i, i + n)

    # Pass 3: punctuation-stripped fuzzy
    tokens_norm = [_normalize(t) for t in tokens]
    value_norm  = [_normalize(t) for t in value_tokens]
    # Drop empty norm tokens that result from pure-punctuation value tokens
    value_norm = [v for v in value_norm if v]
    if not value_norm:
        return None
    n2 = len(value_norm)
    for i in range(len(tokens_norm) - n2 + 1):
        if tokens_norm[i : i + n2] == value_norm:
            return (i, i + n2)

    return None


# ---------------------------------------------------------------------------
# Build one training example from a DB row
# ---------------------------------------------------------------------------

def build_example(row: dict) -> dict | None:
    """
    Returns a dict with:
      id     : label DB id
      tokens : list[str]
      labels : list[str]  (BIO tags, same length as tokens)
    Returns None if the row has no usable OCR text.
    """
    ocr_text = (row.get("ocr_text") or "").strip()
    if not ocr_text:
        return None

    try:
        ground_truth = json.loads(row["ground_truth"])
    except (json.JSONDecodeError, TypeError):
        return None

    tokens = tokenize(ocr_text)
    if not tokens:
        return None

    labels = ["O"] * len(tokens)

    # Sort fields so longer values are matched first (avoids partial overlaps)
    fields_sorted = sorted(
        FIELD_TO_ENTITY.items(),
        key=lambda kv: len(str(ground_truth.get(kv[0]) or "")),
        reverse=True,
    )

    for field, entity in fields_sorted:
        value = str(ground_truth.get(field) or "").strip()
        if not value:
            continue

        span = find_token_span(tokens, value)
        if span is None:
            continue  # Value not found in OCR — skip this field for this example

        start, end = span
        for i in range(start, end):
            # Don't overwrite an already-labeled token
            if labels[i] != "O":
                continue
            if i == start:
                labels[i] = f"B-{entity}"
            elif entity in HAS_CONTINUATION:
                labels[i] = f"I-{entity}"
            else:
                labels[i] = f"B-{entity}"  # Single-token entities repeated

    return {
        "id":     row.get("id", ""),
        "tokens": tokens,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export training data from label_storage.db")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to label_storage.db")
    parser.add_argument("--out", default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    db_path  = Path(args.db)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations from {db_path} ...")
    rows = load_annotated_labels(db_path)
    print(f"  {len(rows)} unique annotated labels found")

    examples = []
    skipped  = 0
    unmatched_fields: dict[str, int] = {f: 0 for f in FIELD_TO_ENTITY}

    for row in rows:
        ex = build_example(row)
        if ex is None:
            skipped += 1
            continue

        # Count how many fields were NOT matched (for diagnostics)
        try:
            gt = json.loads(row["ground_truth"])
        except Exception:
            gt = {}
        for field, entity in FIELD_TO_ENTITY.items():
            value = str(gt.get(field) or "").strip()
            if value:
                tag = f"B-{entity}"
                if tag not in ex["labels"]:
                    unmatched_fields[field] += 1

        examples.append(ex)

    out_file = out_dir / "dataset.jsonl"
    with open(out_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nExport complete → {out_file}")
    print(f"  Examples written : {len(examples)}")
    print(f"  Skipped (no OCR) : {skipped}")
    print(f"\nField match rate (how often ground truth value was found in OCR text):")
    for field in FIELD_TO_ENTITY:
        total   = sum(1 for r in rows if str(json.loads(r["ground_truth"]).get(field) or "").strip())
        missed  = unmatched_fields[field]
        matched = total - missed
        pct     = (matched / total * 100) if total else 0
        print(f"  {field:<20} {matched:>3}/{total:<3}  ({pct:.0f}%)")

    if len(examples) < 50:
        print(
            f"\n⚠️  Only {len(examples)} training examples. "
            "Model accuracy will be low until you reach ~200. "
            "Keep annotating — the pipeline is ready to consume more data."
        )
    elif len(examples) < 200:
        print(f"\n⚡ {len(examples)} examples — model will start showing improvement. Target: 200+")
    else:
        print(f"\n✅ {len(examples)} examples — sufficient for meaningful training.")


if __name__ == "__main__":
    main()
