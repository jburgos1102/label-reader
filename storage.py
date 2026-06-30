import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import config


_DB_PATH = Path(config.STORAGE_DB_PATH)
_CAPTURES_DIR = Path(config.CAPTURES_DIR)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS labels (
    id                          TEXT PRIMARY KEY,
    processed_at                TEXT NOT NULL,
    original_filename           TEXT,
    image_path                  TEXT,
    ocr_text                    TEXT,
    ocr_confidence               REAL,
    ocr_rotations                INTEGER,
    barcode_raw                  TEXT,
    recipient_name              TEXT,
    recipient_name_confidence   REAL,
    recipient_name_source       TEXT,
    street_address              TEXT,
    street_address_confidence   REAL,
    street_address_source       TEXT,
    city                        TEXT,
    city_confidence             REAL,
    city_source                 TEXT,
    state                       TEXT,
    state_confidence            REAL,
    state_source                TEXT,
    zip_code                    TEXT,
    zip_code_confidence         REAL,
    zip_code_source             TEXT,
    tracking_number             TEXT,
    tracking_number_confidence  REAL,
    tracking_number_source      TEXT,
    carrier                     TEXT,
    carrier_confidence          REAL,
    carrier_source              TEXT,
    llm_called                  INTEGER,
    processing_ms               INTEGER,
    ground_truth                TEXT,
    corrections                 TEXT
)
"""

# Columns added after the original schema shipped. CREATE TABLE IF NOT EXISTS
# only applies to brand-new database files, so existing databases are
# migrated forward by adding any of these columns that aren't present yet.
_MIGRATED_COLUMNS = {
    "original_filename": "TEXT",
    "image_path": "TEXT",
    "ocr_confidence": "REAL",
    "ocr_rotations": "INTEGER",
    "barcode_raw": "TEXT",
    "corrections": "TEXT",
}

_INSERT = """
INSERT OR REPLACE INTO labels (
    id, processed_at, original_filename, image_path, ocr_text,
    ocr_confidence, ocr_rotations, barcode_raw,
    recipient_name,   recipient_name_confidence,   recipient_name_source,
    street_address,   street_address_confidence,   street_address_source,
    city,             city_confidence,             city_source,
    state,            state_confidence,            state_source,
    zip_code,         zip_code_confidence,         zip_code_source,
    tracking_number,  tracking_number_confidence,  tracking_number_source,
    carrier,          carrier_confidence,          carrier_source,
    llm_called, processing_ms, ground_truth
) VALUES (
    ?, ?, ?, ?, ?,
    ?, ?, ?,
    ?, ?, ?,
    ?, ?, ?,
    ?, ?, ?,
    ?, ?, ?,
    ?, ?, ?,
    ?, ?, ?,
    ?, ?, ?,
    ?, ?, NULL
)
"""


def _connect():
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the labels table if it does not already exist, and migrate
    in any columns added since the original schema shipped."""
    with _connect() as conn:
        conn.execute(_CREATE_TABLE)
        existing_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(labels)")
        }
        for column, col_type in _MIGRATED_COLUMNS.items():
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE labels ADD COLUMN {column} {col_type}")


def store(
    result,
    ocr_text="",
    image_bytes=None,
    original_filename=None,
    barcode_raw="",
    ocr_confidence=None,
    ocr_rotations=None,
):
    """Persist an ExtractionResult row and return its label_id.

    If image_bytes is provided, the image is saved to
    captures/{label_id}.jpg and its path is stored alongside the row.
    """
    d = result.to_dict()
    ex = d["extracted"]
    meta = d["metadata"]
    now = datetime.now(timezone.utc).isoformat()
    label_id = d["label_id"]

    image_path = None
    if image_bytes:
        _CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        image_path = str(_CAPTURES_DIR / f"{label_id}.jpg")
        with open(image_path, "wb") as f:
            f.write(image_bytes)

    def _v(field):
        return ex[field]["value"]

    def _c(field):
        return ex[field]["confidence"]

    def _s(field):
        return ex[field]["source"]

    with _connect() as conn:
        conn.execute(
            _INSERT,
            [
                label_id,
                now,
                original_filename,
                image_path,
                ocr_text,
                ocr_confidence,
                ocr_rotations,
                barcode_raw,
                _v("recipient_name"),  _c("recipient_name"),  _s("recipient_name"),
                _v("street_address"),  _c("street_address"),  _s("street_address"),
                _v("city"),            _c("city"),            _s("city"),
                _v("state"),           _c("state"),           _s("state"),
                _v("zip_code"),        _c("zip_code"),        _s("zip_code"),
                _v("tracking_number"), _c("tracking_number"), _s("tracking_number"),
                _v("carrier"),         _c("carrier"),         _s("carrier"),
                1 if meta["llm_called"] else 0,
                meta["processing_ms"],
            ],
        )

    return label_id


def get_label(label_id):
    """Return a single label row as a dict, or None if not found.

    ground_truth and corrections are parsed from JSON into Python objects.
    """
    with _connect() as conn:
        row = conn.execute("SELECT * FROM labels WHERE id = ?", (label_id,)).fetchone()

    if row is None:
        return None

    data = dict(row)
    for json_field in ("ground_truth", "corrections"):
        raw = data.get(json_field)
        data[json_field] = json.loads(raw) if raw else None

    return data
