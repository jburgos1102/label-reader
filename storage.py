import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import config


_DB_PATH = Path(config.STORAGE_DB_PATH)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS labels (
    id                          TEXT PRIMARY KEY,
    processed_at                TEXT NOT NULL,
    ocr_text                    TEXT,
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
    ground_truth                TEXT
)
"""

_INSERT = """
INSERT OR REPLACE INTO labels (
    id, processed_at, ocr_text,
    recipient_name,   recipient_name_confidence,   recipient_name_source,
    street_address,   street_address_confidence,   street_address_source,
    city,             city_confidence,             city_source,
    state,            state_confidence,            state_source,
    zip_code,         zip_code_confidence,         zip_code_source,
    tracking_number,  tracking_number_confidence,  tracking_number_source,
    carrier,          carrier_confidence,          carrier_source,
    llm_called, processing_ms, ground_truth
) VALUES (
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
    """Create the labels table if it does not already exist."""
    with _connect() as conn:
        conn.execute(_CREATE_TABLE)


def store(result, ocr_text=""):
    """Persist an ExtractionResult row and return its label_id."""
    d = result.to_dict()
    ex = d["extracted"]
    meta = d["metadata"]
    now = datetime.now(timezone.utc).isoformat()

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
                d["label_id"],
                now,
                ocr_text,
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

    return d["label_id"]
