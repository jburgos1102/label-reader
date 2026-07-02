"""
import_datasets.py
-----------------
Imports pre-labeled images from datasets/ into label_storage.db.

Each carrier subfolder must have:
  datasets/<carrier>/images/<name>.jpeg   — the label image
  datasets/<carrier>/expected/<name>.json — the ground truth

Runs the project's own OCR (ocr.py) on each image, then inserts a row
into label_storage.db with ground_truth set from the expected JSON.
After running this, all dataset images are available to the training
pipeline (export_training_data.py) without any other changes.

Run from the project root:
  python ml/import_datasets.py
  python ml/import_datasets.py --datasets ./datasets --dry-run

Skips images whose tracking_number already exists in label_storage.db
so it is safe to re-run.
"""

import argparse
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Paths (relative to project root, where this script is run from)
# ---------------------------------------------------------------------------

PROJECT_ROOT  = Path(__file__).parent.parent
DATASETS_DIR  = PROJECT_ROOT / "datasets"
DB_PATH       = PROJECT_ROOT / "label_storage.db"

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".JPG", ".PNG", ".JPEG"}

GROUND_TRUTH_FIELDS = [
    "recipient_name",
    "street_address",
    "city",
    "state",
    "zip_code",
    "tracking_number",
    "carrier",
]


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_existing_tracking_numbers(conn) -> set:
    rows = conn.execute("SELECT tracking_number FROM labels WHERE tracking_number IS NOT NULL").fetchall()
    return {r[0] for r in rows}


def insert_dataset_label(conn, *, label_id, filename, ocr_text, ocr_confidence,
                          ocr_rotations, ground_truth: dict):
    """
    Insert a dataset label directly into label_storage.db.
    Field values come from ground_truth; confidence=1.0, source='dataset'
    for all fields since these are human-verified expected outputs.
    ground_truth is also stored as the annotation JSON so evaluate.py
    and export_training_data.py both pick it up.
    """
    now = datetime.now(timezone.utc).isoformat()

    def val(field):
        return ground_truth.get(field, "") or ""

    conn.execute("""
        INSERT OR IGNORE INTO labels (
            id, processed_at, original_filename,
            ocr_text, ocr_confidence, ocr_rotations,
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
            ?, 1.0, 'dataset',
            ?, 1.0, 'dataset',
            ?, 1.0, 'dataset',
            ?, 1.0, 'dataset',
            ?, 1.0, 'dataset',
            ?, 1.0, 'dataset',
            ?, 1.0, 'dataset',
            0, 0, ?
        )
    """, [
        label_id, now, filename,
        ocr_text, ocr_confidence, ocr_rotations,
        val("recipient_name"),
        val("street_address"),
        val("city"),
        val("state"),
        val("zip_code"),
        val("tracking_number"),
        val("carrier"),
        json.dumps({k: val(k) for k in GROUND_TRUTH_FIELDS}),
    ])


# ---------------------------------------------------------------------------
# OCR (uses project's own ocr.py so behaviour matches production)
# ---------------------------------------------------------------------------

def run_ocr(image_path: Path):
    """Returns (ocr_text, ocr_confidence, rotations_tried)."""
    try:
        # Import project OCR module (run from project root)
        from ocr import get_best_ocr_text
        image = Image.open(image_path).convert("RGB")
        return get_best_ocr_text(image)
    except ImportError:
        # Fallback: bare pytesseract if ocr.py not importable
        import pytesseract
        image = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(image)
        return text, 0.0, 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Import datasets/ into label_storage.db")
    parser.add_argument("--datasets", default=str(DATASETS_DIR), help="Path to datasets/ folder")
    parser.add_argument("--db",       default=str(DB_PATH),      help="Path to label_storage.db")
    parser.add_argument("--dry-run",  action="store_true",        help="Show what would be imported without writing")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets)
    db_path      = Path(args.db)

    if not datasets_dir.exists():
        print(f"datasets/ folder not found at {datasets_dir}")
        sys.exit(1)

    if not db_path.exists():
        print(f"label_storage.db not found at {db_path}")
        print("Start the Flask app once to initialize the database, then re-run.")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    existing = get_existing_tracking_numbers(conn)

    imported  = 0
    skipped   = 0
    no_json   = 0
    errors    = 0

    # Walk datasets/<carrier>/images/<file>
    for carrier_dir in sorted(datasets_dir.iterdir()):
        if not carrier_dir.is_dir() or carrier_dir.name.startswith("."):
            continue

        images_dir   = carrier_dir / "images"
        expected_dir = carrier_dir / "expected"

        if not images_dir.exists():
            continue

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix not in IMAGE_SUFFIXES:
                continue

            # Find corresponding expected JSON (same stem, .json extension)
            json_path = expected_dir / (image_path.stem + ".json")
            if not json_path.exists():
                print(f"  [no json]  {image_path.name}")
                no_json += 1
                continue

            # Load ground truth
            try:
                ground_truth = json.loads(json_path.read_text())
            except json.JSONDecodeError as e:
                print(f"  [json err] {json_path.name}: {e}")
                errors += 1
                continue

            tracking = ground_truth.get("tracking_number", "")
            if tracking and tracking in existing:
                print(f"  [skip]     {image_path.name}  (tracking number already in DB)")
                skipped += 1
                continue

            if args.dry_run:
                print(f"  [would import] {image_path.name}  tracking={tracking}")
                imported += 1
                continue

            # Run OCR
            print(f"  [ocr]      {image_path.name} ...", end="", flush=True)
            try:
                ocr_text, ocr_conf, rotations = run_ocr(image_path)
                print(f" conf={ocr_conf:.1f} rotations={rotations}")
            except Exception as e:
                print(f" ERROR: {e}")
                errors += 1
                continue

            # Insert
            label_id = str(uuid.uuid4())
            try:
                insert_dataset_label(
                    conn,
                    label_id=label_id,
                    filename=image_path.name,
                    ocr_text=ocr_text,
                    ocr_confidence=ocr_conf,
                    ocr_rotations=rotations,
                    ground_truth=ground_truth,
                )
                conn.commit()
                existing.add(tracking)
                imported += 1
                print(f"             → imported as {label_id[:8]}...")
            except Exception as e:
                print(f"  [db err]   {image_path.name}: {e}")
                errors += 1

    conn.close()

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Results:")
    print(f"  Imported : {imported}")
    print(f"  Skipped  : {skipped}  (already in DB)")
    print(f"  No JSON  : {no_json}")
    print(f"  Errors   : {errors}")

    if imported and not args.dry_run:
        print(f"\nRe-run export_training_data.py to include these in the training dataset.")


if __name__ == "__main__":
    main()
