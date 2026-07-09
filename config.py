import os

# OpenAI settings
OPENAI_MODEL = "gpt-5"
OPENAI_TIMEOUT = 30.0

# Groq settings (used when GROQ_API_KEY is set; takes priority over OpenAI)
GROQ_TEXT_MODEL = "llama-3.1-8b-instant"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_TIMEOUT = 30.0

# Confidence model for candidate confidence values.
#   "legacy"     — historical heuristic constants (current behavior)
#   "calibrated" — measured P(correct) from calibration/confidence_table.json
# Flipping to "calibrated" changes reported confidence values (not selected
# values/sources) and requires regenerating the golden parity file — do not
# flip without review.
CONFIDENCE_MODE = "legacy"

# Rule-based per-field confidence values
TRACKING_MIN_LENGTH = 15
CONFIDENCE_TRACKING_HIGH = 0.95
CONFIDENCE_TRACKING_LOW = 0.40
CONFIDENCE_NAME_HIGH = 0.85
CONFIDENCE_NAME_LOW = 0.45
CONFIDENCE_STREET_HIGH = 0.70
CONFIDENCE_STREET_LOW = 0.45
CONFIDENCE_CITY_HIGH = 0.85
CONFIDENCE_CITY_LOW = 0.45
CONFIDENCE_STATE_VALID = 0.95
CONFIDENCE_ZIP_OCR = (
    0.78  # 5-digit format match only; single misread digit still passes
)

# LLM cross-validation against OCR text
CONFIDENCE_LLM_OCR_MATCH = 0.85
CONFIDENCE_LLM_OCR_MISMATCH = (
    0.75  # value absent from OCR but plausible (vision reads image)
)

# SQLite storage
STORAGE_DB_PATH = "label_storage.db"
CAPTURES_DIR = "captures"  # saved label images, named {label_id}.jpg

# Camera label detection
CAMERA_MIN_AREA_RATIO = 0.10
CAMERA_STABLE_SECONDS = 2.0
CAMERA_HISTORY_FRAMES = 10
CAMERA_HISTORY_THRESHOLD = 3

# LLM modes /api/scan callers may request. Default is strict: only "off".
# Widen to {"off", "auto", "force_vision"} to allow AI-assisted camera scans.
# Kill switch matters because /api/scan is currently unauthenticated —
# allowing "auto"/"force_vision" lets any caller trigger paid LLM calls.
API_LLM_MODES_ALLOWED = {"off", "auto", "force_vision"}

# Vision LLM trigger thresholds
OCR_CONFIDENCE_VISION_THRESHOLD = (
    60  # Tesseract mean confidence (0–100); below → use vision
)
OCR_TEXT_LENGTH_VISION_THRESHOLD = 50  # chars; shorter OCR output likely needs vision
VISION_TRIGGER_BLANK_FIELDS = 3  # blank/zero-confidence address fields → use vision
CONFIDENCE_TRACKING_CHECKSUM_FAIL = 0.30  # tracking number failed checksum validation

# OCR early-exit threshold — stop rotation loop once this confidence is reached
OCR_CONFIDENCE_EARLY_EXIT = 75

# OCR image size cap — only resizes truly enormous captures (e.g. >4500px)
OCR_MAX_IMAGE_PX = 4500  # trigger: longest edge must exceed this to resize
OCR_TARGET_IMAGE_PX = 4032  # target longest edge after proportional resize

# NER shadow candidate source. Disabled by default; when enabled (env
# NER_ENABLED=true), the DistilBERT NER model emits shadow candidates with
# source="ner" that the legacy Selector IGNORES — they are persisted for
# measurement, and can only affect selected output for recipient_name when
# NER_NAME_SELECTION_ENABLED is also set (see below). Missing model files
# log one warning and disable the source for the process (never fail a scan).
NER_ENABLED = os.getenv("NER_ENABLED", "").strip().lower() == "true"
# Anchored to the repository root: the model must load regardless of the
# process working directory (a relative path made NER silently latch off
# for the process lifetime when the app was launched from elsewhere).
NER_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml", "models")
NER_MAX_TOKENS = 256  # subword cap per inference; matches training max_length regime

# NER recipient_name selection (NerNamePolicy). Requires NER_ENABLED too —
# this flag only controls whether the Selector may USE ner candidates, and
# only ever for recipient_name; every other field is structurally untouched
# (selection.NerNamePolicy returns the legacy decision before any NER logic).
# Rollback: unset this env var -> exact shadow-mode behavior returns.
NER_NAME_SELECTION_ENABLED = (
    os.getenv("NER_NAME_SELECTION_ENABLED", "").strip().lower() == "true"
)
# Gate thresholds chosen by the ner_backtest.py sweep (33 held-out real-scan
# rows): override 0.80 was the zero-loss point (81.8% vs 51.5% base); fill
# applies to blank selections (~0% baseline) so it stays permissive.
# Provisional until eval_candidates.py accumulates fresh ner rows.
NER_NAME_FILL_MIN_CONFIDENCE = 0.50
NER_NAME_OVERRIDE_MIN_CONFIDENCE = 0.80
