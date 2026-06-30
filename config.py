# OpenAI settings
OPENAI_MODEL = "gpt-5"
OPENAI_TIMEOUT = 30.0

# Groq settings (used when GROQ_API_KEY is set; takes priority over OpenAI)
GROQ_TEXT_MODEL = "llama-3.1-8b-instant"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_TIMEOUT = 30.0

# When the overall rule-based confidence falls below this threshold, call the LLM.
LLM_CONFIDENCE_THRESHOLD = 0.70

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
CONFIDENCE_ZIP_VALID = 0.95   # reserved for barcode-sourced zip (future)
CONFIDENCE_ZIP_OCR = 0.78     # 5-digit format match only; single misread digit still passes

# LLM cross-validation against OCR text
CONFIDENCE_LLM_OCR_MATCH = 0.85
CONFIDENCE_LLM_OCR_MISMATCH = 0.30

# SQLite storage
STORAGE_DB_PATH = "label_storage.db"

# Camera label detection
CAMERA_MIN_AREA_RATIO = 0.10
CAMERA_STABLE_SECONDS = 2.0
CAMERA_HISTORY_FRAMES = 10
CAMERA_HISTORY_THRESHOLD = 3
