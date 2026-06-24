import pytesseract
import re


_last_ocr_diagnostics = {
    "selected_text": "",
    "rotations": {},
}


def get_last_ocr_diagnostics():
    """Return a copy of the rotation texts from the most recent OCR call."""
    return {
        "selected_text": _last_ocr_diagnostics["selected_text"],
        "rotations": dict(_last_ocr_diagnostics["rotations"]),
    }


def get_best_ocr_text(image):
    global _last_ocr_diagnostics

    rotations = [0, 90, 180, 270]

    best_text = ""
    best_score = -1
    rotation_texts = {}

    for degrees in rotations:
        rotated_image = image.rotate(degrees, expand=True)
        text = pytesseract.image_to_string(rotated_image)
        rotation_texts[degrees] = text

        score = 0

        upper_text = text.upper()

        if "DELIVER TO" in upper_text:
            score += 5

        if "WARMINSTER" in upper_text:
            score += 3

        if "BAIRD" in upper_text:
            score += 3

        if "UNIUNI" in upper_text:
            score += 2

        if re.search(r"\b[A-Z]{2}\s+\d{5}", text):
            score += 5

        if re.search(r"\b[A-Z]{2}\s+\d{5}[-–— ]?\d{4}", text):
            score += 5

        if "TRACKING" in upper_text:
            score += 2

        if "USPS" in upper_text:
            score += 1

        print(f"OCR ROTATION {degrees} SCORE:", score)
        print(text[:200])
        print("---")

        if score > best_score:
            best_score = score
            best_text = text

    _last_ocr_diagnostics = {
        "selected_text": best_text,
        "rotations": rotation_texts,
    }

    return best_text
