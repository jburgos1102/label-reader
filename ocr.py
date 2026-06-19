import pytesseract
import re


def get_best_ocr_text(image):
    rotations = [0, 90, 180, 270]

    best_text = ""
    best_score = -1

    for degrees in rotations:
        rotated_image = image.rotate(degrees, expand=True)
        text = pytesseract.image_to_string(rotated_image)

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

    return best_text
