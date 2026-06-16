from PIL import Image
from pyzbar.pyzbar import decode
import pytesseract
import re


def extract_tracking_number(image):
    rotations = [0, 90, 180, 270]
    min_tracking_length = 15

    for degrees in rotations:
        rotated_image = image.rotate(degrees, expand=True)
        barcodes = decode(rotated_image)

        if barcodes:
            print(f"\n--- BARCODES FOUND AT ROTATION {degrees} ---")

            for barcode_index, barcode in enumerate(barcodes):
                barcode_data = barcode.data.decode("utf-8")
                print(barcode_index, repr(barcode_data), barcode.type)

                barcode_parts = barcode_data.split("\x1d")

                if len(barcode_parts) > 1:
                    candidate = barcode_parts[1]
                else:
                    candidate = barcode_data

                print("TRACKING CANDIDATE:", repr(candidate))

                if len(candidate) >= min_tracking_length:
                    return candidate

    return ""


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


# --- NORMALIZATION FUNCTION ---
def normalize_extracted_fields(label_data):
    street_address = label_data.get("street_address", "")
    tracking_number = label_data.get("tracking_number", "")

    if street_address:
        street_address = street_address.strip()
        street_address = re.sub(r"\s+", " ", street_address)

        street_address = re.sub(
            r"\b(BE|8T|5T)\b$",
            "ST",
            street_address,
            flags=re.IGNORECASE,
        )

        street_address = re.sub(
            r"\b(st|rd|ave|blvd|dr|ln|ct)\b$",
            lambda match: match.group().upper(),
            street_address,
            flags=re.IGNORECASE,
        )

        label_data["street_address"] = street_address

    if tracking_number:
        tracking_number = tracking_number.strip()
        tracking_number = tracking_number.replace(" ", "")
        tracking_number = (
            tracking_number.replace("O", "0")
            if tracking_number.startswith("1LSDO")
            else tracking_number
        )
        label_data["tracking_number"] = tracking_number

    return label_data


# --- SCORING FUNCTION ---
def score_label_data(label_data):
    confidence = {
        "recipient_name": 0.0,
        "street_address": 0.0,
        "city": 0.0,
        "state": 0.0,
        "zip_code": 0.0,
        "tracking_number": 0.0,
        "overall": 0.0,
    }
    warnings = []

    tracking_number = label_data.get("tracking_number", "")
    recipient_name = label_data.get("recipient_name", "")
    street_address = label_data.get("street_address", "")
    city = label_data.get("city", "")
    state = label_data.get("state", "")
    zip_code = label_data.get("zip_code", "")

    if tracking_number:
        if len(tracking_number) >= 15:
            confidence["tracking_number"] = 0.95
        else:
            confidence["tracking_number"] = 0.40
            warnings.append("tracking_number_short")
    else:
        warnings.append("tracking_number_missing")

    if recipient_name:
        if re.fullmatch(r"[A-Za-z][A-Za-z .'-]+", recipient_name):
            confidence["recipient_name"] = 0.85
        else:
            confidence["recipient_name"] = 0.45
            warnings.append("recipient_name_low_confidence")
    else:
        warnings.append("recipient_name_missing")

    if street_address:
        if re.search(r"\d", street_address):
            confidence["street_address"] = 0.85
        else:
            confidence["street_address"] = 0.45
            warnings.append("street_address_missing_number")
    else:
        warnings.append("street_address_missing")

    if city:
        if re.fullmatch(r"[A-Za-z .'-]+", city):
            confidence["city"] = 0.85
        else:
            confidence["city"] = 0.45
            warnings.append("city_low_confidence")
    else:
        warnings.append("city_missing")

    if re.fullmatch(r"[A-Z]{2}", state):
        confidence["state"] = 0.95
    else:
        warnings.append("state_missing_or_invalid")

    if re.fullmatch(r"\d{5}(-\d{4})?", zip_code):
        confidence["zip_code"] = 0.95
    else:
        warnings.append("zip_code_missing_or_invalid")

    field_scores = [
        confidence["recipient_name"],
        confidence["street_address"],
        confidence["city"],
        confidence["state"],
        confidence["zip_code"],
        confidence["tracking_number"],
    ]

    confidence["overall"] = round(sum(field_scores) / len(field_scores), 2)

    label_data["confidence"] = confidence
    label_data["warnings"] = warnings

    return label_data


def extract_label_data(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")

    label_data = {
        "recipient_name": "",
        "street_address": "",
        "city": "",
        "state": "",
        "zip_code": "",
        "tracking_number": "",
    }

    label_data["tracking_number"] = extract_tracking_number(image)

    text = get_best_ocr_text(image)
    lines = []

    for line in text.splitlines():
        cleaned_line = line.strip()

        if cleaned_line:
            lines.append(cleaned_line)

    print("\n--- RAW OCR TEXT ---")
    print(text)

    print("\n--- OCR LINES ---")
    for line_index, line in enumerate(lines):
        print(line_index, repr(line))

    if not label_data["tracking_number"]:
        for line in lines:
            clean_tracking_line = line.replace("|", "").strip()
            tracking_match = re.fullmatch(r"[A-Z0-9]{14,20}", clean_tracking_line)

            if tracking_match:
                tracking_candidate = tracking_match.group()

                if tracking_candidate.startswith("1BA"):
                    tracking_candidate = "TBA" + tracking_candidate[3:]

                print("OCR TRACKING CANDIDATE:", repr(tracking_candidate))

                label_data["tracking_number"] = tracking_candidate
                break

    used_deliver_to_block = False

    for line_index, line in enumerate(lines):
        if "DELIVER TO" in line.upper():

            if "USPS" not in line.upper():
                print(f"\nFOUND NON-USPS DELIVER TO AT LINE {line_index}")

                for nearby_index in range(line_index, min(line_index + 6, len(lines))):
                    print(nearby_index, repr(lines[nearby_index]))

                if line_index + 4 < len(lines):
                    uniuni_city_state_line = lines[line_index + 2]
                    uniuni_street_line = lines[line_index + 3]
                    uniuni_zip_line = lines[line_index + 4]

                    print("UNIUNI CITY/STATE CANDIDATE:", repr(uniuni_city_state_line))
                    print("UNIUNI STREET CANDIDATE:", repr(uniuni_street_line))
                    print("UNIUNI ZIP CANDIDATE:", repr(uniuni_zip_line))

                    uniuni_city_state_match = re.search(
                        r"([A-Za-z]+),?\s+([A-Z]{2}),?",
                        uniuni_city_state_line,
                    )
                    uniuni_street_match = re.match(r"\d+", uniuni_street_line)
                    uniuni_zip_match = re.search(r"\d{5}", uniuni_zip_line)

                    print("UNIUNI CITY/STATE MATCH:", bool(uniuni_city_state_match))
                    print("UNIUNI STREET MATCH:", bool(uniuni_street_match))
                    print("UNIUNI ZIP MATCH:", bool(uniuni_zip_match))

                    if (
                        uniuni_city_state_match
                        and uniuni_street_match
                        and uniuni_zip_match
                    ):
                        label_data["street_address"] = uniuni_street_line
                        label_data["city"] = uniuni_city_state_match.group(1)
                        label_data["state"] = uniuni_city_state_match.group(2)
                        label_data["zip_code"] = uniuni_zip_match.group()

                        used_deliver_to_block = True

            print(f"\nFOUND DELIVER TO AT LINE {line_index}")

            if line_index + 3 < len(lines):
                print("DELIVER TO HAS ENOUGH LINES")
                deliver_to_name = lines[line_index + 1]
                deliver_to_street = lines[line_index + 2]
                deliver_to_city_line = lines[line_index + 3]
                deliver_to_street = deliver_to_street.replace(
                    "RETURN SERVICE REQUESTED", ""
                )
                deliver_to_street = deliver_to_street.strip()

                print("DELIVER TO NAME:", repr(deliver_to_name))
                print("DELIVER TO STREET:", repr(deliver_to_street))
                print("DELIVER TO CITY LINE:", repr(deliver_to_city_line))

                deliver_to_city_parts = deliver_to_city_line.split()

                print("DELIVER TO CITY PARTS:", deliver_to_city_parts)

                if len(deliver_to_city_parts) >= 3:
                    state_candidate = deliver_to_city_parts[-2]
                    zip_candidate = deliver_to_city_parts[-1]

                    state_match = re.fullmatch(r"[A-Z]{2}", state_candidate)
                    zip_match = re.fullmatch(r"\d{5}", zip_candidate)

                    print("STATE MATCH:", bool(state_match))
                    print("ZIP MATCH:", bool(zip_match))

                    if state_match and zip_match:
                        deliver_to_city = " ".join(deliver_to_city_parts[:-2])
                        deliver_to_city = deliver_to_city.strip(",")

                        label_data["recipient_name"] = deliver_to_name
                        label_data["street_address"] = deliver_to_street
                        label_data["city"] = deliver_to_city
                        label_data["state"] = state_candidate
                        label_data["zip_code"] = zip_candidate

                        used_deliver_to_block = True

        if used_deliver_to_block:
            continue

        parts = line.split()

        if "18974" in line:
            print("ZIP-FIRST PARTS:", parts)

        if len(parts) >= 6:
            zip_match = re.fullmatch(r"\d{5}", parts[0])
            separator_match = re.fullmatch(r"[-–—]", parts[1])
            plus_four_match = re.fullmatch(r"\d{4}", parts[2])
            state_match = re.fullmatch(r"[A-Z]{2}", parts[5])

            print("ZIP-FIRST ZIP:", bool(zip_match))
            print("ZIP-FIRST PLUS4:", bool(plus_four_match))
            print("ZIP-FIRST STATE:", bool(state_match))

            if zip_match and plus_four_match and state_match:
                recipient_name = lines[line_index - 3]
                street_address = lines[line_index - 1]
                city = parts[3]
                state = parts[5]
                full_zip = parts[0] + "-" + parts[2]

                label_data["recipient_name"] = recipient_name
                label_data["street_address"] = street_address
                label_data["city"] = city
                label_data["state"] = state
                label_data["zip_code"] = full_zip

        if len(parts) >= 4:
            zip_first_match = re.fullmatch(r"(\d{5})[-–—](\d{4})", parts[0])
            state_match = re.fullmatch(r"[A-Z]{2}", parts[3])

            print("ZIP-FIRST COMBINED ZIP:", bool(zip_first_match))
            print("ZIP-FIRST COMBINED STATE:", bool(state_match))

            if zip_first_match and state_match:
                recipient_name = lines[line_index - 2]
                street_address = lines[line_index - 1]
                city = parts[1]
                state = parts[3]
                full_zip = zip_first_match.group(1) + "-" + zip_first_match.group(2)

                label_data["recipient_name"] = recipient_name
                label_data["street_address"] = street_address
                label_data["city"] = city
                label_data["state"] = state
                label_data["zip_code"] = full_zip

        country_zip_match = re.fullmatch(
            r"(.+?),\s*([A-Z]{2}),\s*(?:US|USA),?\s*(\d{5})",
            line,
            re.IGNORECASE,
        )

        if country_zip_match:
            print("CITY/STATE/COUNTRY/ZIP MATCH:", country_zip_match.groups())

            if line_index - 2 >= 0:
                recipient_name = lines[line_index - 2]
                street_address = lines[line_index - 1]
                city = country_zip_match.group(1).strip()
                state = country_zip_match.group(2).upper()
                zip_code = country_zip_match.group(3)

                label_data["recipient_name"] = recipient_name
                label_data["street_address"] = street_address
                label_data["city"] = city
                label_data["state"] = state
                label_data["zip_code"] = zip_code

        for index, part in enumerate(parts):
            state_match = re.fullmatch(r"[A-Z]{2}", part)

            if state_match and index + 1 < len(parts):
                zip_part = parts[index + 1]

                zip_match = re.fullmatch(r"(\d{5})[-–—]?(\d{4})?", zip_part)

                if zip_match:
                    zip_code = zip_match.group(1)
                    zip_plus_four = zip_match.group(2)

                    if not zip_plus_four and index + 3 < len(parts):
                        possible_separator = parts[index + 2]
                        possible_plus_four = parts[index + 3]

                        separator_match = re.fullmatch(r"[-–—]", possible_separator)
                        plus_four_match = re.fullmatch(r"\d{4}", possible_plus_four)

                        if separator_match and plus_four_match:
                            zip_plus_four = possible_plus_four

                    city_parts = parts[:index]
                    clean_city_parts = []

                    for city_part in city_parts:
                        if any(character.isalnum() for character in city_part):
                            clean_city_parts.append(city_part)

                    city = " ".join(clean_city_parts)
                    city = city.strip(",")
                    state = part

                    street_address_line = lines[line_index - 1]
                    street_address_parts = street_address_line.split()

                    house_number_index = 0

                    for address_index, address_part in enumerate(street_address_parts):
                        if address_part.isdigit():
                            house_number_index = address_index
                            break

                    street_address = " ".join(street_address_parts[house_number_index:])

                    recipient_name_line = lines[line_index - 2]
                    recipient_name_parts = recipient_name_line.split()

                    recipient_name_index = 0

                    for name_index, name_part in enumerate(recipient_name_parts):
                        if name_part.isalpha():
                            recipient_name_index = name_index
                            break

                    clean_recipient_name_parts = []

                    for recipient_name_part in recipient_name_parts[
                        recipient_name_index:
                    ]:
                        if recipient_name_part.isalpha():
                            clean_recipient_name_parts.append(recipient_name_part)

                    recipient_name = " ".join(clean_recipient_name_parts)

                    if zip_plus_four:
                        full_zip = zip_code + "-" + zip_plus_four
                    else:
                        full_zip = zip_code

                    label_data["recipient_name"] = recipient_name
                    label_data["street_address"] = street_address
                    label_data["city"] = city
                    label_data["state"] = state
                    label_data["zip_code"] = full_zip

    label_data = normalize_extracted_fields(label_data)
    label_data = score_label_data(label_data)

    return label_data


if __name__ == "__main__":
    image_path = "images/USPS_Shipping_Label.JPG"
    label_data = extract_label_data(image_path)
    print(label_data)
