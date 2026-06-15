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

        if re.search(r"\b[A-Z]{2}\s+\d{5}", text):
            score += 5

        if re.search(r"\b[A-Z]{2}\s+\d{5}[-–— ]?\d{4}", text):
            score += 5

        if "TRACKING" in text.upper():
            score += 2

        if "USPS" in text.upper():
            score += 1

        if score > best_score:
            best_score = score
            best_text = text

    return best_text


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

    used_deliver_to_block = False

    for line_index, line in enumerate(lines):
        if "DELIVER TO" in line.upper():
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

    return label_data


if __name__ == "__main__":
    image_path = "images/USPS_Shipping_Label.JPG"
    label_data = extract_label_data(image_path)
    print(label_data)
