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

                candidate = clean_tracking_candidate(candidate)

                if is_valid_tracking_candidate(candidate):
                    return candidate

                barcode_digits = re.sub(r"\D", "", barcode_data)

                if len(barcode_digits) > 22:
                    fedex_candidate = barcode_digits[-12:]

                    if is_valid_tracking_candidate(fedex_candidate):
                        return fedex_candidate

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


# --- IDENTIFY CARRIER ---
def identify_carrier(tracking_number):
    tracking_number = tracking_number.upper()

    if tracking_number.startswith("TBA"):
        return "Amazon"

    if tracking_number.startswith("YWNJC"):
        return "YunExpress"

    if tracking_number.startswith("UUS"):
        return "UniUni"

    if tracking_number.startswith("DDIYS"):
        return "TikTok"

    if tracking_number.startswith("1LSD"):
        return "TikTok"

    if (
        tracking_number.isdigit()
        and tracking_number.startswith(("91", "92", "93", "94", "95"))
        and len(tracking_number) >= 20
    ):
        return "USPS"

    if (
        tracking_number.isdigit()
        and tracking_number.startswith("56")
        and len(tracking_number) >= 20
    ):
        return "USPS"

    if re.fullmatch(r"1Z[A-Z0-9]{16}", tracking_number):
        return "UPS"

    if tracking_number.isdigit() and len(tracking_number) in (12, 15, 20, 22):
        return "FEDEX"

    return "Unknown"


# --- TRACKING HELPER FUNCTIONS ---


def clean_tracking_candidate(candidate):
    candidate = candidate.upper()
    candidate = re.sub(r"[^A-Z0-9]", "", candidate)

    if candidate.startswith("1BA"):
        candidate = "TBA" + candidate[3:]

    if candidate.startswith("1LSDO"):
        candidate = candidate.replace("O", "0", 1)

    return candidate


def is_valid_tracking_candidate(candidate):
    if not candidate:
        return False

    candidate = clean_tracking_candidate(candidate)

    if len(candidate) < 12:
        return False

    if candidate.startswith(("HTTP", "AMZNTO", "WWW")):
        return False

    if candidate.startswith(("TBA", "YWNJC", "UUS", "DDIYS", "1LSD")):
        return True

    if re.fullmatch(r"1Z[A-Z0-9]{16}", candidate):
        return True

    if (
        candidate.isdigit()
        and candidate.startswith(("91", "92", "93", "94", "95"))
        and len(candidate) >= 20
    ):
        return True

    if candidate.isdigit() and candidate.startswith("56") and len(candidate) >= 20:
        return True

    if candidate.isdigit() and len(candidate) in (12, 15, 20, 22):
        return True

    return False


def is_valid_usps_tracking_candidate(candidate):
    candidate = clean_tracking_candidate(candidate)

    if (
        candidate.isdigit()
        and candidate.startswith(("91", "92", "93", "94", "95"))
        and len(candidate) >= 20
    ):
        return True

    if candidate.isdigit() and candidate.startswith("56") and len(candidate) >= 20:
        return True

    return False


def is_valid_usps_ocr_tracking_candidate(candidate):
    candidate = clean_tracking_candidate(candidate)

    if (
        candidate.isdigit()
        and candidate.startswith(("91", "92", "93", "94", "95"))
        and len(candidate) in (22, 26, 34)
    ):
        return True

    if (
        candidate.isdigit()
        and candidate.startswith("56")
        and len(candidate) in (22, 26, 34)
    ):
        return True

    return False


def extract_usps_tracking_candidates_from_text(text):
    candidates = []

    for numeric_match in re.findall(r"(?:\d[\s_\-]*){18,34}", text):
        candidate = clean_tracking_candidate(numeric_match)

        if is_valid_usps_ocr_tracking_candidate(candidate):
            candidates.append(candidate)

    return candidates


def extract_tracking_from_ocr_lines(lines):
    for line_index, line in enumerate(lines):
        upper_line = line.upper()

        if "USPS" not in upper_line or "TRACKING" not in upper_line:
            continue

        nearby_lines = lines[line_index + 1 : min(line_index + 6, len(lines))]

        # Prefer a valid USPS number on one OCR line before combining nearby
        # lines, so trailing noise like "0 0000000" is not appended.
        for nearby_line in nearby_lines:
            for candidate in extract_usps_tracking_candidates_from_text(nearby_line):
                return candidate

        candidate_text = line

        for nearby_line in nearby_lines:
            candidate_text += " " + nearby_line

            for candidate in extract_usps_tracking_candidates_from_text(candidate_text):
                return candidate

    for line_index, line in enumerate(lines):
        upper_line = line.upper()

        if "TRACKING" not in upper_line:
            continue

        candidate_text = line

        if line_index + 1 < len(lines):
            candidate_text += " " + lines[line_index + 1]

        if line_index + 2 < len(lines):
            candidate_text += " " + lines[line_index + 2]

        ups_match = re.search(
            r"1Z(?:[\s_-]*[A-Z0-9]){16}",
            candidate_text,
            re.IGNORECASE,
        )

        if ups_match:
            candidate = clean_tracking_candidate(ups_match.group())

            if is_valid_tracking_candidate(candidate):
                return candidate

        numeric_matches = re.findall(r"(?:\d[\s_\-]*){18,34}", candidate_text)

        for numeric_match in numeric_matches:
            candidate = clean_tracking_candidate(numeric_match)

            if (
                is_valid_tracking_candidate(candidate)
                and identify_carrier(candidate) != "FEDEX"
            ):
                return candidate

    for line in lines:
        candidate = clean_tracking_candidate(line)

        if (
            is_valid_tracking_candidate(candidate)
            and identify_carrier(candidate) != "FEDEX"
        ):
            return candidate

    return ""


def clean_address_service_text(value):
    value = re.sub(
        r"\b(?:CARRIER\s*[—-]\s*LEAVE IF NO RESPONSE|RETURN SERVICE REQUESTED|ADDRESS SERVICE REQUESTED|SERVICE REQUESTED)\b.*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def clean_address_ocr(value):
    value = clean_address_service_text(value)
    value = re.sub(r"(?<=\d)@(?=\d)", "0", value)
    value = re.sub(r"\b1@5\b", "105", value, flags=re.IGNORECASE)
    value = re.sub(r"\bAUB\b", "HUB", value, flags=re.IGNORECASE)
    value = re.sub(r"\bPRRLISLE\b", "CARLISLE", value, flags=re.IGNORECASE)
    value = re.sub(r"\b\\?IVERSITY\b", "UNIVERSITY", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def clean_parser_name(value):
    value = re.sub(r"^\s*(?:TO|SHIP)\s*:?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)

    return value.strip(" |")


def is_noise_recipient_line(value):
    # Recipient candidates come from noisy OCR near the address block; reject
    # label furniture and address-like text before accepting a line as a name.
    clean_value = clean_parser_name(value).upper()

    if not clean_value:
        return True

    if clean_value in (
        "SHIP",
        "TO",
        "PREMIUM",
        "USPS",
        "USA",
        "PARCEL SELECT",
        "PRIORITY MAIL",
        "MEDIA MAIL",
        "OR CURRENT OCCUPANT",
    ):
        return True

    if re.search(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", clean_value):
        return True

    if re.search(
        r"\b(?:EXPECTED DELIVERY|POSTAGE|ORIGIN|MAILED FROM|TRACKING|AMAZON|FULFILLMENT|SERVICES|PRIORITY MAIL|PARCEL SELECT)\b",
        clean_value,
    ):
        return True

    if re.search(
        r"\b(?:STREET|ST|ROAD|RD|AVE|AVENUE|BLVD|DR|LANE|LN|WAY)\b",
        clean_value,
    ) and re.search(r"\d", clean_value):
        return True

    if re.fullmatch(r"[\d\s./#-]+", clean_value):
        return True

    if len(clean_value) <= 2:
        return True

    if re.fullmatch(r"[|.=_\-— ]+", clean_value):
        return True

    return False


def choose_recipient_from_lines(lines, start_index, min_index=0):
    recipient_index = start_index

    while (
        recipient_index >= min_index
        and is_noise_recipient_line(lines[recipient_index])
    ):
        recipient_index -= 1

    if recipient_index < min_index:
        return ""

    recipient_name = clean_parser_name(lines[recipient_index])
    organization_match = re.search(
        r"\b(?:ALUMNI ASSOCIATIO|DEPT OF|DEPARTMENT OF|CURRENT OCCUPANT)\b",
        recipient_name,
        re.IGNORECASE,
    )

    if organization_match:
        previous_index = recipient_index - 1

        while previous_index >= min_index:
            previous_name = clean_parser_name(lines[previous_index])

            if (
                previous_name
                and not is_noise_recipient_line(previous_name)
                and not re.search(r"\d", previous_name)
                and not re.search(
                    r"\b(?:MAILED FROM|POSTAGE|SERVICE|REQUESTED)\b",
                    previous_name,
                    re.IGNORECASE,
                )
            ):
                return previous_name

            previous_index -= 1

    return recipient_name


# --- NORMALIZATION FUNCTION ---
def normalize_extracted_fields(label_data):
    street_address = label_data.get("street_address", "")
    tracking_number = label_data.get("tracking_number", "")
    recipient_name = label_data.get("recipient_name", "")
    city = label_data.get("city", "")

    if street_address:
        street_address = street_address.strip()
        street_address = clean_address_ocr(street_address)
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

    if recipient_name:
        recipient_name = recipient_name.strip()
        recipient_name = re.sub(r"\s+", " ", recipient_name)
        recipient_name = recipient_name.title()
        label_data["recipient_name"] = recipient_name

    if city:
        city = city.strip()
        city = clean_address_ocr(city)
        city = re.sub(r"\s+", " ", city)
        city = city.title()
        label_data["city"] = city

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
        "carrier": "",
        "parser_used": "",
        "parser_matches": [],
    }

    label_data["tracking_number"] = extract_tracking_number(image)
    label_data["carrier"] = identify_carrier(label_data["tracking_number"])

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

    ocr_tracking_candidate = extract_tracking_from_ocr_lines(lines)

    if ocr_tracking_candidate:
        print("OCR TRACKING CANDIDATE:", repr(ocr_tracking_candidate))

        has_usps_tracking_label = any(
            "USPS" in line.upper() and "TRACKING" in line.upper()
            for line in lines
        )
        keep_usps_barcode_tracking = (
            has_usps_tracking_label
            and identify_carrier(label_data["tracking_number"]) == "USPS"
            and identify_carrier(ocr_tracking_candidate) == "UPS"
        )

        if (
            not keep_usps_barcode_tracking
            and (
                not label_data["tracking_number"]
                or identify_carrier(label_data["tracking_number"]) == "Unknown"
                or "TRACKING" in text.upper()
            )
        ):
            label_data["tracking_number"] = ocr_tracking_candidate
            label_data["carrier"] = identify_carrier(ocr_tracking_candidate)

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
                        label_data["parser_used"] = "uniuni_deliver_to"
                        if "uniuni_deliver_to" not in label_data["parser_matches"]:
                            label_data["parser_matches"].append("uniuni_deliver_to")

                        used_deliver_to_block = True

            print(f"\nFOUND DELIVER TO AT LINE {line_index}")

            if line_index + 3 < len(lines):
                print("DELIVER TO HAS ENOUGH LINES")
                deliver_to_name = lines[line_index + 1]
                deliver_to_street = lines[line_index + 2]
                deliver_to_city_line = lines[line_index + 3]
                deliver_to_name = clean_parser_name(deliver_to_name)
                deliver_to_street = clean_address_ocr(deliver_to_street)

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
                        label_data["parser_used"] = "deliver_to"
                        if "deliver_to" not in label_data["parser_matches"]:
                            label_data["parser_matches"].append("deliver_to")

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

            if zip_match and plus_four_match and state_match and line_index >= 3:
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
                label_data["parser_used"] = "zip_first_spaced"
                if "zip_first_spaced" not in label_data["parser_matches"]:
                    label_data["parser_matches"].append("zip_first_spaced")

        if len(parts) >= 4:
            zip_first_match = re.fullmatch(r"(\d{5})[-–—](\d{4})", parts[0])
            state_match = re.fullmatch(r"[A-Z]{2}", parts[3])

            print("ZIP-FIRST COMBINED ZIP:", bool(zip_first_match))
            print("ZIP-FIRST COMBINED STATE:", bool(state_match))

            if zip_first_match and state_match and line_index >= 2:
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
                label_data["parser_used"] = "zip_first_combined"
                if "zip_first_combined" not in label_data["parser_matches"]:
                    label_data["parser_matches"].append("zip_first_combined")

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
                label_data["parser_used"] = "city_state_country_zip"
                if "city_state_country_zip" not in label_data["parser_matches"]:
                    label_data["parser_matches"].append("city_state_country_zip")

        college_mailroom_match = re.fullmatch(
            r"([A-Za-z][A-Za-z .'-]*?),?\s+([A-Z]{2})\s+(\d{5}(?:\s*[-–—]\s*\d{4})?)(?:\s+UNITED STATES)?\W*",
            line,
        )

        if college_mailroom_match and line_index >= 2:
            city = college_mailroom_match.group(1).strip().strip(",")
            state = college_mailroom_match.group(2)
            zip_code = re.sub(
                r"\s*[-–—]\s*",
                "-",
                college_mailroom_match.group(3),
            )

            previous_lines = lines[:line_index]
            mailroom_keywords = (
                "COLLEGE",
                "HUB",
                "UNIVERSITY",
                "PENN STATER",
                "ALUMNI CENTER",
            )
            previous_text = " ".join(previous_lines).upper()
            college_city_match = city.upper() in ("CARLISLE", "UNIVERSITY PARK")
            college_context_match = any(
                keyword in previous_text for keyword in mailroom_keywords
            )

            print(
                "COLLEGE MAILROOM CITY/STATE/ZIP MATCH:",
                college_mailroom_match.groups(),
            )
            print("COLLEGE MAILROOM CONTEXT:", college_city_match, college_context_match)

            if college_city_match or college_context_match:
                address_end_index = line_index - 1

                if re.fullmatch(r"\d{2,5}", lines[address_end_index]):
                    address_end_index -= 1

                address_start_index = address_end_index

                # Build the address from the contiguous mailroom/street lines above
                # city/state/ZIP, stopping before the recipient line.
                while address_start_index - 1 >= 0:
                    previous_line = lines[address_start_index - 1]
                    previous_upper = previous_line.upper()

                    if (
                        any(
                            keyword in previous_upper
                            for keyword in mailroom_keywords
                        )
                        or re.search(r"\bP\.?\s*O\.?\s+BOX\b", previous_upper)
                    ):
                        address_start_index -= 1
                    else:
                        break

                address_lines = lines[address_start_index : address_end_index + 1]

                recipient_name = choose_recipient_from_lines(
                    lines,
                    address_start_index - 1,
                    max(0, address_start_index - 4),
                )

                first_address_line = address_lines[0] if address_lines else ""
                ship_name_address_match = re.match(
                    r"^\s*SHIP\s+(.+?)\s+(HUB\b.*)$",
                    first_address_line,
                    re.IGNORECASE,
                )

                if ship_name_address_match:
                    recipient_name = ship_name_address_match.group(1).strip()
                    address_lines[0] = ship_name_address_match.group(2).strip()

                clean_address_lines = []

                for address_line in address_lines:
                    address_line = re.sub(
                        r"^\s*(?:TO|SHIP)\s*:\s*",
                        "",
                        address_line,
                        flags=re.IGNORECASE,
                    )
                    address_line = re.sub(
                        r"^\s*IP\s+UB\b",
                        "HUB",
                        address_line,
                        flags=re.IGNORECASE,
                    )
                    address_line = re.sub(
                        r"^(\d+)\s*N,\s+",
                        r"\1 N. ",
                        address_line,
                        flags=re.IGNORECASE,
                    )
                    address_line = clean_address_ocr(address_line)

                    if address_line:
                        clean_address_lines.append(address_line)

                recipient_name = clean_parser_name(recipient_name)

                if is_noise_recipient_line(recipient_name):
                    recipient_name = ""

                street_address = " ".join(clean_address_lines)

                label_data["recipient_name"] = recipient_name
                label_data["street_address"] = street_address
                label_data["city"] = city
                label_data["state"] = state
                label_data["zip_code"] = zip_code
                label_data["parser_used"] = "college_mailroom_parser"
                if "college_mailroom_parser" not in label_data["parser_matches"]:
                    label_data["parser_matches"].append("college_mailroom_parser")

        if label_data["parser_used"] == "college_mailroom_parser":
            continue

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
                    label_data["parser_used"] = "generic_city_state_zip"
                    if "generic_city_state_zip" not in label_data["parser_matches"]:
                        label_data["parser_matches"].append("generic_city_state_zip")

    label_data = normalize_extracted_fields(label_data)
    label_data["carrier"] = identify_carrier(label_data["tracking_number"])
    label_data = score_label_data(label_data)

    return label_data


if __name__ == "__main__":
    image_path = "images/USPS_Shipping_Label.JPG"
    label_data = extract_label_data(image_path)
    print(label_data)
