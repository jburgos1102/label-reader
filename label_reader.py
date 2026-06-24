from PIL import Image
import re
from address import (
    choose_recipient_from_lines,
    clean_address_ocr,
    clean_parser_name,
    find_recipient_name_fallback,
    is_deliver_to_marker,
    is_noise_recipient_line,
    normalize_extracted_fields,
    split_ship_recipient_and_hub,
)
from barcodes import extract_tracking_number
from llm_extractor import extract_fields_with_llm
from ocr import get_best_ocr_text
from tracking import extract_tracking_from_ocr_lines, identify_carrier


def normalize_comparison_value(value):
    normalized = str(value or "").upper().strip()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return " ".join(normalized.split())


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
            "USPS" in line.upper() and "TRACKING" in line.upper() for line in lines
        )
        keep_usps_barcode_tracking = (
            has_usps_tracking_label
            and identify_carrier(label_data["tracking_number"]) == "USPS"
            and identify_carrier(ocr_tracking_candidate) == "UPS"
        )

        if not keep_usps_barcode_tracking and (
            not label_data["tracking_number"]
            or identify_carrier(label_data["tracking_number"]) == "Unknown"
            or "TRACKING" in text.upper()
        ):
            label_data["tracking_number"] = ocr_tracking_candidate
            label_data["carrier"] = identify_carrier(ocr_tracking_candidate)

    used_deliver_to_block = False

    for line_index, line in enumerate(lines):
        if is_deliver_to_marker(line):

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
                deliver_to_city_line = clean_address_ocr(deliver_to_city_line)

                print("DELIVER TO NAME:", repr(deliver_to_name))
                print("DELIVER TO STREET:", repr(deliver_to_street))
                print("DELIVER TO CITY LINE:", repr(deliver_to_city_line))

                deliver_to_city_parts = [
                    part.strip(",")
                    for part in deliver_to_city_line.split()
                    if part.strip(",")
                ]

                print("DELIVER TO CITY PARTS:", deliver_to_city_parts)

                if len(deliver_to_city_parts) >= 2:
                    state_candidate = deliver_to_city_parts[-2]
                    zip_candidate = deliver_to_city_parts[-1]

                    state_match = re.fullmatch(r"[A-Z]{2}", state_candidate)
                    zip_match = re.fullmatch(r"\d{5}", zip_candidate)

                    if not (state_match and zip_match):
                        state_candidate = deliver_to_city_parts[-1]
                        zip_candidate = ""
                        state_match = re.fullmatch(r"[A-Z]{2}", state_candidate)

                        for previous_line in lines[max(0, line_index - 8) : line_index]:
                            previous_zip_match = re.search(
                                r"\bSHIP\s+USPS\s+(\d{5})\b",
                                previous_line,
                                re.IGNORECASE,
                            )

                            if previous_zip_match:
                                zip_candidate = previous_zip_match.group(1)
                                break

                        zip_match = re.fullmatch(r"\d{5}", zip_candidate)

                    print("STATE MATCH:", bool(state_match))
                    print("ZIP MATCH:", bool(zip_match))

                    if state_match and zip_match:
                        if deliver_to_city_parts[-1] == zip_candidate:
                            deliver_to_city = " ".join(deliver_to_city_parts[:-2])
                        else:
                            deliver_to_city = " ".join(deliver_to_city_parts[:-1])

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

        city_first_match = re.fullmatch(
            r"([A-Za-z][A-Za-z .'-]*?),?\s+([A-Z]{2})\s+(\d{5}(?:\s*[-–—]\s*\d{4})?)",
            clean_address_ocr(line),
        )

        if city_first_match and line_index == 0:
            for nearby_index in range(line_index + 1, min(line_index + 12, len(lines))):
                if re.search(
                    r"\b(?:USPS\s+)?TRACKING\b",
                    lines[nearby_index],
                    re.IGNORECASE,
                ):
                    break

                if re.match(r"^\s*TO\b", lines[nearby_index], re.IGNORECASE):
                    if nearby_index + 1 < len(lines):
                        possible_street = clean_address_ocr(lines[nearby_index + 1])

                        if re.search(r"\d", possible_street) and not re.search(
                            r"\b(?:USPS\s+)?TRACKING\b",
                            possible_street,
                            re.IGNORECASE,
                        ):
                            label_data["recipient_name"] = ""
                            label_data["street_address"] = possible_street
                            label_data["city"] = city_first_match.group(1).strip(",")
                            label_data["state"] = city_first_match.group(2)
                            label_data["zip_code"] = re.sub(
                                r"\s*[-–—]\s*",
                                "-",
                                city_first_match.group(3),
                            )
                            label_data["parser_used"] = "city_first_to_address"
                            if (
                                "city_first_to_address"
                                not in label_data["parser_matches"]
                            ):
                                label_data["parser_matches"].append(
                                    "city_first_to_address"
                                )

                            break

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
            print(
                "COLLEGE MAILROOM CONTEXT:", college_city_match, college_context_match
            )

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

                    if any(
                        keyword in previous_upper for keyword in mailroom_keywords
                    ) or re.search(r"\bP\.?\s*O\.?\s+BOX\b", previous_upper):
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
                ship_recipient_name, ship_hub_part = split_ship_recipient_and_hub(
                    first_address_line
                )
                drop_split_ship_hub = False

                if ship_recipient_name:
                    recipient_name = ship_recipient_name

                    has_dickinson_address_line = any(
                        re.search(
                            r"\bDICKINSON\s+COLLEGE\b.*\bN\s+COLLEGE\b",
                            address_line,
                            re.IGNORECASE,
                        )
                        for address_line in address_lines[1:]
                    )

                    if has_dickinson_address_line:
                        address_lines = address_lines[1:]
                        drop_split_ship_hub = True
                    else:
                        address_lines[0] = ship_hub_part

                clean_address_lines = []

                for address_line in address_lines:
                    if (
                        drop_split_ship_hub
                        and re.fullmatch(
                            r"\s*TO\s*:\s*ST\s*",
                            address_line,
                            flags=re.IGNORECASE,
                        )
                    ):
                        continue

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

                if zip_match and line_index >= 2:
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

    if not label_data["recipient_name"]:
        fallback_recipient = find_recipient_name_fallback(lines)
        if fallback_recipient:
            label_data["recipient_name"] = fallback_recipient

    label_data = normalize_extracted_fields(label_data)
    label_data["carrier"] = identify_carrier(label_data["tracking_number"])
    label_data = score_label_data(label_data)

    try:
        llm_result = extract_fields_with_llm(text, label_data)
    except Exception:
        llm_result = {
            "recipient_name": label_data.get("recipient_name", ""),
            "street_address": label_data.get("street_address", ""),
            "city": label_data.get("city", ""),
            "state": label_data.get("state", ""),
            "zip_code": label_data.get("zip_code", ""),
            "tracking_number": label_data.get("tracking_number", ""),
            "carrier": label_data.get("carrier", ""),
            "llm_enabled": False,
            "llm_provider": "openai",
            "llm_notes": "OpenAI extraction failed; using the rule-based result.",
        }

    label_data["llm_result"] = llm_result

    selected_fields = (
        "recipient_name",
        "street_address",
        "city",
        "state",
        "zip_code",
        "tracking_number",
        "carrier",
    )
    llm_available = isinstance(llm_result, dict) and bool(
        llm_result.get("llm_enabled")
    )
    selected_result = {
        field: label_data.get(field, "") for field in selected_fields
    }
    selected_sources = {}

    for field in selected_fields:
        rule_based_value = label_data.get(field, "")
        openai_value = llm_result.get(field, "") if llm_available else ""
        values_agree = llm_available and normalize_comparison_value(
            rule_based_value
        ) == normalize_comparison_value(openai_value)
        selected_sources[field] = "agreement" if values_agree else "rule_based"

    selected_result["source"] = selected_sources
    label_data["selected_result"] = selected_result

    comparison = {}

    for field in selected_fields:
        rule_based_value = label_data.get(field, "")
        openai_value = llm_result.get(field, "") if llm_available else ""
        selected_value = selected_result.get(field, "")
        comparison[field] = {
            "rule_based": rule_based_value,
            "openai": openai_value,
            "selected": selected_value,
            "agreement": llm_available
            and normalize_comparison_value(rule_based_value)
            == normalize_comparison_value(openai_value),
        }

    label_data["comparison"] = comparison

    return label_data


if __name__ == "__main__":
    image_path = "images/USPS_Shipping_Label.JPG"
    label_data = extract_label_data(image_path)
    print(label_data)
