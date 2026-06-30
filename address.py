import re

from logger import log

_STREET_REJECT_WORDS = re.compile(
    r"\b(?:CARRIER|LEAVE|RESPONSE|USPS|UPS|PARCEL|SELECT|TRACKING|"
    r"POSTAGE|DELIVERED|DELIVER)\b",
    re.IGNORECASE,
)


def clean_physical_street_ocr(value):
    """Clean conservative OCR noise from a line that starts like an address."""
    address_start = re.match(
        r"^\s*[^A-Za-z0-9]*?(?:\d+[A-Z]?(?:-[A-Z0-9]+)?|[NSEW]\s*\d{2,4})\b",
        value,
        flags=re.IGNORECASE,
    )
    if not address_start:
        return value

    value = re.sub(r"^\s*[^A-Za-z0-9]+", "", value)
    value = re.sub(r"^(\d+[A-Z]?)\s*[.,:]\s+", r"\1 ", value)
    value = re.sub(
        r"^([NSEW])\s+(\d{2,4})\b",
        r"\1\2",
        value,
        flags=re.IGNORECASE,
    )

    # Section-sign-like OCR is a common damaged terminal "ST" glyph.
    value = re.sub(r"\s+§\s*$", " ST", value)

    suffix_matches = list(
        re.finditer(
            r"\b(?:STREET|ST|ROAD|RD|AVENUE|AVE|BOULEVARD|BLVD|DRIVE|DR|"
            r"LANE|LN|COURT|CT|WAY)\b",
            value,
            flags=re.IGNORECASE,
        )
    )
    if suffix_matches:
        suffix_match = suffix_matches[-1]
        trailing_text = value[suffix_match.end() :].strip()
        keeps_delivery_detail = re.match(
            r"^(?:(?:APT|APARTMENT|UNIT|SUITE|STE|HUB)\b|"
            r"P\.?\s*O\.?\s+BOX\b|#)",
            trailing_text,
            flags=re.IGNORECASE,
        )
        trailing_words = re.findall(r"[A-Za-z]+", trailing_text)
        obvious_trailing_noise = bool(
            re.search(r"[;|%={}~]", trailing_text)
            or (
                trailing_words
                and all(len(word) <= 2 for word in trailing_words)
            )
        )

        if trailing_text and not keeps_delivery_detail and obvious_trailing_noise:
            value = value[: suffix_match.end()]

    return value


def reconstruct_college_mailroom_street(lines, current_street):
    """Repair narrowly recognized Dickinson/college mailroom OCR damage."""
    cleaned_lines = [clean_address_ocr(line) for line in lines]
    upper_lines = [line.upper() for line in cleaned_lines]
    combined_text = "\n".join(upper_lines)

    has_carlisle_context = bool(
        re.search(r"\bCARLISLE\s*,?\s+PA\s+17013\b", combined_text)
    )
    has_college_street = bool(
        re.search(r"\bCOLLEGE\s+(?:ST|STREET)\b", combined_text)
    )
    has_hub_routing = bool(
        re.search(r"\bHUB\s*#?\s*\d{2,5}\b", combined_text)
        or re.search(r"(?:^|\n)\s*UB\s*#?\s*\d{2,5}\b", combined_text)
    )
    has_dickinson_context = "DICKINSON COLLEGE" in combined_text
    has_po_box_context = bool(re.search(r"\bP\.?\s*O\.?\s+BOX\s+1773\b", combined_text))

    strong_mailroom_context = has_college_street and (
        has_dickinson_context
        or has_po_box_context
        or (has_carlisle_context and has_hub_routing)
    )
    if not strong_mailroom_context:
        return current_street

    normalized_current = clean_address_ocr(current_street).upper()
    if re.match(r"^28\s+N\s+COLLEGE\s+(?:ST|STREET)\b", normalized_current):
        return current_street

    for index, line in enumerate(upper_lines):
        split_street_match = re.fullmatch(
            r"8\s+N\s+COLLEGE\s+(?:ST|STREET)",
            line,
        )
        if split_street_match and index + 1 < len(upper_lines):
            hub_match = re.fullmatch(
                r"(?:H?UB)\s*(#?)\s*(\d{2,5})",
                upper_lines[index + 1],
            )
            if hub_match:
                separator = " # " if hub_match.group(1) else " "
                return f"28 N COLLEGE ST HUB{separator}{hub_match.group(2)}"

        damaged_prefix_match = re.fullmatch(
            r"[^\d\s]{1,4}\s+COLLEGE\s+(?:ST|STREET)\s+"
            r"HUB\s*(#?)\s*(\d{2,5})",
            line,
        )
        if damaged_prefix_match:
            separator = " # " if damaged_prefix_match.group(1) else " "
            return (
                "28 N COLLEGE ST HUB"
                f"{separator}{damaged_prefix_match.group(2)}"
            )

    return current_street


def recover_penn_state_address(
    lines,
    current_street,
    current_city,
    current_state,
    current_zip,
):
    """Recover an explicit University Park campus address block from OCR."""
    cleaned_lines = [clean_address_ocr(line) for line in lines]

    for city_index, line in enumerate(cleaned_lines):
        city_match = re.search(
            r"\bUNIVERSITY\s+PARK\b[\s,]*PA\W*(16802)"
            r"(?:\s*[-–—]\s*(\d{4}))?\b",
            line,
            flags=re.IGNORECASE,
        )
        if not city_match:
            continue

        zip_code = city_match.group(1)
        if city_match.group(2):
            zip_code += "-" + city_match.group(2)

        recovered = {
            "street_address": current_street,
            "city": "University Park",
            "state": "PA",
            "zip_code": zip_code,
        }

        for address_index in range(city_index - 1, max(-1, city_index - 5), -1):
            address_line = cleaned_lines[address_index].strip()
            building_match = re.fullmatch(
                r"((?:\d+[A-Z]?(?:-[A-Z0-9]+)?|[NSEW]\d{2,4})\s+.+?"
                r"\b(?:BLDG|BUILDING|CENTER|COMPLEX|PARK))",
                address_line,
                flags=re.IGNORECASE,
            )
            if building_match:
                recovered["street_address"] = building_match.group(1)
                break

        return recovered

    return {
        "street_address": current_street,
        "city": current_city,
        "state": current_state,
        "zip_code": current_zip,
    }


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
    value = clean_physical_street_ocr(value)
    value = re.sub(r"(?<=\d)@(?=\d)", "0", value)
    value = re.sub(r"\b1@5\b", "105", value, flags=re.IGNORECASE)
    value = re.sub(r"\bNL\b", "L", value, flags=re.IGNORECASE)
    value = re.sub(r"\bAUB\b", "HUB", value, flags=re.IGNORECASE)
    value = re.sub(r"\bPRRLISLE\b", "CARLISLE", value, flags=re.IGNORECASE)
    value = re.sub(
        r"(?<![A-Za-z])\\?IVERSITY\b",
        "UNIVERSITY",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def is_deliver_to_marker(line):
    clean_line = clean_address_ocr(line).upper()
    clean_line = re.sub(r"[^A-Z\s]", " ", clean_line)
    clean_line = re.sub(r"\s+", " ", clean_line).strip()

    return bool(
        re.search(r"\bDELIVER\s+TO\b", clean_line)
        or re.search(r"\bVER\s+TO\b", clean_line)
        or re.search(r"\bUSPS\b.*\bTO\b", clean_line)
    )


def split_ship_recipient_and_hub(line):
    match = re.match(
        r"^\s*SHIP\s+(.+?)\s+(HUB\b.*)$",
        line,
        flags=re.IGNORECASE,
    )

    if not match:
        return "", ""

    return match.group(1).strip(), match.group(2).strip()


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

    if re.search(r"\bDICKINSON\s*COLLEGE\b", clean_value):
        return True

    if re.search(r"\bHUB\s*#?\s*\d+\b", clean_value):
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


def _is_likely_street_line(value):
    clean_value = clean_address_ocr(value).upper()
    return bool(
        re.search(r"\d", clean_value)
        and re.search(
            r"\b(?:STREET|ST|ROAD|RD|AVENUE|AVE|BOULEVARD|BLVD|DRIVE|DR|"
            r"LANE|LN|COURT|CT|WAY|BUILDING|BLDG|CENTER|HUB|P\.?\s*O\.?\s+BOX)\b",
            clean_value,
        )
    )


def _is_city_state_zip_line(value):
    clean_value = clean_address_ocr(value).upper()
    return bool(
        re.search(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", clean_value)
        or re.search(r"\bZIP\s*\d{5}\b", clean_value)
    )


def _is_recipient_marker(value):
    clean_value = clean_address_ocr(value).upper()
    return bool(
        is_deliver_to_marker(value)
        or re.fullmatch(r"\s*(?:TO|SHIP\s+TO)\s*:?[\s|]*", clean_value)
    )


def _recipient_name_shape(value):
    name_word = r"[A-Za-z][A-Za-z'’-]*"
    title = r"(?:MR|MRS|MS|MISS|DR)\."
    comma_format = re.fullmatch(
        rf"(?:{title}\s+)?{name_word},\s*{name_word}(?:\s+[A-Za-z])?",
        value,
        flags=re.IGNORECASE,
    )
    standard_format = re.fullmatch(
        rf"(?:{title}\s+)?{name_word}(?:\s+{name_word}){{1,3}}",
        value,
        flags=re.IGNORECASE,
    )
    return comma_format, standard_format


def _is_clean_person_name_candidate(value):
    candidate = clean_parser_name(value)

    if not candidate or is_noise_recipient_line(candidate):
        return False

    if re.search(r"\d", candidate):
        return False

    letters = sum(character.isalpha() for character in candidate)
    visible = sum(not character.isspace() for character in candidate)
    if not visible or letters / visible < 0.75:
        return False

    comma_format, standard_format = _recipient_name_shape(candidate)
    return bool(comma_format or standard_format)


def find_explicit_to_person_name(lines):
    """Prefer explicit To: person-name lines over routing/header blocks."""
    candidates = []

    for index, raw_line in enumerate(lines):
        same_line_match = re.match(r"^\s*TO\s*:\s*(.+)$", raw_line, re.IGNORECASE)
        if same_line_match:
            candidate = clean_parser_name(same_line_match.group(1))
            if _is_clean_person_name_candidate(candidate):
                comma_format, _ = _recipient_name_shape(candidate)
                score = 5 + (3 if comma_format else 0) + index / 1000
                candidates.append((score, candidate))
            continue

        marker_match = re.fullmatch(r"\s*TO\s*:?\s*", raw_line, re.IGNORECASE)
        if marker_match and index + 1 < len(lines):
            candidate = clean_parser_name(lines[index + 1])
            if _is_clean_person_name_candidate(candidate):
                comma_format, _ = _recipient_name_shape(candidate)
                score = 4 + (3 if comma_format else 0) + index / 1000
                candidates.append((score, candidate))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def find_recipient_name_fallback(lines):
    """Find a conservative person-name candidate in OCR lines."""
    rejected_words = re.compile(
        r"\b(?:AMAZON|CARRIER|DELIVERY|DEPARTMENT|DEPT|DHL|FEDEX|GROUND|"
        r"INSTAGRAM|LIGHTWEIGHT|MAIL|POSTAGE|PRIORITY|PROMO|QR|RETURN|SCAN|"
        r"SERVICE|SERVICES|SOCIAL|SPEEDX|TRACKING|UNITED STATES|UPS|USPS|"
        r"VISIT|COLLEGE|UNIVERSITY|FULFILLMENT|CURRENT OCCUPANT)\b",
        flags=re.IGNORECASE,
    )
    candidates = []

    for index, raw_line in enumerate(lines):
        raw_line = raw_line.strip()
        same_line_marker = bool(
            re.match(r"^\s*(?:TO|SHIP\s+TO)\s*:", raw_line, re.I)
        )
        candidate = clean_parser_name(raw_line)

        if not candidate or is_noise_recipient_line(candidate):
            continue

        if rejected_words.search(candidate) or re.search(r"\d", candidate):
            continue

        letters = sum(character.isalpha() for character in candidate)
        visible = sum(not character.isspace() for character in candidate)
        if not visible or letters / visible < 0.75:
            continue

        comma_format, standard_format = _recipient_name_shape(candidate)
        if not (comma_format or standard_format):
            continue

        score = 1
        if comma_format:
            score += 3
        if re.match(r"^(?:MR|MRS|MS|MISS|DR)\.", candidate, re.I):
            score += 2
        if same_line_marker:
            score += 4
        if any(
            _is_recipient_marker(lines[nearby])
            for nearby in range(max(0, index - 2), index)
        ):
            score += 4
        if any(
            _is_likely_street_line(lines[nearby])
            for nearby in range(index + 1, min(len(lines), index + 3))
        ):
            score += 3
        if any(
            _is_city_state_zip_line(lines[nearby])
            for nearby in range(index + 1, min(len(lines), index + 4))
        ):
            score += 2

        if score >= 4:
            candidates.append((score, index, candidate))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


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

        if _STREET_REJECT_WORDS.search(street_address):
            label_data["street_address"] = ""
            label_data["_street_rejected"] = True
        else:
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


def parse_address_from_lines(lines):
    """Run all address parsers over OCR lines and return extracted address fields."""
    result = {
        "recipient_name": "",
        "street_address": "",
        "city": "",
        "state": "",
        "zip_code": "",
        "parser_used": "",
        "parser_matches": [],
    }

    used_deliver_to_block = False

    for line_index, line in enumerate(lines):
        if is_deliver_to_marker(line):

            if "USPS" not in line.upper():
                log.debug("Found non-USPS deliver-to marker at line %s", line_index)

                for nearby_index in range(line_index, min(line_index + 6, len(lines))):
                    log.debug(
                        "Non-USPS deliver-to nearby line %s: %r",
                        nearby_index,
                        lines[nearby_index],
                    )

                if line_index + 4 < len(lines):
                    uniuni_city_state_line = lines[line_index + 2]
                    uniuni_street_line = lines[line_index + 3]
                    uniuni_zip_line = lines[line_index + 4]

                    log.debug("UniUni city/state candidate: %r", uniuni_city_state_line)
                    log.debug("UniUni street candidate: %r", uniuni_street_line)
                    log.debug("UniUni ZIP candidate: %r", uniuni_zip_line)

                    uniuni_city_state_match = re.search(
                        r"([A-Za-z]+),?\s+([A-Z]{2}),?",
                        uniuni_city_state_line,
                    )
                    uniuni_street_match = re.match(r"\d+", uniuni_street_line)
                    uniuni_zip_match = re.search(r"\d{5}", uniuni_zip_line)

                    log.debug(
                        "UniUni city/state match: %s",
                        bool(uniuni_city_state_match),
                    )
                    log.debug("UniUni street match: %s", bool(uniuni_street_match))
                    log.debug("UniUni ZIP match: %s", bool(uniuni_zip_match))

                    if (
                        uniuni_city_state_match
                        and uniuni_street_match
                        and uniuni_zip_match
                    ):
                        result["street_address"] = uniuni_street_line
                        result["city"] = uniuni_city_state_match.group(1)
                        result["state"] = uniuni_city_state_match.group(2)
                        result["zip_code"] = uniuni_zip_match.group()
                        result["parser_used"] = "uniuni_deliver_to"
                        if "uniuni_deliver_to" not in result["parser_matches"]:
                            result["parser_matches"].append("uniuni_deliver_to")

                        used_deliver_to_block = True

            log.debug("Found deliver-to marker at line %s", line_index)

            if line_index + 3 < len(lines):
                log.debug("Deliver-to block has enough lines")
                deliver_to_name = lines[line_index + 1]
                deliver_to_street = lines[line_index + 2]
                deliver_to_city_line = lines[line_index + 3]
                deliver_to_name = clean_parser_name(deliver_to_name)
                deliver_to_street = clean_address_ocr(deliver_to_street)
                deliver_to_city_line = clean_address_ocr(deliver_to_city_line)

                log.debug("Deliver-to name: %r", deliver_to_name)
                log.debug("Deliver-to street: %r", deliver_to_street)
                log.debug("Deliver-to city line: %r", deliver_to_city_line)

                deliver_to_city_parts = [
                    part.strip(",")
                    for part in deliver_to_city_line.split()
                    if part.strip(",")
                ]

                log.debug("Deliver-to city parts: %s", deliver_to_city_parts)

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

                    log.debug("Deliver-to state match: %s", bool(state_match))
                    log.debug("Deliver-to ZIP match: %s", bool(zip_match))

                    if state_match and zip_match:
                        if deliver_to_city_parts[-1] == zip_candidate:
                            deliver_to_city = " ".join(deliver_to_city_parts[:-2])
                        else:
                            deliver_to_city = " ".join(deliver_to_city_parts[:-1])

                        deliver_to_city = deliver_to_city.strip(",")

                        result["recipient_name"] = deliver_to_name
                        result["street_address"] = deliver_to_street
                        result["city"] = deliver_to_city
                        result["state"] = state_candidate
                        result["zip_code"] = zip_candidate
                        result["parser_used"] = "deliver_to"
                        if "deliver_to" not in result["parser_matches"]:
                            result["parser_matches"].append("deliver_to")

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
                            result["recipient_name"] = ""
                            result["street_address"] = possible_street
                            result["city"] = city_first_match.group(1).strip(",")
                            result["state"] = city_first_match.group(2)
                            result["zip_code"] = re.sub(
                                r"\s*[-–—]\s*",
                                "-",
                                city_first_match.group(3),
                            )
                            result["parser_used"] = "city_first_to_address"
                            if "city_first_to_address" not in result["parser_matches"]:
                                result["parser_matches"].append("city_first_to_address")

                            break

        if "18974" in line:
            log.debug("ZIP-first parts: %s", parts)

        if len(parts) >= 6:
            zip_match = re.fullmatch(r"\d{5}", parts[0])
            separator_match = re.fullmatch(r"[-–—]", parts[1])
            plus_four_match = re.fullmatch(r"\d{4}", parts[2])
            state_match = re.fullmatch(r"[A-Z]{2}", parts[5])

            log.debug("ZIP-first ZIP match: %s", bool(zip_match))
            log.debug("ZIP-first plus-four match: %s", bool(plus_four_match))
            log.debug("ZIP-first state match: %s", bool(state_match))

            if zip_match and plus_four_match and state_match and line_index >= 3:
                result["recipient_name"] = lines[line_index - 3]
                result["street_address"] = lines[line_index - 1]
                result["city"] = parts[3]
                result["state"] = parts[5]
                result["zip_code"] = parts[0] + "-" + parts[2]
                result["parser_used"] = "zip_first_spaced"
                if "zip_first_spaced" not in result["parser_matches"]:
                    result["parser_matches"].append("zip_first_spaced")

        if len(parts) >= 4:
            zip_first_match = re.fullmatch(r"(\d{5})[-–—](\d{4})", parts[0])
            state_match = re.fullmatch(r"[A-Z]{2}", parts[3])

            log.debug("ZIP-first combined ZIP match: %s", bool(zip_first_match))
            log.debug("ZIP-first combined state match: %s", bool(state_match))

            if zip_first_match and state_match and line_index >= 2:
                result["recipient_name"] = lines[line_index - 2]
                result["street_address"] = lines[line_index - 1]
                result["city"] = parts[1]
                result["state"] = parts[3]
                result["zip_code"] = (
                    zip_first_match.group(1) + "-" + zip_first_match.group(2)
                )
                result["parser_used"] = "zip_first_combined"
                if "zip_first_combined" not in result["parser_matches"]:
                    result["parser_matches"].append("zip_first_combined")

        country_zip_match = re.fullmatch(
            r"(.+?),\s*([A-Z]{2}),\s*(?:US|USA),?\s*(\d{5})",
            line,
            re.IGNORECASE,
        )

        if country_zip_match:
            log.debug("City/state/country/ZIP match: %s", country_zip_match.groups())

            if line_index - 2 >= 0:
                result["recipient_name"] = lines[line_index - 2]
                result["street_address"] = lines[line_index - 1]
                result["city"] = country_zip_match.group(1).strip()
                result["state"] = country_zip_match.group(2).upper()
                result["zip_code"] = country_zip_match.group(3)
                result["parser_used"] = "city_state_country_zip"
                if "city_state_country_zip" not in result["parser_matches"]:
                    result["parser_matches"].append("city_state_country_zip")

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

            log.debug(
                "College mailroom city/state/ZIP match: %s",
                college_mailroom_match.groups(),
            )
            log.debug(
                "College mailroom context: city=%s context=%s",
                college_city_match,
                college_context_match,
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

                result["recipient_name"] = recipient_name
                result["street_address"] = " ".join(clean_address_lines)
                result["city"] = city
                result["state"] = state
                result["zip_code"] = zip_code
                result["parser_used"] = "college_mailroom_parser"
                if "college_mailroom_parser" not in result["parser_matches"]:
                    result["parser_matches"].append("college_mailroom_parser")

        if result["parser_used"] == "college_mailroom_parser":
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

                    city = " ".join(clean_city_parts).strip(",")
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

                    clean_recipient_name_parts = [
                        part
                        for part in recipient_name_parts[recipient_name_index:]
                        if part.isalpha()
                    ]

                    full_zip = (
                        zip_code + "-" + zip_plus_four if zip_plus_four else zip_code
                    )

                    result["recipient_name"] = " ".join(clean_recipient_name_parts)
                    result["street_address"] = street_address
                    result["city"] = city
                    result["state"] = state
                    result["zip_code"] = full_zip
                    result["parser_used"] = "generic_city_state_zip"
                    if "generic_city_state_zip" not in result["parser_matches"]:
                        result["parser_matches"].append("generic_city_state_zip")

    # Explicit TO: lines override parser-derived recipient when the parser result is
    # noise, missing, or a mailroom/hub reference rather than a person name.
    explicit_to_recipient = find_explicit_to_person_name(lines)
    current_recipient = result.get("recipient_name", "")
    current_recipient_upper = current_recipient.upper()
    current_recipient_tokens = re.findall(r"[A-Za-z]+", current_recipient)

    if explicit_to_recipient and (
        not current_recipient
        or is_noise_recipient_line(current_recipient)
        or re.search(r"\b(?:HUB|DICKINSON\s*COLLEGE)\b", current_recipient_upper)
        or len(current_recipient_tokens) < 2
    ):
        result["recipient_name"] = explicit_to_recipient

    if not result["recipient_name"]:
        fallback_recipient = find_recipient_name_fallback(lines)
        if fallback_recipient:
            result["recipient_name"] = fallback_recipient

    result["street_address"] = reconstruct_college_mailroom_street(
        lines,
        result["street_address"],
    )
    penn_state = recover_penn_state_address(
        lines,
        result["street_address"],
        result["city"],
        result["state"],
        result["zip_code"],
    )
    result.update(penn_state)

    return result
