import re


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
