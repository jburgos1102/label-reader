import re


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
