import re
import threading

from pyzbar.pyzbar import decode

from logger import log
from tracking import clean_tracking_candidate, is_valid_tracking_candidate


# Thread-local: Flask serves concurrent requests on separate threads, so
# module-level state here would leak one request's barcodes into another's.
_local = threading.local()


def get_last_raw_barcodes():
    """Return a copy of the raw barcode strings found during the most recent scan
    on the current thread."""
    return list(getattr(_local, "last_raw_barcodes", []))


def extract_tracking_number(image):
    _local.last_raw_barcodes = []

    rotations = [0, 90, 180, 270]
    min_tracking_length = 15

    for degrees in rotations:
        rotated_image = image.rotate(degrees, expand=True)
        barcodes = decode(rotated_image)

        if barcodes:
            log.debug("Barcodes found at rotation %s", degrees)

            for barcode_index, barcode in enumerate(barcodes):
                barcode_data = barcode.data.decode("utf-8")
                if barcode_data not in _local.last_raw_barcodes:
                    _local.last_raw_barcodes.append(barcode_data)
                log.debug(
                    "Barcode %s data=%r type=%s",
                    barcode_index,
                    barcode_data,
                    barcode.type,
                )

                barcode_parts = barcode_data.split("\x1d")

                if len(barcode_parts) > 1:
                    candidate = barcode_parts[1]
                else:
                    candidate = barcode_data

                log.debug("Barcode tracking candidate: %r", candidate)

                candidate = clean_tracking_candidate(candidate)

                if is_valid_tracking_candidate(candidate):
                    return candidate

                barcode_digits = re.sub(r"\D", "", barcode_data)

                if len(barcode_digits) > 22:
                    fedex_candidate = barcode_digits[-12:]

                    if is_valid_tracking_candidate(fedex_candidate):
                        return fedex_candidate

    return ""
