import re
from pyzbar.pyzbar import decode

from tracking import clean_tracking_candidate, is_valid_tracking_candidate


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
