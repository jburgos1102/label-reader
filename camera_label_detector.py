from collections import deque
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np


def find_label_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    minimum_area = frame.shape[0] * frame.shape[1] * 0.10

    # Prefer the largest substantial four-sided region in the frame.
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < minimum_area:
            break

        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            return approximation

    return None


def order_contour_points(contour):
    points = np.asarray(contour, dtype=np.float32).reshape(4, 2)
    coordinate_sums = points.sum(axis=1)
    coordinate_differences = np.diff(points, axis=1).ravel()
    corner_indices = [
        np.argmin(coordinate_sums),
        np.argmin(coordinate_differences),
        np.argmax(coordinate_sums),
        np.argmax(coordinate_differences),
    ]

    if len(set(corner_indices)) != 4:
        raise ValueError("The detected contour does not have four distinct corners.")

    return points[corner_indices]


def crop_label(frame, contour):
    if contour is None or len(contour) != 4:
        raise ValueError("No valid four-point label contour is available.")

    points = order_contour_points(contour)

    # Expand slightly from the center to retain a narrow margin around the label.
    center = points.mean(axis=0)
    points = center + (points - center) * 1.03
    frame_height, frame_width = frame.shape[:2]
    points[:, 0] = np.clip(points[:, 0], 0, frame_width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, frame_height - 1)

    top_left, top_right, bottom_right, bottom_left = points
    crop_width = int(
        round(
            max(
                np.linalg.norm(bottom_right - bottom_left),
                np.linalg.norm(top_right - top_left),
            )
        )
    )
    crop_height = int(
        round(
            max(
                np.linalg.norm(top_right - bottom_right),
                np.linalg.norm(top_left - bottom_left),
            )
        )
    )

    if crop_width < 20 or crop_height < 20:
        raise ValueError("The detected label contour is too small to crop safely.")

    destination = np.array(
        [
            [0, 0],
            [crop_width - 1, 0],
            [crop_width - 1, crop_height - 1],
            [0, crop_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, destination)

    return cv2.warpPerspective(frame, transform, (crop_width, crop_height))


def main():
    capture_directory = Path("captures")
    capture_directory.mkdir(exist_ok=True)

    camera = cv2.VideoCapture(0)
    detection_history = deque(maxlen=10)
    last_valid_contour = None
    detection_started_at = None
    capture_locked = False
    captured_message_until = 0.0

    if not camera.isOpened():
        print("Unable to open the camera. Check that it is connected and available.")
        camera.release()
        return

    try:
        while True:
            frame_available, frame = camera.read()

            if not frame_available:
                print("Unable to read a camera frame. Closing the detector.")
                break

            clean_frame = frame.copy()
            label_contour = find_label_contour(frame)
            frame_detected = label_contour is not None
            detection_history.append(frame_detected)

            if frame_detected:
                last_valid_contour = label_contour
                cv2.polylines(frame, [label_contour], True, (0, 255, 255), 3)

            label_detected = sum(detection_history) >= 3
            current_time = time.monotonic()

            if label_detected:
                if detection_started_at is None:
                    detection_started_at = current_time

                stable_for_two_seconds = current_time - detection_started_at >= 2.0
                if stable_for_two_seconds and not capture_locked:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    capture_path = capture_directory / f"captured_label_{timestamp}.jpg"
                    crop_path = capture_directory / f"cropped_label_{timestamp}.jpg"
                    capture_saved = cv2.imwrite(str(capture_path), clean_frame)
                    capture_locked = True

                    if capture_saved:
                        captured_message_until = current_time + 2.0
                        print(f"Captured label: {capture_path}")
                    else:
                        print(f"Unable to save captured label: {capture_path}")

                    try:
                        cropped_label = crop_label(clean_frame, last_valid_contour)
                        crop_saved = cv2.imwrite(str(crop_path), cropped_label)

                        if crop_saved:
                            print(f"Captured cropped label: {crop_path}")
                        else:
                            print(f"Unable to save cropped label: {crop_path}")
                    except (ValueError, cv2.error) as error:
                        print(f"Unable to crop the label; full frame was kept: {error}")
            else:
                detection_started_at = None
                capture_locked = False

            if label_detected and last_valid_contour is not None:
                cv2.polylines(frame, [last_valid_contour], True, (0, 255, 0), 3)

            if current_time < captured_message_until:
                status_text = "Captured label"
                status_color = (0, 255, 0)
            elif label_detected:
                status_text = "Label detected"
                status_color = (0, 255, 0)
            else:
                status_text = "Looking for label..."
                status_color = (0, 255, 255)
            cv2.putText(
                frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Camera Label Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
