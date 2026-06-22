from collections import deque

import cv2


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


def main():
    camera = cv2.VideoCapture(0)
    detection_history = deque(maxlen=10)
    last_valid_contour = None

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

            label_contour = find_label_contour(frame)
            frame_detected = label_contour is not None
            detection_history.append(frame_detected)

            if frame_detected:
                last_valid_contour = label_contour
                cv2.polylines(frame, [label_contour], True, (0, 255, 255), 3)

            label_detected = sum(detection_history) >= 3

            if label_detected and last_valid_contour is not None:
                cv2.polylines(frame, [last_valid_contour], True, (0, 255, 0), 3)

            status_text = "Label detected" if label_detected else "Looking for label..."
            status_color = (0, 255, 0) if label_detected else (0, 255, 255)
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
