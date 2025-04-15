import cv2
import numpy as np

# Define HSV color ranges
COLOR_RANGES = {
    "Red": [((0, 100, 100), (10, 255, 255)), ((160, 100, 100), (180, 255, 255))],
    "Green": [((40, 70, 70), (80, 255, 255))],
    "Blue": [((100, 150, 0), (140, 255, 255))],
    "Yellow": [((20, 100, 100), (30, 255, 255))],
    "Orange": [((10, 100, 100), (20, 255, 255))],
    "Purple": [((140, 100, 100), (160, 255, 255))],
    "White": [((0, 0, 200), (180, 20, 255))],
    "Black": [((0, 0, 0), (180, 255, 30))],
}

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, hsv_ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype="uint8")

        for lower, upper in hsv_ranges:
            lower_np = np.array(lower, dtype="uint8")
            upper_np = np.array(upper, dtype="uint8")
            mask |= cv2.inRange(hsv, lower_np, upper_np)

        # Morphological operations to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

        # Find contours for the current color
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Filter small noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, color_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Color Detection - Rectangle Overlay", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
