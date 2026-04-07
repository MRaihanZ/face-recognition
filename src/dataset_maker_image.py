import cv2
import os

SAVE_DIR = "dataset_raw"
TARGET_SIZE = 1024

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(3)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access camera")
        break

    h, w, _ = frame.shape

    # Center square crop
    size = min(h, w)
    x = (w - size) // 2
    y = (h - size) // 2

    square = frame[y:y+size, x:x+size]

    # Resize to 1024x1024
    resized = cv2.resize(square, (TARGET_SIZE, TARGET_SIZE))

    cv2.imshow("Camera Preview", resized)

    key = cv2.waitKey(1)

    if key == ord(' '):  # SPACE to capture
        filename = f"{SAVE_DIR}/face_{count:04d}.jpg"
        cv2.imwrite(filename, resized)
        print("Saved:", filename)
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()