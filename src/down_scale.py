import cv2
import os

INPUT_DIR = "dataset_raw"
OUTPUT_DIR = "dataset_512"
TARGET_SIZE = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):

    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(INPUT_DIR, file)

    img = cv2.imread(path)

    if img is None:
        continue

    resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))

    save_path = os.path.join(OUTPUT_DIR, file)

    cv2.imwrite(save_path, resized)

    print("Saved:", save_path)

print("Finished resizing dataset.")