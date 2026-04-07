import cv2

def list_cameras(max_cameras=10):
    available_cameras = []

    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)

        if cap.isOpened():
            ret, frame = cap.read()

            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                print(f"Camera index {index} is available")
                print(f"Resolution: {width}x{height}\n")

                available_cameras.append(index)

            cap.release()

    if not available_cameras:
        print("No cameras detected.")

    return available_cameras


if __name__ == "__main__":
    print("Scanning for cameras...\n")
    cams = list_cameras(10)

    print("Available camera indexes:", cams)