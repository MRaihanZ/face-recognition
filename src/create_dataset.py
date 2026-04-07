import cv2
import numpy as np
import insightface
import pickle

# Load model
app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=0)

cap = cv2.VideoCapture(1)

print("Press S to save face")

while True:
    ret, frame = cap.read()

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow("Enroll Face", frame)

    key = cv2.waitKey(1)

    if key == ord('s') and len(faces) > 0:
        embedding = faces[0].embedding

        with open("authorized_face.pkl", "wb") as f:
            pickle.dump(embedding, f)

        print("Face saved!")
        break

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()