import cv2
import numpy as np
import insightface
import pickle

# Load stored embedding
with open("authorized_face.pkl", "rb") as f:
    authorized_embedding = pickle.load(f)

# Load model
app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=0)

cap = cv2.VideoCapture(3)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

THRESHOLD = 0.5

while True:
    ret, frame = cap.read()

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        embedding = face.embedding

        similarity = cosine_similarity(embedding, authorized_embedding)

        if similarity > THRESHOLD:
            label = "AUTHORIZED"
            color = (0,255,0)
        else:
            label = "UNKNOWN"
            color = (0,0,255)

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f"{label} {similarity:.2f}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()