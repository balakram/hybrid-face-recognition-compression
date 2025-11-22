#!/usr/bin/env python3
"""
Real-time recognizer that loads PCA (eigenfaces), LDA (fisherfaces) and NN classifier from modes/
Displays bounding boxes and predicted name + score on webcam feed.
Usage:
    python examples/recognize_live.py --model_dir modes --camera 0
"""
#!/usr/bin/env python3

import argparse
import pickle
import cv2
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.preprocess import preprocess_face
from src.utils.fileio import flatten_image
from src.recognition.pca_eigenfaces import Eigenfaces
from src.recognition.lda_fisherfaces import Fisherfaces
from src.recognition.classifier import NearestNeighborClassifier


from src.preprocessing.preprocess import preprocess_face # fallback if installed differently

# We'll attempt to import from expected paths; if not found, fallback to local module names in this package
try:
    from src.preprocessing.preprocess import preprocess_face
except Exception:
    def preprocess_face(img, size=(200,200), apply_eq=False):
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        gray = cv2.resize(gray, size)
        if apply_eq:
            gray = cv2.equalizeHist(gray.astype('uint8'))
        arr = gray.astype('float32')
        if arr.max() > 1.1:
            arr = arr / 255.0
        return arr

def load_models(model_dir):
    model_dir = Path(model_dir)
    pca = Eigenfaces.load(model_dir / "eigen_model.pkl")
    lda = Fisherfaces.load(model_dir / "fisher_model.pkl")
    with open(model_dir / "nn_clf.pkl", "rb") as f:
        d = pickle.load(f)
    clf = d["clf"]
    names = d["names"]
    return pca, lda, clf, names

def predict_face(pca, lda, clf, names, face_img):
    # face_img : BGR or gray cropped face (aligned roughly), size ~200x200
    proc = preprocess_face(face_img, size=(200,200), apply_eq=False)
    x = flatten_image(proc)[None, :]
    z_pca = pca.transform(x)
    z_lda = lda.transform(z_pca)
    pred = clf.predict(z_lda)[0]
    # compute simple confidence as inverse distance to nearest
    dists = np.linalg.norm(clf.Z - z_lda, axis=1)
    min_dist = dists.min()
    conf = max(0.0, 1.0 - min_dist / 10.0)
    return names[pred] if names and pred < len(names) else str(pred), conf

def main(model_dir="modes", camera_idx=0):
    pca, lda, clf, names = load_models(model_dir)
    cap = cv2.VideoCapture(int(camera_idx))
    if not cap.isOpened():
        print("Cannot open camera", camera_idx); return
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("[INFO] Real-time recognition started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue
            name, conf = predict_face(pca, lda, clf, names, face_crop)
            text = f"{name} ({conf:.2f})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Real-time Recognizer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="modes")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    main(model_dir=args.model_dir, camera_idx=args.camera)
