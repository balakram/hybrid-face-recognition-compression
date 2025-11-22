#!/usr/bin/env python3
"""
1. Compress image
2. Decompress image
3. Recognize using PCA → LDA → 1NN
"""

import argparse
from pathlib import Path
import pickle
import cv2

from src.compression.full_compression import compress_image, decompress_image
from src.preprocessing.preprocess import preprocess_face
from src.utils.fileio import flatten_image
from src.recognition.pca_eigenfaces import Eigenfaces
from src.recognition.lda_fisherfaces import Fisherfaces


def load_models(model_dir):
    model_dir = Path(model_dir)
    pca = Eigenfaces.load(model_dir/"eigen_model.pkl")
    lda = Fisherfaces.load(model_dir/"fisher_model.pkl")
    d = pickle.load(open(model_dir/"nn_clf.pkl", "rb"))
    clf = d["clf"]
    names = d["names"]
    return pca, lda, clf, names


def main(input_path, model_dir):
    encoded, meta = compress_image(input_path, quality=75)
    recon = decompress_image(encoded, meta)

    pca, lda, clf, names = load_models(model_dir)

    face = preprocess_face(recon)
    x = flatten_image(face)[None, :]
    z1 = pca.transform(x)
    z2 = lda.transform(z1)

    pred = clf.predict(z2)[0]
    print(f"[RESULT] Predicted Person: {names[pred]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model_dir", default="modes")
    args = ap.parse_args()

    main(args.input, args.model_dir)
