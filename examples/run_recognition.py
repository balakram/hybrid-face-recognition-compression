import argparse
from pathlib import Path
import pickle
from src.utils.fileio import load_dataset
from src.recognition.pca_eigenfaces import Eigenfaces
from src.recognition.lda_fisherfaces import Fisherfaces
from src.recognition.classifier import NearestNeighborClassifier
from src.utils.metrics import accuracy_score

def train_pipeline(data_root, model_dir, n_pca=80, n_lda=None):
    X, y, names = load_dataset(data_root, image_size=(200,200))
    pca = Eigenfaces(n_components=n_pca)
    pca.fit(X)
    Z_pca = pca.transform(X)
    if n_lda is None:
        n_lda = min(len(names)-1, Z_pca.shape[1])
    lda = Fisherfaces(n_components=n_lda)
    lda.fit(Z_pca, y)
    Z_lda = lda.transform(Z_pca)
    clf = NearestNeighborClassifier()
    clf.fit(Z_lda, y)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    pca.save(model_dir / "eigen_model.pkl")
    lda.save(model_dir / "fisher_model.pkl")
    with open(model_dir / "nn_clf.pkl", "wb") as f:
        pickle.dump({'clf': clf, 'names': names}, f)
    preds = clf.predict(Z_lda)
    acc = accuracy_score(y, preds)
    print(f"Train-set accuracy: {acc*100:.2f}%")
