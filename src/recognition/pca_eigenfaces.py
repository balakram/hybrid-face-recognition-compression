import numpy as np
from pathlib import Path
import pickle

class Eigenfaces:
    def __init__(self, n_components=100, whiten=False):
        self.n_components = n_components
        self.whiten = whiten
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, X):
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        N, d = X.shape
        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        components = Vt[:self.n_components]
        explained_variance = (S**2) / (N - 1)
        self.components_ = components
        self.explained_variance_ = explained_variance[:self.n_components]
        return self

    def transform(self, X):
        Xc = X - self.mean_
        return np.dot(Xc, self.components_.T)

    def inverse_transform(self, Z):
        return np.dot(Z, self.components_) + self.mean_

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'n_components': self.n_components,
                'mean_': self.mean_,
                'components_': self.components_,
                'explained_variance_': self.explained_variance_,
                'whiten': self.whiten
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls(n_components=data['n_components'], whiten=data.get('whiten', False))
        obj.mean_ = data['mean_']
        obj.components_ = data['components_']
        obj.explained_variance_ = data.get('explained_variance_')
        return obj
