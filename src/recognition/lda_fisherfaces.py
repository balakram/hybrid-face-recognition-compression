import numpy as np
from pathlib import Path
import pickle

class Fisherfaces:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.W = None
        self.class_means_ = None
        self.overall_mean_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X, y):
        labels = np.unique(y)
        n_classes = labels.size
        N, d = X.shape
        if self.n_components is None:
            self.n_components = min(n_classes - 1, d)
        self.overall_mean_ = np.mean(X, axis=0)
        Sw = np.zeros((d, d), dtype=np.float64)
        Sb = np.zeros((d, d), dtype=np.float64)
        class_means = {}
        for cls in labels:
            Xi = X[y == cls]
            mean_i = Xi.mean(axis=0)
            class_means[cls] = mean_i
            Xi_centered = Xi - mean_i
            Sw += Xi_centered.T @ Xi_centered
            ni = Xi.shape[0]
            mean_diff = (mean_i - self.overall_mean_).reshape(-1,1)
            Sb += ni * (mean_diff @ mean_diff.T)
        self.class_means_ = class_means
        Sw_inv = np.linalg.pinv(Sw)
        M = Sw_inv @ Sb
        eigvals, eigvecs = np.linalg.eig(M)
        idx = np.argsort(-np.abs(eigvals.real))
        eigvals = eigvals.real[idx]
        eigvecs = eigvecs.real[:, idx]
        W = eigvecs[:, :self.n_components].T
        self.W = W
        total = np.sum(np.abs(eigvals))
        if total > 0:
            self.explained_variance_ratio_ = np.abs(eigvals[:self.n_components]) / total
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components)
        return self

    def transform(self, X):
        return (X @ self.W.T)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'n_components': self.n_components,
                'W': self.W,
                'class_means_': self.class_means_,
                'overall_mean_': self.overall_mean_,
                'explained_variance_ratio_': self.explained_variance_ratio_
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls(n_components=data['n_components'])
        obj.W = data['W']
        obj.class_means_ = data['class_means_']
        obj.overall_mean_ = data['overall_mean_']
        obj.explained_variance_ratio_ = data.get('explained_variance_ratio_')
        return obj
