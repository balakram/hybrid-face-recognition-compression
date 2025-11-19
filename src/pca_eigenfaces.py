"""
pca_eigenfaces.py - PCA (Eigenfaces) implementation and utilities
"""
import numpy as np
from sklearn.decomposition import PCA
import pickle

class Eigenfaces:
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
        self.mean_face = None

    def fit(self, X):
        \"\"\" Fit PCA on flattened images X (n_samples, n_features) \"\"\"
        self.mean_face = np.mean(X, axis=0)
        Xc = X - self.mean_face
        self.pca.fit(Xc)
        return self

    def transform(self, X):
        Xc = X - self.mean_face
        return self.pca.transform(Xc)

    def inverse_transform(self, X_proj):
        X_rec = self.pca.inverse_transform(X_proj) + self.mean_face
        return X_rec

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'n_components': self.n_components, 'mean_face': self.mean_face, 'pca': self.pca}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls(n_components=data['n_components'])
        obj.mean_face = data['mean_face']
        obj.pca = data['pca']
        return obj
