"""
lda_fisherfaces.py - LDA (Fisherfaces) wrapper
"""
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Fisherfaces:
    def __init__(self, n_components=None):
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.is_fitted = False

    def fit(self, X, y):
        self.lda.fit(X, y)
        self.is_fitted = True
        return self

    def transform(self, X):
        return self.lda.transform(X)

    def predict(self, X):
        return self.lda.predict(X)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.lda, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            lda = pickle.load(f)
        obj = cls()
        obj.lda = lda
        obj.is_fitted = True
        return obj
