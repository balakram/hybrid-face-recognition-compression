import numpy as np

class NearestNeighborClassifier:
    def __init__(self):
        self.Z = None
        self.y = None

    def fit(self, Z, y):
        self.Z = np.asarray(Z)
        self.y = np.asarray(y)
        return self

    def predict(self, Zq):
        Zq = np.atleast_2d(Zq)
        preds = []
        for v in Zq:
            dists = np.linalg.norm(self.Z - v, axis=1)
            idx = np.argmin(dists)
            preds.append(self.y[idx])
        return np.array(preds, dtype=self.y.dtype)

    def score(self, Z, y):
        preds = self.predict(Z)
        return (preds == y).mean()
