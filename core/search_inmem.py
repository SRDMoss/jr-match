import numpy as np

class InMemIndex:
    def __init__(self):
        self.vecs = None
        self.meta = []

    def index(self, vectors: np.ndarray, meta):
        self.vecs = vectors
        self.meta = list(meta)

    def query(self, vector: np.ndarray, k: int = 5):
        sims = (self.vecs @ vector.reshape(-1,))  # cosine if normalized
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i]) for i in idx]
