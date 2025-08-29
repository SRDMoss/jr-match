import numpy as np

class InMemIndex:
    def __init__(self):
        self.vecs = None
        self.meta = []

    def index(self, vectors: np.ndarray, meta):
        if vectors is None or len(vectors) == 0:
            self.vecs = np.zeros((0, 1), dtype="float32")
            self.meta = []
            return
        self.vecs = np.asarray(vectors, dtype="float32")
        self.meta = list(meta)

    def query(self, vector: np.ndarray, k: int = 5):
        if self.vecs is None or len(self.vecs) == 0:
            return []
        v = np.asarray(vector, dtype="float32").reshape(-1,)
        sims = (self.vecs @ v)  # cosine if normalized
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i]) for i in idx]
