# adapters/search_faiss.py
import numpy as np
try:
    import faiss
except ImportError as e:
    raise ImportError("faiss is required for FaissIndex (pip install faiss-cpu)") from e


class FaissIndex:
    def __init__(self):
        self._index = None
        self._meta = []

    def index(self, vectors: np.ndarray, meta):
        # Expect normalized float32 embeddings (same as embed.py)
        if not isinstance(vectors, np.ndarray):
            vectors = np.asarray(vectors, dtype="float32")
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32", copy=False)

        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D array of shape (n, d)")
        if len(meta) != vectors.shape[0]:
            raise ValueError("meta length must match number of vectors")

        d = vectors.shape[1]
        self._index = faiss.IndexFlatIP(d)  # cosine if inputs normalized
        # Ensure C-contiguous
        self._index.add(np.ascontiguousarray(vectors))
        self._meta = list(meta)

    def query(self, vector: np.ndarray, k: int = 5):
        if self._index is None:
            raise RuntimeError("Index is empty. Call index(vectors, meta) first.")

        if not isinstance(vector, np.ndarray):
            vector = np.asarray(vector, dtype="float32")
        if vector.dtype != np.float32:
            vector = vector.astype("float32", copy=False)

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise ValueError("vector must be a 1D array or shape (1, d)")

        # Search
        D, I = self._index.search(np.ascontiguousarray(vector), k)
        scores = D[0].tolist()
        idxs = I[0].tolist()

        out = []
        for s, i in zip(scores, idxs):
            if i == -1:
                continue
            out.append((float(s), self._meta[i]))
        return out
