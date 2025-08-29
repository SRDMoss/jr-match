from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def get_model(name: str = "all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        _model = SentenceTransformer(name)
    return _model

def embed(texts, model_name="all-MiniLM-L6-v2"):
    m = get_model(model_name)
    vecs = m.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")
