# core/search_sqlite.py
from __future__ import annotations
import sqlite3, os, json, time
from typing import Iterable, List, Tuple
import numpy as np

_SQL = """
CREATE TABLE IF NOT EXISTS jr_vecs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    index_name TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,           -- float32, contiguous, normalized
    meta TEXT NOT NULL,          -- JSON-encoded metadata (e.g., the text)
    created_at REAL NOT NULL     -- unix timestamp
);
CREATE INDEX IF NOT EXISTS idx_jr_vecs_index_name ON jr_vecs(index_name);
"""

def _ensure_db(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.executescript(_SQL)
        con.commit()
    finally:
        con.close()

def _as_bytes(arr: np.ndarray) -> bytes:
    a = np.asarray(arr, dtype="float32")
    if a.ndim != 1:
        a = a.reshape(-1,)
    return a.tobytes(order="C")

def _from_bytes(b: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(b, dtype="float32", count=dim)

class SQLiteIndex:
    """
    Simple cosine-sim backend on SQLite.
    API matches other backends: index(), query().
    """
    def __init__(self, db_path: str = "data/jr_match.sqlite3", index_name: str = "default"):
        self.db_path = db_path
        self.index_name = index_name
        _ensure_db(self.db_path)
        self._dim: int | None = None

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def clear(self):
        with self._conn() as con:
            con.execute("DELETE FROM jr_vecs WHERE index_name=?", (self.index_name,))
            con.commit()

    def index(self, vectors: np.ndarray, meta: Iterable[str]):
        vecs = np.asarray(vectors, dtype="float32")
        if vecs.ndim != 2:
            raise ValueError("vectors must be 2D array of shape (n, d)")
        n, d = vecs.shape
        self._dim = d
        # ensure normalized (cosine)
        # (if upstream is already normalized, this is a no-op)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms

        now = time.time()
        rows = [(self.index_name, d, _as_bytes(vecs[i]), json.dumps(m), now)
                for i, m in enumerate(list(meta))]
        with self._conn() as con:
            con.executemany(
                "INSERT INTO jr_vecs (index_name, dim, vec, meta, created_at) VALUES (?,?,?,?,?)",
                rows
            )
            con.commit()

    def query(self, vector: np.ndarray, k: int = 5) -> List[Tuple[float, str]]:
        q = np.asarray(vector, dtype="float32").reshape(-1,)
        # normalize query
        q = q / (np.linalg.norm(q) + 1e-12)

        with self._conn() as con:
            cur = con.execute(
                "SELECT dim, vec, meta FROM jr_vecs WHERE index_name=?",
                (self.index_name,)
            )
            rows = cur.fetchall()

        if not rows:
            return []

        # determine dim from first row (all rows share dim)
        dim = rows[0][0]
        V = np.vstack([_from_bytes(b, dim) for _, b, _ in rows])  # (N, d)
        metas = [m for *_, m in rows]

        sims = V @ q  # cosine on normalized vectors
        idx = np.argsort(-sims)[:k]
        out = [(float(sims[i]), metas[i]) for i in idx]

        # deserialize meta if it’s JSON; fall back to raw string
        def _try(m):
            try:
                return json.loads(m)
            except Exception:
                return m
        return [(s, _try(m)) for s, m in out]
