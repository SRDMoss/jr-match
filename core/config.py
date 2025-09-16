import json, os

USE_FAISS = True
_DEFAULT = {"sim_weight": 0.6, "cov_weight": 0.4}

def load_weights(cfg_path: str | None = None) -> tuple[float, float]:
    """Load (sim_weight, cov_weight) from config.json with safe defaults."""
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        sw = float(cfg.get("sim_weight", _DEFAULT["sim_weight"]))
        cw = float(cfg.get("cov_weight", _DEFAULT["cov_weight"]))
        return sw, cw
    except Exception:
        return _DEFAULT["sim_weight"], _DEFAULT["cov_weight"]

def backend_from_env() -> str:
    """
    Returns one of: 'faiss', 'sqlite', 'inmem'
    Priority: ENV BACKEND → USE_FAISS → default 'inmem'
    """
    b = os.getenv("JR_BACKEND", "").strip().lower()
    if b in {"faiss", "sqlite", "inmem"}:
        return b
    if USE_FAISS:
        return "faiss"
    return "inmem"

def sqlite_path_from_env() -> str:
    return os.getenv("JR_SQLITE_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jr_match.sqlite3"))
