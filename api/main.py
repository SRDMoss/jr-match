from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import os, tempfile, traceback, numpy as np
from core.ontology_loader import load_ontology, id_to_label_map

from core.extractor import extract
from core.embed import embed
from core.search_inmem import InMemIndex
from core.config import USE_FAISS
from core.scoring import score

# Try FAISS import lazily; if missing, we’ll fall back to in-mem
try:
    from adapters.search_faiss import FaissIndex  # type: ignore
except Exception:
    FaissIndex = None  # noqa: N816

app = FastAPI(title="JR Match", docs_url="/docs", redoc_url="/redoc")

ONTO = str((Path(__file__).resolve().parents[1] / "ontology" / "skills.csv"))

def _save_upload_tmp(up: UploadFile) -> str:
    ct = (up.content_type or "").lower()
    ext = ".pdf" if "pdf" in ct else ".docx" if ("word" in ct or "docx" in ct) else ".txt"
    name_ext = os.path.splitext(up.filename or "")[1].lower()
    if name_ext in {".pdf", ".docx", ".txt"}:
        ext = name_ext
    fd, path = tempfile.mkstemp(prefix="jr_", suffix=ext)
    with os.fdopen(fd, "wb") as f:
        f.write(up.file.read())
    up.file.seek(0)
    return path

@app.get("/")
def home():
    return {
        "use": "/docs → POST /match → upload resume + jd",
        "alt": "POST /match with multipart/form-data (resume, jd)"
    }

@app.post("/match")
async def match(resume: UploadFile = File(...), jd: UploadFile = File(...)):
    try:
        rpath = _save_upload_tmp(resume)
        jpath = _save_upload_tmp(jd)

        res = extract(rpath, ONTO, dump_tag="resume")
        job = extract(jpath, ONTO, dump_tag="jd")       

        # Build texts to embed (fallback to sections / whole doc)
        jd_texts = [b.text for b in job.bullets if b.text.strip()]
        if not jd_texts:
            jd_texts = [" ".join(job.sections.skills)] if job.sections.skills else [job.text[:1000] or ""]

        res_texts = [b.text for b in res.bullets if b.text.strip()]
        if not res_texts:
            res_texts = [" ".join(res.sections.skills)] if res.sections.skills else [res.text[:1000] or ""]

        # Choose index backend
        use_faiss = bool(USE_FAISS and FaissIndex is not None)
        idx = (FaissIndex() if use_faiss else InMemIndex())

        # Index JD side
        jd_vecs = embed(jd_texts)
        idx.index(jd_vecs, meta=jd_texts)

        # Per-bullet max sim
        top_sim = 0.0
        for bt in res_texts:
            qv = embed([bt])[0]
            ans = idx.query(qv, k=1)
            if ans:
                top_sim = max(top_sim, ans[0][0])

        # Whole-doc fallback sim
        r_vec = embed([" ".join(res_texts)])[0]
        j_vec = embed([" ".join(jd_texts)])[0]
        doc_sim = float(np.dot(r_vec, j_vec))
        top_sim = max(top_sim, doc_sim)

        detail = score(res, job, top_sim)
        id2label = id_to_label_map(load_ontology(ONTO))


        return {
            **detail.model_dump(),
            "resume_labels":  [id2label.get(x, x) for x in res.skills],
            "jd_labels":      [id2label.get(x, x) for x in job.skills],
            "matched_labels": [id2label.get(x, x) for x in detail.matched_skills],
            "gaps_labels":    [id2label.get(x, x) for x in detail.gaps],
            "_debug": {
                "backend": "faiss" if use_faiss else "inmem",
                "res_bullets": len(res.bullets),
                "jd_bullets": len(job.bullets),
                "res_skills": len(res.skills),
                "jd_skills": len(job.skills),
                "jd_texts_indexed": len(jd_texts),
                "doc_sim": doc_sim,
                "onto_exists": Path(ONTO).exists(),
                "ontology_size": len(id2label),
            },
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback.format_exc(),
            },
        )
