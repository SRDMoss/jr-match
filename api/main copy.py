from fastapi import FastAPI, UploadFile, File
from core.extractor import extract
from core.embed import embed
from core.search_inmem import InMemIndex
from adapters.search_faiss import FaissIndex
from core.config import USE_FAISS
from core.scoring import score
import os, tempfile
import numpy as np

app = FastAPI(title="JR Match", docs_url="/docs", redoc_url="/redoc")

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
    rpath = _save_upload_tmp(resume)
    jpath = _save_upload_tmp(jd)

    res = extract(rpath, "ontology/skills.csv")
    job = extract(jpath, "ontology/skills.csv")

    # -------- ensure we have something to embed for JD bullets --------
    jd_texts = [b.text for b in job.bullets if b.text.strip()]
    if not jd_texts:
        if job.sections.skills:
            jd_texts = [" ".join(job.sections.skills)]
        else:
            jd_texts = [getattr(job, "text", "")[:1000] or ""]

    # -------- ensure we have something to embed for resume bullets -----
    res_texts = [b.text for b in res.bullets if b.text.strip()]
    if not res_texts:
        if res.sections.skills:
            res_texts = [" ".join(res.sections.skills)]
        else:
            res_texts = [getattr(res, "text", "")[:1000] or ""]

    idx = FaissIndex() if USE_FAISS else InMemIndex()

    jd_vecs = embed(jd_texts)
    idx.index(jd_vecs, meta=jd_texts)

    top_sim = 0.0
    for bt in res_texts:
        qv = embed([bt])[0]
        ans = idx.query(qv, k=1)
        if ans:
            top_sim = max(top_sim, ans[0][0])

    # whole-doc fallback
    r_doc = " ".join(res_texts)
    j_doc = " ".join(jd_texts)
    r_vec = embed([r_doc])[0]
    j_vec = embed([j_doc])[0]
    doc_sim = float(np.dot(r_vec, j_vec))
    top_sim = max(top_sim, doc_sim)

    detail = score(res, job, top_sim)

    return {
        **detail.model_dump(),
        "_debug": {
            "res_bullets": len(res.bullets),
            "jd_bullets": len(job.bullets),
            "res_skills": len(res.skills),
            "jd_skills": len(job.skills),
            "jd_texts_indexed": len(jd_texts),
            "doc_sim": doc_sim,
        },
    }