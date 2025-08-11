from fastapi import FastAPI, UploadFile
from core.extractor import extract
from core.embed import embed
from core.search_inmem import InMemIndex
from core.scoring import score

app = FastAPI()

@app.post("/match")
async def match(resume: UploadFile, jd: UploadFile):
    rpath = f"/tmp_{resume.filename}"
    jpath = f"/tmp_{jd.filename}"
    with open(rpath,"wb") as f: f.write(await resume.read())
    with open(jpath,"wb") as f: f.write(await jd.read())
    res = extract(rpath, "ontology/skills.csv")
    job = extract(jpath, "ontology/skills.csv")
    idx = InMemIndex()
    jd_vecs = embed([b.text for b in job.bullets] or [""])
    idx.index(jd_vecs, meta=[b.text for b in job.bullets] or [""])
    top_sim = 0.0
    for bt in res.bullets:
        top_sim = max(top_sim, idx.query(embed([bt.text])[0], k=1)[0][0])
    detail = score(res, job, top_sim)
    return detail.model_dump()
