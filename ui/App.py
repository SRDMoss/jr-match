import os, sys, tempfile
# add project root to sys.path so "core" imports work when Streamlit runs from /ui
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from core.extractor import extract
from core.embed import embed
from core.search_inmem import InMemIndex
from core.scoring import score

st.set_page_config(page_title="JR Match", layout="wide")
st.title("JR Match (MVP)")

ROOT = os.path.dirname(os.path.dirname(__file__))
ONTOLOGY_CSV = os.path.join(ROOT, "ontology", "skills.csv")

resume_file = st.file_uploader("Upload Resume (.pdf/.docx)", type=["pdf", "docx"], key="resume")
jd_file = st.file_uploader("Upload Job Description (.pdf/.docx)", type=["pdf", "docx"], key="jd")

if st.button("Run Match") and resume_file and jd_file:
    # save uploads to temp files
    r_ext = os.path.splitext(resume_file.name)[1].lower()
    j_ext = os.path.splitext(jd_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=r_ext) as rf:
        rf.write(resume_file.read())
        rpath = rf.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=j_ext) as jf:
        jf.write(jd_file.read())
        jpath = jf.name

    # parse
    res = extract(rpath, ONTOLOGY_CSV)
    job = extract(jpath, ONTOLOGY_CSV)

    # bullets and embeddings
    res_bullets = [b.text for b in res.bullets] or [""]
    jd_bullets = [b.text for b in job.bullets] or [""]

    idx = InMemIndex()
    idx.index(embed(jd_bullets), meta=jd_bullets)

    # best semantic similarity from resume bullets to JD bullets
    sims = []
    for bt in res_bullets:
        v = embed([bt])[0]
        sims.append(idx.query(v, k=1)[0][0])
    top_sim = max(sims) if sims else 0.0

    # score and display
    detail = score(res, job, top_sim)
    st.metric("Total Score", f"{detail.total:.3f}")
    st.write("Matched Skills (ontology IDs):", detail.matched_skills)
    st.write("Gaps:", detail.gaps)
