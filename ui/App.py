import os, sys, tempfile
# add project root to sys.path so "core" imports work when Streamlit runs from /ui
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from core.extractor import extract
from core.embed import embed
from core.search_inmem import InMemIndex
from core.scoring import score
from core.ontology_loader import load_ontology, id_to_label_map
from core.explain import find_evidence_for_matches, suggest_gap_phrases
from core.config import load_weights


st.set_page_config(page_title="JR Match", layout="wide")
st.title("JR Match")

ROOT = os.path.dirname(os.path.dirname(__file__))
ONTOLOGY_CSV = os.path.join(ROOT, "ontology", "skills.csv")

skills = load_ontology(ONTOLOGY_CSV)
id2label = id_to_label_map(skills)

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
    matched_pairs = sorted([(sid, id2label.get(sid, sid)) for sid in detail.matched_skills], key=lambda x: x[1].lower())
    gap_pairs = sorted([(sid, id2label.get(sid, sid)) for sid in detail.gaps], key=lambda x: x[1].lower())
    sim_w, cov_w = load_weights()

    st.metric("Total Score", f"{detail.total:.3f}")
    st.subheader("Matched Skills")
    st.write([f"{sid} — {lbl}" for sid, lbl in matched_pairs])
    st.caption(f"Weights — Semantic: {sim_w:.2f}, Coverage: {cov_w:.2f}")

    st.subheader("Gaps")
    st.write([f"{sid} — {lbl}" for sid, lbl in gap_pairs])


    # ---- Explainability ----
    st.subheader("Evidence & Suggestions")

    # Evidence sentences for matched skills
    evidence = find_evidence_for_matches(res.bullets, detail.matched_skills, ONTOLOGY_CSV)
    if any(evidence.values()):
        with st.expander("Evidence sentences (from resume bullets)"):
            for sid, lbl in matched_pairs:
                ev = evidence.get(sid)
                if ev:
                    st.markdown(f"- **{lbl}**: {ev}")
    else:
        st.caption("No direct alias hits found in bullets for matched skills.")

    # Suggested phrasing for gaps
    gap_suggestions = suggest_gap_phrases(detail.gaps, ONTOLOGY_CSV)
    if gap_suggestions:
        with st.expander("Suggested phrasing for missing skills"):
            for sid, lbl in gap_pairs:
                st.markdown(f"- **{lbl}**: {gap_suggestions.get(sid)}")