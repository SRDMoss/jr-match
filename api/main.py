# App.py — Streamlit UI with pluggable vector backends (inmem / sqlite / faiss)

import os, sys, tempfile, uuid

# Make sure "core" is importable whether App.py lives at repo root or in /ui
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import streamlit as st

from core.extractor import extract
from core.embed import embed
from core.search_inmem import InMemIndex
from core.scoring import score
from core.ontology_loader import load_ontology, id_to_label_map
from core.explain import find_evidence_for_matches, suggest_gap_phrases
from core.config import load_weights

# ---- Optional backends (tolerate missing modules) ----
FaissIndex = None
try:
    # Try both potential locations, depending on where you placed the file
    from core.search_faiss import FaissIndex
except Exception:
    try:
        from search_faiss import FaissIndex
    except Exception:
        FaissIndex = None

SQLiteIndex = None
try:
    from core.search_sqlite import SQLiteIndex  # requires you added core/search_sqlite.py
except Exception:
    SQLiteIndex = None

# ---- Streamlit setup ----
st.set_page_config(page_title="JR Match", layout="wide")
st.title("JR Match")

ROOT = REPO_ROOT
ONTOLOGY_CSV = os.path.join(ROOT, "ontology", "skills.csv")
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)

# -------- Sidebar controls --------
backend_choice = st.sidebar.radio(
    "Vector backend",
    options=["inmem", "sqlite", "faiss"],
    index=0,
    help="SQLite persists vectors (zero external deps). FAISS is fastest if available."
)

if backend_choice == "sqlite" and SQLiteIndex is None:
    st.sidebar.warning("SQLite backend not found. Did you add core/search_sqlite.py?")
if backend_choice == "faiss" and FaissIndex is None:
    st.sidebar.warning("FAISS not available. Install faiss-cpu or switch backend.")

db_path = st.sidebar.text_input(
    "SQLite DB path",
    value=os.getenv("JR_SQLITE_PATH", os.path.join(ROOT, "data", "jr_match.sqlite3")),
    help="Only used when backend = sqlite"
)

sim_w, cov_w = load_weights()
st.sidebar.caption(f"Scoring weights → Semantic: **{sim_w:.2f}**  |  Coverage: **{cov_w:.2f}**")

# -------- Load ontology --------
skills = load_ontology(ONTOLOGY_CSV)
id2label = id_to_label_map(skills)

# -------- Inputs --------
resume_file = st.file_uploader("Upload Resume (.pdf/.docx)", type=["pdf", "docx"], key="resume")
jd_file = st.file_uploader("Upload Job Description (.pdf/.docx)", type=["pdf", "docx"], key="jd")

run = st.button("Run Match", type="primary")

if run and resume_file and jd_file:
    # Save uploads to temp files
    r_ext = os.path.splitext(resume_file.name)[1].lower() or ".pdf"
    j_ext = os.path.splitext(jd_file.name)[1].lower() or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=r_ext) as rf:
        rf.write(resume_file.read())
        rpath = rf.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=j_ext) as jf:
        jf.write(jd_file.read())
        jpath = jf.name

    # Parse docs
    res = extract(rpath, ONTOLOGY_CSV, dump_tag="resume")
    job = extract(jpath, ONTOLOGY_CSV, dump_tag="jd")

    # Prepare bullet texts (with sensible fallbacks)
    res_bullets = [b.text for b in res.bullets if b.text.strip()] or (
        [" ".join(res.sections.skills)] if res.sections.skills else [res.text[:1000] or ""]
    )
    jd_bullets = [b.text for b in job.bullets if b.text.strip()] or (
        [" ".join(job.sections.skills)] if job.sections.skills else [job.text[:1000] or ""]
    )

    # -------- Select backend --------
    backend_name = "inmem"
    if backend_choice == "sqlite" and SQLiteIndex is not None:
        idx_name = f"ui_{uuid.uuid4().hex[:8]}"          # short-lived index per run
        idx = SQLiteIndex(db_path=db_path, index_name=idx_name)
        backend_name = "sqlite"
    elif backend_choice == "faiss" and FaissIndex is not None:
        idx = FaissIndex()
        backend_name = "faiss"
    else:
        if backend_choice != "inmem":
            st.info("Falling back to InMem backend.")
        idx = InMemIndex()

    # -------- Index JD bullets --------
    jd_vecs = embed(jd_bullets)
    idx.index(jd_vecs, meta=jd_bullets)

    # -------- Compute best semantic similarity --------
    sims = []
    for bt in res_bullets:
        q = embed([bt])[0]
        ans = idx.query(q, k=1)
        if ans:
            sims.append(ans[0][0])
    top_sim = max(sims) if sims else 0.0

    # -------- Score --------
    detail = score(res, job, top_sim)

    # -------- Display --------
    st.metric("Total Score", f"{detail.total:.3f}")
    st.caption(f"Backend: **{backend_name}**  ·  Resume bullets: {len(res_bullets)}  ·  JD bullets: {len(jd_bullets)}")

    matched_pairs = sorted([(sid, id2label.get(sid, sid)) for sid in detail.matched_skills], key=lambda x: x[1].lower())
    gap_pairs = sorted([(sid, id2label.get(sid, sid)) for sid in detail.gaps], key=lambda x: x[1].lower())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matched Skills")
        st.write([f"{sid} — {lbl}" for sid, lbl in matched_pairs] or "—")
    with col2:
        st.subheader("Gaps")
        st.write([f"{sid} — {lbl}" for sid, lbl in gap_pairs] or "—")

    st.caption(f"Weights — Semantic: {sim_w:.2f}, Coverage: {cov_w:.2f}")

    # -------- Explainability --------
    st.subheader("Evidence & Suggestions")

    evidence = find_evidence_for_matches(res.bullets, detail.matched_skills, ONTOLOGY_CSV)
    if any(evidence.values()):
        with st.expander("Evidence sentences (from resume bullets)"):
            for sid, lbl in matched_pairs:
                ev = evidence.get(sid)
                if ev:
                    st.markdown(f"- **{lbl}**: {ev}")
    else:
        st.caption("No direct alias hits found in bullets for matched skills.")

    gap_suggestions = suggest_gap_phrases(detail.gaps, ONTOLOGY_CSV)
    if gap_suggestions:
        with st.expander("Suggested phrasing for missing skills"):
            for sid, lbl in gap_pairs:
                st.markdown(f"- **{lbl}**: {gap_suggestions.get(sid)}")
