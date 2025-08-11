# JR Match – Ontology-Aware Resume–Job Matcher (MVP)

**JR Match** is a lightweight application that parses resumes and job descriptions, normalizes skills via a local ontology, embeds text for semantic similarity, scores matches using a hybrid (vector + coverage) approach, and displays results in a Streamlit UI with matched skills, evidence, and gaps.

## 🚀 Features
- **Ontology-backed skill matching** – Normalizes aliases (e.g., "JS" → "JavaScript") for consistent scoring.
- **Semantic similarity search** – Embeds resume and JD text with `sentence-transformers` to catch fuzzy matches.
- **Hybrid scoring** – Combines semantic similarity and ontology skill coverage for more accurate results.
- **Explainability** – Displays matched skills, missing skills (gaps), and evidence sentences.
- **Extensible** – Built with swappable backends (FAISS-ready, SQL-ready).

## 🗂 Project Structure

jr-match/
│   README.md
│   requirements.txt
│   .gitignore
│
├── core/                # Core matching logic
│   ├── **init**.py
│   ├── schemas.py
│   ├── config.py
│   ├── ontology_loader.py
│   ├── extractor.py
│   ├── embed.py
│   ├── explain.py
│   ├── search_inmem.py
│   └── scoring.py
│
├── ontology/            # Skills ontology CSV
│   └── skills.csv
│
├── data/
│   └── sample/
│       ├── resumes/     # Sample resume files (PDF/DOCX)
│       └── jds/         # Sample job descriptions (PDF/DOCX)
│
├── ui/                  # Streamlit front-end
│   ├── **init**.py
│   └── app.py
│
├── api/                 # Optional FastAPI backend
│   ├── **init**.py
│   └── main.py
└── notebooks/           # Scratch demos and experiments

## 📦 Requirements

- Python 3.9+  
- Virtual environment recommended

**Python dependencies** (see `requirements.txt`):
- `streamlit`
- `fastapi` (optional API)
- `uvicorn` (optional API server)
- `spacy`
- `sentence-transformers`
- `numpy`
- `pandas`
- `scikit-learn`
- `pdfplumber`
- `python-docx`
- `rapidfuzz`
- `pydantic`
- `typer`

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/yourusername/jr-match.git
cd jr-match
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Make sure the `ontology/skills.csv` file is present (sample provided).  
Place sample resumes and job descriptions in `data/sample/resumes/` and `data/sample/jds/`.

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run ui/app.py
```

This will launch a local web UI where you can:
1. Upload a resume and job description (PDF/DOCX)  
2. See overall match score  
3. View matched skills, missing skills (gaps), and evidence sentences

---

## 💡 Example

With the included sample data:

- Resume: `data/sample/resumes/sample_resume.docx`
- JD: `data/sample/jds/sample_jd.docx`

You should see:
- Hybrid score: semantic + coverage
- Matched skills (ontology-normalized)
- Missing skills with canonical IDs
- Evidence lines from the resume

---

## 📝 TODO

- **UI polish**: Add match score visualization (bar/percentage) and skill category grouping.
- **Evidence enhancement**: Highlight matched skill phrases within evidence sentences.
- **Batch mode**: Compare one resume to multiple JDs at once.
- **Backend options**: Implement FAISS and SQLite/MySQL adapters.
- **Evaluation**: Add simple metrics (precision@k, MRR) for small gold set.
- **Config in UI**: Let users adjust semantic/coverage weight from Streamlit sidebar.
- **Export**: Allow exporting match results as JSON/CSV.

---
