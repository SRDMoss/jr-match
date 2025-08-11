from typing import List, Dict
import pdfplumber
from docx import Document
import re
from .schemas import ParsedDoc, Sections, Bullet
from .ontology_loader import alias_to_id_map, load_ontology

BULLET_RX = re.compile(r"^[\-\u2022\*\•]\s+")
SKILL_SPLIT = re.compile(r"[,\|;/]")

def _read_pdf(p: str) -> str:
    with pdfplumber.open(p) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def _read_docx(p: str) -> str:
    doc = Document(p)
    return "\n".join(p.text for p in doc.paragraphs)

def _split_sections(text: str) -> Sections:
    t = text.splitlines()
    exp, edu, skl = [], [], []
    current = None
    for line in t:
        L = line.strip()
        if not L: 
            continue
        h = L.lower()
        if "experience" in h: current = exp; continue
        if "education" in h: current = edu; continue
        if "skill" in h: current = skl; continue
        (current or exp).append(L)
    return Sections(experience=exp, education=edu, skills=skl)

def extract(path: str, ontology_csv: str) -> ParsedDoc:
    text = _read_pdf(path) if path.lower().endswith(".pdf") else _read_docx(path)
    sections = _split_sections(text)
    bullets: List[Bullet] = []
    for line in sections.experience:
        if BULLET_RX.search(line):
            bullets.append(Bullet(text=BULLET_RX.sub("", line)))
    skills_map = alias_to_id_map(load_ontology(ontology_csv))
    found_ids = set()
    for raw in SKILL_SPLIT.split(" ".join(sections.skills)):
        k = raw.strip().lower()
        if k and k in skills_map:
            found_ids.add(skills_map[k])
    return ParsedDoc(skills=sorted(found_ids), bullets=bullets, sections=sections)
