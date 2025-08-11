from typing import List, Set
import pdfplumber
from docx import Document
import re
import os

from .schemas import ParsedDoc, Sections, Bullet
from .ontology_loader import alias_to_id_map, load_ontology, category_ids

# Regex patterns
BULLET_RX = re.compile(r"^[\-\u2022\*\•]\s+")
SKILL_SPLIT = re.compile(r"[,\|;/]")
TOKEN_RX = re.compile(r"[a-z0-9\+\#\.]+")  # keep +, #, .

def _read_pdf(path: str) -> str:
    """Extracts all text from a PDF file."""
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def _read_docx(path: str) -> str:
    """Extracts all text from a DOCX file."""
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def _split_sections(text: str) -> Sections:
    """Naive section splitter based on common headings."""
    lines = text.splitlines()
    exp, edu, skl = [], [], []
    current = None
    for line in lines:
        L = line.strip()
        if not L:
            continue
        h = L.lower()
        if "experience" in h:
            current = exp
            continue
        if "education" in h:
            current = edu
            continue
        if "skill" in h:
            current = skl
            continue
        (current or exp).append(L)
    return Sections(experience=exp, education=edu, skills=skl)

def _normalize_tokens(text: str) -> List[str]:
    return TOKEN_RX.findall(text.lower())

def _find_bullets(lines: List[str]) -> List[Bullet]:
    """Extract bullet points from a list of lines."""
    bullets = []
    for line in lines:
        if BULLET_RX.search(line):
            clean = BULLET_RX.sub("", line).strip()
            if clean:
                bullets.append(Bullet(text=clean))
    return bullets

# def _match_skills_in_text(text: str, alias_map: dict) -> Set[str]:
#     found_ids = set()
#     tokens = _normalize_tokens(text)
#     token_set = set(tokens)
#     joined = " ".join(tokens)

#     for alias, sid in alias_map.items():
#         alias_tokens = alias.split()
#         if len(alias_tokens) == 1:
#             a = alias_tokens[0]
#             # exact token match (not substring)
#             if a in token_set:
#                 found_ids.add(sid)
#         else:
#             # multi-token phrase match with boundaries
#             pat = r"\b" + r"\s+".join(map(re.escape, alias_tokens)) + r"\b"
#             if re.search(pat, joined):
#                 found_ids.add(sid)
#     return found_ids

def _match_skills_in_text(text: str, alias_map: dict) -> Set[str]:
    found_ids = set()
    tokens = _normalize_tokens(text)  # e.g., ['javascript','vue.js','node.js','docker','kubernetes']
    token_set = set(tokens)

    # Build token variants once (handles dot/suffix cases)
    variant_token_set = set()
    for t in token_set:
        variant_token_set.add(t)
        if "." in t:
            variant_token_set.add(t.replace(".", ""))   # 'vuejs'
            parts = t.split(".")
            if parts[0]:
                variant_token_set.add(parts[0])         # 'vue'
        if "-" in t:
            variant_token_set.add(t.replace("-", " "))  # 'google-cloud' -> 'google cloud'
            variant_token_set.add(t.replace("-", ""))   # 'googlecloud'
        if "/" in t:
            variant_token_set.update(p for p in t.split("/") if p)

    joined = " ".join(tokens)
    joined_variants = " ".join(sorted(variant_token_set))

    for alias, sid in alias_map.items():
        alias_tokens = alias.split()
        if len(alias_tokens) == 1:
            a = alias_tokens[0]
            if a in token_set or a in variant_token_set:
                found_ids.add(sid)
        else:
            # multi-token phrase: try against both normal and variant-joined space text
            pat = r"\b" + r"\s+".join(map(re.escape, alias_tokens)) + r"\b"
            if re.search(pat, joined) or re.search(pat, joined_variants):
                found_ids.add(sid)
    return found_ids

def extract(path: str, ontology_csv: str) -> ParsedDoc:
    """Main extraction function."""
    # Read file
    if path.lower().endswith(".pdf"):
        text = _read_pdf(path)
    elif path.lower().endswith(".docx"):
        text = _read_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    # Split into sections
    sections = _split_sections(text)

    # Collect bullets from experience & education
    bullets = _find_bullets(sections.experience) + _find_bullets(sections.education)

    # Load ontology and alias map
    skills_map = alias_to_id_map(load_ontology(ontology_csv))

    # # Match skills in the "Skills" section
    # skills_from_skills_section = set()
    # for raw in SKILL_SPLIT.split(" ".join(sections.skills)):
    #     k = raw.strip().lower()
    #     if k and k in skills_map:
    #         skills_from_skills_section.add(skills_map[k])
    skills_from_skills_section = _match_skills_in_text(" ".join(sections.skills), skills_map)

    # ALSO match skills anywhere in the document
    skills_from_full_text = _match_skills_in_text(text, skills_map)

    # Combine all found skills
    all_skills = sorted(skills_from_skills_section | skills_from_full_text)

    skills_dict = load_ontology(ontology_csv)
    cat_ids = category_ids(skills_dict)
    filtered_skills = [s for s in all_skills if s not in cat_ids]

    print("RAW Skills Section:", sections.skills)
    print("Skills from skills section:", skills_from_skills_section)
    print("Skills from full text:", skills_from_full_text)

    return ParsedDoc(
        skills=sorted(filtered_skills),
        bullets=bullets,
        sections=sections
    )


