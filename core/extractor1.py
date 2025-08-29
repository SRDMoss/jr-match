from typing import List, Set
import re
import pathlib
from pathlib import Path
from docx import Document
import unicodedata, re
from .schemas import ParsedDoc, Sections, Bullet
from .ontology_loader import alias_to_id_map, load_ontology, category_ids

# Regex patterns
BULLET_RX = re.compile(r"^[\-\u2022\*\•]\s+")
SKILL_SPLIT = re.compile(r"[,\|;/]")
TOKEN_RX = re.compile(r"[\w\+\#\.\-\/]+", re.UNICODE)



def _read_pdf(path: str) -> str:
    """Try pdfplumber → PyMuPDF → pdfminer.six; return best text."""
    text = ""

    # 1) pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages).strip()
    except Exception:
        text = ""

    # 2) PyMuPDF fallback
    if len(text) < 200:
        try:
            import fitz  # PyMuPDF
            t2 = []
            with fitz.open(path) as doc:
                for p in doc:
                    t2.append(p.get_text("text"))
            t2 = "\n".join(t2).strip()
            if len(t2) > len(text):
                text = t2
        except Exception:
            pass

    # 3) pdfminer.six fallback
    if len(text) < 200:
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract_text
            t3 = (pdfminer_extract_text(path) or "").strip()
            if len(t3) > len(text):
                text = t3
        except Exception:
            pass

    # debug: show length so we know if JD is empty
    print(f"[PDF] {path} -> {len(text)} chars")
    return text

def _read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def _read_txt(path: str) -> str:
    try:
        return pathlib.Path(path).read_text(encoding="utf-8")
    except Exception:
        return pathlib.Path(path).read_text(errors="ignore")

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

def _normalize_unicode(text: str) -> str:
    t = text.replace("\u00A0", " ")               # NBSP → space
    t = unicodedata.normalize("NFKD", t)          # decompose accents/ligatures
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return t.lower()

def _normalize_tokens(text: str) -> List[str]:
    # Step 1: preferred path (Unicode-friendly filter)
    t = _normalize_unicode(text)
    t1 = re.sub(r"[^a-z0-9\+\#\.\-\/]+", " ", t)  # keep + # . - /
    toks = [tok for tok in t1.split() if tok]

    # Step 2: fallback (don’t normalize; some PDFs map to odd codepoints)
    if not toks:
        t2 = re.sub(r"[^A-Za-z0-9\+\#\.\-\/]+", " ", text)  # raw text
        toks = [tok.lower() for tok in t2.split() if tok]

    # Step 3: last-resort fallback (whitespace split)
    if not toks and text.strip():
        toks = text.strip().split()

    # Debug once per call if still empty
    if not toks:
        print("[debug] still no tokens; raw head repr:", repr(text[:300]))

    return toks

def _find_bullets(lines: List[str]) -> List[Bullet]:
    bullets = []
    for line in lines:
        if BULLET_RX.search(line):
            clean = BULLET_RX.sub("", line).strip()
            if clean:
                bullets.append(Bullet(text=clean))
    return bullets

def _match_skills_in_text(text: str, alias_map: dict) -> Set[str]:
    """
    Robust matcher:
    - Normalize Unicode
    - Try token-based variants (if tokens exist)
    - Always fall back to substring search over normalized text
    """
    found_ids: Set[str] = set()

    # Normalize once
    norm = _normalize_unicode(text)

    # Token-based path (may be empty for weird PDFs)
    tokens = _normalize_tokens(text)
    token_set = set(tokens)
    variant_token_set = set()

    if token_set:
        for t in token_set:
            variant_token_set.add(t)
            if "." in t:
                variant_token_set.add(t.replace(".", ""))   # 'vuejs'
                parts = t.split(".")
                if parts[0]:
                    variant_token_set.add(parts[0])         # 'vue'
            if "-" in t:
                variant_token_set.add(t.replace("-", " "))  # 'google cloud'
                variant_token_set.add(t.replace("-", ""))   # 'googlecloud'
            if "/" in t:
                variant_token_set.update(p for p in t.split("/") if p)

        joined = " ".join(tokens)
        joined_variants = " ".join(sorted(variant_token_set))
    else:
        joined = ""
        joined_variants = ""

    # 1) Token/phrase route (when tokens exist)
    if token_set:
        for alias, sid in alias_map.items():
            a = alias.lower()
            alias_tokens = a.split()
            if len(alias_tokens) == 1:
                if a in token_set or a in variant_token_set:
                    found_ids.add(sid)
            else:
                pat = r"\b" + r"\s+".join(map(re.escape, alias_tokens)) + r"\b"
                if re.search(pat, joined) or re.search(pat, joined_variants):
                    found_ids.add(sid)

    # 2) Always run a substring fallback (covers tokenization failures)
    #    Use a simple normalized contains check for aliases and a few variants.
    if True:
        for alias, sid in alias_map.items():
            a = alias.lower()
            if a in norm:
                found_ids.add(sid)
                continue
            # minimal variants
            a_dotless = a.replace(".", "")
            a_dashless = a.replace("-", "")
            a_spaced   = a.replace("-", " ")
            if a_dotless and a_dotless in norm:
                found_ids.add(sid); continue
            if a_dashless and a_dashless in norm:
                found_ids.add(sid); continue
            if a_spaced and a_spaced in norm:
                found_ids.add(sid); continue

    return found_ids

def extract(path: str, ontology_csv: str) -> ParsedDoc:
    """Main extraction function."""
    p = path.lower()
    if p.endswith(".pdf"):
        text = _read_pdf(path)
    elif p.endswith(".docx"):
        text = _read_docx(path)
    elif p.endswith(".txt"):
        text = _read_txt(path)
    else:
        # no/unknown extension → probe
        text = _read_pdf(path)
        if len(text) < 50:
            try:
                text = _read_docx(path)
            except Exception:
                pass
        if len(text) < 10:
            text = _read_txt(path)
    
    print("[sanity]", _normalize_tokens("python node.js kubernetes")[:5])
    
    print("[head]", repr(text[:300]))
    dump_path = Path(f"dump_{Path(path).stem}.txt")
    dump_path.write_text(text, encoding="utf-8", errors="ignore")
    print("[dumped]", dump_path)

    print(f"[extract] file={path} lines={text.count(chr(10))} chars={len(text)}")

    sections = _split_sections(text)
    bullets = _find_bullets(sections.experience) + _find_bullets(sections.education)

    skills_dict = load_ontology(ontology_csv)
    skills_map  = _build_normalized_alias_map(skills_dict) 

    skills_from_skills_section = _match_skills_in_text(" ".join(sections.skills), skills_map)
    skills_from_full_text = _match_skills_in_text(text, skills_map)

    all_skills = sorted(skills_from_skills_section | skills_from_full_text)
    cat_ids = category_ids(skills_dict)
    filtered_skills = [s for s in all_skills if s not in cat_ids]

    # optional debug
    print("RAW Skills Section:", sections.skills)
    print("Skills from skills section:", skills_from_skills_section)
    print("Skills from full text:", skills_from_full_text)

    return ParsedDoc(
        text=text,
        skills=sorted(filtered_skills),
        bullets=bullets,
        sections=sections
    )
