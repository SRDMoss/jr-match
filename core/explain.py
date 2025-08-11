from typing import Dict, List, Optional
import re
from .schemas import Bullet
from .ontology_loader import load_ontology

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def _skill_aliases(skills_csv: str) -> Dict[str, List[str]]:
    """id -> aliases (alt_labels + label as fallback for evidence only)."""
    skills = load_ontology(skills_csv)
    id2aliases = {}
    for sk in skills.values():
        alts = [a.strip().lower() for a in sk.alt_labels if a.strip()]
        # include label for evidence match (NOT for extraction)
        if sk.label:
            alts.append(sk.label.lower())
        # dedupe
        id2aliases[sk.id] = sorted(set([a for a in alts if a]))
    return id2aliases

def find_evidence_for_matches(
    resume_bullets: List[Bullet],
    matched_skill_ids: List[str],
    skills_csv: str
) -> Dict[str, Optional[str]]:
    """Return {skill_id: best_resume_bullet_text or None}."""
    id2aliases = _skill_aliases(skills_csv)
    bullets_norm = [(b.text, _normalize(b.text)) for b in resume_bullets]
    evidence = {}

    for sid in matched_skill_ids:
        found = None
        aliases = id2aliases.get(sid, [])
        # prefer alias-containing bullets
        for raw, norm in bullets_norm:
            if any(a in norm for a in aliases if a):
                found = raw
                break
        evidence[sid] = found
    return evidence

def suggest_gap_phrases(gap_ids: List[str], skills_csv: str) -> Dict[str, str]:
    """Return short suggested phrasing per missing skill."""
    skills = load_ontology(skills_csv)
    out = {}
    for sid in gap_ids:
        label = skills[sid].label if sid in skills else sid
        # very short, neutral phrasing
        out[sid] = f"Add a bullet that shows {label} (e.g., “Used {label} to …”)."
    return out

def alias_hits(resume_text: str, matched_skill_ids: List[str], skills_csv: str) -> Dict[str, List[str]]:
    id2aliases = _skill_aliases(skills_csv)
    tokens = re.findall(r"[a-z0-9\+\#\.]+", resume_text.lower())
    token_set = set(tokens)
    joined = " ".join(tokens)
    hits = {}
    for sid in matched_skill_ids:
        hits[sid] = []
        for a in id2aliases.get(sid, []):
            a_tokens = a.split()
            if (len(a_tokens)==1 and a_tokens[0] in token_set) or \
               (len(a_tokens)>1 and re.search(r"\b"+r"\s+".join(map(re.escape,a_tokens))+r"\b", joined)):
                hits[sid].append(a)
    return hits