# ADD or REPLACE these in core/ontology_loader.py

import csv
from typing import Dict, List
from .schemas import Skill

def load_ontology(csv_path: str) -> Dict[str, Skill]:
    skills: Dict[str, Skill] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            alt = [a.strip() for a in row["alt_labels"].split(",")] if row["alt_labels"] else []
            skills[row["id"]] = Skill(
                id=row["id"], label=row["label"], alt_labels=alt, parent_id=row["parent_id"] or None
            )
    return skills

def alias_to_id_map(skills: Dict[str, Skill]) -> Dict[str, str]:
    """
    Build a robust alias→id map. Includes:
      - the canonical label itself
      - alt_labels
      - punctuation variants (., -, / removed or spaced)
      - special handling for *.js → also add base (e.g., 'vue.js' → 'vue')
    """
    def variants(s: str) -> List[str]:
        s = s.strip().lower()
        if not s:
            return []
        outs = {s}
        outs.add(s.replace(".", ""))
        outs.add(s.replace(".", " "))
        outs.add(s.replace("-", " "))
        outs.add(s.replace("/", " "))
        if s.endswith(".js"):
            outs.add(s[:-3])  # 'vue.js' -> 'vue'
        return [v.strip() for v in outs if v.strip()]

    m: Dict[str, str] = {}
    WHITELIST = {"c#", "c++", "js", "go"}  # allow specifically
    AMBIG = {"np","rs","es","tf","ps1","rbac","auth","cache"}  # drop

    for sk in skills.values():
        # include the label as an alias too
        candidates = [sk.label] + (sk.alt_labels or [])

        for cand in candidates:
            for alias in variants(cand):
                # length/ambiguity guard after varianting
                if alias in AMBIG:
                    continue
                if len(alias) < 3 and alias not in WHITELIST:
                    continue
                m[alias] = sk.id
    return m


def id_to_label_map(skills: Dict[str, Skill]) -> Dict[str, str]:
    return {sk.id: sk.label for sk in skills.values()}

def category_ids(skills: Dict[str, Skill]) -> set:
    # Treat root and immediate children of root as categories
    return { "SK000", *[sk.id for sk in skills.values() if sk.parent_id == "SK000"] }
