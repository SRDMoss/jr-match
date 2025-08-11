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
    m = {}
    for sk in skills.values():
        m[sk.label.lower()] = sk.id
        for a in sk.alt_labels:
            if a:
                m[a.lower()] = sk.id
    return m
