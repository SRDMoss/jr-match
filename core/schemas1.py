from pydantic import BaseModel
from typing import List, Dict, Optional

class Skill(BaseModel):
    id: str
    label: str
    alt_labels: List[str] = []
    parent_id: Optional[str] = None

class Bullet(BaseModel):
    text: str
    embedding: Optional[list] = None
    skills: List[str] = []  # ontology IDs

class Sections(BaseModel):
    experience: List[str] = []
    education: List[str] = []
    skills: List[str] = []

class ParsedDoc(BaseModel):
    skills: List[str] = []
    bullets: List[Bullet] = []
    sections: Sections = Sections()

class MatchDetail(BaseModel):
    semantic_sim: float
    coverage: float
    total: float
    gaps: List[str] = []
    matched_skills: List[str] = []
