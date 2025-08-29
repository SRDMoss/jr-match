from pydantic import BaseModel, Field
from typing import List, Optional

class Skill(BaseModel):
    id: str
    label: str
    alt_labels: List[str] = Field(default_factory=list)
    parent_id: Optional[str] = None

class Bullet(BaseModel):
    text: str
    embedding: Optional[list] = None
    skills: List[str] = Field(default_factory=list)  # ontology IDs

class Sections(BaseModel):
    experience: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)

class ParsedDoc(BaseModel):
    text: str
    skills: List[str] = Field(default_factory=list)
    bullets: List[Bullet] = Field(default_factory=list)
    sections: Sections = Sections()

class MatchDetail(BaseModel):
    semantic_sim: float
    coverage: float
    total: float
    gaps: List[str] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)
