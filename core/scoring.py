from typing import List, Dict
from .schemas import MatchDetail, ParsedDoc

def score(resume: ParsedDoc, jd: ParsedDoc, top_sim: float, weights=(0.6, 0.4)) -> MatchDetail:
    w_sim, w_cov = weights
    r = set(resume.skills)
    j = set(jd.skills)
    inter = r & j
    coverage = (len(inter) / max(1, len(j)))
    total = w_sim * top_sim + w_cov * coverage
    gaps = sorted(j - r)
    return MatchDetail(
        semantic_sim=top_sim, coverage=coverage, total=total,
        gaps=gaps, matched_skills=sorted(inter)
    )
