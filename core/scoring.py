from .schemas import MatchDetail, ParsedDoc
from .config import load_weights

def score(resume: ParsedDoc, jd: ParsedDoc, top_sim: float, weights=None) -> MatchDetail:
    if weights is None:
        weights = load_weights()
    w_sim, w_cov = weights
    r = set(resume.skills)
    j = set(jd.skills)
    inter = r & j
    coverage = (len(inter) / max(1, len(j)))
    total = w_sim * top_sim + w_cov * coverage
    gaps = sorted(j - r)
    print("RESUME skills:", r)
    print("JD skills:", j)
    print("Intersection:", inter)
    return MatchDetail(
        semantic_sim=top_sim, coverage=coverage, total=total,
        gaps=gaps, matched_skills=sorted(inter)
    )
