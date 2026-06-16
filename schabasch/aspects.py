"""JD segmentation + aspect-pair coverage scoring (the matcher core).

Pure module — no model needed here. features.py passes pre-computed embeddings in.
Unit-testable with toy numpy vectors (no bge-m3 required).

Key design:
- segment_jd()  → sections dict (must_have / responsibilities / company / offer + _meta)
- score()       → ~24 named features including the headline match_score
- match_score   = fixed monotone combination weighted toward must-have coverage;
                  this IS the cold-start scorer and the interpretable "fit" number.
                  LGBM gate later supersedes it but trains on the same feature vector.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Seniority mapping
# ---------------------------------------------------------------------------

_SENIORITY_LEVELS: dict[str, int] = {
    "intern": -1, "trainee": -1, "berufseinsteiger": -1,
    "junior": 0, "entry": 0, "associate": 0,
    "mid": 1, "medior": 1, "professional": 1,
    "senior": 2, "sr": 2, "expert": 2, "specialist": 2,
    "lead": 3, "principal": 3, "staff": 3, "head": 3, "director": 3,
}

_SENIORITY_TEXT_TO_LEVEL = {
    "junior": 0, "mid": 1, "senior": 2, "lead": 3,
}

# ---------------------------------------------------------------------------
# Header vocabulary for EN + DE JD segmentation
# ---------------------------------------------------------------------------

_HEADER_DEFS: list[tuple[str, str]] = [
    # must_have — EN
    (r"requirements?",                   "must_have"),
    (r"qualifications?",                 "must_have"),
    (r"your profile",                    "must_have"),
    (r"must[\s\-]have",                  "must_have"),
    (r"what (you|we) (bring|expect|need|require)", "must_have"),
    (r"what you('ll)? need",             "must_have"),
    (r"we('re)?( are)? looking for",     "must_have"),
    (r"you (bring|have|offer|should have)",  "must_have"),
    (r"skills?\s*(required|needed|we seek)", "must_have"),
    (r"(minimum |required )?experience",     "must_have"),
    # must_have — DE
    (r"anforderungen?",                  "must_have"),
    (r"(dein|ihr|deine|ihre)\s+profil",  "must_have"),
    (r"wir erwarten",                    "must_have"),
    (r"das bringst du mit",              "must_have"),
    (r"voraussetzungen?",                "must_have"),
    (r"was (du|sie) mitbringst",         "must_have"),
    (r"was wir erwarten",                "must_have"),
    (r"dein knowhow",                    "must_have"),
    (r"deine\s+qualifikation(en)?",      "must_have"),
    (r"dein\s+profil",                   "must_have"),
    (r"ihr\s+profil",                    "must_have"),
    # nice_to_have — EN
    (r"nice[\s\-]to[\s\-]have",          "nice_to_have"),
    (r"preferred qualifications?",       "nice_to_have"),
    (r"(nice|good) to have",             "nice_to_have"),
    (r"bonus (points?|skills?)",         "nice_to_have"),
    # nice_to_have — DE
    (r"wünschenswert",                   "nice_to_have"),
    (r"von vorteil",                     "nice_to_have"),
    (r"idealerweise",                    "nice_to_have"),
    (r"wäre (toll|schön|ein plus)",      "nice_to_have"),
    # responsibilities — EN
    (r"responsibilities",                "responsibilities"),
    (r"(your\s+)?role",                  "responsibilities"),
    (r"tasks?",                          "responsibilities"),
    (r"what you('ll)?( will)? do",       "responsibilities"),
    (r"day[\s\-]to[\s\-]day",           "responsibilities"),
    (r"key (duties|responsibilities|activities)", "responsibilities"),
    (r"job (description|duties)",        "responsibilities"),
    (r"your (day|work|contribution)",    "responsibilities"),
    # responsibilities — DE
    (r"aufgaben?",                       "responsibilities"),
    (r"deine?\s+aufgaben?",             "responsibilities"),
    (r"tätigkeiten?",                    "responsibilities"),
    (r"was (du|sie) (tust|tun)",         "responsibilities"),
    (r"dein\s+aufgabengebiet",           "responsibilities"),
    (r"ihr(e)?\s+aufgaben",             "responsibilities"),
    (r"dein(e)?\s+tag",                 "responsibilities"),
    # company / intro — EN
    (r"about (us|the company|our company)", "company"),
    (r"who (we are|are we)",             "company"),
    (r"our (story|mission|vision|culture)", "company"),
    (r"the company",                     "company"),
    # company / intro — DE
    (r"über uns",                        "company"),
    (r"das unternehmen",                 "company"),
    (r"wer wir sind",                    "company"),
    (r"unser (unternehmen|team)",        "company"),
    # offer — EN
    (r"(what )?we offer",               "offer"),
    (r"benefits?",                       "offer"),
    (r"perks?",                          "offer"),
    (r"why (join|work here|us)",        "offer"),
    (r"what('?s| is) in it for you",   "offer"),
    # offer — DE
    (r"wir bieten",                      "offer"),
    (r"was wir bieten",                  "offer"),
    (r"unser angebot",                   "offer"),
    (r"(ihre?|deine?)\s+benefits?",     "offer"),
]

# Compile patterns: match stripped line (after removing trailing decorators)
_COMPILED_HEADERS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^" + pat + r"$", re.IGNORECASE | re.UNICODE), sec)
    for pat, sec in _HEADER_DEFS
]

_STRIP_TRAIL = re.compile(r"[\s:*\-•▸►✓✔\d\.]+$")
_STRIP_LEAD  = re.compile(r"^[\s\-•▸►*✓✔\d\.]+")
_BULLET_LINE = re.compile(r"^[\-\*•▸►✓✔]\s+(.+)|^\d+[.)]\s+(.+)")


# ---------------------------------------------------------------------------
# Public: JD segmentation
# ---------------------------------------------------------------------------

def _is_header(line: str) -> str | None:
    """Return section key if line matches a known header, else None."""
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return None
    # Remove trailing punctuation / bullets common in German postings
    cleaned = _STRIP_TRAIL.sub("", stripped)
    cleaned = _STRIP_LEAD.sub("", cleaned).strip()
    if not cleaned or len(cleaned) < 3:
        return None
    for pat, section in _COMPILED_HEADERS:
        if pat.match(cleaned):
            return section
    return None


def segment_jd(title: str, description: str) -> dict[str, str]:
    """Split a JD description into named sections using EN+DE header vocabulary.

    Returns dict with keys from:
      must_have, nice_to_have, responsibilities, company, offer
    plus `_meta` with {has_structure: 0|1, title: str}.
    Text before the first header goes into 'company' (intro / boilerplate).
    Headerless JDs: `has_structure=0` and `full` key contains everything.
    """
    sections: dict[str, list[str]] = {k: [] for k in
                                       ("must_have", "nice_to_have", "responsibilities",
                                        "company", "offer")}
    current: str | None = None
    intro: list[str] = []
    found: bool = False

    for raw_line in description.split("\n"):
        matched = _is_header(raw_line)
        if matched is not None:
            current = matched
            found = True
            continue
        stripped = raw_line.strip()
        if current is None:
            if stripped:
                intro.append(stripped)
        else:
            sections[current].append(stripped)

    if intro:
        sections["company"] = intro + sections["company"]

    result: dict[str, str] = {}
    for key, lines in sections.items():
        text = "\n".join(lines).strip()
        if text:
            result[key] = text

    if not found:
        result["full"] = description.strip()

    result["_meta"] = {"has_structure": int(found), "title": str(title or "")}
    return result


def extract_jd_skills(section_text: str) -> list[str]:
    """Extract required skill/requirement phrases from a JD section (bullet points)."""
    phrases: list[str] = []
    for line in section_text.split("\n"):
        m = _BULLET_LINE.match(line.strip())
        if m:
            phrase = (m.group(1) or m.group(2) or "").strip()
            if 3 < len(phrase) < 200:
                phrases.append(phrase)
    # Fallback: comma/semicolon split if no bullets found
    if not phrases and section_text.strip():
        raw = re.sub(r"\n+", " ", section_text)
        for part in re.split(r"[;,]", raw):
            part = part.strip()
            if 3 < len(part) < 100:
                phrases.append(part)
    return phrases[:25]


def detect_jd_seniority(title: str, must_have_text: str = "") -> str:
    """Infer JD seniority level (junior|mid|senior|lead) from title and requirements."""
    text = f"{title} {must_have_text}".lower()
    if any(w in text for w in ("principal", "staff", "vp", "vice president",
                                "head of", "director", "lead", "leiter", "führungskraft")):
        return "lead"
    if any(w in text for w in ("senior", "sr.", "sr ", "expert",
                                "spezialist", "senior")):
        return "senior"
    if any(w in text for w in ("junior", "jr.", "jr ", "entry", "graduate",
                                "trainee", "intern", "berufseinsteiger")):
        return "junior"
    return "mid"


# ---------------------------------------------------------------------------
# Pure vector math helpers
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors. Returns 0.0 on zero-norm inputs."""
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _seniority_level(label: str) -> int:
    label = (label or "").strip().lower()
    if label in _SENIORITY_LEVELS:
        return _SENIORITY_LEVELS[label]
    for key, val in _SENIORITY_LEVELS.items():
        if key in label:
            return val
    return 1  # default to mid


def _keyword_missing(jd_skill_phrases: list[str], cand_skills: list[str]) -> int:
    """Count JD skill phrases not keyword-matched by any CV skill (coarse coverage check)."""
    if not jd_skill_phrases:
        return 0
    cand_tokens = set()
    for s in (cand_skills or []):
        for tok in re.split(r"[\s,/\-]+", s.lower()):
            if len(tok) > 2:
                cand_tokens.add(tok)
    missing = 0
    for phrase in jd_skill_phrases:
        phrase_toks = {tok.lower() for tok in re.split(r"[\s,/\-]+", phrase)
                       if len(tok) > 2}
        if not phrase_toks or not phrase_toks.intersection(cand_tokens):
            missing += 1
    return missing


# ---------------------------------------------------------------------------
# Public: aspect-pair scoring
# ---------------------------------------------------------------------------

def score(
    *,
    # CV side (pre-embedded by features.py)
    cand_vecs: dict[str, np.ndarray],         # {skills, experience, domains, roles, full}
    cand_skills: list[str],                    # raw skill strings for keyword coverage
    cand_seniority: str = "senior",
    # JD side
    jd_sections: dict[str, str],              # text per section (for keyword coverage)
    jd_section_vecs: dict[str, np.ndarray | None],  # dense vecs per section
    jd_full_vec: np.ndarray,                   # whole-JD dense vec
    jd_seniority: str | None = None,           # detected or passed in
    # Optional signals
    sparse_score: float = 0.0,                # pre-computed lexical overlap (features.py)
    colbert_fn: Callable[[], float] | None = None,  # lazy, top-K only
    library=None,                              # PositiveLibrary | None
    gates: dict[str, Any] | None = None,      # pre-computed gate signals
    vacancy_id: int | None = None,            # for library leave-one-out
    company: str | None = None,               # for company_overlap
) -> dict[str, float]:
    """Compute ~24 named match/coverage/taste/gate features + match_score.

    Works with any numpy vectors (toy or bge-m3). All missing sections → 0.0.
    """
    gates = gates or {}

    # Convenience: resolve section vecs, fall back to jd_full_vec
    def _jd(key: str) -> np.ndarray:
        v = jd_section_vecs.get(key)
        return v if v is not None else jd_full_vec

    def _cv(key: str) -> np.ndarray:
        return cand_vecs.get(key, jd_full_vec * 0)  # zero vec if missing

    has_struct = int(jd_sections.get("_meta", {}).get("has_structure", 0) if "_meta" in jd_sections else 0)
    jd_must_text = jd_sections.get("must_have", "")
    jd_skill_phrases = extract_jd_skills(jd_must_text)

    # --- Coverage (asymmetric): does CV cover JD must-haves? ---
    cov_musthave_maxsim = _cosine(_cv("skills"), _jd("must_have"))
    cov_nicetohave_maxsim = _cosine(_cv("skills"), _jd("nice_to_have"))

    tau = 0.6  # coverage threshold (configurable but fine as constant here)
    cov_musthave_frac_above_tau = 1.0 if cov_musthave_maxsim >= tau else 0.0
    n_musthave_missing = _keyword_missing(jd_skill_phrases, cand_skills)

    colbert_req_to_cv = colbert_fn() if colbert_fn is not None else 0.0

    # --- Symmetric similarity (soft fit) ---
    sim_skills_requirements     = _cosine(_cv("skills"),     _jd("must_have"))
    sim_experience_resp         = _cosine(_cv("experience"), _jd("responsibilities"))
    sim_domain_company          = _cosine(_cv("domains"),    _jd("company"))
    sim_title_role              = _cosine(_cv("roles"),      jd_full_vec)
    sim_fullcv_fulljd           = _cosine(_cv("full"),       jd_full_vec)

    # --- Seniority gap (signed: CV level − JD level; negative = underqualified) ---
    if jd_seniority is None:
        jd_seniority = detect_jd_seniority(
            jd_sections.get("_meta", {}).get("title", ""),
            jd_must_text,
        )
    cv_level  = _seniority_level(cand_seniority)
    jd_level  = _seniority_level(jd_seniority)
    seniority_gap = float(cv_level - jd_level)  # +ve = overqualified, -ve = underqualified

    # --- Taste (from PositiveLibrary) ---
    # Pass THIS vacancy's JD embedding (not the constant CV vector) — the library holds liked
    # vacancy embeddings, so taste must vary per-vacancy or every row gets an identical score.
    if library is not None and getattr(library, "n_rows", 0) > 0:
        taste = library.taste_features(
            jd_full_vec, company=company, exclude_vacancy_id=vacancy_id
        )
    else:
        taste = {
            "nearest_liked_cosine": 0.0,
            "positive_centroid_cosine": 0.0,
            "recent_centroid_cosine": 0.0,
            "topic_drift": 0.0,
            "company_overlap_count": 0.0,
        }

    # --- Gate signals as features (cheap pre-LLM signals) ---
    lang_de_required  = float(gates.get("lang_de_required", 0))
    geo_distance_norm = float(gates.get("geo_distance_norm", 0.5))
    is_remote_hint    = float(gates.get("is_remote_hint", 0))
    recency_days      = float(gates.get("recency_days", 30))
    title_text        = jd_sections.get("_meta", {}).get("title", "")
    title_log_len     = math.log1p(len(title_text)) if title_text else 0.0
    full_text         = jd_sections.get("full") or " ".join(
                          jd_sections.get(k, "") for k in
                          ("must_have", "responsibilities", "company", "offer", "nice_to_have")
                        )
    desc_log_len      = math.log1p(len(full_text))

    # --- Enrichment (agent-verified — absent → neutral defaults) ---
    requirements_verified = float(gates.get("requirements_verified", 0))
    company_known         = float(gates.get("company_known", 0))
    salary_vs_target_gap  = float(gates.get("salary_vs_target_gap", 0))

    f: dict[str, float] = {
        # coverage (asymmetric)
        "cov_musthave_maxsim":       cov_musthave_maxsim,
        "cov_musthave_frac_above_tau": cov_musthave_frac_above_tau,
        "n_musthave_missing":        float(n_musthave_missing),
        "cov_nicetohave_maxsim":     cov_nicetohave_maxsim,
        "colbert_req_to_cv":         float(colbert_req_to_cv),
        "sparse_req_vs_cv":          float(sparse_score),
        # symmetric similarity
        "sim_skills_requirements":   sim_skills_requirements,
        "sim_experience_responsibilities": sim_experience_resp,
        "sim_domain_company":        sim_domain_company,
        "sim_title_role":            sim_title_role,
        "sim_fullcv_fulljd":         sim_fullcv_fulljd,
        # seniority
        "seniority_gap":             seniority_gap,
        **taste,
        # gate signals as features
        "lang_de_required":          lang_de_required,
        "geo_distance_norm":         geo_distance_norm,
        "is_remote_hint":            is_remote_hint,
        "recency_days":              recency_days,
        "has_structure":             float(has_struct),
        "title_log_len":             title_log_len,
        "desc_log_len":              desc_log_len,
        # enrichment
        "requirements_verified":     requirements_verified,
        "company_known":             company_known,
        "salary_vs_target_gap":      salary_vs_target_gap,
        # fit signals — placeholders here (0.0); features.rerank_scored fills them for SCORED
        # candidates (cross-encoder + HyRE). Kept in the vector so the trained gate can learn them.
        "fit_hyre":                  float((gates or {}).get("fit_hyre", 0.0)),
        "fit_score":                 float((gates or {}).get("fit_score", 0.0)),
    }

    f["match_score"] = _match_score(f)
    return f


def _match_score(f: dict[str, float]) -> float:
    """Fixed monotone combination of coverage/similarity features.

    Used as the zero-label cold-start score. LGBM gate supersedes this
    once ≥30 labels exist, but trains on the same features.
    Higher coverage of must-haves dominates; seniority gap and missing
    must-haves penalize.
    """
    base = (
        0.35 * f.get("cov_musthave_maxsim", 0.0) +
        0.15 * f.get("cov_musthave_frac_above_tau", 0.0) +
        0.25 * f.get("sim_experience_responsibilities", 0.0) +
        0.15 * f.get("sim_skills_requirements", 0.0) +
        0.10 * f.get("sim_fullcv_fulljd", 0.0)
    )
    # seniority penalty: underqualified (gap < 0)
    gap = f.get("seniority_gap", 0.0)
    if gap < 0:
        base -= 0.08 * min(2.0, abs(gap))
    # missing must-haves penalty
    n_miss = f.get("n_musthave_missing", 0)
    base -= 0.025 * min(6, n_miss)
    return float(max(0.0, min(1.0, base)))


# Stable feature name ordering (for LGBM feature vector)
FEATURE_NAMES: list[str] = [
    "cov_musthave_maxsim", "cov_musthave_frac_above_tau", "n_musthave_missing",
    "cov_nicetohave_maxsim", "colbert_req_to_cv", "sparse_req_vs_cv",
    "sim_skills_requirements", "sim_experience_responsibilities",
    "sim_domain_company", "sim_title_role", "sim_fullcv_fulljd",
    "seniority_gap",
    "nearest_liked_cosine", "positive_centroid_cosine", "recent_centroid_cosine",
    "topic_drift", "company_overlap_count",
    "lang_de_required", "geo_distance_norm", "is_remote_hint",
    "recency_days", "has_structure", "title_log_len", "desc_log_len",
    "requirements_verified", "company_known", "salary_vs_target_gap",
    "fit_hyre", "fit_score",
]
