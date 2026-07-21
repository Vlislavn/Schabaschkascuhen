"""Hard-qualification / eligibility gate — the layer that catches "great skill match but the
candidate is NOT ELIGIBLE" (a PhD position needing a Master's offered to a Bachelor-holder; a
"C1 German required" role; a clearance the user lacks). This is SEPARATE from skill/semantic fit.

Design (grounded in the local SOTA refs):
- ConFit-v2 `src/schema/job.py` splits RequiredQualification {minimum_degree_level, experience,
  languages} from Preferred → we extract only the HARD/required side.
- DualOptimization_jobrec uses ordinal academic-level thresholds on both sides + a separate
  qualification head → we encode education as an ISCED-grounded ordinal and gate it separately
  from the magnet judge (preference) and the cross-encoder (skill fit).
- Education ordering = ISCED 2011 (Bachelor 6 < Master 7 < Doctoral 8); German Diplom/Magister/
  Staatsexamen ≈ Master (EQF/DQR level 7).

Policy: a hard-eligibility miss DOWN-RANKS (multiplicative gate with a floor), never hard-drops —
recall > precision, since extraction is imperfect. Unknown candidate field → NO penalty (warn only).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

# --- ISCED-grounded education ordinal -------------------------------------------------------
EDU_ORDINAL = {"none": 0, "bachelor": 1, "master": 2, "phd": 3}

# surface-term → ordinal (DE + EN). Diplom(Univ.)/Magister/Staatsexamen ≈ Master (EQF/DQR 7).
_EDU_TERMS: list[tuple[str, int]] = [
    (r"\b(ph\.?d|doctora|doctoral|promotion|doktor|dphil)\b", 3),
    (r"\b(master|m\.?sc|m\.?a\.?|m\.?eng|magister|diplom|staatsexamen|mba)\b", 2),
    (r"\b(bachelor|b\.?sc|b\.?a\.?|b\.?eng|bakkalaureat)\b", 1),
    (r"\b(abitur|ausbildung|high school|secondary)\b", 0),
]

CEFR = {"a1": 1, "a2": 2, "b1": 3, "b2": 4, "c1": 5, "c2": 6}


def normalize_education(text: Any) -> int | None:
    """Map an education string to an ordinal (0..3), or None if unrecognized/placeholder."""
    if not text:
        return None
    s = str(text).strip().lower()
    if s in EDU_ORDINAL:
        return EDU_ORDINAL[s]
    if "highest degree" in s or s in ("", "unknown", "n/a"):  # extraction placeholder
        return None
    for pat, ordv in _EDU_TERMS:   # highest match wins (ordered phd→none)
        if re.search(pat, s):
            return ordv
    return None


# --- JD hard-requirement extraction (qwen3:8b structured) -----------------------------------
_SCHEMA_CACHE = """
CREATE TABLE IF NOT EXISTS eligibility_cache (
    content_hash TEXT PRIMARY KEY,
    req_json     TEXT NOT NULL,
    computed_at  TEXT NOT NULL
)
"""

_ELIG_SYSTEM = (
    "You extract ONLY HARD, NON-NEGOTIABLE eligibility requirements from a job posting and return "
    "ONE JSON object. HARD = the candidate is DISQUALIFIED without it (legal/regulatory, or stated "
    "as required/erforderlich/Voraussetzung/must/zwingend). NOT HARD = preferred / von Vorteil / "
    "wünschenswert / ideally / a plus / 'or equivalent experience' / 'oder gleichwertige Erfahrung' "
    "— for those set the matching *_is_hard to false.\n"
    "A PhD position / Doktorandenstelle / Promotion / 'Masterarbeit' is a job to OBTAIN a degree; it "
    "requires a COMPLETED Master's to enrol → set is_phd_or_doctoral_position=true and "
    "education_required=\"master\" (NOT \"phd\"), unless a finished PhD is explicitly required.\n"
    "Do NOT infer requirements that are not written; if a field is not stated use null / [] / false.\n"
    'Output ONLY this JSON: {"education_required": "none|bachelor|master|phd"|null, '
    '"education_is_hard": true|false, "is_phd_or_doctoral_position": true|false, '
    '"min_years_experience": int|null, "years_is_hard": true|false, '
    '"mandatory_credentials": [str], "language_required": [{"lang": str, "cefr": "A1..C2", '
    '"is_hard": true|false}], "reason": "<one short human phrase naming the single hardest '
    'requirement, in Russian>"}'
)


# Deterministic high-precision guards (qwen's is_phd_or_doctoral_position / education_is_hard are
# noisy — it flags "PhD preferred" roles as doctoral positions). A doctoral/student POSITION is
# identifiable from the TITLE; a degree qualified by "preferred/or equivalent/a plus" is NOT hard.
_STUDENT_DOCTORAL_TITLE = re.compile(
    r"\b(ph\.?d|doktorand|doctoral|promotionsstelle|promovier|masterarbeit|master[\s-]?thesis|"
    r"bachelorarbeit|diploma student|working student|werkstudent|werkstudierende|"
    r"praktik(um|ant)|internship|\bintern\b|trainee|ausbildung|studentische)\b", re.I)
_DEGREE_TERM = re.compile(r"(master|m\.?sc|phd|ph\.?d|promotion|doktor|degree|abschluss|diplom)", re.I)
_SOFT_NEAR = re.compile(
    r"(preferred|bevorzugt|von vorteil|wünschenswert|a plus|is a plus|ideally|nice to have|"
    r"or equivalent|oder gleichwertig|advanced degree|gerne|optional|wäre.{0,8}vorteil)", re.I)

# "Master Data", "Scrum Master", "master plan/agreement/thesis/craftsman" etc. are NOT an academic
# degree — qwen routinely mis-reads these as education_required="master" (the literal Merz-439 bug:
# the title "Business Process Owner – Master Data" → phantom master-degree gate on the user's #1 job).
# A "master" mention is degree-context UNLESS it is one of these non-degree noun phrases.
_MASTER_NONDEGREE = re.compile(
    r"(scrum[\s-]+master|product[\s-]+master|master[\s-]+(data|plan|planning|agreement|"
    r"schedul|craftsman|batch|record|file|class|key|node|branch|service|template|builder|"
    r"thesis|study|studies))", re.I)


def _title_is_student_or_doctoral(jd_text: str) -> bool:
    title = (jd_text or "").split("\n", 1)[0]
    return bool(_STUDENT_DOCTORAL_TITLE.search(title))


def _master_is_degree(jd_text: str) -> bool:
    """True iff the JD has at least one MASTER mention in a genuine DEGREE context (not 'master
    data'/'scrum master'/…). Used to suppress a qwen education_required='master' that was triggered
    purely by a non-degree 'Master X' phrase — Merz-439's 'Master Data' false positive."""
    t = jd_text or ""
    for m in re.finditer(r"\bmaster", t, re.I):
        window = t[max(0, m.start() - 12):m.end() + 18]
        if _MASTER_NONDEGREE.search(window):
            continue   # this occurrence is a non-degree 'Master X' / 'Scrum Master'
        return True     # a genuine degree-context master mention exists
    return False


def _degree_requirement_is_soft(jd_text: str) -> bool:
    """True if a degree mention is qualified by preferred/equivalent/a-plus nearby (→ not hard)."""
    t = jd_text or ""
    for m in _DEGREE_TERM.finditer(t):
        if _SOFT_NEAR.search(t[max(0, m.start() - 60):m.end() + 60]):
            return True
    return False


def _ensure_schema(con) -> None:
    con.execute(_SCHEMA_CACHE)
    con.commit()


def _empty_req() -> dict:
    return {"education_required": None, "education_is_hard": False,
            "is_phd_or_doctoral_position": False, "min_years_experience": None,
            "years_is_hard": False, "mandatory_credentials": [], "language_required": [],
            "reason": ""}


def _apply_overrides(req: dict, jd_text: str) -> dict:
    """Layer deterministic high-precision guards over (cached) qwen output — applied every call so
    they never depend on qwen's noisy is_phd_position / education_is_hard fields, AND so a bad cached
    extraction self-corrects on the next read (no re-run needed)."""
    req["is_phd_or_doctoral_position"] = _title_is_student_or_doctoral(jd_text)
    if req.get("education_required") and _degree_requirement_is_soft(jd_text):
        req["education_is_hard"] = False
    # Negative-context degree guard: if qwen says a MASTER is required but every 'master' mention in
    # the JD is a non-degree phrase ('Master Data', 'Scrum Master', …), the master-degree gate is a
    # phantom → drop it entirely (education_required=None). Real 'master's degree' JDs keep the gate.
    # Only fires when the word 'master' is actually present AND all such mentions are non-degree —
    # never on absence (qwen may have read 'M.Sc.'/'Masterabschluss'/German, which lack 'master').
    if (str(req.get("education_required") or "").lower() == "master"
            and re.search(r"\bmaster", jd_text or "", re.I)
            and not _master_is_degree(jd_text)):
        req["education_required"] = None
        req["education_is_hard"] = False
    return req


def req_from_cache(con, *, content_hash: str, jd_text: str) -> dict | None:
    """Read a cached requirement record and re-apply the deterministic guards (incl. the Master-Data
    negative-context guard) — NO LLM call. Lets the gate be recomputed live over stored data so an
    eligibility-logic fix takes effect without a heavy re-run. None on cache miss / corrupt row →
    caller falls back to the persisted elig_score."""
    _ensure_schema(con)
    row = con.execute("SELECT req_json FROM eligibility_cache WHERE content_hash = ?",
                      (content_hash,)).fetchone()
    if not row or not row["req_json"]:
        return None
    try:
        return _apply_overrides({**_empty_req(), **json.loads(row["req_json"])}, jd_text)
    except (TypeError, json.JSONDecodeError):
        return None


def extract_requirements(con, *, content_hash: str, jd_text: str, client) -> dict:
    """Extract the JD's HARD requirements (qwen, cached per content_hash). On any failure → an
    all-empty record (= no eligibility constraint = no penalty). Never gates on a parse failure."""
    _ensure_schema(con)
    row = con.execute("SELECT req_json FROM eligibility_cache WHERE content_hash = ?",
                      (content_hash,)).fetchone()
    if row and row["req_json"]:
        try:
            return _apply_overrides({**_empty_req(), **json.loads(row["req_json"])}, jd_text)
        except (TypeError, json.JSONDecodeError):
            # corrupt cache row → evict it (self-heals) and recompute below, rather than
            # silently re-failing the same parse on every call.
            con.execute("DELETE FROM eligibility_cache WHERE content_hash = ?", (content_hash,))
    req = _empty_req()
    for attempt in range(2):
        try:
            user = jd_text[:5000] if attempt == 0 else \
                jd_text[:5000] + "\n\nReturn ONLY valid JSON matching the schema."
            obj = client.chat_json(_ELIG_SYSTEM, user)
            if isinstance(obj, dict):
                req = {**_empty_req(), **obj}
                break
        except Exception:
            continue
    con.execute("INSERT OR REPLACE INTO eligibility_cache (content_hash, req_json, computed_at) "
                "VALUES (?, ?, ?)",
                (content_hash, json.dumps(req, ensure_ascii=False),
                 datetime.now(timezone.utc).isoformat()))
    con.commit()
    return _apply_overrides(req, jd_text)


# --- candidate side --------------------------------------------------------------------------
def candidate_quals(profile: dict | None) -> dict:
    """{education_ordinal, education_known, years, credentials, languages{lang:cefr_ord}}.
    education_level (enum) is preferred; else normalize the free-text education field."""
    p = profile or {}
    edu = normalize_education(p.get("education_level"))
    if edu is None:
        edu = normalize_education(p.get("education"))
    langs = {}
    for k, v in (p.get("languages") or {}).items():
        cv = str(v or "").strip().lower()
        if cv in CEFR:
            langs[str(k).strip().lower()] = CEFR[cv]
    return {
        "education_ordinal": edu if edu is not None else 0,
        "education_known": edu is not None,
        "years": p.get("years_experience"),
        "credentials": [str(c).strip().lower() for c in (p.get("credentials") or [])],
        "languages": langs,   # {'de': 2, 'en': 5}
    }


# --- shared requirement↔candidate matchers (single source; reused by the gate AND gaps.py) --------
# language surface term / ISO code → normalized code. Bare 2-letter codes are for the gate's exact
# structured-field lookup; free-text detection (meets_language) uses only the ≥4-char NAME keys so it
# never mistakes the English word "it" for Italian.
_LANG_TERMS = {
    "en": "en", "english": "en", "englisch": "en",
    "de": "de", "german": "de", "deutsch": "de",
    "ru": "ru", "russian": "ru", "russisch": "ru",
    "fr": "fr", "french": "fr", "francais": "fr", "französisch": "fr",
    "es": "es", "spanish": "es", "spanisch": "es",
    "it": "it", "italian": "it", "italienisch": "it",
    "zh": "zh", "mandarin": "zh", "chinese": "zh", "chinesisch": "zh",
    "ko": "ko", "korean": "ko", "ja": "ja", "japanese": "ja",
    "pt": "pt", "portuguese": "pt", "nl": "nl", "dutch": "nl",
    "pl": "pl", "polish": "pl", "ar": "ar", "arabic": "ar",
}
# a language NAME signals a proficiency requirement only alongside a cue (guards "English-language
# publications" = a publication gap, not a language gap).
_LANG_CUES = ("fluen", "proficien", "spoken", "written", "command of", "native", "mother tongue",
              "sprachkenntnis", "verhandlungssicher", "knowledge of", "language skill", "communicat",
              "speak", "level")
_CEFR_TEXT_RE = re.compile(r"\b([abc][12])\b")


def meets_education(req_text: Any, cand: dict) -> bool:
    """True if the candidate's KNOWN degree level ≥ the level named in a free-text requirement.
    `cand` is a candidate_quals() dict. Reuses normalize_education — the one degree parser."""
    need = normalize_education(req_text)
    return (need is not None and bool(cand.get("education_known"))
            and cand.get("education_ordinal", 0) >= need)


def meets_language(req_text: Any, cand: dict) -> bool:
    """True if EVERY language named in a free-text proficiency requirement is held at the stated CEFR
    (or a professional B2 floor when none is stated). `cand` is a candidate_quals() dict."""
    low = str(req_text or "").lower()
    hits = {code for term, code in _LANG_TERMS.items()
            if len(term) > 3 and re.search(rf"\b{re.escape(term)}\b", low)}
    if not hits or not (_CEFR_TEXT_RE.search(low) or any(c in low for c in _LANG_CUES)):
        return False
    m = _CEFR_TEXT_RE.search(low)
    need = CEFR[m.group(1)] if m else CEFR["b2"]
    langs = cand.get("languages") or {}
    return all(langs.get(code, 0) >= need for code in hits)


# --- the gate --------------------------------------------------------------------------------


def eligibility_gate(req: dict, cand: dict, *, floor: float = 0.35, mid: float = 0.6,
                     fit_score: float | None = None, soft_lift_threshold: float = 0.55,
                     llm_cov: float | None = None, soft_lift_cov_min: float = 0.0
                     ) -> tuple[float, str, str]:
    """Return (multiplier ∈ [floor, 1.0], human_reason, severity). 1.0 = eligible / unknown.
    severity ∈ {"structural","soft"} — STRUCTURAL (PhD/doctoral position, hard non-EN language) is a
    real STOP → red ⛔; SOFT (a prose degree minimum) is a muted amber note that must NOT sink a
    strong-fit job. The worst single hard miss governs (min of sub-factors). DOWN-RANK, never drop.

    HIGH-FIT LIFT (the user's literal SCHOTT ask — "много мэтча, требование по master degree я бы проигнорил"):
    when fit_score ≥ soft_lift_threshold the SOFT degree penalty is lifted to 1.0 (note still shown,
    no demotion). Structural blockers are NEVER lifted — a PhD position stays a hard no regardless.

    CONSERVATIVE BY DESIGN. Empirically qwen mislabels ordinary SKILLS and things-the user-has
    ("Python proficiency", "Fluent English", "EU work permit") as mandatory_credentials, which
    floored ~all jobs. So the gate acts ONLY on a few HIGH-PRECISION hard blockers and never on
    credentials/skills/years — skills are handled by the cross-encoder fit signal, not here.
    """
    if not cand["education_known"]:
        # don't gate on what we don't know about the candidate (recall > precision)
        return 1.0, "", "structural"
    have = cand["education_ordinal"]
    inv = {v: k for k, v in EDU_ORDINAL.items()}
    factors: list[tuple[float, str, str]] = []   # (multiplier, reason, severity)

    # 1) PhD / doctoral / Masterarbeit / "Diploma Student" POSITION → requires a completed Master's
    #    to enrol. HIGH precision (structural: title/role), and exactly the user's reported bug.
    if req.get("is_phd_or_doctoral_position") and have < 2:
        factors.append((floor, "докторская/студенческая позиция требует Master", "structural"))

    # 2) explicit HARD degree minimum from prose (noisier extraction) → SOFT down-rank, never floor.
    #    A 1-step gap (Bachelor missing a Master) is SOFT and lifted to 1.0 for a strong-fit job
    #    (the user's SCHOTT ask). A 2+-step gap (Bachelor missing a PhD) is a REAL blocker → STRUCTURAL and
    #    NEVER lifted (fit≈0.64 for ~every JD here, so without this guard the lift would always fire
    #    and could rescue a genuinely PhD-required role the user cannot qualify for).
    req_edu = req.get("education_required")
    if req_edu and req.get("education_is_hard"):
        need = EDU_ORDINAL.get(str(req_edu).lower())
        if need is not None and have < need:
            reason = f"требуется {req_edu}, у тебя {inv.get(have, '—')}"
            if need - have >= 2:
                factors.append((mid, reason, "structural"))   # 2-step gap → red ⛔, not liftable
            else:
                soft_mult = mid
                # HIGH-FIT lift, GATED on honest coverage: only lift the soft degree gap when fit is
                # high AND (when known) the per-requirement coverage llm_cov clears soft_lift_cov_min.
                # Without the cov gate an aspiration-magnet job (high semantic fit, ~0 real coverage,
                # e.g. VINFAST ML-Engineer fit 0.76 / cov 0.12) gets its master gap wrongly lifted.
                cov_ok = llm_cov is None or float(llm_cov) >= soft_lift_cov_min
                if fit_score is not None and float(fit_score) >= soft_lift_threshold and cov_ok:
                    soft_mult = 1.0   # high-fit lift: don't sink a 1-step job the user would apply to (SCHOTT)
                factors.append((soft_mult, reason, "soft"))

    # 3) hard NON-English language clearly above the user's KNOWN level (English never gates — the user is C1).
    for L in (req.get("language_required") or []):
        if not isinstance(L, dict) or not L.get("is_hard"):
            continue
        lang = _LANG_TERMS.get(str(L.get("lang") or "").strip().lower())
        need = CEFR.get(str(L.get("cefr") or "").strip().lower())
        if (lang and lang != "en" and need is not None
                and lang in cand["languages"] and cand["languages"][lang] < need):
            factors.append((mid, f"нужен {L.get('lang')} {L.get('cefr')}", "structural"))

    # mandatory_credentials and years gates are DELIBERATELY DROPPED — see docstring.
    if not factors:
        return 1.0, "", "structural"
    factors.sort(key=lambda x: x[0])
    return factors[0]


def jd_hard_blocker(jd_text: str, patterns, *, floor: float = 0.35) -> tuple[float, str, str] | None:
    """Deterministic STRUCTURAL blocker straight from the JD text — for hard LEGAL barriers the
    candidate cannot satisfy (active US security clearance / TS-SCI, US-citizenship-only, etc.). These
    don't come through the qwen requirement extractor and are 0% skill-matchable, yet they topped the
    slate via the domain magnet. `patterns` = config `eligibility.hard_blockers` (regex list); EMPTY =
    OFF (behaviour-preserving). Returns (floor, reason, 'structural') on the first match, else None."""
    for pat in patterns or ():
        try:
            m = re.search(pat, jd_text or "", re.IGNORECASE)
        except re.error:
            continue   # a malformed config pattern must not crash the gate
        if m:
            return (floor, f"жёсткий барьер: {m.group(0)}", "structural")
    return None
