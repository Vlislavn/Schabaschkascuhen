"""Structured résumé intake: text/PDF → CandidateProfile with aspect fields.

Extracts skills/experience/domains/seniority aspects from a candidate description
or CV text. These aspects become the reference vectors in aspect-based JD matching
(features.py uses them to score coverage: "do the user's skills cover the JD requirements?").

One profile row at a time; re-run to update. aspect_vecs BLOB is filled later by features.py.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm import LLMError
from .models import ErrorClass

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidate_profile (
    id            INTEGER PRIMARY KEY,
    created_at    TEXT NOT NULL,
    raw_input     TEXT,
    profile_json  TEXT NOT NULL,
    aspect_texts  TEXT NOT NULL,
    doc_hash      TEXT NOT NULL,
    aspect_vecs   BLOB
)
"""

_SYSTEM = """You are a résumé analyst. Extract structured candidate information from the text below.
Return ONLY a JSON object with EXACTLY these keys:
{
  "skills": ["list", "of", "discrete", "skill", "phrases"],
  "experience": "prose summary of work background and responsibilities",
  "domains": ["list", "of", "industries/domains"],
  "seniority": "junior|mid|senior|lead",
  "years_experience": 6,
  "education": "highest degree and field",
  "education_level": "none|bachelor|master|phd",
  "languages": {"de": "A2", "en": "C1"},
  "target_roles": ["list", "of", "target job titles"],
  "locations": ["Heidelberg", "Frankfurt"],
  "magnets": ["what excites the user about a role"],
  "repellents": ["hard-stop factors"],
  "summary": "2-3 sentence overview"
}
No markdown, no comments, no extra keys. Output must be a single valid JSON object.
"""

# Only skills+experience are truly essential; everything else is default-filled so a missing
# optional key (e.g. the unused 'repellents') never crashes intake.
_REQUIRED = {"skills", "experience"}
_LIST_FIELDS = ("skills", "domains", "target_roles", "locations", "magnets", "repellents")
_DEFAULTS: dict[str, Any] = {
    "domains": [], "seniority": "", "years_experience": None, "education": "", "education_level": "",
    "languages": {}, "target_roles": [], "locations": [], "magnets": [],
    "repellents": [], "summary": "",
}


def _coerce_list(v: Any) -> list[str]:
    """Coerce an LLM list-field to a real list. A bare string would otherwise char-iterate into
    garbage (['b','i','o',...]); split it on commas/newlines instead."""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        parts = [p.strip() for p in re.split(r"[,\n;]", v) if p.strip()]
        return parts or ([v.strip()] if v.strip() else [])
    return [str(v)]


def _ensure_schema(con) -> None:
    con.execute(_SCHEMA)
    con.commit()


def _read_cv(path: str) -> str:
    """Extract text from a PDF or plain-text CV file."""
    p = Path(path)
    if p.suffix.lower() != ".pdf":
        return p.read_text(encoding="utf-8")
    # Prefer pdfminer; fall back to pypdf. Catch ONLY the import (the optional backend may be
    # absent) — any real extraction error propagates instead of being silently dropped.
    try:
        import pdfminer.high_level as _pdf  # type: ignore
    except ImportError:
        _pdf = None
    if _pdf is not None:
        return _pdf.extract_text(str(p))
    try:
        import pypdf  # type: ignore
    except ImportError as e:
        raise RuntimeError("No PDF library — install pdfminer.six or pypdf") from e
    reader = pypdf.PdfReader(str(p))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _validate(d: dict[str, Any]) -> dict[str, Any]:
    missing = _REQUIRED - set(d.keys())
    if missing:
        raise ValueError(f"candidate profile missing keys: {missing}")
    # Coerce every list-typed field (LLM sometimes returns a bare string) and default-fill the
    # optional keys so downstream aspect-text building never char-iterates or KeyErrors.
    for f in _LIST_FIELDS:
        d[f] = _coerce_list(d.get(f))
    for k, dv in _DEFAULTS.items():
        if d.get(k) is None:
            d[k] = dv
    return d


def _build_aspect_texts(profile: dict[str, Any]) -> dict[str, str]:
    """Build the embeddable strings for each résumé aspect."""
    skills = profile.get("skills") or []
    skills_text = ", ".join(str(s) for s in skills)

    experience_text = str(profile.get("experience") or "")

    domains = profile.get("domains") or []
    domains_text = ", ".join(str(d) for d in domains)

    roles = profile.get("target_roles") or []
    roles_text = ", ".join(str(r) for r in roles)

    seniority = str(profile.get("seniority") or "")
    education = str(profile.get("education") or "")
    summary = str(profile.get("summary") or "")
    magnets = ", ".join(str(m) for m in (profile.get("magnets") or []))

    full_doc = "\n".join(filter(None, [
        f"Skills: {skills_text}",
        f"Experience: {experience_text}",
        f"Domains: {domains_text}",
        f"Target roles: {roles_text}",
        f"Seniority: {seniority}",
        f"Education: {education}",
        f"Summary: {summary}",
        f"Looking for: {magnets}",
    ]))

    return {
        "skills_text": skills_text,
        "experience_text": experience_text,
        "domains_text": domains_text,
        "roles_text": roles_text,
        "full_doc": full_doc,
    }


def extract_candidate(cfg: dict, con, *,
                      description: str | None = None,
                      cv_path: str | None = None) -> dict:
    """Extract structured CandidateProfile from a freeform description or CV file.

    Calls the LLM normalizer once, persists to candidate_profile sidecar table,
    returns the stored dict (profile_json + aspect_texts merged).
    """
    _ensure_schema(con)

    if cv_path:
        raw_text = _read_cv(cv_path)
        raw_input = f"cv_path:{cv_path}"
    elif description:
        raw_text = description
        raw_input = description[:500]
    else:
        raise ValueError("Provide description= or cv_path=")

    # Guard empty input (e.g. a scanned/image-only PDF that yielded no text) — sending an empty
    # string to the LLM produces a fully hallucinated profile.
    if not raw_text or not raw_text.strip():
        raise RuntimeError("empty CV/description text — nothing to extract "
                           "(scanned/image-only PDF or unreadable file?)")

    from .llm_clients import make_llm_client
    client = make_llm_client(cfg, "candidate")  # role-routed; defaults to ollama normalizer_model

    try:
        raw_profile = client.chat_json(_SYSTEM, raw_text[:8000])
    except LLMError as e:
        raise RuntimeError(f"LLM extraction failed ({e.error_class}): {e.details}") from e

    try:
        profile = _validate(raw_profile)
    except ValueError as e:
        raise RuntimeError(f"profile schema violation: {e}") from e

    # optional ESCO skill normalization: skip cleanly only when the module is ABSENT. A real
    # error inside normalize_skills is a bug and must surface — not be swallowed as "no ESCO".
    esco_csv = cfg.get("features", {}).get("esco_csv")
    if esco_csv:
        try:
            from . import esco  # type: ignore
        except ImportError:
            esco = None  # type: ignore
        if esco is not None:
            profile["skills"] = esco.normalize_skills(profile["skills"])

    aspect_texts = _build_aspect_texts(profile)
    doc_hash = hashlib.sha256(json.dumps(aspect_texts, sort_keys=True).encode()).hexdigest()[:16]

    now = datetime.now(timezone.utc).isoformat()
    con.execute(
        """INSERT INTO candidate_profile
           (created_at, raw_input, profile_json, aspect_texts, doc_hash)
           VALUES (?, ?, ?, ?, ?)""",
        (now, raw_input, json.dumps(profile, ensure_ascii=False),
         json.dumps(aspect_texts, ensure_ascii=False), doc_hash),
    )
    con.commit()

    return {**profile, "aspect_texts": aspect_texts, "doc_hash": doc_hash}


def load_candidate(con) -> dict | None:
    """Return the most recently stored CandidateProfile, or None."""
    _ensure_schema(con)
    row = con.execute(
        "SELECT profile_json, aspect_texts, doc_hash FROM candidate_profile ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    profile = json.loads(row["profile_json"])
    aspect_texts = json.loads(row["aspect_texts"])
    return {**profile, "aspect_texts": aspect_texts, "doc_hash": row["doc_hash"]}


def aspect_texts(con) -> dict[str, str] | None:
    """Return aspect embed strings for the latest profile."""
    _ensure_schema(con)
    row = con.execute(
        "SELECT aspect_texts FROM candidate_profile ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return json.loads(row["aspect_texts"]) if row else None


def candidate_doc(con) -> str | None:
    """Return the full-doc embed string for the latest profile."""
    texts = aspect_texts(con)
    return texts.get("full_doc") if texts else None
