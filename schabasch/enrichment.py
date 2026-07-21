"""Zotero-style card enrichment — extracted JD snippets + pros/cons + deep company + slop re-parse.

The card showed too little, so the user opened the full JD every time. This adds a "paper-render"-like
block (the zotero-summarizer pattern): a deterministic EXTRACTIVE pass (bge-reranker ranks JD sentences
against goal queries → the few sentences that matter) plus an ABSTRACTIVE pass via the model cascade
(35B-MLX by default, a remote `sota` tier for «глупые» high-slop descriptions) producing pros/cons, a
deeper company read, and a clean re-parse of muddy ads — all grounded in the JD (anti-fabrication).

Frozen-contract-safe: a NEW sidecar (vacancy_enrichment) owned here; keyed by content_hash so reposts
share the result. Runs on the daily SLATE SET ONLY (cap), memory-gated, after rerank. Degrades: if no
abstractive tier is reachable, the deterministic snippets still render; abstractive is best-effort.

Reuses: features._load_reranker / _content_hash, llm_clients.make_llm_client / role_available.
zotero-summarizer refs: services/library/quality_review.py (anti-fabrication rubric),
_paper_goal_summaries.py (goal-focused extraction), _review_text.py (snippet ranking).
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from . import db
from .features import _content_hash, _load_reranker
from .llm import LLMError
from .llm_clients import client_label, make_llm_client, role_available

log = logging.getLogger("schabasch.enrichment")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vacancy_enrichment (
    vacancy_id     INTEGER PRIMARY KEY,
    content_hash   TEXT NOT NULL,
    key_snippets   TEXT,            -- JSON [{snippet, goal}]
    pros           TEXT,            -- JSON [str]
    cons           TEXT,            -- JSON [str]
    company_review TEXT,            -- deeper 1-2 sentence read (abstractive, grounded)
    clean_summary  TEXT,            -- plain-language re-parse of a muddy/high-slop ad ('' if clear)
    model_used     TEXT,            -- provenance: qwen3:8b | Qwen3.6-35B… | sota
    extracted_at   TEXT NOT NULL
)
"""

# Goal queries for the extractive pass — what the user wants to see without opening the JD.
_SNIPPET_GOALS = [
    ("требования", "key requirements, must-have skills and qualifications"),
    ("что за работа", "main responsibilities and what the role actually involves day to day"),
    ("плюсы/условия", "benefits, salary, remote/hybrid, team language, visa or relocation support"),
    ("подвох", "downsides, red flags, hard requirements like a specific degree or fluent German"),
]

def _system_prompt(cfg: dict | None) -> str:
    """Per-user enrichment prompt: the old module constant CLAIMED «профиль задан в системе» but
    never injected it, and the cons examples were user #1's repellents hardcoded («скрытый
    немецкий, мастер/PhD, hands-on») — user #2 got HER minuses on his cards (multi-user fix)."""
    p = (cfg or {}).get("profile") or {}
    summary = str(p.get("summary") or "").strip()
    reps = ", ".join(str(r) for r in (p.get("repellents") or [])) or "нерелевантный домен, рутина"
    profile_block = f"ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ:\n{summary}\n\n" if summary else ""
    return f"""Ты помогаешь пользователю быстро понять вакансию по её ТЕКСТУ, не открывая оригинал.
{profile_block}Цель — найти dream job ДЛЯ ЭТОГО пользователя, а не просто любую вакансию. Верни ТОЛЬКО JSON:
{{
 "pros": [str],            // 2-4 плюса для ЭТОГО пользователя, КАЖДЫЙ обоснован текстом вакансии
 "cons": [str],            // 1-3 минуса/подвоха именно для него (его отталкиватели: {reps})
 "company_review": str,    // 1-2 предложения: что за компания/команда и как это ложится на его цели
 "clean_summary": str      // если описание мутное/«AI-слоп» — перескажи простыми словами что это за \
работа и кого ищут; если и так понятно — пустая строка ""
}}
ПРАВИЛА: только по тексту вакансии ниже — НЕ выдумывай зарплату/льготы/факты, которых нет в тексте; \
если чего-то нет — не пиши. Кратко, по-русски. Без markdown, один валидный JSON-объект."""


def ensure_schema(con) -> None:
    con.execute(_SCHEMA)
    con.commit()


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+|•|·|•")


def _split_sentences(text: str, *, max_sentences: int = 60) -> list[str]:
    """Coarse sentence/bullet split for the extractive pass (bge-reranker scores each)."""
    parts = [re.sub(r"\s+", " ", p).strip() for p in _SENT_SPLIT.split(text or "")]
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if len(p) < 16 or len(p) > 320:   # drop tiny fragments + giant blobs (keep terse req lines)
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= max_sentences:
            break
    return out


def extract_snippets(reranker, jd_text: str, *, per_goal: int = 1, min_score: float = 0.15) -> list[dict]:
    """Deterministic extractive snippets: rank JD sentences against each goal query with the
    cross-encoder, keep the top per goal (deduped, above a relevance floor)."""
    sentences = _split_sentences(jd_text)
    if not sentences:
        return []
    out: list[dict] = []
    used: set[str] = set()
    for label, query in _SNIPPET_GOALS:
        pairs = [[query, s] for s in sentences]
        try:
            scores = reranker.compute_score(pairs, normalize=True)
        except Exception as exc:  # noqa: BLE001 — a reranker failure must not sink enrichment
            log.warning("snippet rerank failed: %s", exc)
            return out
        ranked = sorted(zip(sentences, scores), key=lambda t: -t[1])
        kept = 0
        for sent, sc in ranked:
            if sc < min_score or sent.lower() in used:
                continue
            out.append({"snippet": sent, "goal": label, "score": round(float(sc), 3)})
            used.add(sent.lower())
            kept += 1
            if kept >= per_goal:
                break
    return out


def _deep_chain(cfg: dict) -> list:
    """Ordered client chain for the single-shot abstractive pass: ONE small tier (``deep_reasoning``,
    e.g. ollama qwen3.5:4b) plus the always-available ollama ``normalizer`` as a reliability fallback.

    Escalation stays WITHIN small local models — **no** ``sota`` and **no** slop-based tier-jump (the
    heavy MLX 35B is reserved for the supervised multi-turn AGENT and must never co-load with the bulk
    pool). The normalizer tail is resilience (used only if ``deep_reasoning`` errors), not escalation."""
    chain = []
    if role_available(cfg, "deep_reasoning"):
        chain.append(make_llm_client(cfg, "deep_reasoning"))
    chain.append(make_llm_client(cfg, "normalizer"))   # ollama qwen3:8b — always reachable
    return chain


def abstractive(cfg: dict, *, jd_title: str, jd_text: str, slop: int) -> tuple[dict | None, str | None]:
    """Pros/cons + company review + clean re-parse via the single small tier. Returns (obj,
    model_label) or (None, None) if every tier failed. Tries each tier in order, catching LLMError →
    next tier. `slop` is surfaced to the model as context but no longer triggers escalation."""
    truncate = int((cfg.get("llm", {}) or {}).get("desc_truncate_chars", 6000))
    user = f"ВАКАНСИЯ:\nTITLE: {jd_title}\nslop_score: {slop}\n\nОПИСАНИЕ:\n{(jd_text or '')[:truncate]}"
    system = _system_prompt(cfg)
    for client in _deep_chain(cfg):
        try:
            obj = client.chat_json(system, user)
        except LLMError as e:
            log.info("deep tier %s failed (%s); trying next", client_label(client), e.error_class)
            continue
        return obj, client_label(client)
    return None, None


def _coerce_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()][:4]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def enrich_one(cfg: dict, con, vid: int, *, reranker, do_abstractive: bool = True) -> str:
    """Enrich ONE vacancy (idempotent per content_hash). Returns 'cached'|'enriched'|'skipped'."""
    ensure_schema(con)
    row = con.execute("SELECT title, description FROM vacancy WHERE id = ?", (vid,)).fetchone()
    if not row or not (row["description"] or "").strip():
        return "skipped"
    ch = _content_hash(row["title"] or "", row["description"] or "")
    cached = con.execute(
        "SELECT content_hash FROM vacancy_enrichment WHERE vacancy_id = ?", (vid,)).fetchone()
    if cached and cached["content_hash"] == ch:
        return "cached"

    snippets = extract_snippets(reranker, f"{row['title'] or ''}\n{row['description'] or ''}") if reranker else []
    pros = cons = company_review = clean = None
    model_used = None
    if do_abstractive:
        # slop_score lives on the normalizer Card (stored in vacancy.card_json)
        slop = 0
        cj = con.execute("SELECT card_json FROM vacancy WHERE id = ?", (vid,)).fetchone()
        if cj and cj["card_json"]:
            try:
                slop = int(json.loads(cj["card_json"]).get("slop_score", 0) or 0)
            except (TypeError, ValueError, json.JSONDecodeError):
                slop = 0
        obj, model_used = abstractive(cfg, jd_title=row["title"] or "",
                                      jd_text=row["description"] or "", slop=slop)
        if obj:
            pros = _coerce_list(obj.get("pros"))
            cons = _coerce_list(obj.get("cons"))
            company_review = (str(obj.get("company_review") or "").strip() or None)
            clean = (str(obj.get("clean_summary") or "").strip() or None)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con.execute(
        """INSERT OR REPLACE INTO vacancy_enrichment
               (vacancy_id, content_hash, key_snippets, pros, cons, company_review, clean_summary,
                model_used, extracted_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (vid, ch, json.dumps(snippets, ensure_ascii=False),
         json.dumps(pros, ensure_ascii=False) if pros else None,
         json.dumps(cons, ensure_ascii=False) if cons else None,
         company_review, clean, model_used, now))
    con.commit()
    return "enriched"


def enrich_slate(cfg: dict, con, *, slate_date: str, top_n: int | None = None) -> dict:
    """Enrich the daily slate cards (cap = deep.enrich_top_n). Loads the bge-reranker once. Heavy →
    pipeline runs this via _heavy_step (memory-gated)."""
    ensure_schema(con)
    if top_n is None:
        top_n = int((cfg.get("deep", {}) or {}).get("enrich_top_n", 12))
    vids = [int(r["vacancy_id"]) for r in con.execute(
        "SELECT vacancy_id FROM slate_entry WHERE slate_date = ? ORDER BY rank LIMIT ?",
        (slate_date, top_n)).fetchall()]
    if not vids:
        return {"enriched": 0, "cached": 0, "skipped": 0}
    f_cfg = cfg.get("features", {}) or {}
    try:
        reranker = _load_reranker(f_cfg.get("reranker", "BAAI/bge-reranker-v2-m3"))
    except Exception as exc:  # noqa: BLE001 — no reranker → abstractive-only enrichment
        log.warning("reranker load failed (%s); snippets disabled this run", exc)
        reranker = None
    out = {"enriched": 0, "cached": 0, "skipped": 0}
    for vid in vids:
        try:
            res = enrich_one(cfg, con, vid, reranker=reranker)
        except Exception as exc:  # noqa: BLE001 — one card must not sink the batch
            log.warning("enrich vid=%s failed: %s", vid, exc)
            res = "skipped"
        out[res] = out.get(res, 0) + 1
    db.log_funnel(con, "enrich", out["enriched"],
                  detail=f"enriched={out['enriched']} cached={out['cached']} skipped={out['skipped']}")
    return out


def enrichments(con) -> dict[int, dict]:
    """{vacancy_id: enrichment dict} for rendering. Degrades to {} if the table is absent."""
    try:
        rows = con.execute(
            "SELECT vacancy_id, key_snippets, pros, cons, company_review, clean_summary, model_used "
            "FROM vacancy_enrichment").fetchall()
    except Exception:
        return {}
    out: dict[int, dict] = {}
    for r in rows:
        def _j(v):
            try:
                return json.loads(v) if v else None
            except (TypeError, json.JSONDecodeError):
                return None
        out[int(r["vacancy_id"])] = {
            "key_snippets": _j(r["key_snippets"]) or [],
            "pros": _j(r["pros"]) or [], "cons": _j(r["cons"]) or [],
            "company_review": r["company_review"], "clean_summary": r["clean_summary"],
            "model_used": r["model_used"]}
    return out
