"""Slate: сборка до 10 карточек (8 exploit + 2 explore) + самодостаточный HTML.

8 exploit = топ по оценке судьи, ≤3 на компанию (tie-break: integration_potential, −slop).
2 explore = random из SCORED вне топа, seed = slate_date (детерминизм пересборки).
Меньше 10 — норма; мусором не добиваем. Рендер в стиле render_job() из jobspy_playground.
"""
from __future__ import annotations

import json
import random
from html import escape

from . import db, eligibility as _elig, features as _features, geo as _geo, role_kind as _rk, \
    triage as _triage
from .candidate import load_candidate
from .geo import _normalize_city
from .i18n import DEFAULT_LANG, available_langs, t
from .models import Status, normalize_company, normalize_title


def _investigations(con) -> dict[int, dict]:
    """{vacancy_id: enrichment dict (+verdict)} from the investigator sidecar — the deeper
    company review (size, salary, English-team, still-open, notes). Degrade if table absent."""
    try:
        rows = con.execute(
            "SELECT vacancy_id, enrichment_json, verdict FROM investigation").fetchall()
    except Exception:
        return {}
    out: dict[int, dict] = {}
    for r in rows:
        try:
            enr = json.loads(r["enrichment_json"]) if r["enrichment_json"] else {}
        except (TypeError, json.JSONDecodeError):
            enr = {}
        if not isinstance(enr, dict):
            enr = {}
        enr["verdict"] = r["verdict"]
        out[int(r["vacancy_id"])] = enr
    return out


def _enrichments(con) -> dict[int, dict]:
    """{vacancy_id: Zotero-style enrichment} (snippets / pros / cons / company / clean re-parse).
    Degrades to {} when the sidecar is absent (no enrich run yet) — the card just omits the block."""
    try:
        from . import enrichment as _enr
        return _enr.enrichments(con)
    except Exception:
        return {}


def _user_notes(con) -> dict[int, str]:
    """{vacancy_id: her saved free-text note} from the slate-source labels, so a typed note
    re-renders in the textarea on reload (insert_label upserts on (vacancy_id, source))."""
    rows = con.execute(
        "SELECT vacancy_id, why_freetext FROM label "
        "WHERE source = 'slate' AND why_freetext IS NOT NULL AND TRIM(why_freetext) != ''"
    ).fetchall()
    return {int(r["vacancy_id"]): r["why_freetext"] for r in rows}


def _cand_quals(con) -> dict:
    """Candidate education/years/languages for the live eligibility recompute (loaded once per
    slate build, passed down so the CV isn't re-read per card)."""
    return _elig.candidate_quals(load_candidate(con))


def _fit_fields(con, vacancy_id: int, cfg: dict, cand_quals: dict | None = None) -> dict:
    """{fit_score, fit_note, llm_cov, llm_cov_reqs, elig_score, elig_note, elig_severity} —
    fit_score + the eligibility gate are recomputed LIVE from stored caches under the CURRENT
    fit_weights + gate logic (no model load), so a re-tune / eligibility fix takes effect on reload
    without a rerank. llm_cov breakdown comes straight from the stored feature row."""
    if cand_quals is None:
        cand_quals = _cand_quals(con)
    live = _features.recompute_live(con, vacancy_id, cfg, cand_quals=cand_quals)
    feat = _features.feature_row(con, vacancy_id) or {}
    return {"fit_score": live["fit_score"], "fit_note": live["fit_note"],
            "llm_cov": feat.get("llm_cov"),
            "llm_cov_reqs": feat.get("llm_cov_reqs") or [],
            "elig_score": live["elig_score"], "elig_note": live["elig_note"],
            "elig_severity": live["elig_severity"]}


def _load_scored(con, rubric_version: str | None = None, *, max_age_days: int | None = None) -> list[dict]:
    # Latest judge score PER VACANCY, restricted to the active rubric so stale scores from a
    # previous persona/rubric never surface (re-judging assigns a fresh current-rubric row).
    mid_sql = "SELECT vacancy_id, MAX(id) mid FROM judge_score"
    params: list = []
    if rubric_version is not None:
        mid_sql += " WHERE rubric_version = ?"
        params.append(rubric_version)
    mid_sql += " GROUP BY vacancy_id"
    # status IN (SCORED, SLATED): a vacancy slated but never labelled (Alina skipped that
    # morning) re-enters the next slate on equal footing — USE_CASE 9a.
    where = "v.status IN (?, ?)"
    wparams: list = [Status.SCORED.value, Status.SLATED.value]
    # FRESHNESS ceiling (daily slate only — annotation_batch passes max_age_days=None to keep the
    # full backlog for labeling): once judged, a job otherwise re-enters the slate forever. Bound it
    # to jobs re-seen within max_age_days so the daily view stays current.
    if max_age_days is not None:
        from datetime import datetime, timedelta, timezone
        where += " AND v.last_seen >= ?"
        wparams.append((datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat())
    rows = con.execute(
        f"""SELECT v.id, v.title, v.company, v.city, v.url, v.card_json, v.date_posted, v.first_seen,
                  js.score, js.why_tag, js.why_freetext, js.explanation
           FROM vacancy v
           JOIN ({mid_sql}) m ON m.vacancy_id = v.id
           JOIN judge_score js ON js.id = m.mid
           WHERE {where}""",
        (*params, *wparams),
    ).fetchall()
    inv_map = _investigations(con)        # deeper company review, attached per card
    enr_map = _enrichments(con)           # Zotero-style snippets + pros/cons + deep company
    notes = _user_notes(con)              # her saved free-text per vacancy (re-render on reload)
    items: list[dict] = []
    for r in rows:
        vid = int(r["id"])
        # NOTE: a deterministically-closed job (still_open=False, _check_still_open) is NOT dropped —
        # it stays with a "⚠ вакансия закрыта" note (recall > a one-click check; the check can miss
        # a transient). still_open is now a real HTTP/AA check, not the old qwen guess.
        try:
            card = json.loads(r["card_json"]) if r["card_json"] else {}
        except (TypeError, json.JSONDecodeError):
            card = {}
        items.append({
            "vacancy_id": vid,
            "title": r["title"], "company": r["company"], "city": r["city"], "url": r["url"],
            "score": int(r["score"]),
            "why": r["why_tag"] or r["why_freetext"] or "",
            "explanation": r["explanation"] or "",
            "summary": card.get("summary_2lines", ""),
            "work_mode": card.get("work_mode", "unknown"),
            "integration_potential": int(card.get("integration_potential", 0) or 0),
            "slop_score": int(card.get("slop_score", 0) or 0),
            "date_posted": r["date_posted"],
            "first_seen": r["first_seen"],
            "investigation": inv_map.get(vid),
            "enrichment": enr_map.get(vid),
            "user_note": notes.get(vid, ""),
        })
    return items


def _collapse_reposts(items: list[dict]) -> list[dict]:
    """Collapse cross-account reposts — the SAME role+city posted under different recruiter names
    (455 Laveer ≡ 906 Westinghouse) — to ONE display card, preserving sort order. The kept (best-
    scored, first in the list) card gains `also_at` = the other companies. Display only; no DB change.

    Guard against false-merging two genuinely-different jobs that share a GENERIC title ('Data
    Analyst') at different employers: collapse a cross-company pair only when the normalized title is
    SPECIFIC (≥3 tokens, like the 5-token 'GRO Data Analytics and Reporting'); a same-company dup
    (pipeline dedup missed it) collapses at any length. Reuses normalize_title / _normalize_city."""
    seen: dict[tuple[str, str], dict] = {}
    out: list[dict] = []
    for it in items:
        title_n = normalize_title(it.get("title") or "")
        if not title_n:
            out.append(it)
            continue
        key = (title_n, _normalize_city(it.get("city")))
        kept = seen.get(key)
        same_company = kept is not None and (
            normalize_company(kept.get("company") or "") == normalize_company(it.get("company") or ""))
        specific = len(title_n.split()) >= 3
        if kept is not None and (specific or same_company):
            other = (it.get("company") or "").strip()
            if other and other.lower() != (kept.get("company") or "").strip().lower():
                lst = kept.setdefault("also_at", [])
                if other not in lst:
                    lst.append(other)
            continue   # collapsed → not shown separately
        seen.setdefault(key, it)
        out.append(it)
    return out


def build_slate(cfg: dict, con, slate_date: str) -> list[dict]:
    """Собрать slate на дату. INSERT slate_entry, set_status(SLATED). Возвращает карточки."""
    s_cfg = cfg.get("slate", {})
    n_exploit = int(s_cfg.get("exploit", 8))
    n_explore = int(s_cfg.get("explore", 2))
    max_per_company = int(s_cfg.get("max_per_company", 3))
    rubric_version = cfg.get("judge", {}).get("rubric_version")

    # если slate на эту дату уже собран — вернуть его (идемпотентность утреннего открытия)
    mid_sql = "SELECT vacancy_id, MAX(id) mid FROM judge_score"
    rub_params: tuple = ()
    if rubric_version is not None:
        mid_sql += " WHERE rubric_version = ?"
        rub_params = (rubric_version,)
    mid_sql += " GROUP BY vacancy_id"
    existing = con.execute(
        f"""SELECT se.vacancy_id, se.rank, se.slot_type, se.feedback, v.title, v.company,
                  v.city, v.url, v.card_json, v.date_posted, v.first_seen,
                  js.score, js.why_tag, js.why_freetext, js.explanation
           FROM slate_entry se JOIN vacancy v ON v.id = se.vacancy_id
           LEFT JOIN ({mid_sql}) m ON m.vacancy_id = v.id
           LEFT JOIN judge_score js ON js.id = m.mid
           WHERE se.slate_date = ? ORDER BY se.rank""",
        (*rub_params, slate_date),
    ).fetchall()
    if existing:
        inv_map = _investigations(con)
        enr_map = _enrichments(con)
        cand_quals = _cand_quals(con)
        notes = _user_notes(con)
        out = []
        for r in existing:
            try:
                card = json.loads(r["card_json"]) if r["card_json"] else {}
            except (TypeError, json.JSONDecodeError):
                card = {}
            mark = _geo.geo_mark(r["city"], cfg)   # far/dist_km/anchor re-marked on reopen
            out.append({
                "vacancy_id": int(r["vacancy_id"]), "rank": int(r["rank"]),
                "slot_type": r["slot_type"], "title": r["title"], "company": r["company"],
                "city": r["city"], "url": r["url"],
                "score": int(r["score"]) if r["score"] is not None else None,
                "why": r["why_tag"] or r["why_freetext"] or "",
                "explanation": r["explanation"] or "", "summary": card.get("summary_2lines", ""),
                "work_mode": card.get("work_mode", "unknown"),
                "feedback": r["feedback"],   # persisted → card renders as done on reload
                "date_posted": r["date_posted"],
                "first_seen": r["first_seen"],
                "investigation": inv_map.get(int(r["vacancy_id"])),
                "enrichment": enr_map.get(int(r["vacancy_id"])),
                "far": mark["far"], "dist_km": mark["dist_km"], "geo_anchor": mark["anchor"],
                "user_note": notes.get(int(r["vacancy_id"]), ""),
                **_fit_fields(con, int(r["vacancy_id"]), cfg, cand_quals),
            })
        return out

    # FRESHNESS: the daily slate shows only jobs re-seen within slate.fresh_days (default 14) —
    # /annotate (annotation_batch) passes no ceiling and keeps the full backlog for labeling.
    items = _load_scored(con, rubric_version=rubric_version,
                         max_age_days=int(s_cfg.get("fresh_days", 14)))
    ts_map = _triage.scores_by_vacancy(con)
    cand_quals = _cand_quals(con)
    for it in items:
        it["triage_score"] = ts_map.get(it["vacancy_id"], 0.0)
        feat = _features.feature_row(con, it["vacancy_id"]) or {}
        it["xenc_score"] = float(feat.get("xenc_full") or 0.0)
        it["fit_hyre"] = float(feat.get("fit_hyre") or 0.0)
        # fit_score + eligibility recomputed LIVE under the current fit_weights + gate logic (no
        # model load) so a re-tune / the Master-Data + high-fit-lift eligibility fix take effect
        # without a rerank. llm_cov breakdown is read straight from the stored feature row.
        it.update(_fit_fields(con, it["vacancy_id"], cfg, cand_quals))
        # geo MARK (not a filter): far-but-in-Germany jobs are shown with a quiet 📍 tag + preferred
        # for the explore slots, never dropped (the user's geo ask).
        mark = _geo.geo_mark(it.get("city"), cfg)
        it["far"], it["dist_km"], it["geo_anchor"] = mark["far"], mark["dist_km"], mark["anchor"]
    # Observability: if most candidates lack a fit score, rerank hasn't run for this set — the
    # de-conflate degrades to judge-order. Log it so a missing-rerank/tick is visible, not silent.
    n_nofit = sum(1 for it in items if it.get("fit_score", 0.0) == 0.0)
    if items and n_nofit >= len(items) * 0.5:
        db.log_funnel(con, "slate_fit_missing", n_nofit,
                      detail=f"{n_nofit}/{len(items)} candidates lack fit_score — run rerank/tick first")
    # De-conflate qualification from preference (DualOptimization_jobrec — `s_final = s_pref + λ·s_qual`,
    # the card's #1 mistake is letting the 1–5 magnet judge swamp fit). RE-DERIVED on Alina's 37 REAL
    # labels 2026-06-15: the old judge-LED formula was ~random (pairwise 0.564). Now FIT LEADS —
    #   effective = fit_score · (1 + β·judge_norm) · elig_score   (β = slate.judge_blend_beta).
    # On real labels β=0 measures best (the magnet judge is near-random as a magnitude term); the
    # magnet instead DIFFERENTIATES comparable-fit jobs as the tie-break key below + drives explore
    # selection + the card emoji. judge_norm = (score-1)/4 ∈ [0,1]. eligibility stays multiplicative.
    beta = float(s_cfg.get("judge_blend_beta", 0.0))

    def _effective(x: dict) -> float:
        fit = x.get("fit_score", 0.0)
        judge_norm = max(0.0, (float(x["score"]) - 1.0) / 4.0) if x.get("score") is not None else 0.0
        elig = x.get("elig_score", 1.0)
        # role-kind soft down-rank (W1): hands-on-engineer / intern roles she repeatedly rejected get
        # a config-driven multiplier (<1) so they sink out of the exploit slots — never a hard drop
        # (still explore-eligible; one engineer role she rated 4). Measured on real labels (W4).
        rk = _rk.multiplier(_rk.classify(x.get("title"), x.get("summary")), cfg)
        return fit * (1.0 + beta * judge_norm) * elig * rk

    for it in items:
        it["_eff"] = _effective(it)
    # tie-break among comparable-fit jobs: magnet judge (the preference differentiator), then triage,
    # integration potential, freshness-via-slop. This is where the magnet earns its keep at β=0.
    items.sort(key=lambda x: (
        -x["_eff"],
        -(float(x["score"]) if x.get("score") is not None else 0.0),
        -x["triage_score"],
        -x["integration_potential"],
        x["slop_score"],
    ))

    # Cross-account repost collapse (display only): the SAME role+city posted under two recruiter
    # names (455 Laveer ≡ 906 Westinghouse) is collapsed to ONE card here — the pipeline dedup blocks
    # by company by design (a false merge kills a live vacancy), so it never compares these. Best-
    # scored (first in sorted order) is kept; the others ride along as `also_at`. No DB/status change.
    items = _collapse_reposts(items)

    # QUALITY FLOOR (the "много неподходящих" fix): an exploit slot must clear slate.quality_floor on
    # effective — on a thin day the slate shows FEWER exploit cards rather than padding with unsuitable
    # ones (extends the existing "no junk padding" principle). Below-floor jobs are NOT hidden: they
    # remain eligible for the explore/"test-interest" slots. Default 0.0 = off until measured.
    quality_floor = float(s_cfg.get("quality_floor", 0.0))
    exploit: list[dict] = []
    company_count: dict[str, int] = {}
    chosen_ids: set[int] = set()
    for it in items:
        if len(exploit) >= n_exploit:
            break
        if it["_eff"] < quality_floor:
            continue   # below the quality floor → not an exploit card (still explore-eligible)
        ckey = normalize_company(it["company"] or "")
        if company_count.get(ckey, 0) >= max_per_company:
            continue
        exploit.append(it)
        chosen_ids.add(it["vacancy_id"])
        company_count[ckey] = company_count.get(ckey, 0) + 1

    # explore = "test-interest" slots. Prefer FAR-but-in-Germany jobs here (a strong-magnet München
    # role she'd consider), then fill with random others — deterministic per slate_date.
    remaining = [it for it in items if it["vacancy_id"] not in chosen_ids]
    rng = random.Random(slate_date)
    rng.shuffle(remaining)
    remaining.sort(key=lambda x: 0 if x.get("far") else 1)   # stable: far first, shuffle order kept
    explore = remaining[:n_explore]

    slate: list[dict] = []
    rank = 1
    for it in exploit:
        slate.append({**it, "rank": rank, "slot_type": "exploit"})
        rank += 1
    for it in explore:
        slate.append({**it, "rank": rank, "slot_type": "explore"})
        rank += 1

    for entry in slate:
        con.execute(
            "INSERT OR IGNORE INTO slate_entry (slate_date, vacancy_id, rank, slot_type) "
            "VALUES (?,?,?,?)",
            (slate_date, entry["vacancy_id"], entry["rank"], entry["slot_type"]),
        )
        db.set_status(con, entry["vacancy_id"], Status.SLATED)
    con.commit()
    db.log_funnel(con, "slate", len(slate), detail=f"exploit={len(exploit)} explore={len(explore)}")

    # вернуть без внутренних полей сортировки
    keys = ("vacancy_id", "rank", "slot_type", "title", "company", "city", "url",
            "score", "why", "explanation", "summary", "work_mode", "date_posted", "first_seen",
            "investigation", "enrichment", "fit_score", "fit_note", "llm_cov", "llm_cov_reqs",
            "elig_score", "elig_note", "elig_severity", "far", "dist_km", "geo_anchor", "also_at",
            "user_note")
    return [{k: e.get(k) for k in keys} for e in slate]


# Von Restorff: exactly TWO strong-emphasis channels — the blue score and the red ⛔ STOP
# alert (hard eligibility miss, the one thing that should halt her). Everything else that used
# to compete for attention (green "verified", teal why-badge, purple explore border, the fit-gap
# warning) is demoted to a quiet neutral gray so the ⛔ actually stands out.
_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:780px;margin:0 auto;
padding:20px;background:#f5f6f8;color:#1a1a1a}
h1{font-size:20px}.muted{color:#595959;font-size:13px}
.topbar{text-align:right;font-size:12px;margin-bottom:4px}
.langsw a{color:#1F4E79;margin-left:6px}.langsw b{margin-left:6px;color:#1a1a1a}
.nav{font-size:13px;margin:2px 0 12px}.nav a{margin-right:10px}
.progress{font-size:13px;color:#1F4E79;font-weight:600;margin:4px 0 14px}
.card{background:#fff;border:1px solid #e2e4e8;border-radius:10px;padding:16px 18px;margin:14px 0;
box-shadow:0 1px 2px rgba(0,0,0,.04)}
.card.explore{border-left:3px solid #cdd2da}
.card.done{opacity:.45}
.alert{background:#fdecea;border:1px solid #f5c6cb;color:#a12622;border-radius:8px;
padding:10px 14px;margin:10px 0;font-size:13px}
.alert.warn{background:#f3f4f6;border-color:#e2e4e8;color:#595959}
.row{display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.title{font-size:16px;font-weight:600;margin:0}
.score{font-size:20px;line-height:1;padding:6px 12px;border-radius:999px;white-space:nowrap;
box-shadow:0 1px 3px rgba(0,0,0,.12)}
.meta{font-size:13px;color:#555;margin:4px 0}
.meta .meta2{color:#595959}
.why{display:inline-block;background:#eef0f2;color:#44474c;border-radius:6px;padding:1px 8px;
font-size:12px;margin-right:6px}
.summary{white-space:pre-line;margin:8px 0;font-size:14px}
.expl{font-size:13px;color:#444;font-style:italic}
.btns{margin-top:10px}
button{font-size:15px;border:1px solid #ccc;background:#fafafa;border-radius:8px;padding:6px 12px;
margin-right:6px;cursor:pointer}
button:hover{background:#eee}
button:focus-visible,summary:focus-visible,a:focus-visible{outline:2px solid #1F4E79;outline-offset:2px}
a.open{font-size:13px}.tag-explore{color:#595959;font-size:11px;font-weight:600}
.posted{color:#595959;font-size:12px}
.undo{display:none;font-size:12px;color:#595959;cursor:pointer;text-decoration:underline}
.card.done .undo{display:inline}
.verified{background:#f3f4f6;border:1px solid #e2e4e8;color:#44474c;border-radius:6px;
padding:6px 10px;margin:6px 0;font-size:12px}
.verified .suspect{color:#a85b00;font-weight:600}
.vnote{color:#4a4a4a;font-style:italic;margin-top:3px}
.headline{font-size:18px;font-weight:700;color:#26303a;margin:8px 0}
.metrics{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
.metrics th,.metrics td{text-align:left;padding:6px 8px;border-bottom:1px solid #e2e4e8}
.metrics th{color:#595959;font-weight:600}
.leaky{color:#a85b00;font-size:12px}
details.skills{margin:6px 0;font-size:13px}
details.skills summary{cursor:pointer;color:#2a2a2a;font-weight:600;list-style:revert}
details.skills ul{margin:6px 0 0;padding-left:18px;color:#444}
details.skills li{margin:1px 0}
.sk-missing{color:#a85b00}.sk-present{color:#1a6b34}
.far{color:#595959;font-size:12px}
details.enrich{margin:6px 0;font-size:13px}
details.enrich summary{cursor:pointer;color:#2a2a2a;font-weight:600;list-style:revert}
details.enrich .ej-clean{margin:6px 0;color:#333;background:#f7f9fc;border-left:3px solid #cdd6e6;
padding:6px 10px;border-radius:6px}
details.enrich .ej-company{margin:5px 0;color:#444}
.ej-pc{display:flex;gap:14px;margin:6px 0;flex-wrap:wrap}
.ej-pros,.ej-cons{flex:1;min-width:200px}
.ej-pros ul,.ej-cons ul,.ej-snip ul{margin:4px 0 0;padding-left:18px}
.ej-pros li{color:#1a6b34}.ej-cons li{color:#a85b00}
.ej-snip{margin:6px 0}.ej-snip li{color:#444;margin:2px 0}
.ej-goal{color:#595959;font-size:11px;text-transform:uppercase}
.ej-prov{color:#6b6b6b;font-size:11px;margin-top:6px}
.note-toggle{display:inline-block;font-size:12px;color:#1F4E79;cursor:pointer;margin-top:8px;
text-decoration:underline;background:none;border:none;padding:0}
.note{display:block;width:100%;box-sizing:border-box;margin:8px 0 2px;padding:6px 8px;
border:1px solid #e2e4e8;border-radius:8px;font:13px -apple-system,Segoe UI,Roboto,sans-serif;
resize:vertical;background:#fcfcfd;color:#1a1a1a}
.note:focus{outline:none;border-color:#9fb3d6}
.chips{margin:2px 0 12px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.chips .lbl{color:#595959;font-size:12px;margin-right:2px}
.chip{font-size:12px;font-family:inherit;margin:0;border:1px solid #d7dbe0;background:#fff;
color:#555;border-radius:999px;padding:3px 10px;cursor:pointer;user-select:none}
.chip:hover{background:#f0f2f5}
.chip.on{background:#1F4E79;border-color:#1F4E79;color:#fff;font-weight:600}
.card.hidden{display:none}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
clip:rect(0,0,0,0);white-space:nowrap;border:0}
details.legend{margin:2px 0 12px;font-size:13px}
details.legend summary{cursor:pointer;color:#595959;list-style:revert}
details.legend .leg{display:flex;flex-wrap:wrap;gap:4px 14px;margin-top:6px;color:#444;font-size:12px}
details.legend .leg span{white-space:nowrap}
@media (max-width:480px){
  body{padding:12px}
  .meta .meta2{display:block;margin-top:2px}
  .ej-pc{flex-direction:column;gap:6px}
  .ej-pros,.ej-cons{min-width:0}
  button{min-height:44px;padding:8px 14px}
  table.metrics thead{position:absolute;left:-9999px}
  table.metrics tr{display:block;border:1px solid #e2e4e8;border-radius:8px;margin:8px 0;
  padding:6px 10px;background:#fff}
  table.metrics td{display:flex;justify-content:space-between;gap:10px;border:0;padding:3px 0;
  text-align:right}
  table.metrics td::before{content:attr(data-label);color:#595959;font-weight:600;text-align:left}
}
"""

_JS_BODY = """
async function fb(id, action, el){
  const card = document.getElementById('card-'+id);
  // WS2: include the free-text note so a correction ('Master Data ≠ degree', 'люблю lead') becomes
  // durable judge-visible signal (why_freetext → few-shot). Empty note → null (don't wipe a prior).
  const noteEl = document.getElementById('note-'+id);
  const note = noteEl && noteEl.value.trim() ? noteEl.value.trim() : null;
  try{
    const r = await fetch('/feedback', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({vacancy_id:id, action:action, note:note})});
    if(r.ok){ card.classList.add('done');
      card.querySelector('.fbstate').textContent = ' ✓ '+action; bump(); }
  }catch(e){ alert('feedback failed: '+e); }
}
// Undo is client-side only: re-enable the card so a fresh click overwrites the label (the
// insert_label upsert keys on (vacancy_id, source) — re-rating replaces, no delete path needed).
function undo(id){
  const card = document.getElementById('card-'+id);
  card.classList.remove('done');
  const st = card.querySelector('.fbstate'); if(st) st.textContent='';
  bump();
}
// Goal-Gradient: live "N/M отмечено" counter, recomputed from the VISIBLE DOM so fb()+undo()+filter agree.
function bump(){
  const p = document.getElementById('progress'); if(!p) return;
  const total = +p.dataset.total;
  const done = document.querySelectorAll('.card.done').length;
  p.textContent = (done>=total && total>0) ? (T.done+' '+done+'/'+total)
                                            : (T.marked+' '+done+'/'+total);
}
// WS3 client-side filter chips: time window (data-days) + order (перспективные=fit DOM order /
// свежие=by days) + far toggle. No new page/endpoint (Hick/Tesler — subtract). Cards carry data-*.
var _F = {days: 1e9, order: 'fit', far: true, hideWeak: false};
var _ORDER = [];   // original card ids in effective (fit-led) order, for restoring 'перспективные'
function _initFilter(){
  _ORDER = Array.prototype.map.call(document.querySelectorAll('#cards .card'), function(c){return c.id;});
  applyFilter();
}
function chip(group, val, el){
  if(group==='far'){ _F.far = !_F.far; el.classList.toggle('on', _F.far); }
  else if(group==='weak'){ _F.hideWeak = !_F.hideWeak; el.classList.toggle('on', _F.hideWeak); }
  else {
    var sel = document.querySelectorAll('.chip[data-group="'+group+'"]');
    Array.prototype.forEach.call(sel, function(c){c.classList.remove('on');});
    if(el) el.classList.add('on');
    if(group==='time') _F.days = val; else if(group==='order') _F.order = val;
  }
  applyFilter();
}
function applyFilter(){
  var cont = document.getElementById('cards'); if(!cont) return;
  var cards = Array.prototype.slice.call(cont.querySelectorAll('.card'));
  cards.forEach(function(c){
    var days = +c.dataset.days, far = c.dataset.far==='1';
    // "скрыть слабые": weak = judge score ≤2 OR fit below the slate quality floor (0.45)
    var weak = (+c.dataset.score <= 2) || (+c.dataset.fit < 0.45);
    var show = days <= _F.days && !(far && !_F.far) && !(weak && _F.hideWeak);
    c.classList.toggle('hidden', !show);
  });
  var visible = cards.filter(function(c){return !c.classList.contains('hidden');});
  if(_F.order==='fresh'){
    visible.sort(function(a,b){return (+a.dataset.days) - (+b.dataset.days);});
  } else {
    visible.sort(function(a,b){return _ORDER.indexOf(a.id) - _ORDER.indexOf(b.id);});
  }
  visible.forEach(function(c){cont.appendChild(c);});
}
// UI-triggered fetch (full pipeline, async + single-flight server-side). Polls /fetch-status.
function triggerFetch(){
  var btn = document.getElementById('fetch-btn'), st = document.getElementById('fetch-status');
  if(btn) btn.disabled = true;
  if(st) st.textContent = T.fetch_starting;
  fetch('/fetch?lang='+LANG, {method:'POST'}).then(function(r){ return r.json().then(function(d){return {ok:r.ok, d:d};}); })
   .then(function(x){
     if(!x.ok){ if(st) st.textContent = ' ⚠ ' + (x.d.error || T.fetch_busy); if(btn) btn.disabled = false; return; }
     if(st) st.textContent = T.fetch_running;
     var poll = setInterval(function(){
       fetch('/fetch-status?lang='+LANG).then(function(r){return r.json();}).then(function(s){
         if(s.running){
           var label = s.stage_human || s.stage || T.fetch_working;
           var pos = (s.stage_index >= 0) ? (' (' + (s.stage_index+1) + '/' + s.n_stages + ')') : '';
           var heavy = s.heavy ? ' 🧠' : '';   // model-loading stage (qwen/bge)
           if(st) st.textContent = ' ⏳ ' + label + pos + heavy + '…';
         }
         else { clearInterval(poll); if(btn) btn.disabled = false;
                if(st) st.textContent = s.error ? (' ⚠ ' + s.error) : T.fetch_done; }
       }).catch(function(){ clearInterval(poll); if(btn) btn.disabled = false; });
     }, 5000);
   }).catch(function(e){ if(st) st.textContent = ' ⚠ ' + e; if(btn) btn.disabled = false; });
}
// /tasks: flip a comment-task's status (open|accounted|wontfix) → POST /task-status, update inline.
function taskStatus(id, status){
  fetch('/task-status', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({task_id:id, status:status})})
   .then(function(r){return r.json();}).then(function(d){
     if(!d.ok) return;
     var row = document.getElementById('task-'+id); if(!row) return;
     row.dataset.status = status;
     var b = row.querySelector('.tstate');
     if(b) b.textContent = status==='accounted' ? T.task_accounted
                         : status==='wontfix' ? T.task_wontfix : T.task_open;
   });
}
// Note toggle (Selective Attention): reveal the hidden note field after the buttons + focus it.
function toggleNote(id){
  var t = document.getElementById('note-'+id); if(!t) return;
  t.style.display = 'block'; t.focus();
  var b = document.getElementById('notebtn-'+id); if(b) b.style.display = 'none';
}
_initFilter();
"""


def _js(lang: str) -> str:
    """Client script with a localized strings object `T` + `LANG` injected at the top (the static
    body references T.key / LANG, so client-side text follows the page language)."""
    keys = ["js.marked", "js.done", "js.fetch_starting", "js.fetch_busy", "js.fetch_running",
            "js.fetch_working", "js.fetch_done", "js.task_accounted", "js.task_wontfix", "js.task_open"]
    tbl = {k.split(".", 1)[1]: t(lang, k) for k in keys}
    head = (f"const LANG={json.dumps(lang)};\n"
            f"const T={json.dumps(tbl, ensure_ascii=False)};\n")
    return head + _JS_BODY


def degraded_sources(con) -> list[str]:
    """Sources whose LATEST canary verdict is dead/degraded — surfaced in the slate header so
    a silent scraper death (the 'тихий Google' case) is never invisible (USE_CASE 1a)."""
    try:
        rows = con.execute(
            """SELECT source, verdict FROM canary_log c
               WHERE id = (SELECT MAX(id) FROM canary_log WHERE source = c.source)"""
        ).fetchall()
    except Exception:
        return []
    bad = []
    for r in rows:
        v = (r["verdict"] or "").lower()
        if v in ("dead_scraper", "degraded"):
            bad.append(f"{r['source']} → {v}")
    return bad


def _posted_ago(date_posted: str | None, first_seen: str | None = None,
                lang: str = DEFAULT_LANG) -> str:
    """Human 'N days ago' from the ISO posting date. Falls back to `first_seen` (when we first
    scraped it) labelled 'found' so a date ALWAYS shows — most board rows lack a real posting
    date (LinkedIn never returns one). '' only if neither parses."""
    from datetime import datetime, timezone
    verb, raw = t(lang, "posted.published"), date_posted
    if not raw:
        verb, raw = t(lang, "posted.found"), first_seen   # no posting date → show when we found it
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(str(raw))
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    days = (datetime.now(timezone.utc) - dt).days
    if days <= 0:
        return t(lang, "posted.today", verb=verb)
    if days == 1:
        return t(lang, "posted.yesterday", verb=verb)
    return t(lang, "posted.days_ago", verb=verb, days=days)


def _days_ago(date_posted: str | None, first_seen: str | None = None) -> int:
    """Numeric days-ago for the WS3 time-filter chip's data-days attribute (date_posted, else
    first_seen). 9999 when neither parses → such a card only shows under the 'всё' chip."""
    from datetime import datetime, timezone
    for raw in (date_posted, first_seen):
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(str(raw))
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - dt).days)
    return 9999


def _to_int(v: object) -> int | None:
    """Coerce a value to int, or None if it isn't numeric — lets callers skip a malformed salary
    by validating at the boundary instead of wrapping the format in a swallowing try/except."""
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _verified_html(inv: dict | None, lang: str = DEFAULT_LANG) -> str:
    """Compact 'deeper review' line from the investigator enrichment (company size/salary/English
    team/still-open/notes). Returns '' when there's no investigation for this card."""
    if not isinstance(inv, dict):
        return ""
    parts: list[str] = []
    size = inv.get("company_size")
    if size and str(size).lower() != "unknown":
        parts.append(t(lang, "verified.company", size=escape(str(size))))
    elif inv.get("company_known"):
        parts.append(t(lang, "verified.known_employer"))
    kmin, kmax = _to_int(inv.get("salary_eur_min")), _to_int(inv.get("salary_eur_max"))
    if kmin and kmax:
        parts.append(f"€{kmin // 1000}–{kmax // 1000}k")
    elif kmin or kmax:
        parts.append(f"€{(kmin or kmax) // 1000}k")
    if inv.get("english_team_signal"):
        parts.append(t(lang, "verified.english_team"))
    if inv.get("german_rooted"):
        parts.append(t(lang, "verified.german_rooted"))   # integration signal (validated company)
    src = inv.get("validation_source")
    if src and str(src).startswith("wikipedia"):
        parts.append(t(lang, "verified.wikipedia"))       # independently verified on a known site
    # still_open is now a DETERMINISTIC check (investigate._check_still_open): True=verified open,
    # False=verified gone (404/410/AA-API), None+key-present=checked-but-unverified (blocked/timeout).
    # Never a false "closed": an unverified link reads as calm info, not an alarm.
    so = inv.get("still_open")
    if so is True:
        parts.append(t(lang, "verified.open"))
    elif so is False:
        parts.append(t(lang, "verified.closed"))
    elif "still_open" in inv:
        parts.append(t(lang, "verified.unchecked"))
    verdict = inv.get("verdict") or ""
    badge = f'<span class="suspect">{t(lang, "verified.suspect")}</span> ' if verdict == "suspect" else ""
    notes = escape(str(inv.get("notes") or "").strip())
    desc = escape(str(inv.get("company_description") or "").strip())
    if not parts and not notes and not badge and not desc:
        return ""
    desc_html = f'<div class="vnote">{desc}</div>' if desc else ""   # deeper company description
    note_html = f'<div class="vnote">{notes}</div>' if notes else ""
    return f'<div class="verified">🔎 {badge}{" · ".join(parts)}{desc_html}{note_html}</div>'


def _enrichment_html(enr: dict | None, lang: str = DEFAULT_LANG) -> str:
    """Zotero-style enrichment block (collapsible): a clean re-parse of muddy ads, pros/cons two-
    column, key extracted JD snippets, + the model that produced it (provenance). '' when absent —
    so a card with no enrich run just omits it (graceful degrade)."""
    if not isinstance(enr, dict):
        return ""
    pros = [escape(str(p)) for p in (enr.get("pros") or []) if str(p).strip()]
    cons = [escape(str(c)) for c in (enr.get("cons") or []) if str(c).strip()]
    snippets = [s for s in (enr.get("key_snippets") or []) if isinstance(s, dict) and s.get("snippet")]
    clean = escape(str(enr.get("clean_summary") or "").strip())
    company = escape(str(enr.get("company_review") or "").strip())
    if not (pros or cons or snippets or clean or company):
        return ""
    parts: list[str] = []
    if clean:   # re-parse of a «глупый»/AI-slop description (the smarter-model win)
        parts.append(f'<div class="ej-clean">📝 {clean}</div>')
    if company:
        parts.append(f'<div class="ej-company">🏢 {company}</div>')
    if pros or cons:
        pl = "".join(f"<li>{p}</li>" for p in pros)
        cl = "".join(f"<li>{c}</li>" for c in cons)
        parts.append(
            '<div class="ej-pc">'
            f'<div class="ej-pros"><b>{t(lang, "enrich.pros")}</b><ul>{pl or "<li>—</li>"}</ul></div>'
            f'<div class="ej-cons"><b>{t(lang, "enrich.cons")}</b><ul>{cl or "<li>—</li>"}</ul></div></div>')
    if snippets:
        lis = "".join(
            f'<li><span class="ej-goal">{escape(str(s.get("goal") or ""))}:</span> '
            f'{escape(str(s.get("snippet") or ""))}</li>' for s in snippets[:2])
        parts.append(f'<div class="ej-snip"><b>{t(lang, "enrich.from_desc")}</b><ul>{lis}</ul></div>')
    model = escape(str(enr.get("model_used") or ""))
    prov = f'<div class="ej-prov">{t(lang, "enrich.review_by", model=model)}</div>' if model else ""
    return (f'<details class="enrich"><summary>{t(lang, "enrich.deep_dive")}</summary>'
            f'{"".join(parts)}{prov}</details>')


def _skills_html(e: dict, lang: str = DEFAULT_LANG) -> str:
    """Collapsible per-skill match: '🎯 Skills {cov}% · n✓·n◐·n✗' summary + the ✓/◐/✗ requirement
    list (matched first). Headline % = llm_cov (the honest skill-coverage), NOT the xenc-compressed
    fit_score. '' when there's no per-requirement data (llm_cov off / pre-rerank)."""
    reqs = e.get("llm_cov_reqs") or []
    cov = e.get("llm_cov")
    if not reqs or cov is None:
        return ""
    sym = {"present": "✓", "partial": "◐", "missing": "✗"}
    order = {"present": 0, "partial": 1, "missing": 2}
    n = {v: sum(1 for r in reqs if r.get("verdict") == v) for v in sym}
    lis = "".join(
        f'<li class="sk-{r.get("verdict")}">{sym.get(r.get("verdict"), "·")} '
        f'{escape(str(r.get("requirement") or ""))}</li>'
        for r in sorted(reqs, key=lambda r: order.get(r.get("verdict"), 3)))
    summary = t(lang, "skills.summary", cov=f"{float(cov):.0%}",
                present=n["present"], partial=n["partial"], missing=n["missing"])
    return (f'<details class="skills"><summary>{summary}</summary>'
            f'<ul>{lis}</ul></details>')


# Persona scale, 1 «офисная мышь» (drab) → 5 «шабашка» (vibrant). Themed emoji + a gradient that
# ENCODES magnitude — Von Restorff (the score stays the one bright element) + Aesthetic-Usability.
# One dict, trivially tweakable.
_SCORE_EMOJI = {1: "💻🐀", 2: "🐭", 3: "😐", 4: "😎", 5: "👸✨🧚"}
_SCORE_GRADIENT = {
    1: "linear-gradient(135deg,#9aa0a6,#c4c8cc)",   # офисная мышь — drab grey
    2: "linear-gradient(135deg,#8b93a3,#b9c1cf)",
    3: "linear-gradient(135deg,#6f8fd0,#a9c2ea)",   # neutral blue
    4: "linear-gradient(135deg,#e98b3a,#f6c463)",   # warming up
    5: "linear-gradient(135deg,#ff5fa2,#ffd24d)",   # шабашка — pink→gold
}


def _score_badge(score, lang: str = DEFAULT_LANG) -> str:
    """Themed score chip: 💻🐀 (1, office mouse) → 👸✨🧚 (5, шабашка) over a grey→gold/pink gradient
    encoding magnitude. '' when unscored."""
    if score is None:
        return ""
    s = max(1, min(5, int(score)))
    label = (t(lang, "label.office_mouse") if s == 1
             else t(lang, "label.shabashka") if s == 5 else f"{s}/5")
    # role=img + aria-label so a screen reader announces "score N of 5" instead of the raw emoji
    # (WCAG 4.1.2 accessible name; the emoji+gradient is decorative once the score is announced).
    return (f'<div class="score" role="img" aria-label="{escape(t(lang, "score.aria", s=s))}" '
            f'title="{escape(t(lang, "score.title", s=s, label=label))}" '
            f'style="background:{_SCORE_GRADIENT[s]}">{_SCORE_EMOJI[s]}</div>')


def _card_block(e: dict, *, show_applied: bool = True, lang: str = DEFAULT_LANG) -> str:
    """One HTML card block — shared by the daily slate and the annotation page (one data model, one
    template, no fork). Strong emphasis on exactly two things (Von Restorff): the score chip
    (💻🐀→👸✨🧚, gradient) and the red ⛔ STOP; the skill-gap ⚠ is muted.
    data-* attrs (days/score/fit/far) drive the client-side filter chips (WS3, no new endpoint)."""
    done = e.get("feedback")   # actioned today → render as done (persisted state)
    cls = "card explore" if e.get("slot_type") == "explore" else "card"
    if done:
        cls += " done"
    vid = e["vacancy_id"]
    tag_explore = (f'<span class="tag-explore">{t(lang, "card.explore_tag")}</span>'
                   if e.get("slot_type") == "explore" else "")
    why = escape(str(e.get("why") or ""))
    score = e.get("score")
    score_html = _score_badge(score, lang)
    fbstate = f' ✓ {escape(str(done))}' if done else ""
    posted = _posted_ago(e.get("date_posted"), e.get("first_seen"), lang)
    posted_html = f' · <span class="posted">{escape(posted)}</span>' if posted else ""
    # 📍 far-but-in-Germany marker (neutral gray, never competing with the score/⛔ — Von Restorff).
    far_html = ""
    if e.get("far"):
        dist = e.get("dist_km")
        anchor = e.get("geo_anchor")
        tail = (t(lang, "card.far_tail", km=int(round(float(dist))), anchor=escape(str(anchor).title()))
                if dist is not None and anchor else "")
        far_html = f' · <span class="far">{t(lang, "card.far")}{tail}</span>'
    # role-kind flag (W1): «🛠 hands-on» / «🎓 intern» — quiet gray, never competes with the score/⛔
    # (Von Restorff). Computed from the title so it shows on both build + reopen paths.
    kind = _rk.classify(e.get("title"), e.get("summary"))
    role_flag = t(lang, f"roleflag.{kind}") if kind in ("hands_on_engineer", "junior") else ""
    role_flag_html = f' · <span class="far">{escape(role_flag)}</span>' if role_flag else ""
    verified_html = _verified_html(e.get("investigation"), lang)   # deeper company review
    # cross-account repost note (455 Laveer ≡ 906 Westinghouse): same role at another employer.
    also = [escape(str(c)) for c in (e.get("also_at") or []) if str(c).strip()]
    also_html = (f'<div class="vnote">{t(lang, "card.also", names=", ".join(also))}</div>'
                 if also else "")
    # Eligibility severity (WS1b/1c): STRUCTURAL (PhD position / hard non-EN language) is the one red
    # ⛔ STOP; a SOFT prose-degree note renders muted amber and never sinks a strong-fit job.
    elig_note = escape(str(e.get("elig_note") or "").strip())
    severity = e.get("elig_severity") or "structural"
    stop_html = ""
    if elig_note:
        if severity == "soft":
            stop_html = f'<div class="alert warn">{t(lang, "card.stop_soft", note=elig_note)}</div>'
        else:
            stop_html = f'<div class="alert">{t(lang, "card.stop_hard", note=elig_note)}</div>'
    # Per-skill match (collapsible). When present it supersedes the one-line fit ⚠ gap note.
    skills_html = _skills_html(e, lang)
    enrich_html = _enrichment_html(e.get("enrichment"), lang)   # Zotero-style snippets + pros/cons
    fit_note = escape(str(e.get("fit_note") or "").strip())
    fit_pct = f"{float(e.get('fit_score') or 0.0):.0%}"
    warn_html = "" if skills_html else (
        f'<div class="alert warn">{t(lang, "card.fit_warn", note=fit_note, pct=fit_pct)}</div>'
        if fit_note else "")
    applied_btn = (f'<button aria-label="{escape(t(lang, "btn.applied_aria"))}" '
                   f'title="{escape(t(lang, "btn.applied_title"))}" '
                   f"onclick=\"fb({vid},'applied',this)\">{escape(t(lang, 'btn.applied_label'))}</button>"
                   ) if show_applied else ""
    # free-text feedback (WS2): a note per card → why_freetext → judge few-shot. Pre-filled on reload.
    # Selective Attention (laws-of-ux gate): hidden behind a "+ note" toggle placed AFTER the rating
    # buttons so it never pre-empts the primary action; opens by default if a prior note exists.
    user_note = escape(str(e.get("user_note") or ""))
    note_open = bool((e.get("user_note") or "").strip())
    note_hidden = "" if note_open else ' style="display:none"'
    note_toggle = ("" if note_open else
                   f'<button type="button" class="note-toggle" id="notebtn-{vid}" '
                   f'onclick="toggleNote({vid})">{t(lang, "card.note_toggle")}</button>')
    note_html = (f'<textarea class="note" id="note-{vid}" rows="2" '
                 f'aria-label="{escape(t(lang, "card.note_aria"))}" '
                 f'placeholder="{escape(t(lang, "card.note_placeholder"))}"{note_hidden}>{user_note}</textarea>')
    # data-* for the client-side filter chips (WS3)
    data_days = _days_ago(e.get("date_posted"), e.get("first_seen"))
    data_score = int(score) if score is not None else 0
    data_fit = float(e.get("fit_score") or 0.0)
    data_far = 1 if e.get("far") else 0
    return f"""
<div class="{cls}" id="card-{vid}" data-days="{data_days}" data-score="{data_score}" \
data-fit="{data_fit:.4f}" data-far="{data_far}">
  <div class="row">
    <div>
      <p class="title">{escape(str(e.get('title') or ''))} {tag_explore}</p>
      <p class="meta">{escape(str(e.get('company') or '—'))} · {escape(str(e.get('city') or '—'))}<span \
class="meta2"> · {escape(str(e.get('work_mode') or ''))}{posted_html}{far_html}{role_flag_html}</span></p>
    </div>
    {score_html}
  </div>
  {stop_html}{warn_html}
  <div>{'<span class="why">'+why+'</span>' if why else ''}</div>
  <div class="summary">{escape(str(e.get('summary') or ''))}</div>
  {skills_html}
  {enrich_html}
  {verified_html}{also_html}
  <div class="expl">{escape(str(e.get('explanation') or ''))}</div>
  <div class="btns">
    <button aria-label="{escape(t(lang, "btn.bad_aria"))}" title="{escape(t(lang, "btn.bad_title"))}" onclick="fb({vid},'bad',this)">💻🐀</button>
    <button aria-label="{escape(t(lang, "btn.good_aria"))}" title="{escape(t(lang, "btn.good_title"))}" onclick="fb({vid},'good',this)">😎</button>
    <button aria-label="{escape(t(lang, "btn.star_aria"))}" title="{escape(t(lang, "btn.star_title"))}" onclick="fb({vid},'star',this)">👸✨🧚</button>
    {applied_btn}
    <a class="open" href="{escape(str(e.get('url') or '#'))}" target="_blank">{t(lang, "card.open_original")}</a>
    <span class="fbstate muted">{fbstate}</span>
    <span class="undo" onclick="undo({vid})">{t(lang, "card.undo")}</span>
  </div>
  {note_toggle}
  {note_html}
</div>"""


def _chip_row(lang: str = DEFAULT_LANG) -> str:
    """WS3 filter chips (client-side, no endpoint): time window + relevance order + far toggle.
    Default-on chips reflect _JS._F defaults (all time, promising order, far shown) — nothing is
    hidden until she narrows, so a curated slate is never silently truncated (Aesthetic-Usability).
    """
    # Chips are <button type=button> (not <span>) so they are keyboard-focusable + Enter/Space work
    # natively (WCAG 2.1.1; Jakob — native semantics). .chip CSS overrides the default button look.
    return (
        '<div class="chips">'
        f'<span class="lbl">{t(lang, "chip.time_label")}</span>'
        f'<button type="button" class="chip" data-group="time" onclick="chip(\'time\',1,this)">{t(lang, "chip.today")}</button>'
        f'<button type="button" class="chip" data-group="time" onclick="chip(\'time\',3,this)">{t(lang, "chip.3days")}</button>'
        f'<button type="button" class="chip" data-group="time" onclick="chip(\'time\',7,this)">{t(lang, "chip.week")}</button>'
        f'<button type="button" class="chip on" data-group="time" onclick="chip(\'time\',1000000000,this)">{t(lang, "chip.all")}</button>'
        '<span class="lbl">·</span>'
        f'<button type="button" class="chip on" data-group="order" onclick="chip(\'order\',\'fit\',this)">{t(lang, "chip.promising")}</button>'
        f'<button type="button" class="chip" data-group="order" onclick="chip(\'order\',\'fresh\',this)">{t(lang, "chip.fresh")}</button>'
        '<span class="lbl">·</span>'
        f'<button type="button" class="chip on" data-group="far" onclick="chip(\'far\',0,this)">{t(lang, "chip.far")}</button>'
        '<span class="lbl">·</span>'
        f'<button type="button" class="chip" data-group="weak" onclick="chip(\'weak\',0,this)">{t(lang, "chip.hide_weak")}</button>'
        '</div>'
    )


# Compact COLLAPSED emoji legend (Jakob / Mental Model) — one shared header affordance so a
# non-technical user can decode the card glyphs, without per-card noise (Von Restorff / Cognitive Load).
_LEGEND_KEYS = ["legend.score", "legend.stop", "legend.warn", "legend.skills", "legend.deepdive",
                "legend.hands_on", "legend.junior", "legend.far", "legend.german",
                "legend.research", "legend.explore"]


def _legend_html(lang: str = DEFAULT_LANG) -> str:
    return (f'<details class="legend"><summary>{t(lang, "legend.summary")}</summary><div class="leg">'
            + "".join(f"<span>{escape(t(lang, k))}</span>" for k in _LEGEND_KEYS) + "</div></details>")


def _lang_toggle(lang: str) -> str:
    """Header language switcher: links to ?lang=<other> for every available locale (a new locale JSON
    appears here for free). The active language is shown bold-disabled."""
    links = []
    for code in available_langs():
        name = escape(t(code, "lang.name"))
        if code == lang:
            links.append(f'<b>{name}</b>')
        else:
            links.append(f'<a href="?lang={code}">{name}</a>')
    return f'<span class="langsw">{" · ".join(links)}</span>'


def _page(h1: str, top_html: str, body: str, lang: str = DEFAULT_LANG) -> str:
    """Self-contained HTML shell shared by every page. `lang` sets the <html> lang + the toggle."""
    return f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(h1)}</title><style>{_CSS}</style></head>
<body>
<div class="topbar">{_lang_toggle(lang)}</div>
<h1>{escape(h1)}</h1>
{top_html}
{body}
<script>{_js(lang)}</script>
</body></html>"""


# UI fetch trigger (the "how do I fetch new vacancies from the UI" answer). Runs the full pipeline
# async server-side (single-flight); the button polls /fetch-status via triggerFetch().
def _fetch_btn(lang: str = DEFAULT_LANG) -> str:
    return (f'<button id="fetch-btn" onclick="triggerFetch()" '
            f'title="{escape(t(lang, "fetch.btn_title"))}">{t(lang, "fetch.btn")}</button>'
            '<span id="fetch-status" class="muted"></span>')


def render_html(slate: list[dict], slate_date: str, alerts: list[str] | None = None,
                dedup_count: int | None = None, lang: str = DEFAULT_LANG) -> str:
    """Daily slate. Buttons → POST /feedback. alerts → a degradation banner. dedup_count → a visible
    'dedup happened' marker (logged-not-merged is otherwise invisible). The 🔄 button starts a fetch.

    The score legend is collapsed out of the header (Working Memory) — the button's meaning lives in
    its title tooltip; the header carries the N/M progress (Goal-Gradient) + nav to the other pages."""
    body = ('<div id="cards">' + "\n".join(_card_block(e, lang=lang) for e in slate) + "</div>") \
        if slate else f'<p class="muted">{t(lang, "slate.empty")}</p>'
    banner = ""
    if alerts:
        items = "; ".join(escape(str(a)) for a in alerts)
        banner = f'<div class="alert">{t(lang, "slate.degraded", items=items, lang=lang)}</div>'
    total = len(slate)
    done = sum(1 for e in slate if e.get("feedback"))
    pkey = "progress.done" if (done >= total and total) else "progress.marked"
    progress = (f'<p class="progress" id="progress" data-total="{total}">'
                f'{t(lang, pkey, done=done, total=total)}</p>' if total else "")
    dedup = (f'<p class="muted">{t(lang, "slate.dedup", n=dedup_count, lang=lang)}</p>'
             if dedup_count else "")
    nav = (f'<p class="nav muted"><a href="/annotate?lang={lang}">{t(lang, "nav.annotate_more")}</a>'
           f'<a href="/eval?lang={lang}">{t(lang, "nav.eval")}</a>'
           f'<a href="/gaps?lang={lang}">{t(lang, "nav.gaps")}</a>'
           f'<a href="/tasks?lang={lang}">{t(lang, "nav.tasks")}</a>'
           f'<a href="/funnel?lang={lang}">{t(lang, "nav.funnel")}</a> {_fetch_btn(lang)}</p>')
    chips = _chip_row(lang) if slate else ""
    legend = _legend_html(lang) if slate else ""
    return _page(t(lang, "slate.title", date=slate_date),
                 banner + nav + progress + dedup + legend + chips, body, lang)


def annotation_batch(cfg: dict, con, slate_date: str) -> tuple[list[dict], int]:
    """Очередь разметки: оценённые судьёй вакансии, ещё не размеченные (status SCORED/SLATED;
    LABELED отпадает сам, исчезая из очереди по мере разметки — Goal-Gradient к нулю).

    Возвращает (батч ≤ slate.annotate_batch, всего_в_очереди). Reuse _load_scored + _fit_fields —
    тот же источник данных, что и у дневного slate: одна модель данных, две проекции (Occam)."""
    n = int(cfg.get("slate", {}).get("annotate_batch", 30))
    rubric_version = cfg.get("judge", {}).get("rubric_version")
    items = _load_scored(con, rubric_version=rubric_version)
    cand_quals = _cand_quals(con)
    for it in items:
        it.update(_fit_fields(con, it["vacancy_id"], cfg, cand_quals))
        mark = _geo.geo_mark(it.get("city"), cfg)   # 📍 far marking on the annotate queue too
        it["far"], it["dist_km"], it["geo_anchor"] = mark["far"], mark["dist_km"], mark["anchor"]
    total = len(items)
    rng = random.Random(slate_date)   # стабильно в пределах дня, перемешано для разнообразия
    rng.shuffle(items)
    return items[:n], total


def render_annotate_html(items: list[dict], slate_date: str, *, total_pending: int,
                         lang: str = DEFAULT_LANG) -> str:
    """The single annotation surface (the xlsx pack is retired). Same card + same 💻🐀/😎/👸✨🧚
    buttons (no 'applied' — there's nothing to "apply to" for a random queued job)."""
    body = ('<div id="cards">' + "\n".join(_card_block(e, show_applied=False, lang=lang)
            for e in items) + "</div>") if items \
        else f'<p class="muted">{t(lang, "annotate.empty")}</p>'
    total = len(items)
    shown = t(lang, "annotate.shown", shown=total, total=total_pending) if total_pending > total else ""
    progress = (f'<p class="progress" id="progress" data-total="{total}">'
                f'{t(lang, "progress.marked", done=0, total=total)}</p>' if total else "")
    nav = (f'<p class="nav muted"><a href="/?lang={lang}">{t(lang, "nav.slate_today")}</a>'
           f'<a href="/eval?lang={lang}">{t(lang, "nav.eval")}</a>'
           f'<a href="/gaps?lang={lang}">{t(lang, "nav.gaps")}</a>'
           f'<a href="/tasks?lang={lang}">{t(lang, "nav.tasks")}</a>'
           f'<a href="/funnel?lang={lang}">{t(lang, "nav.funnel")}</a> {_fetch_btn(lang)}</p>')
    # Tesler/Jakob (laws-of-ux gate): the slate's time/order/far filter chips are dropped here — they
    # are meaningless on an unlabeled shuffle queue. Keep the shared emoji legend.
    legend = _legend_html(lang) if items else ""
    return _page(t(lang, "annotate.title", n=total_pending, shown=shown), nav + progress + legend,
                 body, lang)


def render_eval_html(report: dict, lang: str = DEFAULT_LANG) -> str:
    """Validation dashboard: matcher ranking-quality vs Alina's REAL labels (schabasch.validation).
    Von Restorff: one headline (the clean fit_score). Goal-Gradient: a "rate more in /annotate"
    banner until enough labels accrue. No inputs — the page only reads (Tesler)."""
    nav = (f'<p class="nav muted"><a href="/?lang={lang}">{t(lang, "nav.slate")}</a>'
           f'<a href="/annotate?lang={lang}">{t(lang, "nav.annotate")}</a>'
           f'<a href="/gaps?lang={lang}">{t(lang, "nav.gaps")}</a>'
           f'<a href="/funnel?lang={lang}">{t(lang, "nav.funnel")}</a></p>')
    if report["n_labels"] == 0:
        body = f'<p class="muted">{t(lang, "eval.empty", lang=lang)}</p>'
        return _page(t(lang, "eval.title"), nav, body, lang)

    h = report["headline"]
    pw = f'{h["pairwise_acc"]:.0%}'
    nd = f'{h["ndcg@10"]:.2f}'
    headline = f'<p class="headline">{t(lang, "eval.headline", pairwise=pw, ndcg=nd)}</p>'
    banner = ""
    if not report["reliable"]:
        msg = t(lang, "eval.banner", n=report["n_labels"], pairs=report["n_comparable_pairs"],
                min=report["min_pairs"], lang=lang)
        banner = f'<div class="alert warn">{msg}</div>'
    c_sig, c_pw, c_nd, c_sp, c_n = (t(lang, "eval.col.signal"), t(lang, "eval.col.pairwise"),
                                    t(lang, "eval.col.ndcg"), t(lang, "eval.col.spearman"),
                                    t(lang, "eval.col.n"))
    leaky = t(lang, "eval.leaky")
    trs = []
    for r in report["rows"]:
        tag = "" if r.get("clean") else f' <span class="leaky">{leaky}</span>'
        trs.append(f'<tr><td data-label="{c_sig}">{escape(str(r.get("label") or r["name"]))}{tag}</td>'
                   f'<td data-label="{c_pw}">{r["pairwise_acc"]:.0%}</td>'
                   f'<td data-label="{c_nd}">{r["ndcg@10"]:.2f}</td>'
                   f'<td data-label="{c_sp}">{r["spearman"]:.2f}</td>'
                   f'<td data-label="{c_n}">{r["n"]}</td></tr>')
    table = (f'<table class="metrics"><thead><tr><th>{c_sig}</th><th>{c_pw}</th><th>{c_nd}</th>'
             f'<th>{c_sp}</th><th>{c_n}</th></tr></thead><tbody>' + "".join(trs) + "</tbody></table>")
    note = f'<p class="muted">{t(lang, "eval.note")}</p>'
    return _page(t(lang, "eval.title"), nav, headline + banner + table + note, lang)


def _theme_label(lang: str, theme: str) -> str:
    """Localized theme label; falls back to the raw theme tag if no locale key exists."""
    lbl = t(lang, f"theme.{theme}")
    return theme if lbl == f"theme.{theme}" else lbl


def render_tasks_html(tasks: list[dict], summary: dict, lang: str = DEFAULT_LANG) -> str:
    """Comment-tracker page: every review comment as a theme-grouped task with an open|accounted|
    wontfix toggle — the "which feedback did the product act on, which not" audit (W1). Reuses the
    shared shell + JS (taskStatus). Read-and-toggle only (Tesler)."""
    nav = (f'<p class="nav muted"><a href="/?lang={lang}">{t(lang, "nav.slate")}</a>'
           f'<a href="/annotate?lang={lang}">{t(lang, "nav.annotate")}</a>'
           f'<a href="/eval?lang={lang}">{t(lang, "nav.eval")}</a>'
           f'<a href="/gaps?lang={lang}">{t(lang, "nav.gaps")}</a>'
           f'<a href="/funnel?lang={lang}">{t(lang, "nav.funnel")}</a></p>')
    if not tasks:
        body = f'<p class="muted">{t(lang, "tasks.empty", lang=lang)}</p>'
        return _page(t(lang, "tasks.title"), nav, body, lang)

    badge_of = {"accounted": t(lang, "tasks.status.accounted"),
                "wontfix": t(lang, "tasks.status.wontfix"), "open": t(lang, "tasks.status.open")}
    col_comment, col_acc = t(lang, "tasks.col.comment"), t(lang, "tasks.col.accounted")
    col_acc_s, col_status = t(lang, "tasks.col.accounted_short"), t(lang, "tasks.col.status")
    groups: dict[str, list[dict]] = {}
    for tk in tasks:
        groups.setdefault(tk["theme_tag"], []).append(tk)
    sections = []
    # show acted/recurring themes first; "other"/"pref" last
    order = ["jd-slop", "engineer-repellent", "junior-floor", "gap-too-big", "degree-misread",
             "degree-gap", "hidden-de", "duplicate", "pref", "other"]
    for theme in sorted(groups, key=lambda th: order.index(th) if th in order else 99):
        items = groups[theme]
        rows = []
        for tk in items:
            tid = tk["id"]
            st = tk["task_status"]
            meta = " · ".join(filter(None, [
                escape(str(tk.get("company") or "")),
                (t(lang, "tasks.score", s=tk["score_1_5"]) if tk.get("score_1_5") is not None else "")]))
            rows.append(
                f'<tr id="task-{tid}" data-status="{escape(st)}">'
                f'<td data-label="{col_comment}">{escape(str(tk["comment_text"]))}'
                f'{("<div class=\"muted\">" + meta + "</div>") if meta else ""}</td>'
                f'<td data-label="{col_acc_s}">{escape(str(tk.get("product_change") or ""))}</td>'
                f'<td data-label="{col_status}"><span class="tstate">{badge_of.get(st, st)}</span><br>'
                f'<button aria-label="{escape(t(lang, "tasks.btn.accounted_aria"))}" title="{escape(t(lang, "tasks.btn.accounted_title"))}" onclick="taskStatus({tid},\'accounted\')">✅</button>'
                f'<button aria-label="{escape(t(lang, "tasks.btn.open_aria"))}" title="{escape(t(lang, "tasks.btn.open_title"))}" onclick="taskStatus({tid},\'open\')">⏳</button>'
                f'<button aria-label="{escape(t(lang, "tasks.btn.wontfix_aria"))}" title="{escape(t(lang, "tasks.btn.wontfix_title"))}" onclick="taskStatus({tid},\'wontfix\')">🚫</button></td></tr>')
        sections.append(
            f'<h2 style="font-size:15px;margin:16px 0 4px">{escape(_theme_label(lang, theme))} '
            f'({len(items)})</h2>'
            f'<table class="metrics"><thead><tr><th>{col_comment}</th><th>{col_acc}</th>'
            f'<th>{col_status}</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")
    hl = t(lang, "tasks.headline", accounted=summary.get("accounted", 0),
           open=summary.get("open", 0), wontfix=summary.get("wontfix", 0),
           total=summary.get("total", 0))
    headline = f'<p class="headline">{hl}</p>'
    note = f'<p class="muted">{t(lang, "tasks.note")}</p>'
    return _page(t(lang, "tasks.title"), nav, headline + "".join(sections) + note, lang)


def render_gaps_html(report: dict, lang: str = DEFAULT_LANG) -> str:
    """Skill-gap dashboard (schabasch.gaps): across the jobs Alina WANTS (😎/👸✨🧚/applied), which
    requirements recur as ✗ missing / ◐ partial. A 'not on my CV → add it or learn it' list."""
    nav = (f'<p class="nav muted"><a href="/?lang={lang}">{t(lang, "nav.slate")}</a>'
           f'<a href="/annotate?lang={lang}">{t(lang, "nav.annotate")}</a>'
           f'<a href="/eval?lang={lang}">{t(lang, "nav.eval")}</a>'
           f'<a href="/funnel?lang={lang}">{t(lang, "nav.funnel")}</a></p>')
    n_wanted = report.get("n_wanted", 0)
    rows = report.get("rows") or []
    if n_wanted == 0 or not rows:
        body = f'<p class="muted">{t(lang, "gaps.empty", lang=lang)}</p>'
        return _page(t(lang, "gaps.title"), nav, body, lang)
    headline = (f'<p class="headline">'
                f'{t(lang, "gaps.headline", n=n_wanted, parsed=report.get("n_jobs_with_reqs", 0))}</p>')
    banner = ""
    if not report.get("reliable"):
        banner = f'<div class="alert warn">{t(lang, "gaps.banner", n=n_wanted, lang=lang)}</div>'
    c_req = t(lang, "gaps.td.req")
    trs = []
    for r in rows:
        trs.append(f'<tr><td data-label="{c_req}">{escape(str(r["requirement"]))}</td>'
                   f'<td data-label="{t(lang, "gaps.col.missing")}">{r["missing"]} ✗</td>'
                   f'<td data-label="{t(lang, "gaps.col.partial")}">{r["partial"]} ◐</td>'
                   f'<td data-label="{t(lang, "gaps.col.present")}">{r["present"]} ✓</td>'
                   f'<td data-label="{t(lang, "gaps.col.jobs")}">{r["jobs"]}</td></tr>')
    table = (f'<table class="metrics"><thead><tr><th>{t(lang, "gaps.col.req")}</th>'
             f'<th>{t(lang, "gaps.col.missing")}</th><th>{t(lang, "gaps.col.partial")}</th>'
             f'<th>{t(lang, "gaps.col.present")}</th><th>{t(lang, "gaps.col.jobs")}</th></tr></thead>'
             f'<tbody>' + "".join(trs) + "</tbody></table>")
    note = f'<p class="muted">{t(lang, "gaps.note")}</p>'
    return _page(t(lang, "gaps.title"), nav, headline + banner + table + note, lang)
