"""Оркестрация: ночной tick + разовые импорты (спайк-данные, bootstrap-метки).

Идемпотентность: каждый шаг в try/except — сбой шага не валит tick, вакансии остаются в
предыдущем FSM-статусе и подхватываются следующей ночью. Порядок строго hard-before-soft:
канарейки → сбор → details → гео → жёсткие фильтры → нормализация → судья → slate.
"""
from __future__ import annotations

from datetime import date

from . import db, dedup, features, geo, hardfilters, investigate, judge, memory_guard, normalize, \
    slate, triage
from .config import ROOT
from .models import Status
from .sources import arbeitsagentur, jobspy_source


def _step(name: str, fn, summary: dict):
    try:
        summary[name] = fn()
    except Exception as e:  # noqa: BLE001 — шаг не валит tick
        summary[name] = {"error": f"{type(e).__name__}: {e}"}
    return summary


def _heavy_step(con, name: str, fn, summary: dict, label: str):
    """A _step that loads a model (qwen / bge). Gated on free-RAM headroom BEFORE the load so a
    low-memory moment SKIPS that stage gracefully (logged, tick continues) instead of starting a
    swap death spiral on this constrained Mac — the memory-safety tripwire the user asked for. Shared
    by the cron tick AND the UI /fetch worker. Degrades to a plain _step where the probe is absent."""
    try:
        memory_guard.require_headroom(f"tick: {label}")
    except memory_guard.MemoryHeadroomError as e:
        summary[name] = {"skipped_low_memory": str(e)}
        db.log_funnel(con, "memory_skip", 0, source=name, detail=str(e)[:300])
        return summary
    return _step(name, fn, summary)


def expire_stale(cfg: dict, con, *, days: int | None = None) -> dict[str, int]:
    """Mark PRE-slate vacancies not seen in `days` as EXPIRED, bounding the active set so the
    triage drop-bucket and unprocessed rows don't accumulate and reprocess every night forever.
    Slate-relevant states (NORMALIZED/SCORED/SLATED/LABELED) are never expired."""
    from datetime import datetime, timedelta, timezone
    d = int(days if days is not None else cfg.get("retention", {}).get("expiry_days", 30))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=d)).isoformat()
    cur = con.execute(
        "UPDATE vacancy SET status = ? WHERE last_seen < ? AND status IN (?, ?, ?)",
        (Status.EXPIRED.value, cutoff, Status.NEW.value, Status.PREFILTERED.value,
         Status.DESCRIBED.value),
    )
    con.commit()
    n = cur.rowcount
    db.log_funnel(con, "expire", n, detail=f"days={d}")
    return {"expired": n}


def nightly_tick(cfg: dict, con, *, german_queries: bool = False, budget: int | None = None,
                 tertiary: bool = False) -> dict:
    """Полный ночной прогон. Возвращает (и печатает) сводку воронки."""
    search = cfg.get("search", {})
    queries = search.get("queries_de" if german_queries else "queries_en", [])
    summary: dict = {"date": date.today().isoformat(), "german": german_queries}

    _step("canary", lambda: jobspy_source.canary(cfg, con), summary)
    _step("scrape_jobspy", lambda: jobspy_source.scrape(cfg, con, queries=queries), summary)
    _step("scrape_aa", lambda: arbeitsagentur.search(cfg, con, queries=queries), summary)
    # Agentic discovery is OPT-IN: with the local qwen3:8b + DuckDuckGo it reliably finds ~0 real
    # postings (free-text search returns aggregator listing pages, which the quality gate discards;
    # JS-heavy careers pages can't be parsed within budget) while costing ~2 min/tick. It stays
    # bounded+gated for when a stronger backend (SEARXNG_URL / bigger model) is configured. The
    # board scrapers + the investigator are the workhorses. Enable via search.agent_discovery: true.
    if cfg.get("search", {}).get("agent_discovery", False):
        from .sources import agent_discovery as _ad
        _step("scrape_agent", lambda: _ad.scrape(cfg, con), summary)
    if tertiary:
        from .sources import tertiary as _t
        _step("scrape_arbeitnow", lambda: _t.fetch_arbeitnow(cfg, con), summary)
        _step("scrape_gtj", lambda: _t.fetch_germantechjobs(cfg, con), summary)
    _step("details_aa", lambda: arbeitsagentur.fetch_details(cfg, con), summary)
    _step("expire_stale", lambda: expire_stale(cfg, con), summary)
    _step("prefilter_geo", lambda: geo.prefilter(cfg, con), summary)
    _step("hardfilters", lambda: hardfilters.apply_hard_filters(cfg, con), summary)
    _step("dedup_fuzzy", lambda: dedup.dedup_fuzzy(cfg, con), summary)
    # Heavy stages (load bge-m3 / qwen3:8b / bge-reranker) are memory-gated: a low-RAM moment skips
    # the stage (logged 'memory_skip', tick continues) so a UI/cron tick can't swap-spiral the Mac.
    _heavy_step(con, "features", lambda: features.extract_features(cfg, con), summary, "признаки (bge-m3)")
    _step("triage", lambda: triage.triage_pending(cfg, con), summary)
    _heavy_step(con, "normalize", lambda: normalize.normalize_pending(cfg, con, budget=budget),
                summary, "нормализация (qwen3:8b)")
    _heavy_step(con, "judge", lambda: judge.judge_pending(cfg, con), summary, "оценка (qwen3:8b)")
    _heavy_step(con, "rerank", lambda: features.rerank_scored(cfg, con), summary, "rerank (bge-reranker)")
    # Investigate only the top few cards (the ones the user sees first). Each deep-dive is a bounded
    # but ~60-120s agent run, so investigating all 8 exploit slots would dominate the tick; default
    # 3 keeps the cost ~4 min while still enriching the highest-ranked jobs. Tune via slate.investigate_top_n.
    _heavy_step(con, "investigate", lambda: investigate.investigate_top(
        cfg, con, slate_date=summary["date"],
        top_n=int(cfg.get("slate", {}).get("investigate_top_n", 3))), summary, "глубокий поиск (агент)")
    _step("slate", lambda: {"size": len(slate.build_slate(cfg, con, summary["date"]))}, summary)
    # Zotero-style enrichment of the just-built slate cards (extractive snippets + abstractive
    # pros/cons + deep company + slop re-parse). Heavy (bge-reranker + cascade model) → gated; runs
    # on the slate set only; degrades to snippets-only when no abstractive tier is reachable.
    if (cfg.get("deep", {}) or {}).get("enable", True):
        from . import enrichment as _enrich
        _heavy_step(con, "enrich",
                    lambda: _enrich.enrich_slate(cfg, con, slate_date=summary["date"]),
                    summary, "обогащение карточек (сниппеты + pros/cons)")

    summary["status_counts"] = dict(
        con.execute("SELECT status, COUNT(*) FROM vacancy GROUP BY status").fetchall()
    )
    _print_summary(summary)
    return summary


def _print_summary(s: dict) -> None:
    print(f"\n=== nightly tick {s['date']} (german={s['german']}) ===")
    for k in ("canary", "scrape_jobspy", "scrape_aa", "scrape_agent", "details_aa",
              "expire_stale", "prefilter_geo", "hardfilters", "dedup_fuzzy", "features", "triage",
              "normalize", "judge", "rerank", "investigate", "slate", "enrich"):
        print(f"  {k:16s} {s.get(k)}")
    print(f"  status_counts    {s.get('status_counts')}\n")


# ---------------------------------------------------------------- разовые импорты

def _city0(loc) -> str | None:
    import pandas as pd
    if loc is None or (isinstance(loc, float) and pd.isna(loc)):
        return None
    return str(loc).split(",")[0].strip()


def import_spike_data(cfg: dict, con) -> dict[str, int]:
    """Разовый импорт уже собранного спайком пула (с описаниями → DESCRIBED)."""
    import pandas as pd

    base = ROOT / "spike" / "data"
    counts: dict[str, int] = {}

    def _present(v) -> bool:
        return v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v).strip() != ""

    # --- Indeed ---
    p = base / "indeed.csv"
    if p.exists():
        df = pd.read_csv(p)
        n = 0
        for _, r in df.iterrows():
            url = r.get("job_url")
            if not _present(url):
                continue
            db.upsert_vacancy(con, {
                "source": "indeed", "url": str(url), "title": str(r.get("title") or ""),
                "company": r.get("company") if _present(r.get("company")) else None,
                "city": _city0(r.get("location")),
                "is_remote_hint": 1 if r.get("is_remote") is True or str(r.get("is_remote")).lower() == "true"
                                  else (0 if str(r.get("is_remote")).lower() == "false" else None),
                "description": str(r.get("description")) if _present(r.get("description")) else None,
                "query_term": r.get("query_term"), "query_city": r.get("query_city"),
            })
            n += 1
        counts["indeed"] = n

    # --- LinkedIn ---
    p = base / "linkedin_described.csv"
    if p.exists():
        df = pd.read_csv(p)
        n = 0
        for _, r in df.iterrows():
            url = r.get("job_url")
            if not _present(url):
                continue
            db.upsert_vacancy(con, {
                "source": "linkedin", "url": str(url), "title": str(r.get("title") or ""),
                "company": r.get("company") if _present(r.get("company")) else None,
                "city": _city0(r.get("location")),
                "is_remote_hint": 1 if r.get("is_remote") is True or str(r.get("is_remote")).lower() == "true"
                                  else (0 if str(r.get("is_remote")).lower() == "false" else None),
                "description": str(r.get("description")) if _present(r.get("description")) else None,
                "query_term": None, "query_city": None,
            })
            n += 1
        counts["linkedin"] = n

    # --- Arbeitsagentur (details) ---
    p = base / "arbeitsagentur_details.csv"
    if p.exists():
        df = pd.read_csv(p)
        n = 0
        for _, r in df.iterrows():
            refnr = r.get("refnr")
            if not _present(refnr):
                continue
            ext = r.get("externeURL")
            url = str(ext) if _present(ext) else f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{refnr}"
            it = r.get("istArbeitnehmerUeberlassung")
            is_temp = None if not _present(it) else (1 if it is True or str(it).lower() in ("true", "1", "1.0") else 0)
            db.upsert_vacancy(con, {
                "source": "arbeitsagentur", "url": url, "refnr": str(refnr),
                "title": str(r.get("title") or r.get("search_title") or ""),
                "company": r.get("employer") if _present(r.get("employer")) else None,
                "city": r.get("city") if _present(r.get("city")) else None,
                "is_temp_agency": is_temp,
                "description": str(r.get("description")) if _present(r.get("description")) else None,
                "query_term": r.get("query"), "query_city": r.get("search_city"),
            })
            n += 1
        counts["arbeitsagentur"] = n

    db.log_funnel(con, "import_spike", sum(counts.values()), detail=str(counts))
    return counts
