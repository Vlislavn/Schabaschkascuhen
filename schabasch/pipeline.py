"""Оркестрация: ночной tick + разовые импорты (спайк-данные, bootstrap-метки).

Идемпотентность: каждый шаг в try/except — сбой шага не валит tick, вакансии остаются в
предыдущем FSM-статусе и подхватываются следующей ночью. Порядок строго hard-before-soft:
канарейки → сбор → details → гео → жёсткие фильтры → нормализация → судья → slate.
"""
from __future__ import annotations

import threading
import time
from datetime import date

from . import db, dedup, features, geo, hardfilters, investigate, judge, memory_guard, \
    normalize, slate, triage
from .config import ROOT
from .models import FilterReason, Status
from .sources import arbeitsagentur, indeed, jobspy_source


# Rich per-stage console logging (the user can't tell what a long `serve`/`tick` is doing). On by
# default for every tick caller (serve, cron, /fetch); `serve --quiet` sets VERBOSE=False.
VERBOSE = True


def _fmt_result(r) -> str:
    """Compact one-line stage result for the ✓ line."""
    if isinstance(r, dict):
        nums = [f"{k}:{v}" for k, v in r.items() if isinstance(v, (int, float)) and v]
        if nums:
            return "{" + ", ".join(nums[:6]) + "}"
        if "error" in r:
            return f"ERROR {r['error']}"
        if "skipped_low_memory" in r:
            return "skipped (low memory)"
        return str(r)[:80]
    return str(r)[:80]


def _run_stage(name: str, fn, summary: dict, *, heavy: bool = False, label: str | None = None):
    """Run one pipeline stage with rich console logging: a ▶ start line, a 30 s heartbeat so a slow
    stage (LinkedIn scrape, qwen normalize/judge) never looks frozen, and a ✓ finish line with the
    compact result + elapsed seconds. Logging only — counts / funnel / FSM are unchanged."""
    if not VERBOSE:
        try:
            summary[name] = fn()
        except Exception as e:  # noqa: BLE001 — шаг не валит tick
            summary[name] = {"error": f"{type(e).__name__}: {e}"}
        return summary
    t0 = time.monotonic()
    print(f"  ▶ {label or name}{' ⏳ (loads a model)' if heavy else ''} …", flush=True)
    done = threading.Event()

    def _beat():
        while not done.wait(30.0):
            print(f"    … {name} still running ({int(time.monotonic() - t0)}s)", flush=True)
    threading.Thread(target=_beat, daemon=True).start()
    try:
        summary[name] = fn()
    except Exception as e:  # noqa: BLE001 — шаг не валит tick
        summary[name] = {"error": f"{type(e).__name__}: {e}"}
    finally:
        done.set()
    print(f"  ✓ {name} → {_fmt_result(summary[name])} ({int(time.monotonic() - t0)}s)", flush=True)
    return summary


def _step(name: str, fn, summary: dict):
    return _run_stage(name, fn, summary)


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
        if VERBOSE:
            print(f"  ⏭ {name} SKIPPED — low memory ({label})", flush=True)
        return summary
    return _run_stage(name, fn, summary, heavy=True, label=label)


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


def verify_liveness(cfg: dict, con) -> dict[str, int]:
    """Actively re-verify SCORED/SLATED postings whose `last_seen` has gone stale (older than the
    publication window) and EXPIRE the ones confirmed gone — Arbeitsagentur via its API (reliable),
    Indeed via the TLS+`"expired"`-JSON check (the user's «expired Indeed → убрать»). LinkedIn and
    the rest have no reliable per-URL liveness check (anti-bot), so they're left to the slate
    freshness window. Bounded by `retention.liveness_recheck_max` + throttled; ONLY a definitive gone
    verdict (False) expires — None (couldn't verify) never false-closes (mirrors _check_still_open)."""
    from datetime import datetime, timedelta, timezone
    import time as _time
    ret = cfg.get("retention", {}) or {}
    # A live board job within its posting window is re-seen every tick; once last_seen exceeds the
    # scrape window it has aged OUT (status ambiguous — live-but-old OR expired), so verify it. Keyed
    # to the scrape cadence (ceil(hours_old/24)+1 ≈ 3d), NOT the 7d publication window — else the
    # 5-day-stale expired Indeed jobs the user hit would never be checked.
    hours = int((cfg.get("search", {}) or {}).get("hours_old", 48) or 48)
    stale_days = int(ret.get("liveness_stale_days", max(1, -(-hours // 24) + 1)))
    cap = int(ret.get("liveness_recheck_max", 30))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=stale_days)).isoformat()
    # Verify the highest-scored stale cards first — those are the ones about to surface in the slate.
    rows = con.execute(
        """SELECT v.id, v.source, v.url, v.refnr
           FROM vacancy v
           JOIN (SELECT vacancy_id, MAX(id) mid FROM judge_score GROUP BY vacancy_id) m
             ON m.vacancy_id = v.id
           JOIN judge_score js ON js.id = m.mid
           WHERE v.status IN (?, ?) AND v.last_seen < ? AND v.source IN ('arbeitsagentur', 'indeed')
           ORDER BY js.score DESC, v.last_seen ASC
           LIMIT ?""",
        (Status.SCORED.value, Status.SLATED.value, cutoff, cap),
    ).fetchall()
    counts = {"checked": 0, "expired": 0, "unknown": 0}
    for r in rows:
        if r["source"] == "arbeitsagentur":
            alive = arbeitsagentur.check_open(r["refnr"])
        else:  # indeed
            alive = indeed.check_open(indeed.jk_from_url(r["url"]))
        counts["checked"] += 1
        if alive is False:
            db.set_status(con, r["id"], Status.EXPIRED, filter_reason=FilterReason.EXPIRED_GONE)
            counts["expired"] += 1
        elif alive is None:
            counts["unknown"] += 1
        _time.sleep(0.4)   # polite throttle between liveness probes
    con.commit()
    db.log_funnel(con, "verify_liveness", counts["expired"],
                  detail=f"checked={counts['checked']} expired={counts['expired']} "
                         f"unknown={counts['unknown']}")
    return counts


def nightly_tick(cfg: dict, con, *, german_queries: bool = False, budget: int | None = None,
                 tertiary: bool = False, run_investigate: bool = True) -> dict:
    """Полный ночной прогон. Возвращает (и печатает) сводку воронки.

    run_investigate=False skips the upfront agentic investigate batch (the serve-start path runs it
    progressively top→down instead, so the bot can greet as soon as the slate is built)."""
    search = cfg.get("search", {})
    queries = search.get("queries_de" if german_queries else "queries_en", [])
    summary: dict = {"date": date.today().isoformat(), "german": german_queries}
    import time as _time
    from datetime import datetime as _dt, timezone as _tz
    _t0 = _time.monotonic()
    _started_iso = _dt.now(_tz.utc).isoformat()   # new-vs-reseen delta (first_seen is INSERT-only)

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
    # Funnel-expansion sources (2026-07-03 research, все keyless): person-posts из соцсетей
    # (Bluesky/Mastodon/Telegram), HN Who-is-hiring, прямые ATS-борды. Каждый opt-in через
    # config-флаг (не CLI — multi-user tick читает per-user cfg).
    if search.get("social", False):
        from .sources import social as _soc
        _step("scrape_social", lambda: _soc.scrape(cfg, con), summary)
    if search.get("hn_hiring", False):
        from .sources import hn_hiring as _hn
        _step("scrape_hn", lambda: _hn.scrape(cfg, con), summary)
    if search.get("ats", False):
        from .sources import ats_boards as _ats
        _step("scrape_ats", lambda: _ats.scrape(cfg, con), summary)
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
    # Liveness: re-verify stale top SCORED/SLATED cards (AA API + Indeed TLS check) and EXPIRE the
    # confirmed-gone before the slate is built — so an "expired Indeed" never reaches the slate.
    # HTTP-only (no model), so a plain _step (not memory-gated).
    _step("verify_liveness", lambda: verify_liveness(cfg, con), summary)
    # Investigate only the top few cards (the ones the user sees first). Each deep-dive is a bounded
    # but ~60-120s agent run, so investigating all 8 exploit slots would dominate the tick; default
    # 3 keeps the cost ~4 min while still enriching the highest-ranked jobs. Tune via slate.investigate_top_n.
    if run_investigate:
        _heavy_step(con, "investigate", lambda: investigate.investigate_top(
            cfg, con, slate_date=summary["date"],
            top_n=int(cfg.get("slate", {}).get("investigate_top_n", 3))), summary, "глубокий поиск (агент)")
    # rebuild=True: re-rank today's slate now that fresh jobs are scored + stale ones expired, so a
    # mid-day /fetch reflects the new batch instead of returning the frozen morning slate. Labels
    # live in `label` (not slate_entry), so feedback is never lost; max_reshows + label-dedup keep
    # already-rated cards out.
    _step("slate", lambda: {"size": len(slate.build_slate(cfg, con, summary["date"], rebuild=True))},
          summary)
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
    # Honest tick DELTA (the "why nothing new / where did the time go" fix): genuinely-NEW vacancies
    # (first_seen set THIS tick) vs total scraped; slate size + how many slate cards are new today.
    def _sum_counts(v):
        if isinstance(v, dict):
            return int(sum(x for x in v.values() if isinstance(x, (int, float))))
        return int(v) if isinstance(v, (int, float)) else 0
    today = summary["date"]
    scraped = sum(_sum_counts(summary.get(k)) for k in
                  ("scrape_jobspy", "scrape_aa", "scrape_agent", "scrape_arbeitnow", "scrape_gtj",
                   "scrape_social", "scrape_hn", "scrape_ats"))
    new = int(con.execute("SELECT COUNT(*) FROM vacancy WHERE first_seen >= ?",
                          (_started_iso,)).fetchone()[0])
    slate_size = summary.get("slate", {}).get("size") if isinstance(summary.get("slate"), dict) else None
    slate_new = int(con.execute(
        "SELECT COUNT(*) FROM slate_entry se JOIN vacancy v ON v.id = se.vacancy_id "
        "WHERE se.slate_date = ? AND substr(v.first_seen, 1, 10) = ?", (today, today)).fetchone()[0])
    summary["delta"] = {"scraped": scraped, "new": new, "reseen": max(0, scraped - new),
                        "slate_size": slate_size, "slate_new": slate_new,
                        "wall_seconds": int(_time.monotonic() - _t0)}
    db.log_funnel(con, "tick_delta", new, detail=str(summary["delta"])[:300])
    _print_summary(summary)
    return summary


def _print_summary(s: dict) -> None:
    print(f"\n=== nightly tick {s['date']} (german={s['german']}) ===")
    for k in ("canary", "scrape_jobspy", "scrape_aa", "scrape_agent", "scrape_social",
              "scrape_hn", "scrape_ats", "details_aa",
              "expire_stale", "prefilter_geo", "hardfilters", "dedup_fuzzy", "features", "triage",
              "normalize", "judge", "rerank", "verify_liveness", "investigate", "slate", "enrich"):
        print(f"  {k:16s} {s.get(k)}")
    print(f"  status_counts    {s.get('status_counts')}")
    d = s.get("delta") or {}
    if d:
        print(f"  delta            scraped={d.get('scraped')} NEW={d.get('new')} "
              f"reseen={d.get('reseen')} slate={d.get('slate_size')} (new today {d.get('slate_new')}) "
              f"in {d.get('wall_seconds')}s\n")


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
