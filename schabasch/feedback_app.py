"""Surface 2: локальная FastAPI-страница фидбека (копия паттерна VerdictPanel).

4 кнопки 👎/👍/⭐/applied пишут напрямую в SQLite проекта (нулевой impedance mismatch).
Клик ≤1 с. Соединение на запрос (sqlite не делится между потоками uvicorn).
"""
from __future__ import annotations

import re
import threading
import time
from datetime import date, datetime, timezone

from fastapi import Request   # module-level so `from __future__ annotations` strings resolve for FastAPI
from pydantic import BaseModel

from . import db, memory_guard, slate
from .i18n import normalize_lang, t
from .models import FEEDBACK_TO_SCORE, FilterReason, Status

# 5b: a feedback note flagging the posting as expired/closed/too-late is a HARD removal signal — the
# user literally read «Diese Stellenanzeige ist auf Indeed abgelaufen, не могу податься». Curated +
# conservative (word boundaries; deliberately NOT bare `alt`/`old` — salt/alternative false-match).
# ponytail: extend this list as the user coins new phrasings — that's the upgrade path.
_EXPIRY_NOTE_RE = re.compile(
    r"(\babgelaufen\b|\bexpired\b|no longer (available|accepting)|too late|не могу податься|"
    r"\bистёк\w*|\bустарел\w*|\bзакрыт\w*|снят\w* с публикац)", re.IGNORECASE)


def note_signals_expired(note: str | None) -> bool:
    """True when a feedback note flags the posting as gone → EXPIRE it (not just hide via dedup)."""
    return bool(note and _EXPIRY_NOTE_RE.search(note))


class Feedback(BaseModel):
    """Тело POST /feedback (на module-level, иначе FastAPI трактует как query-параметр)."""

    vacancy_id: int
    action: str  # bad | good | star | applied
    note: str | None = None  # WS2: free-text → label.why_freetext → judge few-shot (durable signal)


class TaskStatusBody(BaseModel):
    """Тело POST /task-status — переключить статус задачи-из-комментария (W1)."""

    task_id: int
    status: str  # open | accounted | wontfix


class RoleFeedback(BaseModel):
    """Тело POST /role-feedback — the ROLE axis ('good domain, wrong role'). fits=True → ✅ role fits,
    False → 🙅 wrong role. Separate from the 1–5 domain score (Delphi two-axis design)."""

    vacancy_id: int
    fits: bool


def _con(cfg: dict):
    return db.connect(cfg["paths"]["db"])


# ── async fetch (UI-triggered full pipeline) — SINGLE-FLIGHT, background thread ───────────────
# The tick is long + model-heavy (qwen normalize/judge + bge rerank). It must NOT block the request
# thread and only ONE may run at a time (a second click / a concurrent run would double-load models
# on this memory-constrained Mac). An in-process lock guards it; the run is idempotent (pipeline).
_FETCH_LOCK = threading.Lock()
# slate_ready: flips True once the slate is built + top-2 investigated (the bot's greet trigger,
# distinct from `running` which stays True through the slow progressive investigation that follows).
# fetch_skipped: the startup fetch was skipped because data was still fresh (refetch_after_hours) →
# the UI + bot prompt "data is N h old, не обновлялось — refetch?". data_age_hours = how old.
_FETCH_STATE: dict = {"running": False, "started": None, "finished": None, "summary": None,
                      "error": None, "slate_ready": False, "fetch_skipped": False, "data_age_hours": None}

# Human-readable pipeline stages for the /fetch progress (the "написано scrape, непонятно что это"
# fix). Ordered as nightly_tick runs them; `heavy` marks model-loading stages (qwen/bge). Only some
# log to funnel_log (the source of the live `stage`); the rest are shown for transparency in the
# checklist. Codes mirror db.log_funnel(...) stage strings.
_FETCH_STAGES: list[dict] = [
    {"code": "scrape",      "human": "Собираю вакансии: Indeed, LinkedIn, Arbeitsagentur", "heavy": False},
    {"code": "details",     "human": "Догружаю полные описания (Arbeitsagentur)", "heavy": False},
    {"code": "expire",      "human": "Помечаю устаревшие вакансии", "heavy": False},
    {"code": "prefilter",   "human": "Гео-фильтр (Heidelberg / Frankfurt ±40 км)", "heavy": False},
    {"code": "hardfilter",  "human": "Жёсткие фильтры (язык, временные агентства)", "heavy": False},
    {"code": "dedup_fuzzy", "human": "Дедупликация похожих вакансий", "heavy": False},
    {"code": "features",    "human": "Извлекаю признаки и фит (модель bge-m3)", "heavy": True},
    {"code": "normalize",   "human": "Читаю описания моделью qwen3:8b → карточки", "heavy": True},
    {"code": "judge",       "human": "Оцениваю вакансии моделью qwen3:8b (1–5)", "heavy": True},
    {"code": "rerank",      "human": "Переранжирую (модель bge-reranker)", "heavy": True},
    {"code": "investigate", "human": "Глубокий поиск про компании (агент)", "heavy": True},
    {"code": "slate",       "human": "Собираю итоговый слейт (10 рекомендаций)", "heavy": False},
]
_STAGE_BY_CODE = {s["code"]: s for s in _FETCH_STAGES}
_STAGE_INDEX = {s["code"]: i for i, s in enumerate(_FETCH_STAGES)}


def _run_tick_background(cfg: dict, *, german: bool, tertiary: bool) -> None:
    # EVERYTHING (the import + db.connect too) is inside the try so the finally ALWAYS releases the
    # single-flight lock + clears `running` — a failure in connect/import must not leak the lock and
    # wedge /fetch into a permanent 409 with running=True forever.
    con = None
    try:
        from . import memory_guard, pipeline
        # Memory safety (the "подключить модуль с управлением памятью" ask): start the watchdog and
        # refuse to START a model-heavy tick when free RAM is already below the hard floor — a clear
        # message instead of a swap death spiral. Per-stage gating inside nightly_tick covers mid-run
        # degradation. MemoryHeadroomError is surfaced via _FETCH_STATE["error"] (button shows it).
        memory_guard.configure_from_cfg(cfg)
        memory_guard.start_watchdog()
        memory_guard.require_headroom("fetch new jobs (qwen + bge models)")
        con = db.connect(cfg["paths"]["db"])
        summary = pipeline.nightly_tick(cfg, con, german_queries=german, tertiary=tertiary)
        _FETCH_STATE["summary"] = {k: summary.get(k) for k in
                                   ("scrape_jobspy", "scrape_aa", "dedup_fuzzy", "normalize",
                                    "judge", "rerank", "slate", "delta")}
        _FETCH_STATE["error"] = None
    except memory_guard.MemoryHeadroomError:
        # store a CODE; /fetch-status localizes it to the page language (the thread has no lang)
        _FETCH_STATE["error"] = "@memory"
    except Exception as e:  # noqa: BLE001 — surface, don't crash the worker thread
        _FETCH_STATE["error"] = f"{type(e).__name__}: {e}"
    finally:
        if con is not None:
            con.close()
        _FETCH_STATE["running"] = False
        _FETCH_STATE["finished"] = datetime.now(timezone.utc).isoformat()
        _FETCH_LOCK.release()


def _hours_since_last_fetch(con) -> float | None:
    """Hours since the last COMPLETED fetch (the `slate` funnel stage), or None if never fetched."""
    row = con.execute("SELECT MAX(run_at) FROM funnel_log WHERE stage = 'slate'").fetchone()
    if not row or not row[0]:
        return None
    try:
        ts = datetime.fromisoformat(row[0])
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0


def _refetch_guard(con, cfg: dict, *, force: bool) -> tuple[bool, str | None]:
    """Decide whether to SKIP the startup fetch because data is still fresh — so a quick restart does
    not re-scrape for ~20 min. Returns (skip, log_message). `serve --refetch` / refetch_after_hours<=0
    force a fetch."""
    if force:
        return False, None
    after = float((cfg.get("serve") or {}).get("refetch_after_hours", 12))
    if after <= 0:
        return False, None
    age = _hours_since_last_fetch(con)
    if age is not None and age < after:
        return True, (f"↻ data fetched {age:.1f}h ago (< {after:.0f}h) — skipping startup fetch; the "
                      f"recent slate stands. `schabasch serve --refetch` to force a fresh fetch.")
    return False, None


def _run_startup_pipeline(cfg: dict, *, seed: int = 2, dry: bool = False, quiet: bool = False) -> None:
    """On `schabasch serve` start: retrain (checkpointing the previous model) → full fetch (no upfront
    investigate) → build slate → investigate the top-`seed` cards, flip slate_ready (the bot greets) →
    keep investigating the rest TOP→DOWN so cards enrich 'on the go'. Single-flight via _FETCH_LOCK;
    heavy stages are memory-gated inside nightly_tick + before each progressive agent run.

    Emits RICH console logging (banner + phase headers + per-card lines) so a ~15-30 min run is legible;
    `quiet=True` mutes it. `nightly_tick` prints the per-stage ▶/✓ lines itself."""
    if not _FETCH_LOCK.acquire(blocking=False):
        return   # a /fetch is already running — don't double-load models

    def _say(msg: str = "") -> None:
        if not quiet:
            print(msg, flush=True)

    con = None
    t_start = time.monotonic()
    try:
        import logging
        from datetime import date as _date

        from . import investigate, memory_guard, pipeline, slate as _slate, triage
        _FETCH_STATE.update(running=True, started=datetime.now(timezone.utc).isoformat(),
                            finished=None, summary=None, error=None, slate_ready=False,
                            fetch_skipped=False)
        con = db.connect(cfg["paths"]["db"])
        memory_guard.configure_from_cfg(cfg)
        memory_guard.start_watchdog()

        # ── startup banner: what's about to happen, the scale, the long pole ──────────────────────
        counts = dict(con.execute("SELECT status, COUNT(*) FROM vacancy GROUP BY status").fetchall())
        plan = " · ".join(("⏳" if s["heavy"] else "") + s["code"] for s in _FETCH_STAGES)
        _say("")
        _say("═══ schabasch startup pipeline" + ("  (DRY — feedback NOT saved)" if dry else "") + " ═══")
        _say("  plan:  retrain → fetch → slate → investigate (top→down)")
        _say("  pending:  " + " · ".join(f"{v} {k}" for k, v in sorted(counts.items())))
        _say(f"  stages:  {plan}      (⏳ = loads a model)")
        _say("  est ~15-30 min — the LinkedIn scrape (~10-15 min) is the long pole. Follow the ▶/✓ lines below.")
        _say("")

        # 1) retrain the feedback model, checkpointing the previous one (no-op if labels unchanged)
        _say("▶ [1/3] retrain — model on your feedback …")
        try:
            r = triage.retrain_checkpointed(cfg, con)
            _FETCH_STATE["retrain"] = r
            rmsg = ("skipped (labels unchanged)" if r.get("skipped")
                    else f"trained on {r.get('n_rows')} labels" if r.get("trained") else str(r)[:60])
            _say(f"  ✓ retrain → {rmsg}")
        except Exception as e:  # noqa: BLE001 — a retrain failure must not block the fetch/serve
            _FETCH_STATE["retrain"] = {"error": f"{type(e).__name__}: {e}"}
            _say(f"  ✓ retrain → error: {e}")

        # 2) full fetch (always), WITHOUT the upfront agentic batch — progressive does it below.
        #    nightly_tick prints the per-stage ▶/✓ lines (pipeline.VERBOSE).
        _say("▶ [2/3] fetch — scrape → features → normalize → judge → rerank → slate → enrich")
        summary = pipeline.nightly_tick(cfg, con, run_investigate=False)
        _FETCH_STATE["summary"] = {k: summary.get(k) for k in
                                   ("scrape_jobspy", "scrape_aa", "dedup_fuzzy", "normalize",
                                    "judge", "rerank", "slate", "delta")}

        # 3) progressive investigation, top→down; greet after the seed is done
        today = _date.today().isoformat()
        cards = _slate.build_slate(cfg, con, today)
        n = len(cards)
        _say(f"▶ [3/3] investigate top→down — {n} cards (agent ~1-2 min each)")
        for i, c in enumerate(cards):
            vid = c["vacancy_id"]
            if i >= seed:   # the seed is ready → let the bot greet; keep enriching the rest
                if not _FETCH_STATE["slate_ready"]:
                    _FETCH_STATE["slate_ready"] = True
                    _say(f"  ✅ slate ready ({n} cards) — the bot greets now; enriching the rest in the background")
                try:
                    memory_guard.require_headroom("investigate (agent)")
                except memory_guard.MemoryHeadroomError:
                    _say("  ⏭ low memory — stopping the progressive pass (cards keep their enrichment)")
                    break   # low RAM → stop the progressive pass (cards still show enrichment)
            ti = time.monotonic()
            _say(f"  🔎 [{i+1}/{n}] {(c.get('company') or '?')[:30]} ({(c.get('title') or '')[:34]}) …")
            try:
                verdict = investigate.investigate_one(cfg, con, vid)
            except Exception as e:  # noqa: BLE001 — one card must not sink the pass
                verdict = f"error: {e}"
                logging.getLogger(__name__).warning("startup investigate vid=%s: %s", vid, e)
            _say(f"     {'· cached' if verdict == 'cached' else '✓ ' + str(verdict)} ({int(time.monotonic() - ti)}s)")
        _FETCH_STATE["error"] = None
    except memory_guard.MemoryHeadroomError:  # type: ignore[name-defined]
        _FETCH_STATE["error"] = "@memory"
        _say("✗ startup aborted — low memory")
    except Exception as e:  # noqa: BLE001 — surface, never crash the serve process
        _FETCH_STATE["error"] = f"{type(e).__name__}: {e}"
        _say(f"✗ startup error: {type(e).__name__}: {e}")
    finally:
        if con is not None:
            con.close()
        _FETCH_STATE["running"] = False
        _FETCH_STATE["slate_ready"] = True   # never leave the bot polling forever
        _FETCH_STATE["finished"] = datetime.now(timezone.utc).isoformat()
        _FETCH_LOCK.release()
        m, s = divmod(int(time.monotonic() - t_start), 60)
        err = _FETCH_STATE.get("error")
        _say(f"✓ startup complete in {m}m {s:02d}s" + (f" (with error: {err})" if err else ""))
        _say("")


def create_app(cfg: dict, dry: bool = False):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="Schabaschkascuhen Slate")

    def _lang(request: Request) -> str:
        return normalize_lang(request.query_params.get("lang"))

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> str:
        lang = _lang(request)
        con = _con(cfg)
        try:
            today = date.today().isoformat()
            entries = slate.build_slate(cfg, con, today)
            alerts = slate.degraded_sources(con)   # surface dead/degraded scrapers (UC 1a)
            if _FETCH_STATE.get("fetch_skipped"):   # data wasn't refreshed this start → offer a refetch
                _age = _FETCH_STATE.get("data_age_hours") or 0
                alerts = [f"⚠ Данные от {_age:.0f} ч назад — при запуске не обновлялись. Нажми кнопку "
                          f"обновления ниже, чтобы поискать свежие вакансии (займёт ~15–20 мин)."] + (alerts or [])
            dr = con.execute("SELECT count FROM funnel_log WHERE stage='dedup_fuzzy' "
                             "ORDER BY id DESC LIMIT 1").fetchone()
            dedup_count = int(dr["count"]) if dr else 0
            return slate.render_html(entries, today, alerts=alerts, dedup_count=dedup_count, lang=lang)
        finally:
            con.close()

    @app.get("/annotate", response_class=HTMLResponse)
    def annotate(request: Request) -> str:
        # Single annotation surface (the xlsx bootstrap pack is retired): the judged-but-unrated
        # queue, same card + same 💻🐀/😎/👸✨🧚 buttons, writing to the label table via /feedback.
        lang = _lang(request)
        con = _con(cfg)
        try:
            today = date.today().isoformat()
            items, total = slate.annotation_batch(cfg, con, today)
            return slate.render_annotate_html(items, today, total_pending=total, lang=lang)
        finally:
            con.close()

    @app.get("/eval", response_class=HTMLResponse)
    def eval_page(request: Request) -> str:
        # Live match-quality vs the user's REAL labels (the label table) — updates as the user rates in
        # /annotate. No manual code re-point; the synthetic GOLD stays a CLI-only dev floor.
        from . import validation
        lang = _lang(request)
        con = _con(cfg)
        try:
            return slate.render_eval_html(validation.eval_report(cfg, con), lang=lang)
        finally:
            con.close()

    @app.get("/gaps", response_class=HTMLResponse)
    def gaps_page(request: Request) -> str:
        # Recurring skill gaps across the jobs the user WANTS (😎/👸✨🧚/applied) — what to add to the CV
        # or learn. Pure aggregation of stored llm_cov_reqs (no LLM call on the web path).
        from . import gaps
        lang = _lang(request)
        con = _con(cfg)
        try:
            return slate.render_gaps_html(gaps.gap_report(cfg, con), lang=lang)
        finally:
            con.close()

    @app.get("/tasks", response_class=HTMLResponse)
    def tasks_page(request: Request) -> str:
        # Comment-tracker: every review comment as a theme-tagged task with an open|accounted|wontfix
        # status — the "which feedback did we act on" audit (W1). Pure read; toggles via /task-status.
        from . import tasks as _tasks
        lang = _lang(request)
        con = _con(cfg)
        try:
            return slate.render_tasks_html(_tasks.all_tasks(con), _tasks.summary(con), lang=lang)
        finally:
            con.close()

    @app.post("/task-status")
    def task_status(body: TaskStatusBody):
        from . import tasks as _tasks
        if body.status not in _tasks.STATUSES:
            return JSONResponse({"ok": False, "error": "bad status"}, status_code=400)
        con = _con(cfg)
        try:
            ok = _tasks.set_status(con, body.task_id, body.status)
            return JSONResponse({"ok": ok}, status_code=200 if ok else 404)
        finally:
            con.close()

    @app.post("/feedback")
    def feedback(fb: Feedback):
        if fb.action not in ("bad", "good", "star", "applied", "direction"):
            return JSONResponse({"ok": False, "error": "bad action"}, status_code=400)
        if dry:
            # `serve --dry`: testing mode — ack so the bot flow works, but DON'T touch the golden
            # label table. Validation above still runs (catches bot bugs); only the write is skipped.
            print(f"[dry] feedback vacancy_id={fb.vacancy_id} action={fb.action} note={fb.note!r}")
            return {"ok": True, "dry": True}
        con = _con(cfg)
        try:
            # Reject feedback for a non-existent vacancy — FKs are off, so without this an
            # unknown id silently corrupts the golden label table.
            if not con.execute("SELECT 1 FROM vacancy WHERE id = ?", (fb.vacancy_id,)).fetchone():
                return JSONResponse({"ok": False, "error": "unknown vacancy_id"}, status_code=404)
            # WS2: a typed note rides on EITHER branch → why_freetext (insert_label COALESCEs, so an
            # empty note never wipes a prior one; applied+note keeps the prior score).
            note = (fb.note or "").strip() or None
            if fb.action == "applied":
                # 'applied' is a FLAG on top of the score (USE_CASE glossary), NOT a 5-star
                # rating. Keep any existing score; never fabricate a 5 for an unrated job.
                row = con.execute(
                    "SELECT score_1_5, interview FROM label WHERE vacancy_id = ? AND source = 'slate'",
                    (fb.vacancy_id,),
                ).fetchone()
                prior = row["score_1_5"] if row else None
                prior_iv = row["interview"] if row else None
                lab = {"score_1_5": prior, "applied": 1, "source": "slate", "why_freetext": note,
                       "interview": prior_iv if prior_iv is not None
                       else (1 if (prior or 0) >= 4 else None)}
            elif fb.action == "direction":
                # 🧭 «направление интересно, но не эта вакансия»: a low score removes THIS posting from
                # the slate, while a positive role-fit + the magnet why_tag boost the DIRECTION/domain.
                import json as _json

                from . import role_feedback as _rolefb, role_kind as _rk
                vrow = con.execute("SELECT title, card_json FROM vacancy WHERE id = ?",
                                   (fb.vacancy_id,)).fetchone()
                try:
                    summ = (_json.loads(vrow["card_json"]) if vrow and vrow["card_json"] else {}
                            ).get("summary_2lines", "")
                except (TypeError, _json.JSONDecodeError):
                    summ = ""
                kind = _rk.classify(vrow["title"] if vrow else "", summ)
                jt = con.execute("SELECT why_tag FROM judge_score WHERE vacancy_id = ? "
                                 "ORDER BY id DESC LIMIT 1", (fb.vacancy_id,)).fetchone()
                lab = {"score_1_5": FEEDBACK_TO_SCORE["direction"], "applied": 0, "source": "slate",
                       "why_tag": (jt["why_tag"] if jt else None),
                       "why_freetext": note or "🧭 направление интересно (домен ок, не эта вакансия)",
                       "interview": 0}
            else:
                score = FEEDBACK_TO_SCORE[fb.action]
                lab = {"score_1_5": score, "applied": 0, "source": "slate", "why_freetext": note,
                       "interview": 1 if score >= 4 else (0 if score <= 2 else None)}
            db.insert_label(con, fb.vacancy_id, lab)  # сам ставит status=LABELED
            if note_signals_expired(note):
                # 5b: the note says the posting is gone (e.g. «…abgelaufen», «too late») → EXPIRE it so
                # it leaves every future slate (not just dedup-hidden). Fires regardless of score (the
                # user gave an expired job 👍). The golden label row stays as training signal.
                db.set_status(con, fb.vacancy_id, Status.EXPIRED,
                              filter_reason=FilterReason.EXPIRED_GONE)
            if fb.action == "direction":
                _rolefb.record(con, fb.vacancy_id, kind, True, source="slate")  # direction-fits → boost
            # отметить фидбек в сегодняшнем slate_entry
            con.execute(
                "UPDATE slate_entry SET feedback = ? WHERE vacancy_id = ? AND slate_date = ?",
                (fb.action, fb.vacancy_id, date.today().isoformat()),
            )
            con.commit()
            return {"ok": True}
        finally:
            con.close()

    @app.post("/role-feedback")
    def role_feedback(rf: RoleFeedback):
        """The ROLE axis — 'good domain, wrong role'. Classifies the role server-side (authoritative)
        and records it in the `label_role` sidecar (golden), separate from the 1–5 domain score.
        --dry: logged, not persisted (same firewall as /feedback)."""
        import json as _json

        from . import role_feedback as _rolefb, role_kind as _rk
        con = _con(cfg)
        try:
            row = con.execute("SELECT title, card_json FROM vacancy WHERE id = ?",
                              (rf.vacancy_id,)).fetchone()
            if not row:
                return JSONResponse({"ok": False, "error": "unknown vacancy_id"}, status_code=404)
            try:
                summary = (_json.loads(row["card_json"]) if row["card_json"] else {}).get("summary_2lines", "")
            except (TypeError, _json.JSONDecodeError):
                summary = ""
            kind = _rk.classify(row["title"], summary)
            if dry:
                print(f"[dry] role-feedback vacancy_id={rf.vacancy_id} kind={kind} fits={rf.fits}")
                return {"ok": True, "dry": True, "role_kind": kind}
            _rolefb.record(con, rf.vacancy_id, kind, rf.fits, source="slate")
            return {"ok": True, "role_kind": kind}
        finally:
            con.close()

    @app.post("/fetch")
    def fetch(request: Request, german: bool = False, tertiary: bool = False):
        """UI-triggered full pipeline (scrape→…→slate), async + SINGLE-FLIGHT. Returns immediately;
        poll /fetch-status. 409 if a fetch is already running (prevents double model-load)."""
        lang = _lang(request)
        if not _FETCH_LOCK.acquire(blocking=False):
            return JSONResponse({"ok": False, "error": t(lang, "fetch.busy409")}, status_code=409)
        _FETCH_STATE.update(running=True, started=datetime.now(timezone.utc).isoformat(),
                            finished=None, summary=None, error=None, fetch_skipped=False)
        threading.Thread(target=_run_tick_background, args=(cfg,),
                         kwargs={"german": german, "tertiary": tertiary}, daemon=True).start()
        return {"ok": True, "message": t(lang, "fetch.started")}

    @app.get("/fetch-status")
    def fetch_status(request: Request):
        """Poll the background fetch: running flag + the latest funnel stage as progress (localized)."""
        lang = _lang(request)
        con = _con(cfg)
        try:
            latest = con.execute(
                "SELECT stage, count FROM funnel_log ORDER BY id DESC LIMIT 1").fetchone()
        finally:
            con.close()
        code = latest["stage"] if latest else None
        if code == "memory_skip":
            stage_human, heavy, idx = t(lang, "fetch.memory_skip"), False, -1
        else:
            info = _STAGE_BY_CODE.get(code, {})
            stage_human = t(lang, f"fetch.stage.{code}") if code in _STAGE_BY_CODE else (code or None)
            heavy = bool(info.get("heavy"))
            idx = _STAGE_INDEX.get(code, -1)
        # localized error: a "@memory" sentinel from the worker → the friendly low-memory message
        err = _FETCH_STATE["error"]
        if err == "@memory":
            err = t(lang, "fetch.err_memory")
        stages = [{"code": s["code"], "heavy": s["heavy"], "human": t(lang, f"fetch.stage.{s['code']}")}
                  for s in _FETCH_STAGES]
        return {"running": _FETCH_STATE["running"], "started": _FETCH_STATE["started"],
                "finished": _FETCH_STATE["finished"], "summary": _FETCH_STATE["summary"],
                "error": err,
                "slate_ready": _FETCH_STATE.get("slate_ready", False),
                "fetch_skipped": _FETCH_STATE.get("fetch_skipped", False),
                "data_age_hours": _FETCH_STATE.get("data_age_hours"),
                "stage": code,
                "stage_count": (latest["count"] if latest else None),
                "stage_human": stage_human, "heavy": heavy, "stage_index": idx,
                "n_stages": len(_FETCH_STAGES),
                "stages": stages}

    @app.get("/funnel")
    def funnel():
        import json as _json
        con = _con(cfg)
        try:
            rows = con.execute(
                "SELECT run_at, stage, source, count, detail FROM funnel_log "
                "ORDER BY id DESC LIMIT 40"
            ).fetchall()
            canaries = con.execute(
                "SELECT run_at, source, verdict, rows, detail FROM canary_log "
                "ORDER BY id DESC LIMIT 12"
            ).fetchall()
            status_counts = dict(con.execute(
                "SELECT status, COUNT(*) FROM vacancy GROUP BY status"
            ).fetchall())
            # Surface DE-DUP so it's visible (it's logged-not-merged → otherwise invisible): parse the
            # latest dedup_fuzzy funnel detail into a candidate list (sim / titles / sources / company).
            dedup_candidates: list = []
            dr = con.execute(
                "SELECT count, detail FROM funnel_log WHERE stage='dedup_fuzzy' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if dr and dr["detail"]:
                try:
                    dedup_candidates = _json.loads(dr["detail"])
                except (TypeError, ValueError):
                    dedup_candidates = []
            return {
                "funnel": [dict(r) for r in rows],
                "canaries": [dict(r) for r in canaries],
                "status_counts": status_counts,
                "dedup_candidates": dedup_candidates,
                "dedup_count": (dr["count"] if dr else 0),
            }
        finally:
            con.close()

    @app.get("/slate.json")
    def slate_json():
        # Same data the HTML index renders (index() above), as JSON — the Telegram bot's only read.
        import json as _json
        con = _con(cfg)
        try:
            entries = slate.build_slate(cfg, con, date.today().isoformat())
            # ponytail: build_slate has never been JSON'd (HTML-only) → fit_score may be a numpy float.
            # Coerce non-native types (numpy → float) instead of 500-ing; typed view if it ever grows.
            return JSONResponse(content=_json.loads(_json.dumps(entries, default=float)))
        finally:
            con.close()

    @app.get("/backlog.json")
    def backlog_json():
        # The judged-but-unrated BACKLOG (the /annotate pool) as JSON — the bot's "/more" beyond the
        # ≤10 daily slate. Reuses slate.annotation_batch (no freshness ceiling). Returns {cards, total}.
        import json as _json
        con = _con(cfg)
        try:
            cards, total = slate.annotation_batch(cfg, con, date.today().isoformat())
            return JSONResponse(content=_json.loads(_json.dumps(
                {"cards": cards, "total": total}, default=float)))
        finally:
            con.close()

    return app


def serve(cfg: dict, dry: bool = False, fetch_on_start: bool | None = None, quiet: bool = False,
          refetch: bool = False):
    import importlib.util
    import os
    import subprocess
    import sys

    import uvicorn

    from . import pipeline
    # CAPA belt-and-suspenders: cap the torch/OpenMP intra-op pool from the SAME config knob
    # (features.torch_num_threads) BEFORE any torch import, so libomp inits single-threaded and the
    # features-stage deadlock can't recur even before features._cap_torch_threads runs. setdefault →
    # an explicit operator OMP_NUM_THREADS still wins. See features._cap_torch_threads / the CAPA.
    _nt = (cfg.get("features") or {}).get("torch_num_threads", 1)
    if _nt and int(_nt) > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(int(_nt)))
    pipeline.VERBOSE = not quiet   # rich per-stage console logging (the startup pipeline + any tick)
    if dry:
        print("⚠️  DRY MODE — POST /feedback + /role-feedback are logged, NOT written.")
    if fetch_on_start is None:
        fetch_on_start = bool((cfg.get("serve") or {}).get("fetch_on_start", True))
    # Freshness guard: don't re-scrape on startup if a fetch completed within refetch_after_hours
    # (default 12) — a quick restart reuses the recent slate. `--refetch` forces it.
    if fetch_on_start:
        _c = db.connect(cfg["paths"]["db"])
        try:
            _skip, _msg = _refetch_guard(_c, cfg, force=refetch)
            _age = _hours_since_last_fetch(_c) if _skip else None
        finally:
            _c.close()
        if _skip:
            print(_msg, flush=True)
            fetch_on_start = False
            # surface "data was NOT updated this start" so the UI + bot can offer a refetch
            _FETCH_STATE.update(fetch_skipped=True, data_age_hours=_age)
    port = int(cfg.get("slate", {}).get("port", 8787))
    # Optionally bring up the Telegram bot (separate repo, installed editable) as a child process.
    # Soft: a missing token / disabled flag / uninstalled `schabasch_bot` just skips it — never crashes.
    tg = cfg.get("telegram", {})
    token = tg.get("token") or os.environ.get("TELEGRAM_BOT_TOKEN")
    # Per-mode chat lock: prod (no --dry) → the real user (golden labels); --dry → the debug chat
    # (you, while debugging — feedback is not persisted anyway). 0/empty falls back to auto-lock.
    locked_chat = (tg.get("chat_id_debug") if dry else tg.get("chat_id")) or 0
    child = None
    if tg.get("enabled") and token and importlib.util.find_spec("schabasch_bot"):
        env = {**os.environ, "TELEGRAM_BOT_TOKEN": token,
               "SCHABASCH_BASE_URL": f"http://127.0.0.1:{port}",
               "TELEGRAM_CHAT_ID": str(locked_chat)}
        child = subprocess.Popen([sys.executable, "-m", "schabasch_bot"], env=env)
    if fetch_on_start:
        # retrain + full fetch + progressive investigate, in a daemon thread so the server comes up
        # immediately and the bot can poll /fetch-status (it greets when slate_ready flips).
        seed = int((cfg.get("serve") or {}).get("investigate_seed", 2))
        threading.Thread(target=_run_startup_pipeline, args=(cfg,),
                         kwargs={"seed": seed, "dry": dry, "quiet": quiet}, daemon=True).start()
    else:
        _FETCH_STATE["slate_ready"] = True   # no auto-fetch → bot greets with the existing slate
    app = create_app(cfg, dry=dry)
    try:
        uvicorn.run(app, host="127.0.0.1", port=port)
    finally:
        if child is not None:
            child.terminate()
