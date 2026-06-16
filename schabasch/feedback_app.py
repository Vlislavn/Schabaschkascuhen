"""Surface 2: локальная FastAPI-страница фидбека (копия паттерна VerdictPanel).

4 кнопки 👎/👍/⭐/applied пишут напрямую в SQLite проекта (нулевой impedance mismatch).
Клик ≤1 с. Соединение на запрос (sqlite не делится между потоками uvicorn).
"""
from __future__ import annotations

import threading
from datetime import date, datetime, timezone

from pydantic import BaseModel

from . import db, memory_guard, slate
from .models import FEEDBACK_TO_SCORE


class Feedback(BaseModel):
    """Тело POST /feedback (на module-level, иначе FastAPI трактует как query-параметр)."""

    vacancy_id: int
    action: str  # bad | good | star | applied
    note: str | None = None  # WS2: free-text → label.why_freetext → judge few-shot (durable signal)


class TaskStatusBody(BaseModel):
    """Тело POST /task-status — переключить статус задачи-из-комментария (W1)."""

    task_id: int
    status: str  # open | accounted | wontfix


def _con(cfg: dict):
    return db.connect(cfg["paths"]["db"])


# ── async fetch (UI-triggered full pipeline) — SINGLE-FLIGHT, background thread ───────────────
# The tick is long + model-heavy (qwen normalize/judge + bge rerank). It must NOT block the request
# thread and only ONE may run at a time (a second click / a concurrent run would double-load models
# on this memory-constrained Mac). An in-process lock guards it; the run is idempotent (pipeline).
_FETCH_LOCK = threading.Lock()
_FETCH_STATE: dict = {"running": False, "started": None, "finished": None, "summary": None, "error": None}

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
        memory_guard.require_headroom("обновление вакансий (модели qwen + bge)")
        con = db.connect(cfg["paths"]["db"])
        summary = pipeline.nightly_tick(cfg, con, german_queries=german, tertiary=tertiary)
        _FETCH_STATE["summary"] = {k: summary.get(k) for k in
                                   ("scrape_jobspy", "scrape_aa", "dedup_fuzzy", "normalize",
                                    "judge", "rerank", "slate")}
        _FETCH_STATE["error"] = None
    except memory_guard.MemoryHeadroomError:
        # friendly RU message for the button (not the verbose English headroom text)
        _FETCH_STATE["error"] = "мало памяти — закрой тяжёлые приложения и попробуй снова"
    except Exception as e:  # noqa: BLE001 — surface, don't crash the worker thread
        _FETCH_STATE["error"] = f"{type(e).__name__}: {e}"
    finally:
        if con is not None:
            con.close()
        _FETCH_STATE["running"] = False
        _FETCH_STATE["finished"] = datetime.now(timezone.utc).isoformat()
        _FETCH_LOCK.release()


def create_app(cfg: dict):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="Schabaschkascuhen Slate")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        con = _con(cfg)
        try:
            today = date.today().isoformat()
            entries = slate.build_slate(cfg, con, today)
            alerts = slate.degraded_sources(con)   # surface dead/degraded scrapers (UC 1a)
            dr = con.execute("SELECT count FROM funnel_log WHERE stage='dedup_fuzzy' "
                             "ORDER BY id DESC LIMIT 1").fetchone()
            dedup_count = int(dr["count"]) if dr else 0
            return slate.render_html(entries, today, alerts=alerts, dedup_count=dedup_count)
        finally:
            con.close()

    @app.get("/annotate", response_class=HTMLResponse)
    def annotate() -> str:
        # Single annotation surface (the xlsx bootstrap pack is retired): the judged-but-unrated
        # queue, same card + same 👎/👍/⭐ buttons, writing to the same label table via /feedback.
        con = _con(cfg)
        try:
            today = date.today().isoformat()
            items, total = slate.annotation_batch(cfg, con, today)
            return slate.render_annotate_html(items, today, total_pending=total)
        finally:
            con.close()

    @app.get("/eval", response_class=HTMLResponse)
    def eval_page() -> str:
        # Live match-quality vs Alina's REAL labels (the label table) — updates as she rates in
        # /annotate. No manual code re-point; the synthetic GOLD stays a CLI-only dev floor.
        from . import validation
        con = _con(cfg)
        try:
            return slate.render_eval_html(validation.eval_report(cfg, con))
        finally:
            con.close()

    @app.get("/gaps", response_class=HTMLResponse)
    def gaps_page() -> str:
        # Recurring skill gaps across the jobs she WANTS (👍/💅💸/applied) — what to add to the CV
        # or learn. Pure aggregation of stored llm_cov_reqs (no LLM call on the web path).
        from . import gaps
        con = _con(cfg)
        try:
            return slate.render_gaps_html(gaps.gap_report(cfg, con))
        finally:
            con.close()

    @app.get("/tasks", response_class=HTMLResponse)
    def tasks_page() -> str:
        # Comment-tracker: every review comment as a theme-tagged task with an open|accounted|wontfix
        # status — the "which feedback did we act on" audit (W1). Pure read; toggles via /task-status.
        from . import tasks as _tasks
        con = _con(cfg)
        try:
            return slate.render_tasks_html(_tasks.all_tasks(con), _tasks.summary(con))
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
        if fb.action not in ("bad", "good", "star", "applied"):
            return JSONResponse({"ok": False, "error": "bad action"}, status_code=400)
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
            else:
                score = FEEDBACK_TO_SCORE[fb.action]
                lab = {"score_1_5": score, "applied": 0, "source": "slate", "why_freetext": note,
                       "interview": 1 if score >= 4 else (0 if score <= 2 else None)}
            db.insert_label(con, fb.vacancy_id, lab)  # сам ставит status=LABELED
            # отметить фидбек в сегодняшнем slate_entry
            con.execute(
                "UPDATE slate_entry SET feedback = ? WHERE vacancy_id = ? AND slate_date = ?",
                (fb.action, fb.vacancy_id, date.today().isoformat()),
            )
            con.commit()
            return {"ok": True}
        finally:
            con.close()

    @app.post("/fetch")
    def fetch(german: bool = False, tertiary: bool = False):
        """UI-triggered full pipeline (scrape→…→slate), async + SINGLE-FLIGHT. Returns immediately;
        poll /fetch-status. 409 if a fetch is already running (prevents double model-load)."""
        if not _FETCH_LOCK.acquire(blocking=False):
            return JSONResponse({"ok": False, "error": "уже выполняется — подожди завершения"},
                                status_code=409)
        _FETCH_STATE.update(running=True, started=datetime.now(timezone.utc).isoformat(),
                            finished=None, summary=None, error=None)
        threading.Thread(target=_run_tick_background, args=(cfg,),
                         kwargs={"german": german, "tertiary": tertiary}, daemon=True).start()
        return {"ok": True, "message": "фетч запущен (займёт несколько минут — модели грузятся)"}

    @app.get("/fetch-status")
    def fetch_status():
        """Poll the background fetch: running flag + the latest funnel stage as progress."""
        con = _con(cfg)
        try:
            latest = con.execute(
                "SELECT stage, count FROM funnel_log ORDER BY id DESC LIMIT 1").fetchone()
        finally:
            con.close()
        code = latest["stage"] if latest else None
        if code == "memory_skip":
            stage_human, heavy, idx = "пропускаю тяжёлый этап (мало памяти)", False, -1
        else:
            info = _STAGE_BY_CODE.get(code, {})
            stage_human = info.get("human") or (code or None)
            heavy = bool(info.get("heavy"))
            idx = _STAGE_INDEX.get(code, -1)
        return {"running": _FETCH_STATE["running"], "started": _FETCH_STATE["started"],
                "finished": _FETCH_STATE["finished"], "summary": _FETCH_STATE["summary"],
                "error": _FETCH_STATE["error"],
                "stage": code,
                "stage_count": (latest["count"] if latest else None),
                "stage_human": stage_human, "heavy": heavy, "stage_index": idx,
                "n_stages": len(_FETCH_STAGES),
                "stages": _FETCH_STAGES}

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

    return app


def serve(cfg: dict):
    import uvicorn
    app = create_app(cfg)
    uvicorn.run(app, host="127.0.0.1", port=int(cfg.get("slate", {}).get("port", 8787)))
