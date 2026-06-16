"""CLI (typer): точка входа `python -m schabasch.cli ...` и console_script `schabasch`."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from . import candidate, config, db, dedup, features as _features, geo, hardfilters, judge, normalize, pipeline, slate, triage as _triage
from .sources import arbeitsagentur, jobspy_source

app = typer.Typer(add_completion=False, help="Schabaschkascuhen — личный пайплайн поиска работы")


def _ctx():
    cfg = config.load()
    con = db.connect(cfg["paths"]["db"])
    return cfg, con


def _echo(obj) -> None:
    typer.echo(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


@app.command("candidate")
def candidate_cmd(
    description: str = typer.Argument(None, help="Freeform description of the candidate"),
    cv_path: str = typer.Option(None, "--cv-path", help="Path to PDF/text CV"),
):
    """Extract structured CandidateProfile (skills, aspects) from description or CV file."""
    if not description and not cv_path:
        typer.echo("Provide a description argument or --cv-path", err=True)
        raise typer.Exit(1)
    cfg, con = _ctx()
    result = candidate.extract_candidate(cfg, con, description=description, cv_path=cv_path)
    _echo({
        "seniority": result.get("seniority"),
        "years_experience": result.get("years_experience"),
        "skills_count": len(result.get("skills") or []),
        "skills_preview": (result.get("skills") or [])[:8],
        "target_roles": result.get("target_roles"),
        "domains": result.get("domains"),
        "doc_hash": result.get("doc_hash"),
        "full_doc_chars": len((result.get("aspect_texts") or {}).get("full_doc", "")),
    })


@app.command("import-spike")
def import_spike():
    """Разовый импорт спайк-пула (indeed/linkedin/arbeitsagentur)."""
    cfg, con = _ctx()
    _echo(pipeline.import_spike_data(cfg, con))


@app.command()
def tick(german: bool = typer.Option(False, "--german"),
         budget: int = typer.Option(None, "--budget"),
         tertiary: bool = typer.Option(False, "--tertiary")):
    """Полный ночной прогон (canary→scrape→details→geo→hard→normalize→judge→slate)."""
    cfg, con = _ctx()
    pipeline.nightly_tick(cfg, con, german_queries=german, budget=budget, tertiary=tertiary)


@app.command()
def scrape(german: bool = typer.Option(False, "--german")):
    """Скрейп Indeed/LinkedIn + поиск Arbeitsagentur."""
    cfg, con = _ctx()
    q = cfg["search"].get("queries_de" if german else "queries_en")
    _echo({"jobspy": jobspy_source.scrape(cfg, con, queries=q),
           "arbeitsagentur": arbeitsagentur.search(cfg, con, queries=q)})


@app.command()
def details():
    """Добыть полные описания Arbeitsagentur (v3 jobdetails)."""
    cfg, con = _ctx()
    _echo({"described": arbeitsagentur.fetch_details(cfg, con)})


@app.command()
def tertiary():
    """Третичные фетчеры (Should): Arbeitnow API + GermanTechJobs RSS, регион-фильтр."""
    cfg, con = _ctx()
    from .sources import tertiary as t
    _echo({"arbeitnow": t.fetch_arbeitnow(cfg, con),
           "germantechjobs": t.fetch_germantechjobs(cfg, con)})


@app.command()
def prefilter():
    """Геокодированный грубый гео-фильтр."""
    cfg, con = _ctx()
    _echo(geo.prefilter(cfg, con))


@app.command("hard")
def hard():
    """Жёсткие фильтры (скрытый немецкий, Zeitarbeit) до LLM."""
    cfg, con = _ctx()
    _echo(hardfilters.apply_hard_filters(cfg, con))


@app.command("dedup")
def dedup_cmd(threshold: int = typer.Option(88, "--threshold")):
    """Fuzzy кросс-борд дедупликация (RapidFuzz, logged-not-merged)."""
    cfg, con = _ctx()
    _echo(dedup.dedup_fuzzy(cfg, con, threshold=threshold))


@app.command("normalize")
def normalize_(budget: int = typer.Option(None, "--budget")):
    """Нормализация DESCRIBED → карточки (qwen3:8b)."""
    cfg, con = _ctx()
    _echo(normalize.normalize_pending(cfg, con, budget=budget))


@app.command("judge")
def judge_():
    """Оценка карточек судьёй."""
    cfg, con = _ctx()
    _echo(judge.judge_pending(cfg, con))


@app.command("slate")
def slate_cmd(date: str = typer.Option(None, "--date")):
    """Собрать slate (8 exploit + 2 explore) и напечатать карточки."""
    from datetime import date as _date
    cfg, con = _ctx()
    d = date or _date.today().isoformat()
    s = slate.build_slate(cfg, con, d)
    _echo({"slate_date": d, "n": len(s),
           "cards": [{"rank": c["rank"], "slot": c["slot_type"], "score": c["score"],
                      "title": c["title"], "company": c["company"], "why": c["why"]} for c in s]})


@app.command()
def render(date: str = typer.Option(None, "--date"), out: str = typer.Option(None, "--out")):
    """Отрендерить HTML сегодняшнего slate в файл (по умолчанию data/slates/<date>.html)."""
    from datetime import date as _date
    cfg, con = _ctx()
    d = date or _date.today().isoformat()
    s = slate.build_slate(cfg, con, d)
    html = slate.render_html(s, d)
    out_path = Path(out) if out else Path(cfg["paths"]["slate_dir"]) / f"{d}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    typer.echo(str(out_path))


@app.command()
def canary():
    """Канарейки источников (min-row assertion)."""
    cfg, con = _ctx()
    _echo(jobspy_source.canary(cfg, con))


@app.command()
def serve():
    """Поднять локальную страницу фидбека (uvicorn)."""
    cfg, _ = _ctx()
    from . import feedback_app
    feedback_app.serve(cfg)


@app.command("rerank")
def rerank_cmd(top_k: int = typer.Option(None, "--top-k", help="Max SCORED vacancies to rerank")):
    """Cross-encoder re-rank top-K SCORED vacancies (bge-reranker-v2-m3, requires FlagEmbedding)."""
    cfg, con = _ctx()
    _echo(_features.rerank_scored(cfg, con, top_k=top_k))


@app.command("export-golden")
def export_golden(out: str = typer.Option(None, "--out")):
    """Экспорт golden dataset в CSV."""
    cfg, con = _ctx()
    out_path = Path(out) if out else Path(cfg["paths"]["golden_csv"])
    n = db.export_golden_csv(con, out_path)
    typer.echo(f"{n} rows -> {out_path}")


@app.command()
def funnel():
    """Показать последний лог воронки + канарейки + статусы."""
    cfg, con = _ctx()
    rows = con.execute(
        "SELECT run_at, stage, source, count, detail FROM funnel_log ORDER BY id DESC LIMIT 25"
    ).fetchall()
    canaries = con.execute(
        "SELECT run_at, source, verdict, rows FROM canary_log ORDER BY id DESC LIMIT 8"
    ).fetchall()
    status = dict(con.execute("SELECT status, COUNT(*) FROM vacancy GROUP BY status").fetchall())
    _echo({"funnel": [dict(r) for r in rows], "canaries": [dict(r) for r in canaries],
           "status_counts": status})


@app.command()
def cv(folds: int = typer.Option(5, "--folds"), runs: int = typer.Option(3, "--runs")):
    """5-fold CV agreement судья↔метки (gate ≥75% и > majority+10pp). См. calibration.py."""
    from . import calibration
    cfg, con = _ctx()
    _echo(calibration.cross_validate(cfg, con, folds=folds, runs=runs))


@app.command()
def gaps():
    """Какие навыки регулярно отсутствуют под ЖЕЛАННЫЕ вакансии (👍/💅💸/applied) — отчёт пробелов."""
    from . import gaps as _gaps
    cfg, con = _ctx()
    rep = _gaps.gap_report(cfg, con)
    _echo({"n_wanted": rep["n_wanted"], "n_jobs_with_reqs": rep["n_jobs_with_reqs"],
           "reliable": rep["reliable"], "top_gaps": rep["rows"][:15]})


@app.command("features")
def features_cmd(limit: int = typer.Option(None, "--limit", help="Max vacancies to embed")):
    """Embed DESCRIBED vacancies with bge-m3 and compute aspect-pair features."""
    cfg, con = _ctx()
    _echo(_features.extract_features(cfg, con, limit=limit))


@app.command("triage")
def triage_cmd(limit: int = typer.Option(None, "--limit", help="Max vacancies to score")):
    """Score DESCRIBED vacancies → bucket (must/should/could/drop) via ML gate or match_score."""
    cfg, con = _ctx()
    _echo(_triage.triage_pending(cfg, con, limit=limit))


@app.command("triage-train")
def triage_train_cmd(force: bool = typer.Option(False, "--force", help="Retrain even if labels unchanged")):
    """Train LightGBM triage gate on labeled+featured vacancies."""
    cfg, con = _ctx()
    _echo(_triage.train(cfg, con, force=force))


@app.command("triage-eval")
def triage_eval_cmd():
    """Temporal holdout evaluation of the triage gate (train on oldest 80%, test on newest 20%)."""
    cfg, con = _ctx()
    _echo(_triage.evaluate(cfg, con))


@app.command("discover")
def discover_cmd(max_results: int = typer.Option(20, "--max-results", help="Max postings to discover")):
    """Run ReAct agent to find jobs beyond fixed boards (requires kl_agent_builder + ollama)."""
    try:
        from .sources import agent_discovery
    except ImportError as e:
        typer.echo(f"agent_discovery not available: {e}", err=True)
        raise typer.Exit(1)
    cfg, con = _ctx()
    _echo(agent_discovery.scrape(cfg, con, max_results=max_results))


@app.command("investigate")
def investigate_cmd(
    date: str = typer.Option(None, "--date"),
    top_n: int = typer.Option(5, "--top-n", help="Number of top vacancies to investigate"),
):
    """Deep-dive top-N SCORED vacancies via ReAct agent (requires kl_agent_builder + ollama)."""
    from datetime import date as _date
    try:
        from . import investigate
    except ImportError as e:
        typer.echo(f"investigate not available: {e}", err=True)
        raise typer.Exit(1)
    cfg, con = _ctx()
    d = date or _date.today().isoformat()
    _echo(investigate.investigate_top(cfg, con, slate_date=d, top_n=top_n))


if __name__ == "__main__":
    app()
