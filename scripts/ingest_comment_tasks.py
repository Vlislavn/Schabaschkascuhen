"""Ingest ALL of the user's review comments into the session_comment_task tracker (W1).

Idempotent. Pulls every label.why_freetext (slate + the 15-June backfill) joined to its vacancy, plus
the session-md-only notes that never became a label row (two standalone preferences + two jobs whose
comments live only in 15JuneSession.md: Peraton "плохой мэтч", Schrödinger "PhD"). Each comment is
theme-tagged and gets an initial open|accounted status (see schabasch/tasks.THEME_PRODUCT); re-run
after every review session and new comments appear on /tasks automatically.

Run:  .venv/bin/python -m scripts.ingest_comment_tasks        (from the repo root)
"""
from __future__ import annotations

from schabasch import config, db, tasks

# Comments that exist ONLY in schabasch/15JuneSession.md (no label.why_freetext row): two standalone
# preferences (affect ranking globally, not one job) + two jobs the user commented on without a slate label.
SESSION_MD_EXTRA: list[dict] = [
    {"comment_text": "Люблю когда в вакансиях lead/principal/ownership — тянут вверх.",
     "theme": "pref"},
    {"comment_text": "Не люблю monthly / жёсткие дедлайны / рутину — это стресс, минус.",
     "theme": "pref"},
    {"comment_text": "Peraton (Cyber Threat Analyst): «не знаю таких слов, правда плохой мэтч "
                     "по резюме» — нулевое совпадение навыков.",
     "company": "Peraton", "title": "Senior Cyber Threat Analyst", "theme": "gap-too-big"},
    {"comment_text": "Schrödinger (Materials Science Applications Scientist): «не подходит "
                     "из-за PhD» — требуется PhD, структурный разрыв.",
     "company": "Schrödinger", "title": "Materials Science Applications Scientist",
     "theme": "degree-gap"},
]


def main() -> None:
    cfg = config.load()
    con = db.connect(cfg["paths"]["db"])
    print("Ingesting review comments into session_comment_task …")
    res = tasks.ingest_from_db(con, extra=SESSION_MD_EXTRA)
    s = tasks.summary(con)
    by_theme = {k: v for k, v in res.items() if k != "ingested"}
    print(f"  ingested: {res['ingested']} comments")
    print(f"  by theme: {by_theme}")
    print(f"  status:   {s['accounted']} accounted · {s['open']} open · {s['wontfix']} wontfix "
          f"(total {s['total']})")


if __name__ == "__main__":
    main()
