"""Session-comment → tracked-task sidecar (frozen-contract-safe: a NEW table + helpers owned here,
db.py untouched, mirroring how vacancy_feature / candidate_profile own their own DDL).

Alina + the user leave free-text review comments while triaging vacancies (label.why_freetext, plus
hand notes in 15JuneSession.md). They were wired into the judge few-shot but never TRACKED — there
was no "which of these did the product actually act on?" audit. This module turns every comment into
a theme-tagged task with an open|accounted|wontfix status, so the `/tasks` page shows what's been
handled and what's still open. ``ingest_from_db`` is idempotent — re-run after each review session and
new comments appear automatically.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_comment_task (
    id             INTEGER PRIMARY KEY,
    comment_text   TEXT NOT NULL,
    vacancy_id     INTEGER,                 -- nullable: standalone prefs / session-md-only notes
    company        TEXT,
    title          TEXT,
    score_1_5      INTEGER,
    theme_tag      TEXT NOT NULL,           -- engineer-repellent | junior-floor | gap-too-big | …
    product_change TEXT,                    -- HOW the product accounts for it
    task_status    TEXT NOT NULL DEFAULT 'open',   -- open | accounted | wontfix
    resolved_note  TEXT,
    source         TEXT NOT NULL DEFAULT 'label',  -- label | session_md
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    UNIQUE (comment_text, vacancy_id)
);
"""

STATUSES = ("open", "accounted", "wontfix")

# theme → (how the product accounts for it, initial status). "accounted" themes are either shipped
# earlier (degree/hidden-de/duplicate) or shipped in THIS pass (engineer/junior/gap/slop) — the /tasks
# toggles let the user override any verdict. "other" stays open (needs a human decision).
THEME_PRODUCT: dict[str, tuple[str, str]] = {
    "engineer-repellent": ("Понижение role_kind + флаг «🛠 hands-on — не твоё» в slate (мягко, не drop)", "accounted"),
    "junior-floor":       ("Понижение intern/working-student + флаг «🎓 стажёр» в slate", "accounted"),
    "gap-too-big":        ("Явный флаг «❗ большой гэп» + показ eligibility/fit-разрыва на карточке", "accounted"),
    "jd-slop":            ("Перечитывание «глупого» описания моделью посильнее (deep_reasoning tier)", "accounted"),
    "degree-misread":     ("Eligibility: Master-Data ≠ degree guard + soft-lift при сильном фите", "accounted"),
    "degree-gap":         ("Eligibility: структурный 2-шаговый разрыв (Bachelor→PhD никогда не lift)", "accounted"),
    "hidden-de":          ("Условный немецкий → language_reality='de' (де-факто требование)", "accounted"),
    "duplicate":          ("Свёртка кросс-аккаунтных репостов (also_at)", "accounted"),
    "pref":               ("Персона судьи: lead↑, monthly/дедлайны↓ (тай-брейк, β=0 — не влияет на ранг)", "accounted"),
    "other":              ("—", "open"),
}

# Ordered keyword → theme. First match wins (priority order matters: slop/engineer before the
# generic degree/gap fallbacks). Deterministic + generalizable to FUTURE comments (no per-vid hardcode).
_THEME_RULES: list[tuple[str, str]] = [
    (r"слоп|непонятн|ничего не понятн|ai[- ]?слоп", "jd-slop"),
    (r"инженер|руками|hands[- ]?on|разработчик|кодить|code|developer", "engineer-repellent"),
    (r"стаж[её]р|интерн|working student|werkstudent|trainee|diplomand|diploma student", "junior-floor"),
    (r"master data", "degree-misread"),
    (r"\bphd\b|пхд", "degree-gap"),
    (r"master|degree|мастер|дегри|bachelor", "degree-misread"),
    (r"немецк|german|скрыт", "hidden-de"),
    (r"дублир|продублир|та же|same position", "duplicate"),
    (r"\blead\b|monthly|дедлайн|рутин", "pref"),
    (r"не подход|не проход|плохой м[эе]тч|большой г[эе]п|gap", "gap-too-big"),
]


def ensure_schema(con) -> None:
    con.execute(_SCHEMA)
    con.commit()


def theme_for(text: str) -> str:
    """Classify a comment into a theme (deterministic keyword match, first rule wins)."""
    low = (text or "").lower()
    for pattern, theme in _THEME_RULES:
        if re.search(pattern, low):
            return theme
    return "other"


def upsert_task(con, *, comment_text: str, vacancy_id: int | None, company: str | None = None,
                title: str | None = None, score_1_5: int | None = None,
                theme: str | None = None, source: str = "label") -> str:
    """Idempotent: insert a comment-task (theme + product_change + initial status auto-derived).

    Re-running NEVER clobbers a status the user has since changed: ON CONFLICT updates only the
    derived metadata (theme/product/company/title/score), preserving task_status + resolved_note.
    Returns the resolved theme.
    """
    ensure_schema(con)
    theme = theme or theme_for(comment_text)
    product, init_status = THEME_PRODUCT.get(theme, THEME_PRODUCT["other"])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # NULL-safe existence check (`IS`, not `=`): the table UNIQUE can't dedup vacancy_id=NULL rows
    # (SQLite treats NULLs as distinct), so a standalone-pref / session-md note would re-insert every
    # run. `vacancy_id IS ?` matches both a value and NULL → idempotent for every source.
    existing = con.execute(
        "SELECT id FROM session_comment_task WHERE comment_text = ? AND vacancy_id IS ?",
        (comment_text, vacancy_id)).fetchone()
    if existing:
        # preserve the user's manual status/resolved_note; refresh only derived metadata
        con.execute(
            "UPDATE session_comment_task SET company = ?, title = ?, score_1_5 = ?, theme_tag = ?, "
            "product_change = ?, updated_at = ? WHERE id = ?",
            (company, title, score_1_5, theme, product, now, existing["id"]))
    else:
        con.execute(
            """INSERT INTO session_comment_task
                   (comment_text, vacancy_id, company, title, score_1_5, theme_tag, product_change,
                    task_status, resolved_note, source, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (comment_text, vacancy_id, company, title, score_1_5, theme, product,
             init_status, None, source, now, now))
    con.commit()
    return theme


def set_status(con, task_id: int, status: str, *, resolved_note: str | None = None) -> bool:
    """Flip a task's status (open|accounted|wontfix). Returns False on unknown status / missing id."""
    if status not in STATUSES:
        return False
    ensure_schema(con)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cur = con.execute(
        "UPDATE session_comment_task SET task_status = ?, "
        "resolved_note = COALESCE(?, resolved_note), updated_at = ? WHERE id = ?",
        (status, resolved_note, now, task_id))
    con.commit()
    return cur.rowcount > 0


def all_tasks(con) -> list[dict]:
    """All comment-tasks, grouped-friendly order: by theme then newest comment first."""
    ensure_schema(con)
    rows = con.execute(
        "SELECT id, comment_text, vacancy_id, company, title, score_1_5, theme_tag, "
        "product_change, task_status, resolved_note, source FROM session_comment_task "
        "ORDER BY theme_tag, id DESC").fetchall()
    return [dict(r) for r in rows]


def summary(con) -> dict[str, int]:
    """Counts per status for the page header (Goal-Gradient)."""
    ensure_schema(con)
    rows = con.execute(
        "SELECT task_status, COUNT(*) c FROM session_comment_task GROUP BY task_status").fetchall()
    out = {s: 0 for s in STATUSES}
    for r in rows:
        out[r["task_status"]] = r["c"]
    out["total"] = sum(out[s] for s in STATUSES)
    return out


def ingest_from_db(con, *, extra: list[dict] | None = None) -> dict[str, int]:
    """Ingest EVERY comment into the tracker (idempotent). Sources:
      1. label.why_freetext (slate + backfilled session notes) joined to the vacancy,
      2. ``extra`` — session-md-only notes with no label row (standalone prefs / unmapped jobs).
    Returns {ingested, themes...}.
    """
    ensure_schema(con)
    out: dict[str, int] = {"ingested": 0}
    rows = con.execute(
        "SELECT l.vacancy_id, l.score_1_5, l.why_freetext, v.company, v.title "
        "FROM label l JOIN vacancy v ON v.id = l.vacancy_id "
        "WHERE l.why_freetext IS NOT NULL AND TRIM(l.why_freetext) != ''").fetchall()
    for r in rows:
        theme = upsert_task(con, comment_text=r["why_freetext"].strip(),
                            vacancy_id=int(r["vacancy_id"]), company=r["company"],
                            title=r["title"], score_1_5=r["score_1_5"], source="label")
        out["ingested"] += 1
        out[theme] = out.get(theme, 0) + 1
    for item in (extra or []):
        theme = upsert_task(con, comment_text=item["comment_text"],
                            vacancy_id=item.get("vacancy_id"), company=item.get("company"),
                            title=item.get("title"), score_1_5=item.get("score_1_5"),
                            theme=item.get("theme"), source="session_md")
        out["ingested"] += 1
        out[theme] = out.get(theme, 0) + 1
    return out
