"""Role-fit feedback — the second axis of "good domain, WRONG role" (Delphi panel, 2026-06-16).

A 1–5 rating conflates DOMAIN interest (score_1_5) with ROLE fit. This sidecar captures the role axis
separately: on a *positive* rating of an ambiguous card (role_kind ∈ {hands_on_engineer, junior, lead})
the user may tap ✅ role fits / 🙅 wrong role. The vote feeds a LEARNED per-kind multiplier (P2) so the
ranker stops surfacing aspiration-but-wrong-role jobs — WITHOUT touching the honest domain score.

Frozen-contract-safe: a NEW sidecar table owned here (additive; `label`/`db`/`models` untouched).
`source` firewalls debug feedback ('debug') out of the golden learner/eval ('slate').
"""
from __future__ import annotations

from . import db

AMBIGUOUS_KINDS = ("hands_on_engineer", "junior", "lead")   # non-neutral → the role-fit row applies


def ensure_schema(con) -> None:
    con.execute(
        """CREATE TABLE IF NOT EXISTS label_role (
            id          INTEGER PRIMARY KEY,
            vacancy_id  INTEGER NOT NULL REFERENCES vacancy(id),
            role_kind   TEXT NOT NULL,        -- snapshot of role_kind.classify at vote time
            fits        INTEGER NOT NULL,     -- 1 = ✅ role fits | 0 = 🙅 wrong role
            source      TEXT NOT NULL,        -- 'slate' (golden) | 'debug' (firewalled out of gold)
            created_at  TEXT NOT NULL,
            UNIQUE (vacancy_id, source)
        )""")
    con.commit()


def record(con, vacancy_id: int, role_kind: str, fits: bool, *, source: str = "slate") -> None:
    """Upsert a role-fit vote. (vacancy_id, source) is unique so a re-vote overwrites."""
    ensure_schema(con)
    con.execute(
        """INSERT INTO label_role (vacancy_id, role_kind, fits, source, created_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT (vacancy_id, source) DO UPDATE SET
               role_kind = excluded.role_kind, fits = excluded.fits, created_at = excluded.created_at""",
        (vacancy_id, role_kind, int(bool(fits)), source, db.now_iso()))
    con.commit()


def fit_counts(con, *, source: str = "slate") -> dict[str, tuple[int, int]]:
    """{role_kind: (n_total, n_fits)} from golden role votes — input to the learned multiplier (P2)."""
    ensure_schema(con)
    out: dict[str, tuple[int, int]] = {}
    for r in con.execute(
            "SELECT role_kind, COUNT(*) n, COALESCE(SUM(fits),0) f FROM label_role "
            "WHERE source = ? GROUP BY role_kind", (source,)):
        out[r[0]] = (int(r[1]), int(r[2]))
    return out


def veto_map(con, *, source: str = "slate") -> dict[int, int]:
    """{vacancy_id: fits} — the role mask for the eval's veto-aware gold (P1)."""
    ensure_schema(con)
    return {int(r[0]): int(r[1]) for r in con.execute(
        "SELECT vacancy_id, fits FROM label_role WHERE source = ?", (source,))}
