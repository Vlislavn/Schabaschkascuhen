"""expire_stale: pre-slate stale vacancies → EXPIRED; slate-relevant states untouched."""
from __future__ import annotations

from schabasch import db, pipeline
from schabasch.models import Status


def _seed(con, url, status, last_seen):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": url, "title": "T",
                                  "company": "C", "city": "Frankfurt", "description": "x" * 500})
    con.execute("UPDATE vacancy SET status = ?, last_seen = ? WHERE id = ?",
                (status.value, last_seen, vid))
    con.commit()
    return vid


def test_expire_stale_only_pre_slate(con, cfg):
    old = "2020-01-01T00:00:00+00:00"   # far in the past
    fresh = "2999-01-01T00:00:00+00:00"  # far future → never stale
    v_old_desc = _seed(con, "u/old_desc", Status.DESCRIBED, old)
    v_old_scored = _seed(con, "u/old_scored", Status.SCORED, old)   # protected
    v_old_slated = _seed(con, "u/old_slated", Status.SLATED, old)   # protected
    v_fresh_desc = _seed(con, "u/fresh_desc", Status.DESCRIBED, fresh)

    res = pipeline.expire_stale(cfg, con, days=30)
    assert res["expired"] == 1   # only the stale DESCRIBED one

    def status(vid):
        return con.execute("SELECT status FROM vacancy WHERE id=?", (vid,)).fetchone()[0]

    assert status(v_old_desc) == Status.EXPIRED.value
    assert status(v_old_scored) == Status.SCORED.value     # untouched
    assert status(v_old_slated) == Status.SLATED.value     # untouched
    assert status(v_fresh_desc) == Status.DESCRIBED.value  # too fresh
