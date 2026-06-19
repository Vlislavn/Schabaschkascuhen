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


def _seed_scored_src(con, url, *, source, refnr=None, last_seen, score=5):
    vid = db.upsert_vacancy(con, {"source": source, "url": url, "refnr": refnr, "title": "T",
                                  "company": "C", "city": "Frankfurt", "description": "x" * 500})
    con.execute("UPDATE vacancy SET status = ?, last_seen = ? WHERE id = ?",
                (Status.SCORED.value, last_seen, vid))
    db.insert_judge_score(con, vid, {"score": score, "model": "qwen3:8b", "rubric_version": "v1",
                                     "explanation": "e"})
    con.commit()
    return vid


def test_verify_liveness_expires_confirmed_gone(con, cfg, monkeypatch):
    """Stale SCORED AA + Indeed cards: a definitive-gone verdict EXPIRES; alive/unknown stay."""
    from schabasch.sources import arbeitsagentur, indeed
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)   # verify_liveness throttles 0.4s/probe
    old = "2020-01-01T00:00:00+00:00"
    aa_gone = _seed_scored_src(con, "aa/gone", source="arbeitsagentur", refnr="R_GONE", last_seen=old)
    aa_alive = _seed_scored_src(con, "aa/alive", source="arbeitsagentur", refnr="R_ALIVE", last_seen=old)
    in_gone = _seed_scored_src(con, "https://de.indeed.com/viewjob?jk=JKGONE", source="indeed", last_seen=old)
    in_unk = _seed_scored_src(con, "https://de.indeed.com/viewjob?jk=JKUNK", source="indeed", last_seen=old)
    monkeypatch.setattr(arbeitsagentur, "check_open",
                        lambda refnr: {"R_GONE": False, "R_ALIVE": True}[refnr])
    monkeypatch.setattr(indeed, "check_open", lambda jk: {"JKGONE": False, "JKUNK": None}[jk])

    res = pipeline.verify_liveness(cfg, con)

    def st(v):
        return con.execute("SELECT status FROM vacancy WHERE id=?", (v,)).fetchone()[0]
    assert st(aa_gone) == Status.EXPIRED.value
    assert st(in_gone) == Status.EXPIRED.value          # the user's «expired Indeed → убрать»
    assert st(aa_alive) == Status.SCORED.value          # alive untouched
    assert st(in_unk) == Status.SCORED.value            # None never false-closes
    assert res["checked"] == 4 and res["expired"] == 2 and res["unknown"] == 1


def test_verify_liveness_skips_fresh_and_unverifiable(con, cfg, monkeypatch):
    """A still-fresh card (recent last_seen) and LinkedIn (no reliable per-URL check) aren't probed."""
    from schabasch.sources import arbeitsagentur
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    old, fresh = "2020-01-01T00:00:00+00:00", "2999-01-01T00:00:00+00:00"
    _seed_scored_src(con, "aa/fresh", source="arbeitsagentur", refnr="R_FRESH", last_seen=fresh)
    _seed_scored_src(con, "li/old", source="linkedin", last_seen=old)
    called: list = []
    monkeypatch.setattr(arbeitsagentur, "check_open", lambda refnr: called.append(refnr) or True)
    res = pipeline.verify_liveness(cfg, con)
    assert res["checked"] == 0 and called == []   # fresh skipped (cutoff), linkedin excluded (source)
