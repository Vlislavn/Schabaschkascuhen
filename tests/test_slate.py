"""Slate: 8 exploit + 2 explore, ≤3 на компанию, детерминизм по seed=date."""
from __future__ import annotations

from schabasch import slate
from schabasch.models import Status
from tests.conftest import seed_scored


def _seed_many(con, n=20):
    # n вакансий, оценки убывают, компании размазаны (по 1, чтобы cap не мешал по умолчанию)
    for i in range(n):
        seed_scored(con, f"u/{i}", score=5 - (i % 5), company=f"Co{i}")


def test_slate_size_8_plus_2(cfg, con):
    _seed_many(con, 20)
    s = slate.build_slate(cfg, con, "2026-06-13")
    assert len([x for x in s if x["slot_type"] == "exploit"]) == 8
    assert len([x for x in s if x["slot_type"] == "explore"]) == 2
    assert len(s) == 10


def test_effective_score_zero_cov_floor():
    """0%-skill floor: a job with llm_cov==0 is multiplied down when zero_cov_mult<1; cov=None is never floored."""
    base = {"fit_score": 0.8, "llm_cov": 0.0, "score": 5, "elig_score": 1.0, "title": "Analyst", "summary": ""}
    assert slate.effective_score(base, {"slate": {"zero_cov_mult": 1.0}}) > 0      # off
    assert slate.effective_score(base, {"slate": {"zero_cov_mult": 0.0}}) == 0.0   # floored
    nocov = dict(base, llm_cov=None)
    assert slate.effective_score(nocov, {"slate": {"zero_cov_mult": 0.0}}) > 0     # uncomputed → not floored


def test_slate_card_carries_role_kind(cfg, con):
    """Cards expose role_kind so the bot can ask the role-fit follow-up on ambiguous roles."""
    eng = seed_scored(con, "u/eng", score=4, company="A", title="Senior Software Engineer")
    s = slate.build_slate(cfg, con, "2026-06-13", rebuild=True)
    card = next(c for c in s if c["vacancy_id"] == eng)
    assert card["role_kind"] == "hands_on_engineer"


def test_slate_excludes_already_rated(cfg, con):
    """De-dup: a vacancy the user already RATED is never re-served on the daily slate; a fresh
    unrated one is. (rebuild=True so the change applies to an existing slate_date.)"""
    from schabasch import db
    rated = seed_scored(con, "u/rated", score=5, company="A")
    fresh = seed_scored(con, "u/fresh", score=5, company="B")
    db.insert_label(con, rated, {"score_1_5": 4, "applied": 0, "source": "slate"})
    ids = [c["vacancy_id"] for c in slate.build_slate(cfg, con, "2026-06-13", rebuild=True)]
    assert fresh in ids and rated not in ids


def test_company_cap_max_3(cfg, con):
    # 10 вакансий одной компании с высокой оценкой → в exploit максимум 3
    for i in range(10):
        seed_scored(con, f"big/{i}", score=5, company="MonopolyCorp")
    # плюс прочие компании, чтобы добить exploit
    for i in range(10):
        seed_scored(con, f"oth/{i}", score=4, company=f"Other{i}")
    s = slate.build_slate(cfg, con, "2026-06-13")
    exploit = [x for x in s if x["slot_type"] == "exploit"]
    mono = [x for x in exploit if x["company"] == "MonopolyCorp"]
    assert len(mono) <= 3


def test_determinism_same_seed(cfg, con):
    _seed_many(con, 20)
    s1 = slate.build_slate(cfg, con, "2026-06-13")
    ids1 = [x["vacancy_id"] for x in s1]
    # пересборка той же даты → идемпотентно (из slate_entry), тот же порядок
    s2 = slate.build_slate(cfg, con, "2026-06-13")
    assert [x["vacancy_id"] for x in s2] == ids1


def test_explore_seed_reproducible(cfg, con):
    # два отдельных in-memory: тот же seed=date даёт ту же explore-выборку
    import sqlite3
    from schabasch import db

    def build(date):
        c = db.connect(":memory:")
        _seed_many(c, 20)
        out = slate.build_slate(cfg, c, date)
        ids = [x["vacancy_id"] for x in out if x["slot_type"] == "explore"]
        c.close()
        return ids

    assert build("2026-06-13") == build("2026-06-13")


def test_thin_day_no_padding(cfg, con):
    # всего 3 scored → slate из 3, мусором не добивается
    for i in range(3):
        seed_scored(con, f"few/{i}", score=4, company=f"C{i}")
    s = slate.build_slate(cfg, con, "2026-06-13")
    assert len(s) == 3


def test_slated_status_set(cfg, con):
    _seed_many(con, 12)
    s = slate.build_slate(cfg, con, "2026-06-13")
    vid = s[0]["vacancy_id"]
    row = con.execute("SELECT status FROM vacancy WHERE id=?", (vid,)).fetchone()
    assert row["status"] == Status.SLATED.value


def test_render_html_contains_buttons(cfg, con):
    _seed_many(con, 12)
    s = slate.build_slate(cfg, con, "2026-06-13")
    html = slate.render_html(s, "2026-06-13")
    assert "applied" in html and "/feedback" in html
    assert "💻🐀" in html and "👸✨🧚" in html  # themed scale: office mouse → шабашка


# ---------------------------------------------------------------------------
# Posting date (Workstream C) + investigator company-research surfacing (B)
# ---------------------------------------------------------------------------

def _seed_investigation(con, vid, enrichment, verdict="ok"):
    import json
    con.execute(
        """CREATE TABLE IF NOT EXISTS investigation (
               vacancy_id INTEGER PRIMARY KEY, enrichment_json TEXT NOT NULL,
               verdict TEXT, investigated_at TEXT NOT NULL)""")
    con.execute("INSERT OR REPLACE INTO investigation VALUES (?,?,?,?)",
                (vid, json.dumps(enrichment), verdict, "2026-06-14T00:00:00+00:00"))
    con.commit()


def test_slate_card_shows_posting_date(cfg, con):
    vid = seed_scored(con, "u/dated", score=5, company="Co")
    con.execute("UPDATE vacancy SET date_posted='2020-01-01' WHERE id=?", (vid,))
    con.commit()
    s = slate.build_slate(cfg, con, "2026-06-13")
    assert any(x["vacancy_id"] == vid and x.get("date_posted") == "2020-01-01" for x in s)
    html = slate.render_html(s, "2026-06-13")
    assert "freshbadge" in html  # posting age surfaced via the freshness badge by the title


def test_slate_card_shows_company_research(cfg, con):
    vid = seed_scored(con, "u/inv", score=5, company="Co")
    _seed_investigation(con, vid, {
        "company_size": "large", "salary_eur_min": 70000, "salary_eur_max": 90000,
        "english_team_signal": True, "still_open": True, "notes": "Great role at Co."}, verdict="ok")
    s = slate.build_slate(cfg, con, "2026-06-13")
    item = next(x for x in s if x["vacancy_id"] == vid)
    assert item.get("investigation", {}).get("company_size") == "large"
    html = slate.render_html(s, "2026-06-13")
    assert "🔎" in html and "€70" in html
    assert "English-speaking team" in html and "Great role at Co." in html


def test_slate_warns_closed_but_does_not_drop(cfg, con):
    """A DETERMINISTICALLY-closed listing (still_open=False) must NOT drop the job — it stays with
    a '⚠ вакансия закрыта' note (down-rank, not hide). Closure is now the HTTP/AA check, not a model
    guess, so the message is truthful."""
    v_ok = seed_scored(con, "u/open", score=5, company="A")
    v_closed = seed_scored(con, "u/closed", score=5, company="B")
    _seed_investigation(con, v_closed, {"still_open": False, "notes": "removed from board"}, verdict="ok")
    s = slate.build_slate(cfg, con, "2026-06-13")
    ids = [x["vacancy_id"] for x in s]
    assert v_ok in ids and v_closed in ids          # closed is NOT dropped (recall > one-click check)
    html = slate.render_html(s, "2026-06-13")
    assert "listing closed" in html                 # truthfully flagged


def test_slate_unverified_listing_is_calm_not_alarm(cfg, con):
    """A listing the check COULDN'T verify (still_open=None, blocked/timeout) reads as calm info,
    never a false 'closed' — directly the user's complaint ('no active listing' for a blocked link)."""
    vid = seed_scored(con, "u/blocked", score=5, company="C")
    _seed_investigation(con, vid, {"still_open": None, "notes": "link blocked"}, verdict="ok")
    html = slate.render_html(slate.build_slate(cfg, con, "2026-06-13"), "2026-06-13")
    assert "listing not checked" in html and "listing closed" not in html


def test_slate_no_investigation_renders_clean(cfg, con):
    """A card with no investigation must render without a verified block (no crash, no 🔎)."""
    seed_scored(con, "u/plain", score=4, company="Co")
    s = slate.build_slate(cfg, con, "2026-06-13")
    html = slate.render_html(s, "2026-06-13")
    # scope to the CARD structure, not the page (the header emoji legend now lists 🔎 deliberately)
    assert 'class="verified"' not in html  # nothing investigated → no verified line


# ---------------------------------------------------------------------------
# WS1c: cross-account repost collapse (455 Laveer ≡ 906 Westinghouse)
# ---------------------------------------------------------------------------

def test_cross_account_repost_collapsed(cfg, con):
    """Same SPECIFIC role+city posted under two recruiter names collapses to ONE card; the other
    company rides along as `also_at`. (Pipeline dedup blocks by company → never compares these.)"""
    t = "GRO Data Analytics and Reporting"
    a = seed_scored(con, "u/laveer", score=5, company="Laveer Engineering", title=t, city="Mannheim")
    b = seed_scored(con, "u/westh", score=5, company="Westinghouse Electric", title=t, city="Mannheim")
    s = slate.build_slate(cfg, con, "2026-06-13")
    ids = [x["vacancy_id"] for x in s]
    assert (a in ids) ^ (b in ids)          # exactly one of the pair shown
    kept = next(x for x in s if x["vacancy_id"] in (a, b))
    assert kept.get("also_at")              # the other employer attached
    html = slate.render_html(s, "2026-06-13")
    assert "also:" in html


def test_generic_title_not_collapsed_across_companies(cfg, con):
    """Guard: two genuinely-different jobs sharing a GENERIC 2-word title at different employers must
    NOT be merged (only specific ≥3-token titles or same-company dups collapse)."""
    a = seed_scored(con, "u/da1", score=5, company="SAP", title="Data Analyst", city="Frankfurt")
    b = seed_scored(con, "u/da2", score=5, company="Allianz", title="Data Analyst", city="Frankfurt")
    s = slate.build_slate(cfg, con, "2026-06-13")
    ids = [x["vacancy_id"] for x in s]
    assert a in ids and b in ids            # both kept (different real jobs)


# ---------------------------------------------------------------------------
# WS4: far-but-in-Germany shown + marked + preferred for explore
# ---------------------------------------------------------------------------

def test_far_job_marked_not_dropped(cfg, con):
    seed_scored(con, "u/near", score=5, company="NearCo", city="Frankfurt")
    far = seed_scored(con, "u/far", score=5, company="FarCo", city="München")
    s = slate.build_slate(cfg, con, "2026-06-13")
    ids = [x["vacancy_id"] for x in s]
    assert far in ids                                   # shown, not dropped
    far_item = next(x for x in s if x["vacancy_id"] == far)
    assert far_item["far"] is True and far_item["dist_km"] is not None
    html = slate.render_html(s, "2026-06-13")
    assert "📍 far" in html


def test_far_preferred_for_explore(cfg, con):
    # 8 high-score near jobs fill exploit; a lower-score far job should win an explore slot.
    for i in range(8):
        seed_scored(con, f"near/{i}", score=5, company=f"N{i}", city="Frankfurt")
    far = seed_scored(con, "u/far", score=2, company="FarCo", city="Berlin")
    seed_scored(con, "u/nearlow", score=2, company="NearLow", city="Heidelberg")
    s = slate.build_slate(cfg, con, "2026-06-13")
    explore = [x for x in s if x["slot_type"] == "explore"]
    assert any(x["vacancy_id"] == far for x in explore)   # far routed to explore


# ---------------------------------------------------------------------------
# WS3: filter chips + WS2: free-text note textarea render
# ---------------------------------------------------------------------------

def test_filter_chips_and_data_attrs_render(cfg, con):
    _seed_many(con, 12)
    html = slate.render_html(slate.build_slate(cfg, con, "2026-06-13"), "2026-06-13")
    assert 'class="chips"' in html and "promising" in html and "📍 far" in html
    assert 'id="cards"' in html and "data-days=" in html and "data-fit=" in html
    assert "applyFilter" in html   # the client-side filter is wired


def test_note_textarea_renders(cfg, con):
    vid = seed_scored(con, "u/note", score=4, company="Co")
    html = slate.render_html(slate.build_slate(cfg, con, "2026-06-13"), "2026-06-13")
    assert 'class="note"' in html and f'note-{vid}' in html


def test_quality_floor_keeps_low_effective_out_of_exploit(cfg, con):
    """W2: an exploit slot must clear slate.quality_floor on effective; a below-floor job is NOT shown
    as exploit (still explore-eligible) — the 'много неподходящих' fix (no junk padding)."""
    import json
    from schabasch import features
    cfg = dict(cfg); cfg["slate"] = dict(cfg["slate"]); cfg["slate"]["quality_floor"] = 0.45
    cfg["features"] = dict(cfg["features"]); cfg["features"]["fit_weights"] = {"hyre": 0.7, "sparse": 0.3}
    features._ensure_schema(con)

    def seed_fit(url, score, fit):
        vid = seed_scored(con, url, score=score, company=url, title=f"Role {url}")
        feat = {"match_score": fit, "fit_score": fit, "fit_hyre": fit, "bgem3_sparse": fit * 0.45,
                "elig_score": 1.0}
        con.execute("INSERT OR REPLACE INTO vacancy_feature (vacancy_id, match_score, feature_json,"
                    " computed_at) VALUES (?,?,?,datetime('now'))", (vid, fit, json.dumps(feat)))
        con.commit()
        return vid
    hi = seed_fit("hi", 5, 0.70)    # effective ≈ 0.70 ≥ floor
    lo = seed_fit("lo", 5, 0.20)    # effective ≈ 0.20 < floor → out of exploit
    s = slate.build_slate(cfg, con, "2026-06-13")
    exploit_ids = [x["vacancy_id"] for x in s if x["slot_type"] == "exploit"]
    assert hi in exploit_ids and lo not in exploit_ids


def test_soft_eligibility_renders_amber_not_red(cfg, con):
    """WS1c: a SOFT (prose-degree) eligibility note renders muted amber, never the red ⛔ STOP."""
    import json
    from schabasch import features
    vid = seed_scored(con, "u/soft", score=4, company="Co")
    features._ensure_schema(con)
    feat = {"match_score": 0.5, "fit_score": 0.7, "fit_hyre": 0.7, "xenc_full": 0.7,
            "elig_score": 1.0, "elig_note": "требуется master, у тебя bachelor",
            "elig_severity": "soft"}
    con.execute("INSERT OR REPLACE INTO vacancy_feature (vacancy_id, match_score, feature_json,"
                " computed_at) VALUES (?,?,?,datetime('now'))", (vid, 0.5, json.dumps(feat)))
    con.commit()
    html = slate.render_html(slate.build_slate(cfg, con, "2026-06-13"), "2026-06-13")
    assert "Borderline requirement" in html and "alert warn" in html
    # the red STOP is the alert block, not the ⛔ glyph (which the header legend lists deliberately)
    assert '<div class="alert">⛔' not in html   # soft → never the red STOP alert


# ---------------------------------------------------------------------------
# Freshness re-rank (recent on top) + accent posting-date badge
# ---------------------------------------------------------------------------

def test_recency_mult_off_by_default_decays_when_on():
    """recency_mult: floor 1.0 (absent/default) = OFF (==1.0 any age); when on it decays
    today→1.0 > halflife > old→floor, never below floor, and clamps negative days to fresh."""
    off = {"slate": {}}
    assert slate._recency_mult(0, off) == 1.0 and slate._recency_mult(99, off) == 1.0
    on = {"slate": {"recency_floor": 0.6, "recency_halflife_days": 7}}
    assert slate._recency_mult(0, on) == 1.0                       # today = full weight
    assert abs(slate._recency_mult(7, on) - 0.8) < 1e-9            # one halflife → midway to floor
    assert slate._recency_mult(0, on) > slate._recency_mult(7, on) > slate._recency_mult(60, on)
    assert slate._recency_mult(10_000, on) >= 0.6                  # never below floor
    assert slate._recency_mult(-5, on) == 1.0                      # negative days clamped to fresh


def _seed_fit_dated(con, url, fit, date_posted):
    """A scored vacancy with a live fit (fit_hyre + bge-m3 sparse → fit_eff==fit) and a posting date."""
    import json
    vid = seed_scored(con, url, score=5, company=url, title=f"Role {url}")
    feat = {"match_score": fit, "fit_score": fit, "fit_hyre": fit, "bgem3_sparse": fit * 0.45,
            "elig_score": 1.0}
    con.execute("INSERT OR REPLACE INTO vacancy_feature (vacancy_id, match_score, feature_json,"
                " computed_at) VALUES (?,?,?,datetime('now'))", (vid, fit, json.dumps(feat)))
    con.execute("UPDATE vacancy SET date_posted=? WHERE id=?", (date_posted, vid))
    con.commit()
    return vid


def test_recency_rerank_fresh_outranks_stale_great(cfg, con):
    """The user's "recent on top": a FRESH slightly-worse job outranks a STALE great one in slate
    order (recency reorders _eff). quality_floor still gates on pure _eff so both stay in exploit."""
    from datetime import datetime, timezone, timedelta
    from schabasch import features
    cfg = dict(cfg); cfg["slate"] = dict(cfg["slate"])
    cfg["slate"]["recency_floor"] = 0.6; cfg["slate"]["recency_halflife_days"] = 7
    cfg["features"] = dict(cfg["features"]); cfg["features"]["fit_weights"] = {"hyre": 0.7, "sparse": 0.3}
    features._ensure_schema(con)
    old = (datetime.now(timezone.utc) - timedelta(days=14)).date().isoformat()
    today = datetime.now(timezone.utc).date().isoformat()
    stale_great = _seed_fit_dated(con, "great", 0.90, old)    # _eff .90 × recency(14d)≈.70 → rank ≈.63
    fresh_good = _seed_fit_dated(con, "good", 0.72, today)    # _eff .72 × recency(0d)=1.0  → rank ≈.72
    order = [x["vacancy_id"] for x in slate.build_slate(cfg, con, "2026-06-13", rebuild=True)]
    assert order.index(fresh_good) < order.index(stale_great)


def test_recency_off_keeps_quality_order(cfg, con):
    """Behaviour-preserving: with recency OFF (floor 1.0, the default) the great job leads again —
    proves the re-rank, not some other change, is what moved the fresh one up."""
    from datetime import datetime, timezone, timedelta
    from schabasch import features
    cfg = dict(cfg); cfg["slate"] = dict(cfg["slate"]); cfg["slate"]["recency_floor"] = 1.0
    cfg["features"] = dict(cfg["features"]); cfg["features"]["fit_weights"] = {"hyre": 0.7, "sparse": 0.3}
    features._ensure_schema(con)
    old = (datetime.now(timezone.utc) - timedelta(days=14)).date().isoformat()
    today = datetime.now(timezone.utc).date().isoformat()
    stale_great = _seed_fit_dated(con, "great", 0.90, old)
    fresh_good = _seed_fit_dated(con, "good", 0.72, today)
    order = [x["vacancy_id"] for x in slate.build_slate(cfg, con, "2026-06-13", rebuild=True)]
    assert order.index(stale_great) < order.index(fresh_good)


def test_fresh_card_shows_accent_badge(cfg, con):
    """A ≤2-day job renders the accent freshness badge by the title; the old muted gray .posted
    date no longer sits in the .meta2 line (laws-of-UX subtract-before-add)."""
    from datetime import datetime, timezone
    vid = seed_scored(con, "u/fresh", score=5, company="Co")
    con.execute("UPDATE vacancy SET date_posted=? WHERE id=?",
                (datetime.now(timezone.utc).date().isoformat(), vid))
    con.commit()
    html = slate.render_html(slate.build_slate(cfg, con, "2026-06-13"), "2026-06-13")
    assert "freshbadge fresh" in html        # accent badge for today's posting
    assert 'class="posted"' not in html      # the old gray meta date is gone


def test_bilingual_toggle_en_default_ru_on_request(cfg, con):
    """English by default; ?lang=ru flips the UI to Russian; both keep «шабашка» + the toggle."""
    _seed_many(con, 6)
    s = slate.build_slate(cfg, con, "2026-06-13")
    en = slate.render_html(s, "2026-06-13")                 # default = English
    ru = slate.render_html(s, "2026-06-13", lang="ru")
    assert '<html lang="en"' in en and "office mouse" in en and "annotate more ↗" in en
    assert '<html lang="ru"' in ru and "офисная мышь" in ru and "разметить ещё ↗" in ru
    assert "шабашка" in en and "шабашка" in ru             # signature term kept in both
    assert "langsw" in en and 'href="?lang=ru"' in en      # toggle to Russian present on the EN page


def test_recent_reject_penalty(cfg, con):
    """5c: ≥2 recent score-≤2 labels of a role_kind → a sub-1.0 demotion multiplier for that kind;
    neutral (analyst/manager) kinds never penalised; disabled (days=0) returns {}."""
    import json
    from schabasch import db
    from schabasch.models import Status
    cfg = dict(cfg)
    cfg["slate"] = dict(cfg["slate"], pref_penalty_days=7, pref_penalty_floor=0.3)

    def seed_label(url, title, score):
        vid = db.upsert_vacancy(con, {"source": "linkedin", "url": url, "title": title,
                                      "company": "C", "city": "F", "description": "x" * 100})
        db.set_status(con, vid, Status.SCORED, card_json=json.dumps({"summary_2lines": "a\nb"}))
        db.insert_label(con, vid, {"score_1_5": score, "source": "slate"})
        return vid

    seed_label("l/e1", "Software Engineer", 2)
    seed_label("l/e2", "Backend Engineer", 2)            # 2 engineer rejections → penalised
    seed_label("l/a1", "Business Analyst", 4)            # neutral kind, liked → no entry

    pen = slate.recent_reject_penalty(con, cfg)
    assert "hands_on_engineer" in pen and pen["hands_on_engineer"] == 0.3   # all-reject → floor
    assert "" not in pen                                  # neutral analyst kind never penalised

    # a single rejection is below the ≥2 noise guard → no penalty for that kind
    seed_label("l/j1", "Werkstudent (m/w/d)", 2)
    assert "junior" not in slate.recent_reject_penalty(con, cfg)

    cfg["slate"]["pref_penalty_days"] = 0                 # disabled → off
    assert slate.recent_reject_penalty(con, cfg) == {}
