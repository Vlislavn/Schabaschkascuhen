"""Multi-user isolation layer (schabasch.users) + per-user web routing (feedback_app)."""
from __future__ import annotations

import copy
import json

import pytest
import yaml

from schabasch import config, db, users
from schabasch.models import Status


@pytest.fixture
def user_env(cfg, tmp_path, monkeypatch):
    """A world with two users: 'default' (base cfg on a tmp file DB) + 'testbob' (overlay yaml).
    config.load is stubbed so tests never touch the real config/profile.yaml or data/."""
    base = copy.deepcopy(cfg)
    base["paths"] = {"db": str(tmp_path / "default.sqlite3"),
                     "golden_csv": str(tmp_path / "golden.csv"),
                     "slate_dir": str(tmp_path / "slates"),
                     "model_dir": str(tmp_path / "models"),
                     "agent_workdir": str(tmp_path / "agent")}
    monkeypatch.setattr(config, "load", lambda path=None: copy.deepcopy(base))
    udir = tmp_path / "users"
    udir.mkdir()
    overlay = {
        "profile": {"summary": "Bob the builder"},
        "telegram": {"chat_id": 424242},
        "paths": {"db": str(tmp_path / "bob" / "bob.sqlite3"),
                  "golden_csv": str(tmp_path / "bob" / "golden.csv"),
                  "slate_dir": str(tmp_path / "bob" / "slates"),
                  "model_dir": str(tmp_path / "bob" / "models")},
    }
    (udir / "testbob.yaml").write_text(yaml.safe_dump(overlay), encoding="utf-8")
    monkeypatch.setattr(users, "USERS_DIR", udir)
    return base


# ── users.load / registry ─────────────────────────────────────────────────────────────────────

def test_load_default_is_base_config(user_env):
    assert users.load()["paths"]["db"] == user_env["paths"]["db"]
    assert users.load("default")["paths"]["db"] == user_env["paths"]["db"]


def test_overlay_merges_and_isolates_paths(user_env):
    bob = users.load("testbob")
    # overlay wins where set
    assert bob["profile"]["summary"] == "Bob the builder"
    assert bob["paths"]["db"].endswith("bob/bob.sqlite3")
    # siblings inherited from base (deep merge, not replace)
    assert bob["search"]["queries_en"] == user_env["search"]["queries_en"]
    assert bob["profile"]["magnets"] == user_env["profile"]["magnets"]
    # every mutable per-user path differs from the default user's (guards C2: a shared
    # model_dir would let bob's retrain overwrite the default user's triage model)
    for name in ("db", "golden_csv", "slate_dir", "model_dir"):
        assert bob["paths"][name] != user_env["paths"][name], name


def test_forced_per_user_paths_when_overlay_omits_them(user_env, tmp_path):
    (users.USERS_DIR / "carol.yaml").write_text(
        yaml.safe_dump({"telegram": {"chat_id": 7}}), encoding="utf-8")
    carol = users.load("carol")
    for name in ("db", "golden_csv", "slate_dir", "model_dir"):
        assert "data/users/carol" in carol["paths"][name].replace("\\", "/"), name


def test_list_users_excludes_example(user_env):
    (users.USERS_DIR / "example.yaml").write_text("{}", encoding="utf-8")
    assert users.list_users() == ["default", "testbob"]


def test_by_telegram_id(user_env):
    assert users.by_telegram_id(424242) == "testbob"
    assert users.by_telegram_id(999) is None
    assert users.by_telegram_id(0) is None   # 0 = the base auto-lock sentinel, never an identity


def test_registry_lists_users_with_chat_ids(user_env):
    reg = {r["user"]: r["chat_id"] for r in users.registry()}
    assert reg["testbob"] == 424242
    assert "default" in reg


def test_invalid_and_unknown_keys_rejected(user_env):
    with pytest.raises(ValueError):
        users.load("../evil")
    with pytest.raises(FileNotFoundError):
        users.load("nosuch")


# ── per-user web routing (feedback_app) ───────────────────────────────────────────────────────

def _seed_scored(con, url, *, score=4, company="ACME"):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": url, "title": f"T {url}",
                                  "company": company, "city": "Frankfurt", "description": "x" * 500})
    card = dict(role="r", company=company, domain="aerospace", city="Frankfurt",
                work_mode="hybrid", language_posting="en", language_reality="en",
                integration_potential=2, summary_2lines="a\nb", slop_score=5, temp_agency_guess=False)
    db.set_status(con, vid, Status.SCORED, card_json=json.dumps(card, ensure_ascii=False))
    db.insert_judge_score(con, vid, {"score": score, "model": "qwen3:8b", "rubric_version": "v1",
                                     "explanation": "e"})
    return vid


def _client(base_cfg, monkeypatch):
    from fastapi.testclient import TestClient
    from schabasch import feedback_app
    monkeypatch.setattr(feedback_app, "_FETCH_STATE", {})   # fresh per-user state store per test
    return TestClient(feedback_app.create_app(base_cfg))


def test_feedback_isolated_per_user(user_env, monkeypatch):
    """The core multi-user guarantee: bob's rating lands in bob's DB, never in the default's."""
    bob_cfg = users.load("testbob")
    con_d = db.connect(user_env["paths"]["db"])
    vid_d = _seed_scored(con_d, "u/shared")
    con_d.close()
    con_b = db.connect(bob_cfg["paths"]["db"])
    vid_b = _seed_scored(con_b, "u/shared")
    con_b.close()

    client = _client(user_env, monkeypatch)
    r = client.post("/feedback", json={"vacancy_id": vid_b, "action": "good", "user": "testbob"})
    assert r.status_code == 200 and r.json()["ok"] is True

    con_b = db.connect(bob_cfg["paths"]["db"])
    assert con_b.execute("SELECT COUNT(*) FROM label").fetchone()[0] == 1
    con_b.close()
    con_d = db.connect(user_env["paths"]["db"])
    assert con_d.execute("SELECT COUNT(*) FROM label").fetchone()[0] == 0   # default untouched
    con_d.close()

    # no user field → backward-compatible default-user write
    r = client.post("/feedback", json={"vacancy_id": vid_d, "action": "good"})
    assert r.status_code == 200
    con_d = db.connect(user_env["paths"]["db"])
    assert con_d.execute("SELECT COUNT(*) FROM label").fetchone()[0] == 1
    con_d.close()


def test_html_threads_user_through_links_and_js(user_env, monkeypatch):
    """Guards C1: a page opened as ?user=testbob must carry the user in every nav link AND in the
    JS payloads — else feedback silently lands in the default user's golden label table."""
    bob_cfg = users.load("testbob")
    con_b = db.connect(bob_cfg["paths"]["db"])
    _seed_scored(con_b, "u/bobjob")
    con_b.close()
    bob_cfg["judge"] = {**bob_cfg.get("judge", {}), "rubric_version": "v1"}

    client = _client(user_env, monkeypatch)
    html = client.get("/?user=testbob").text
    assert 'const USER="testbob"' in html
    for page in ("/annotate", "/eval", "/gaps", "/tasks", "/funnel"):
        assert f"{page}?lang=en&user=testbob" in html, page
    # default page stays link-compatible: no user link fragments, USER const is "default"
    # (the static JS body legitimately contains the literal '&user='+USER — not a link)
    html_d = client.get("/").text
    assert 'const USER="default"' in html_d and "?lang=en&user=" not in html_d


def test_unknown_user_404_never_falls_through(user_env, monkeypatch):
    """A typo'd ?user= must 404 — not silently read/write another user's DB."""
    client = _client(user_env, monkeypatch)
    assert client.get("/?user=nosuch").status_code == 404
    r = client.post("/feedback", json={"vacancy_id": 1, "action": "good", "user": "nosuch"})
    assert r.status_code == 404


def test_fetch_status_isolated_per_user(user_env, monkeypatch):
    """Guards C3: bob polling /fetch-status must not see the default user's running fetch."""
    from schabasch import feedback_app
    client = _client(user_env, monkeypatch)
    feedback_app._fetch_state("default")["running"] = True
    assert client.get("/fetch-status?user=testbob").json()["running"] is False
    assert client.get("/fetch-status").json()["running"] is True


def test_users_json_registry_endpoint(user_env, monkeypatch):
    body = _client(user_env, monkeypatch).get("/users.json").json()
    reg = {r["user"]: r["chat_id"] for r in body["users"]}
    assert reg.get("testbob") == 424242
