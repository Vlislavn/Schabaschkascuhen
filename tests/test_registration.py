"""Self-service registration (schabasch.registration + POST /register). LLM mocked — no ollama."""
from __future__ import annotations

import copy

import pytest
import yaml

from schabasch import config, registration, users


@pytest.fixture
def reg_env(cfg, tmp_path, monkeypatch):
    """Isolated world: stubbed base config, tmp USERS_DIR + tmp ROOT (forced per-user paths land
    under tmp_path/data/users/<key>, never the real repo)."""
    base = copy.deepcopy(cfg)
    base["paths"] = {"db": str(tmp_path / "default.sqlite3"),
                     "golden_csv": str(tmp_path / "golden.csv"),
                     "slate_dir": str(tmp_path / "slates"),
                     "model_dir": str(tmp_path / "models"),
                     "agent_workdir": str(tmp_path / "agent")}
    base["telegram"] = {"chat_id": 111}
    monkeypatch.setattr(config, "load", lambda path=None: copy.deepcopy(base))
    monkeypatch.setattr(config, "ROOT", tmp_path)
    udir = tmp_path / "config" / "users"
    monkeypatch.setattr(users, "USERS_DIR", udir)
    return base


_CV_PROFILE = {"skills": ["sql", "python"], "experience": ["5y BA"],
               "target_roles": ["Business Analyst", "Product Manager"],
               "domains": ["fintech", "Business Analyst"],   # dupe (case) — must dedupe
               "summary": "Senior BA, processes + SQL."}


@pytest.fixture
def mock_extract(monkeypatch):
    """Stand-in for the qwen CV extraction; records the DB it was pointed at."""
    calls = {}

    def fake(ucfg, con, *, description=None, cv_path=None):
        calls["db"] = ucfg["paths"]["db"]
        calls["description"] = description
        calls["cv_path"] = cv_path
        return dict(_CV_PROFILE)

    monkeypatch.setattr("schabasch.candidate.extract_candidate", fake)
    return calls


# ── resolve_city ──────────────────────────────────────────────────────────────────────────────

def test_resolve_city_exact_and_normalized(reg_env):
    r = registration.resolve_city("Berlin")
    assert r["ok"] and r["name"] == "Berlin" and r["lat"] and r["lon"]
    assert registration.resolve_city("berlin, Germany")["ok"]        # Postel: messy input
    assert registration.resolve_city("10115 Berlin")["ok"]           # PLZ prefix stripped


def test_resolve_city_fuzzy_suggestions(reg_env):
    r = registration.resolve_city("Berln")
    assert not r["ok"] and "Berlin" in r["suggestions"]


# ── derive_key ────────────────────────────────────────────────────────────────────────────────

def test_derive_key_slug_fallback_and_collision(reg_env):
    assert users.derive_key("Bob Smith", 42) == "bob-smith"
    assert users.derive_key("Маша К.", 42) == "u42"                  # non-ASCII → chat_id fallback
    users.USERS_DIR.mkdir(parents=True, exist_ok=True)
    (users.USERS_DIR / "bob-smith.yaml").write_text("telegram: {chat_id: 9}\n", encoding="utf-8")
    assert users.derive_key("Bob Smith", 43) == "bob-smith2"         # collision → suffix


# ── register_user ─────────────────────────────────────────────────────────────────────────────

def test_register_happy_path_pins_overlay_schema(reg_env, mock_extract, tmp_path):
    out = registration.register_user(reg_env, chat_id=222, name="Bob Smith",
                                     city="Berlin", cv_text="I am a BA...")
    assert out["user"] == "bob-smith"
    # derived queries: target_roles first, domains deduped case-insensitively, ≤4
    assert out["queries_en"] == ["Business Analyst", "Product Manager", "fintech"]
    assert out["summary"] == _CV_PROFILE["summary"]
    # extraction ran against the NEW user's own DB
    assert "data/users/bob-smith" in mock_extract["db"].replace("\\", "/")
    # pinned overlay schema — the latent-bug guards: queries_en/_de (not `queries`),
    # geo.anchors as a DICT with lat/lon/radius_km, cities as "City, Germany"
    ov = yaml.safe_load(users.user_file("bob-smith").read_text(encoding="utf-8"))
    assert ov["search"]["queries_en"] == out["queries_en"]
    assert ov["search"]["queries_de"] == out["queries_en"]
    assert "queries" not in ov["search"]
    assert ov["search"]["cities"] == ["Berlin, Germany"]
    assert ov["geo"]["anchors"]["Berlin"]["lat"] and ov["geo"]["anchors"]["Berlin"]["radius_km"] == 40
    assert ov["telegram"]["chat_id"] == 222
    assert ov["profile"]["summary"] == out["summary"]
    # the isolated DB was materialized, and the resulting cfg loads
    assert (tmp_path / "data" / "users" / "bob-smith" / "bob-smith.sqlite3").is_file()
    assert users.load("bob-smith")["search"]["queries_en"] == out["queries_en"]


def test_register_explicit_queries_win(reg_env, mock_extract):
    out = registration.register_user(reg_env, chat_id=222, name="B", city="Berlin",
                                     cv_text="x", queries=["data engineer"])
    assert out["queries_en"] == ["data engineer"]


def test_register_cap_duplicate_city_and_missing_cv(reg_env, mock_extract):
    registration.register_user(reg_env, chat_id=222, name="B", city="Berlin", cv_text="x")
    # cap: default + 1 registered = MAX_USERS(2)
    with pytest.raises(registration.RegistrationError) as e:
        registration.register_user(reg_env, chat_id=333, name="C", city="Berlin", cv_text="x")
    assert e.value.status == 409 and "cap" in str(e.value)
    # duplicate chat_id (also capped, but dup fires first)
    with pytest.raises(registration.RegistrationError) as e:
        registration.register_user(reg_env, chat_id=222, name="B", city="Berlin", cv_text="x")
    assert e.value.status == 409 and "already registered" in str(e.value)


def test_register_unknown_city_422_with_suggestions(reg_env, mock_extract):
    with pytest.raises(registration.RegistrationError) as e:
        registration.register_user(reg_env, chat_id=222, name="B", city="Berln", cv_text="x")
    assert e.value.status == 422 and "Berlin" in e.value.suggestions


def test_register_rollback_on_extraction_failure(reg_env, monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("scanned pdf, no text")
    monkeypatch.setattr("schabasch.candidate.extract_candidate", boom)
    before = users.list_users()
    with pytest.raises(registration.RegistrationError) as e:
        registration.register_user(reg_env, chat_id=222, name="B", city="Berlin", cv_text="x")
    assert e.value.status == 500
    assert users.list_users() == before                                # no half-registered user
    assert not users.user_file("b").is_file()
    assert not (tmp_path / "data" / "users" / "b").exists()            # DB dir removed too


def test_register_rollback_when_no_queries_derivable(reg_env, monkeypatch):
    monkeypatch.setattr("schabasch.candidate.extract_candidate",
                        lambda *a, **k: {"skills": ["x"], "experience": ["y"]})
    with pytest.raises(registration.RegistrationError) as e:
        registration.register_user(reg_env, chat_id=222, name="B", city="Berlin", cv_text="x")
    assert e.value.status == 422 and users.list_users() == ["default"]


# ── HTTP routes ───────────────────────────────────────────────────────────────────────────────

def _client(base_cfg):
    from fastapi.testclient import TestClient
    from schabasch import feedback_app
    return TestClient(feedback_app.create_app(base_cfg))


def test_register_route_and_geo_resolve(reg_env, mock_extract):
    client = _client(reg_env)
    r = client.get("/geo-resolve", params={"city": "Berlin"})
    assert r.status_code == 200 and r.json()["ok"]
    # unknown city carries suggestions in the body (checked BEFORE the slot fills — once the cap
    # is reached, 409 deliberately fires before city validation)
    r = client.post("/register", json={"chat_id": 999, "name": "C", "city": "Berln", "cv_text": "x"})
    assert r.status_code == 422 and "Berlin" in r.json()["suggestions"]
    r = client.post("/register", json={"chat_id": 222, "name": "Bob Smith",
                                       "city": "Berlin", "cv_text": "I am a BA"})
    assert r.status_code == 200 and r.json()["ok"] and r.json()["user"] == "bob-smith"
    assert {u["user"] for u in client.get("/users.json").json()["users"]} == {"default", "bob-smith"}
    # duplicate now 409 through the route
    r = client.post("/register", json={"chat_id": 222, "name": "Bob Smith",
                                       "city": "Berlin", "cv_text": "x"})
    assert r.status_code == 409


def test_register_route_cv_path_guard(reg_env):
    r = _client(reg_env).post("/register", json={"chat_id": 222, "name": "B", "city": "Berlin",
                                                 "cv_path": "/nonexistent/cv.pdf"})
    assert r.status_code == 422 and "cv_path" in r.json()["error"]


# ── live-run fixes 2026-07-03 (real Telegram smoke test findings) ─────────────────────────────

def test_resolve_city_cyrillic_aliases(reg_env):
    """Live finding: «Франкфурт» was rejected 4× — RU-speaking users type Cyrillic city names."""
    for ru, latin in [("Франкфурт", "Frankfurt"), ("Мюнхен", "München"), ("Берлин", "Berlin"),
                      ("Гейдельберг", "Heidelberg")]:
        r = registration.resolve_city(ru)
        assert r["ok"] and r["name"] == latin, (ru, r)


def test_resolve_cities_multi_and_nationwide(reg_env):
    """Live findings: «Франкфурт, Мюнхен» (multi) and «Вся германия» (nationwide) were rejected."""
    r = registration.resolve_cities("Франкфурт, Мюнхен")
    assert r["ok"] and [c["name"] for c in r["cities"]] == ["Frankfurt", "München"]
    for phrase in ("Вся германия", "germany", "везде", "remote"):
        r = registration.resolve_cities(phrase)
        assert r["ok"] and r["nationwide"], phrase
    r = registration.resolve_cities("Франкфурт, Xyzzberg")
    assert not r["ok"] and r["failed"] == "Xyzzberg"


def test_register_multicity_and_nationwide(reg_env, mock_extract):
    out = registration.register_user(reg_env, chat_id=222, name="B", city="Франкфурт, Мюнхен",
                                     cv_text="x" * 50)
    import yaml as _yaml
    ov = _yaml.safe_load(users.user_file(out["user"]).read_text(encoding="utf-8"))
    assert ov["search"]["cities"] == ["Frankfurt, Germany", "München, Germany"]
    assert set(ov["geo"]["anchors"]) == {"Frankfurt", "München"}
    users.delete_user(out["user"])
    out = registration.register_user(reg_env, chat_id=333, name="C", city="вся Германия",
                                     cv_text="y" * 50)
    ov = _yaml.safe_load(users.user_file(out["user"]).read_text(encoding="utf-8"))
    assert ov["search"]["cities"] == ["Germany"] and ov["geo"]["anchors"] is None
    # falsy anchors = no geo preference: geo_check passes, geo_mark never tags far
    from schabasch import geo
    ucfg = users.load(out["user"])
    assert geo.geo_check("Berlin", ucfg) == (True, None)
    assert geo.geo_mark("Berlin", ucfg)["far"] is False


def test_register_merges_cv_file_plus_addendum(reg_env, mock_extract, tmp_path):
    """Live finding: the document caption («…agentic AI…») was lost — file + text now merge."""
    cv_file = tmp_path / "cv.txt"
    cv_file.write_text("Base CV body: oncologist, data quality." , encoding="utf-8")
    registration.register_user(reg_env, chat_id=222, name="B", city="Berlin",
                               cv_path=str(cv_file), cv_text="Addendum: agentic AI researcher")
    assert "Base CV body" in mock_extract["description"]
    assert "agentic AI researcher" in mock_extract["description"]
    assert mock_extract["cv_path"] is None   # merged into description


def test_update_queries_route_add_and_replace(reg_env, mock_extract):
    client = _client(reg_env)
    r = client.post("/register", json={"chat_id": 222, "name": "Bob Smith",
                                       "city": "Berlin", "cv_text": "x" * 50})
    key = r.json()["user"]
    # add mode (the «помимо тех, ещё…» live ask)
    r = client.post("/update-queries", json={"user": key, "queries": ["agentic AI"], "mode": "add"})
    assert r.status_code == 200
    assert r.json()["queries_en"][-1] == "agentic AI" and len(r.json()["queries_en"]) == 4
    # takes effect IMMEDIATELY on the running app (the _cfg_for staleness fix)
    assert "agentic AI" in " ".join(client.get("/queries", params={"user": key}).json()["queries_en"])
    # replace mode
    r = client.post("/update-queries", json={"user": key, "queries": ["ml engineer"]})
    assert r.json()["queries_en"] == ["ml engineer"]
    # default user is refused (profile.yaml is hand-commented; a yaml round-trip would strip it)
    r = client.post("/update-queries", json={"user": "default", "queries": ["x"]})
    assert r.status_code == 400
    # empty list refused
    r = client.post("/update-queries", json={"user": key, "queries": ["  "]})
    assert r.status_code == 422


def test_geo_resolve_route_is_multi(reg_env):
    r = _client(reg_env).get("/geo-resolve", params={"city": "Франкфурт, Мюнхен"})
    body = r.json()
    assert body["ok"] and [c["name"] for c in body["cities"]] == ["Frankfurt", "München"]


def test_refetch_guard_ignores_fresh_stamp_on_empty_db(reg_env, mock_extract):
    """Live finding: GET /slate.json on a just-registered (empty) user logs a `slate` funnel row,
    which the restart guard then read as 'fetched 0.1h ago' → his REAL first fetch was skipped.
    A fresh stamp without any data must never skip."""
    from schabasch import db, feedback_app
    client = _client(reg_env)
    r = client.post("/register", json={"chat_id": 222, "name": "Bob Smith",
                                       "city": "Berlin", "cv_text": "x" * 50})
    key = r.json()["user"]
    # the bot's greet reads his slate → build_slate logs the poison `slate` funnel row
    assert client.get("/slate.json", params={"user": key}).status_code == 200
    ucfg = users.load(key)
    con = db.connect(ucfg["paths"]["db"])
    try:
        age = feedback_app._hours_since_last_fetch(con)
        assert age is not None and age < 1          # the stamp IS there (the poison)
        skip, _ = feedback_app._refetch_guard(con, ucfg, force=False)
        assert skip is False                        # …but an empty DB must still fetch
    finally:
        con.close()


def test_register_writes_persona_keys(reg_env, mock_extract):
    """De-personalization (2026-07-03): a new user must NOT inherit the base profile's taste —
    the overlay carries explicit scale/rubric/role_kind_mult + CV-derived magnets/repellents."""
    out = registration.register_user(reg_env, chat_id=222, name="Bob Smith",
                                     city="Berlin", cv_text="x" * 50)
    ov = yaml.safe_load(users.user_file(out["user"]).read_text(encoding="utf-8"))
    assert ov["judge"]["rubric_version"] == f"v1-{out['user']}"
    assert ov["slate"]["role_kind_mult"] == {"hands_on_engineer": 1.0, "junior": 1.0}
    assert ov["profile"]["scale"]["5"].startswith("Шабашка")
    # safety repellents always present; magnets fall back when the CV extraction has none
    for must in ("hidden-german", "slop-text", "temp-agency"):
        assert must in ov["profile"]["repellents"]
    assert ov["profile"]["magnets"] == ["new-domain"]
