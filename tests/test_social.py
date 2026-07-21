"""Person-post источники (bluesky/mastodon/telegram): ворота + маппинг поста в вакансию (без сети)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from schabasch.models import Status
from schabasch.sources import social


def _recent(hours: int = 1) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


HIRING_TEXT = ("We're hiring an aerospace systems engineer for our Frankfurt office. "
               "Hybrid, visa sponsorship possible. DM me or apply via the link below.")
NOISE_TEXT = ("I visited the aerospace museum today and it was absolutely amazing, "
              "so many rockets and satellites on display, highly recommend it to everyone.")


def test_hiring_gate_en_de():
    assert social._HIRING_RE.search("We're hiring a data analyst")
    assert social._HIRING_RE.search("Wir suchen Verstärkung im Team (m/w/d)")
    assert social._HIRING_RE.search("bewirb dich jetzt")
    assert not social._HIRING_RE.search("I love my new job at ACME")


def test_bluesky_gate_and_idempotency(cfg, con, monkeypatch):
    posts = {"posts": [
        {"uri": "at://did:plc:x/app.bsky.feed.post/abc",
         "author": {"handle": "alice.bsky.social"},
         "record": {"text": HIRING_TEXT, "createdAt": _recent()}},
        {"uri": "at://did:plc:y/app.bsky.feed.post/def",
         "author": {"handle": "bob.bsky.social"},
         "record": {"text": NOISE_TEXT, "createdAt": _recent()}},          # нет hiring-лексикона
        {"uri": "at://did:plc:z/app.bsky.feed.post/ghi",
         "author": {"handle": "carol.bsky.social"},
         "record": {"text": HIRING_TEXT + " (old)",
                    "createdAt": _recent(hours=24 * 10)}},                  # 10 дней → freshness-дроп
    ]}
    monkeypatch.setattr(social, "http_get_json", lambda *a, **kw: posts)
    monkeypatch.setattr("time.sleep", lambda s: None)
    n = social._bluesky(cfg, con, max_age=2)
    assert n == 1
    row = con.execute("SELECT source, status, date_posted, description FROM vacancy "
                      "WHERE url LIKE '%alice%'").fetchone()
    assert row["source"] == "bluesky" and row["status"] == Status.DESCRIBED.value
    assert row["date_posted"] and "alice.bsky.social" in row["description"]
    # повторный прогон: те же url → reseen, не new
    assert social._bluesky(cfg, con, max_age=2) == 0


def test_mastodon_relevance_tokens(cfg, con, monkeypatch):
    statuses = [
        {"url": "https://mastodon.social/@a/1", "created_at": _recent(),
         "content": f"<p>{HIRING_TEXT}</p>", "account": {"acct": "a"}},     # токен 'aerospace' есть
        {"url": "https://mastodon.social/@b/2", "created_at": _recent(),
         "content": "<p>We are hiring a senior pastry chef for our bakery team, "
                     "great benefits, apply now, join our team today!</p>",
         "account": {"acct": "b"}},                                          # нет токена запросов
    ]
    monkeypatch.setattr(social, "http_get_json", lambda *a, **kw: statuses)
    monkeypatch.setattr("time.sleep", lambda s: None)
    cfg["search"]["social_instances"] = ["mastodon.social"]
    cfg["search"]["social_hashtags"] = ["hiring"]
    n = social._mastodon(cfg, con, max_age=2, tokens=social._query_tokens(cfg))
    assert n == 1
    assert con.execute("SELECT 1 FROM vacancy WHERE url LIKE '%@a/1'").fetchone()
    assert con.execute("SELECT 1 FROM vacancy WHERE url LIKE '%@b/2'").fetchone() is None


_TG_HTML = f"""<html><body>
<div class="tgme_widget_message" data-post="testchan/42">
  <div class="tgme_widget_message_text">{HIRING_TEXT}</div>
  <time datetime="{_recent()}"></time>
</div>
<div class="tgme_widget_message" data-post="testchan/43">
  <div class="tgme_widget_message_text">short hiring</div>
</div>
</body></html>"""


class _FakeResp:
    def __init__(self, text):
        self.text = text
    def raise_for_status(self):
        pass


def test_telegram_parse_and_no_preview(cfg, con, monkeypatch):
    monkeypatch.setattr("requests.get", lambda url, **kw:
                        _FakeResp(_TG_HTML if "testchan" in url else "<html>join</html>"))
    monkeypatch.setattr("time.sleep", lambda s: None)
    cfg["search"]["telegram_channels"] = ["testchan", "nopreview"]
    n = social._telegram(cfg, con, max_age=2, tokens=social._query_tokens(cfg))
    assert n == 1  # 43 отрезан по длине; nopreview дал 0 блоков
    row = con.execute("SELECT source, url FROM vacancy WHERE url = 'https://t.me/testchan/42'").fetchone()
    assert row is not None and row["source"] == "telegram"
    detail = con.execute("SELECT detail FROM funnel_log WHERE source='telegram' "
                         "AND detail LIKE 'no preview%'").fetchone()
    assert detail is not None


def test_mk_title_and_tokens(cfg):
    assert social._mk_title("  Строка один\nстрока два  ") == "Строка один"
    assert len(social._mk_title("x" * 500)) == 120
    toks = social._query_tokens(cfg)
    assert "aerospace" in toks and "raumfahrt" in toks
