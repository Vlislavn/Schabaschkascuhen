"""indeed.check_open — parse the embedded "expired" JSON boolean from the (TLS-fetched) viewjob page.
The human banner ("abgelaufen" / "This job has expired") is React boilerplate present even on a LIVE
page, so the parser must key on the JSON boolean, NOT a substring (spike 2026-06-18)."""
from __future__ import annotations

from schabasch.sources import indeed

# real pages are ~0.5 MB; the parser ignores anything below _MIN_REAL_PAGE_BYTES (the 403 shell).
_PAD = "x" * (indeed._MIN_REAL_PAGE_BYTES + 10)
# a LIVE page STILL contains the human banner — the false-positive trap the parser must avoid.
OPEN_PAGE = _PAD + 'Diese Stellenanzeige ist auf Indeed abgelaufen … "expired":false …'
EXPIRED_PAGE = _PAD + '… "expired":true …'
SHELL = "Security Check - Indeed.com" + "x" * 100   # short, no "expired" token


def test_expired_true_is_closed(monkeypatch):
    monkeypatch.setattr(indeed, "_fetch_html", lambda url: EXPIRED_PAGE)
    assert indeed.check_open("jk1") is False


def test_expired_false_is_open_despite_banner(monkeypatch):
    monkeypatch.setattr(indeed, "_fetch_html", lambda url: OPEN_PAGE)
    assert indeed.check_open("jk1") is True   # the banner text must NOT trigger a false close


def test_shell_is_unknown(monkeypatch):
    monkeypatch.setattr(indeed, "_fetch_html", lambda url: SHELL)
    assert indeed.check_open("jk1", attempts=3) is None   # all attempts hit the wall → None


def test_fetch_error_is_unknown(monkeypatch):
    monkeypatch.setattr(indeed, "_fetch_html", lambda url: None)
    assert indeed.check_open("jk1", attempts=2) is None


def test_no_jk_is_unknown():
    assert indeed.check_open(None) is None
    assert indeed.check_open("") is None


def test_jk_from_url():
    assert indeed.jk_from_url("https://de.indeed.com/viewjob?jk=abc123&x=1") == "abc123"
    assert indeed.jk_from_url("https://de.indeed.com/viewjob") is None
    assert indeed.jk_from_url(None) is None
