"""Deterministic still-open check for an Indeed posting (the user's «expired Indeed → убрать» ask).

Indeed `viewjob` URLs answer a plain GET with an HTTP 403 anti-bot "Security Check" shell, so the
generic `llm.http_get_status` probe can only ever say "unknown" for Indeed. Spike (2026-06-18) found
the working path: jobspy's TLS-impersonating session (`create_session(is_tls=True)`) bypasses the
wall *intermittently* (≈1 in N attempts returns the real ~525 KB page; the rest return the shell).

The reliable signal is NOT the human banner — the strings `abgelaufen` / "This job has expired" are
React-bundle boilerplate present even on a LIVE page (an `"expired":false` page still contains them),
so a substring match is a false-positive generator. The page embeds a JSON state with a clean
`"expired":true|false` boolean; the spike confirmed it against the user's own ground truth (both
`abgelaufen`-noted jobs → `"expired":true`, both seen-today jobs → `"expired":false`).

Mirrors `arbeitsagentur.check_open`: True = open, False = verified gone, None = couldn't verify
(never a false "closed" from a blocked/odd fetch).
"""
from __future__ import annotations

import logging
import re
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

DEFAULT_ATTEMPTS = 4            # TLS retries to get past the intermittent 403 shell
_EXPIRED_JSON = re.compile(r'"expired"\s*:\s*(true|false)')
# the real job page is ~0.5 MB; the 403 "Security Check" shell is ~60 KB and carries no JSON state.
_MIN_REAL_PAGE_BYTES = 200_000


def jk_from_url(url: str | None) -> str | None:
    """Extract the Indeed job key (`jk`) from a `…/viewjob?jk=…` URL. None if absent."""
    if not url:
        return None
    jk = parse_qs(urlparse(str(url)).query).get("jk")
    return jk[0] if jk else None


def _fetch_html(url: str) -> str | None:
    """One TLS-impersonated GET via jobspy's session. None on any error (import-local so the package
    imports without jobspy and the call is easy to mock in tests)."""
    try:
        from jobspy.util import create_session
        s = create_session(is_tls=True, has_retry=False)
        return s.get(url).text or ""
    except Exception as exc:  # network / tls_client error → couldn't verify, not "closed"
        logger.debug("indeed.check_open fetch failed for %s: %s", url, exc)
        return None


def check_open(jk: str | None, *, attempts: int = DEFAULT_ATTEMPTS) -> bool | None:
    """True = live, False = expired ("expired":true), None = couldn't verify (all attempts hit the
    anti-bot shell, or an ambiguous page). Retries because the TLS session passes the 403 only
    intermittently — the FIRST attempt that returns a real page (carrying the `"expired"` JSON token)
    decides; the shell carries no token, so it just costs a retry."""
    if not jk:
        return None
    url = f"https://de.indeed.com/viewjob?jk={jk}"
    for _ in range(max(1, attempts)):
        body = _fetch_html(url)
        if not body or len(body) < _MIN_REAL_PAGE_BYTES:
            continue  # 403 shell / empty → retry
        vals = set(_EXPIRED_JSON.findall(body))
        if vals == {"true"}:
            return False        # the posting itself is expired → verified gone
        if vals == {"false"}:
            return True         # verified live
        # token absent (odd page) or BOTH true+false (a similar-jobs rail) → don't guess; retry/None
    return None
