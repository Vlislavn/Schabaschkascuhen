"""P0: role-fit sidecar (the 'good domain, wrong role' axis) + the debug→golden firewall."""
from __future__ import annotations

from schabasch import role_feedback
from tests.conftest import seed_scored


def test_record_counts_and_firewall(cfg, con):
    a = seed_scored(con, "u/a", score=4, company="A", title="Senior ML Engineer")
    b = seed_scored(con, "u/b", score=4, company="B", title="Data Engineer")
    role_feedback.record(con, a, "hands_on_engineer", False, source="slate")   # wrong role
    role_feedback.record(con, b, "hands_on_engineer", True, source="slate")    # fits
    role_feedback.record(con, a, "hands_on_engineer", True, source="debug")    # FIREWALLED debug vote

    counts = role_feedback.fit_counts(con, source="slate")
    assert counts["hands_on_engineer"] == (2, 1)               # debug vote excluded from golden
    assert role_feedback.fit_counts(con, source="debug")["hands_on_engineer"] == (1, 1)

    vm = role_feedback.veto_map(con, source="slate")
    assert vm[a] == 0 and vm[b] == 1                           # a=wrong role, b=fits


def test_record_is_upsert(cfg, con):
    a = seed_scored(con, "u/up", score=4, company="A", title="Junior Developer")
    role_feedback.record(con, a, "junior", False, source="slate")
    role_feedback.record(con, a, "junior", True, source="slate")   # re-vote overwrites (UNIQUE)
    assert role_feedback.veto_map(con)[a] == 1
