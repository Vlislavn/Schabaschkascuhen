"""Rich per-stage console logging (the user couldn't tell what a long serve/tick was doing)."""
from __future__ import annotations

import schabasch.pipeline as pipe


def test_step_logs_start_and_finish(capsys):
    pipe.VERBOSE = True
    summary: dict = {}
    pipe._step("scrape_demo", lambda: {"indeed": 5}, summary)
    out = capsys.readouterr().out
    assert "▶ scrape_demo" in out          # start line
    assert "✓ scrape_demo" in out          # finish line
    assert "indeed:5" in out               # compact result
    assert summary["scrape_demo"] == {"indeed": 5}   # counts unchanged (logging only)


def test_heavy_marker_via_run_stage(capsys):
    pipe.VERBOSE = True
    pipe._run_stage("feat", lambda: {"featured": 3}, {}, heavy=True, label="признаки (bge-m3)")
    out = capsys.readouterr().out
    assert "⏳" in out and "признаки" in out and "✓ feat" in out


def test_quiet_mutes(capsys):
    pipe.VERBOSE = False
    try:
        pipe._step("quiet_demo", lambda: 7, {})
        assert capsys.readouterr().out == ""
    finally:
        pipe.VERBOSE = True


def test_fmt_result_compact():
    assert pipe._fmt_result({"a": 0, "b": 3, "c": 0}) == "{b:3}"
    assert pipe._fmt_result({"error": "boom"}) == "ERROR boom"
    assert pipe._fmt_result({"skipped_low_memory": "x"}) == "skipped (low memory)"
