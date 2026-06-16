"""Regression tests for the ReAct agent harness fix (2026-06-16).

Covers two root causes proven via CAPA on the kather `sota` / qwen3:8b agent runs:

  1. parse_json_output only recognized the '[max turns exhausted]' sentinel, so kl's OTHER
     non-finalize sentinels ('[budget exceeded]') and the salvage preamble reached ast.literal_eval
     and surfaced as a confusing 'invalid syntax / forgot a comma' SyntaxError.
  2. build_agent did not teach the model the kl final_answer protocol, so a strong model returned a
     bare result object → kl mapped it to a final_answer with empty content → degenerate → the loop
     (max_turns=6 < MAX_DEGENERATE=8) never accepted it → '[max turns exhausted]'. Strong (non-qwen)
     models also need budgets above kl's give-up guards so the salvage net can fire.
"""
from __future__ import annotations

import pytest

from schabasch import agent_runtime


# ---------------------------------------------------------------------------
# parse_json_output — non-finalize sentinels raise a CLEAN error (not SyntaxError)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentinel", [
    "[max turns exhausted]",
    "max turns exhausted",
    "[budget exceeded]",                       # qwen hit this; was 'forgot a comma' SyntaxError
    "[no answer]",
    "[unparseable response]",
    "Partial answer (synthesized from tool records, no LLM grounding produced).\nTask: x\n- turn 2: ...",
])
def test_parse_json_output_nonfinalize_sentinels_raise_valueerror(sentinel):
    with pytest.raises(ValueError, match="did not finalize"):
        agent_runtime.parse_json_output(sentinel)


def test_parse_json_output_budget_exceeded_is_not_a_syntaxerror():
    """The exact qwen symptom: '[budget exceeded]' must NOT crash ast.literal_eval."""
    try:
        agent_runtime.parse_json_output("[budget exceeded]")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "did not finalize" in str(e)
    except SyntaxError:  # pragma: no cover - this is the pre-fix bug
        pytest.fail("regressed: '[budget exceeded]' reached ast.literal_eval as SyntaxError")


def test_parse_json_output_still_parses_valid_json_and_tolerant_forms():
    assert agent_runtime.parse_json_output('{"verdict":"ok"}') == {"verdict": "ok"}
    # fenced + prose preamble (extract first balanced object)
    assert agent_runtime.parse_json_output(
        'Here is the result:\n```json\n{"a": 1}\n```') == {"a": 1}
    # single-quoted Python-dict style qwen emits (ast.literal_eval fallback preserved)
    assert agent_runtime.parse_json_output("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}
    # arrays still parse (discover agent returns a JSON array)
    assert agent_runtime.parse_json_output('[{"x": 1}]') == [{"x": 1}]


# ---------------------------------------------------------------------------
# build_agent — finalize-protocol contract + role-aware budgets
# ---------------------------------------------------------------------------

@pytest.fixture
def _capture_react(monkeypatch):
    """Mock kl's react_agent to capture the kwargs build_agent passes it."""
    pytest.importorskip("kl_agent_builder")
    captured: dict = {}

    def _fake_react_agent(**kwargs):
        captured.update(kwargs)
        return lambda _inp: None

    monkeypatch.setattr("kl_agent_builder.react_agent", _fake_react_agent)
    return captured


def _qwen_cfg() -> dict:
    return {"llm": {"roles": {"agent": {"client": "ollama", "model": "qwen3:8b",
                                        "provider": "openai", "base_url": "http://localhost:11434/v1",
                                        "api_key": "ollama"}}},
            "paths": {"agent_workdir": "/tmp/schabasch_agent_test"}}


def _strong_cfg(**overrides) -> dict:
    agent_role = {"client": "openai", "provider": "openai", "model": "test-strong-model",
                  "base_url": "https://example.invalid/v1", "api_key": "k"}
    cfg = {"llm": {"roles": {"agent": agent_role}},
           "paths": {"agent_workdir": "/tmp/schabasch_agent_test"}}
    if overrides:
        cfg["agent"] = overrides
    return cfg


def test_build_agent_injects_finalize_contract_for_all_models(_capture_react):
    agent_runtime.build_agent(_qwen_cfg(), system_prompt="TASK PROMPT BODY")
    sp = _capture_react["system_prompt"]
    assert "OUTPUT PROTOCOL" in sp and "final_answer" in sp
    assert "TASK PROMPT BODY" in sp


def test_build_agent_qwen_keeps_no_think_first_and_tight_budgets(_capture_react):
    agent_runtime.build_agent(_qwen_cfg(), system_prompt="TASK")
    sp = _capture_react["system_prompt"]
    assert sp.startswith("/no_think\n")             # qwen directive must lead
    assert _capture_react["max_turns"] == 6          # qwen unchanged (local cost)
    assert _capture_react["max_tool_calls"] == 6


def test_build_agent_strong_model_no_no_think_and_shares_tight_budget(_capture_react):
    """Non-qwen must NOT get the qwen /no_think directive, DOES get the finalize contract, and shares
    the same TIGHT budget — a looser strong budget was measured to regress sota (see _AGENT_DEFAULTS)."""
    agent_runtime.build_agent(_strong_cfg(), system_prompt="TASK")
    sp = _capture_react["system_prompt"]
    assert not sp.startswith("/no_think")            # non-qwen must NOT get the qwen directive
    assert "OUTPUT PROTOCOL" in sp
    assert _capture_react["max_turns"] == 6           # tight bound, same as local (no bump)
    assert _capture_react["max_tool_calls"] == 6


def test_build_agent_max_turns_argument_wins(_capture_react):
    agent_runtime.build_agent(_strong_cfg(), system_prompt="TASK", max_turns=3)
    assert _capture_react["max_turns"] == 3          # explicit per-call argument still overrides
