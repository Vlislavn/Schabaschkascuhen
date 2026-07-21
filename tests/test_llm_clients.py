"""Cascade clients + role router (schabasch/llm_clients.py)."""
from __future__ import annotations

import json

import pytest

from schabasch import llm_clients as lc
from schabasch.llm import LLMError, OllamaClient


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


def _ok(content):
    return _Resp(200, {"choices": [{"message": {"content": content}}]})


def test_router_defaults_to_ollama():
    """No llm.roles → ollama with the legacy model knob (byte-for-byte unchanged behaviour)."""
    cfg = {"llm": {"normalizer_model": "qwen3:8b", "judge_model": "qwen3:8b"}}
    c = lc.make_llm_client(cfg, "normalizer")
    assert isinstance(c, OllamaClient) and c.model == "qwen3:8b"
    # an unknown role still falls back to ollama, never crashes
    assert isinstance(lc.make_llm_client(cfg, "deep_reasoning"), OllamaClient)


def test_router_builds_openai_client():
    cfg = {"llm": {"roles": {"deep_reasoning": {
        "client": "openai", "model": "Qwen3.6-35B", "base_url": "http://localhost:8082/v1/",
        "api_key": "x", "max_tokens": 9000}}}}
    c = lc.make_llm_client(cfg, "deep_reasoning")
    assert isinstance(c, lc.OpenAIClient)
    assert c.model == "Qwen3.6-35B" and c.base_url == "http://localhost:8082/v1"  # trailing / stripped
    assert c.max_tokens == 9000


def test_role_available_local_needs_no_key():
    cfg = {"llm": {"roles": {
        "deep_reasoning": {"client": "openai", "base_url": "http://localhost:8082/v1"},
        "sota": {"client": "openai", "base_url": "https://api.example.com/v1", "api_key_env": "NOPE_MISSING"},
    }}}
    assert lc.role_available(cfg, "deep_reasoning") is True   # localhost → assume up
    assert lc.role_available(cfg, "sota") is False            # remote, no key → unavailable
    assert lc.role_available(cfg, "nonexistent") is False


def test_env_file_key_parse_strips_inline_comment(tmp_path):
    p = tmp_path / ".env"
    p.write_text('OPENAI_API_KEY=sk-secret123              # shared key\nOTHER=1\n')
    assert lc._read_env_file_key(str(p), "OPENAI_API_KEY") == "sk-secret123"
    assert lc._read_env_file_key(str(p), "MISSING") is None


def test_env_file_key_quoted_value(tmp_path):
    p = tmp_path / ".env"
    p.write_text('HF_TOKEN="hf_abc#notacomment"\n')
    assert lc._read_env_file_key(str(p), "HF_TOKEN") == "hf_abc#notacomment"


def test_resolve_key_prefers_env_then_file(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("OPENAI_API_KEY=from-file\n")
    rc = {"api_key_env": "OPENAI_API_KEY", "env_file": str(p)}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert lc._resolve_key({}, rc) == "from-file"
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert lc._resolve_key({}, rc) == "from-env"   # env wins


def test_extract_json_obj_tolerates_fences_and_think():
    assert lc._extract_json_obj('```json\n{"a": 1}\n```') == {"a": 1}
    assert lc._extract_json_obj('<think>reasoning…</think>\n{"score": 5}') == {"score": 5}
    assert lc._extract_json_obj("prefix {\"x\": 2} suffix") == {"x": 2}
    assert lc._extract_json_obj("no json here") is None


def test_openai_chat_json_happy_path(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        captured["auth"] = headers.get("Authorization")
        return _ok('{"score": 4, "why": "ok"}')

    monkeypatch.setattr(lc.requests, "post", fake_post)
    c = lc.OpenAIClient(model="sota", base_url="https://api.example.com/v1", api_key="sk-x")
    out = c.chat_json("sys", "user")
    assert out == {"score": 4, "why": "ok"}
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-x"
    assert captured["body"]["model"] == "sota"


def test_openai_retries_without_response_format(monkeypatch):
    """A 400 mentioning response_format → retry once without it (older mlx_lm servers)."""
    calls = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        if "response_format" in json:
            return _Resp(400, text="unsupported parameter: response_format")
        return _ok('{"ok": true}')

    monkeypatch.setattr(lc.requests, "post", fake_post)
    c = lc.OpenAIClient(model="m", api_key="")
    assert c.chat_json("s", "u") == {"ok": True}
    assert calls["n"] == 2


def test_openai_raises_classified_error_on_http_500(monkeypatch):
    monkeypatch.setattr(lc.requests, "post",
                        lambda *a, **k: _Resp(500, text="boom"))
    c = lc.OpenAIClient(model="m", api_key="")
    with pytest.raises(LLMError):
        c.chat_json("s", "u")


def test_openai_reasoning_content_fallback(monkeypatch):
    """Reasoning models can leave content empty and put the answer in reasoning_content."""
    payload = {"choices": [{"message": {"content": "", "reasoning_content": '{"a": 9}'}}]}
    monkeypatch.setattr(lc.requests, "post", lambda *a, **k: _Resp(200, payload))
    c = lc.OpenAIClient(model="m", api_key="")
    assert c.chat_json("s", "u") == {"a": 9}


# ---------------------------------------------------------------------------
# agent_client_params — 35B preferred, auto-fallback to ollama when :8082 is down
# ---------------------------------------------------------------------------

_MLX_AGENT = {"llm": {"judge_model": "qwen3:8b", "roles": {"agent": {
    "client": "openai", "provider": "openai", "model": "/models/Qwen3.6-35B-OptiQ-4bit",
    "base_url": "http://localhost:8082/v1", "api_key": "ollama"}}}}


def test_agent_params_uses_35b_when_reachable(monkeypatch):
    monkeypatch.setattr(lc, "_endpoint_reachable", lambda *a, **k: True)
    p = lc.agent_client_params(_MLX_AGENT)
    assert p["base_url"] == "http://localhost:8082/v1"
    assert p["model"] == "/models/Qwen3.6-35B-OptiQ-4bit"


def test_agent_params_falls_back_to_ollama_when_35b_down(monkeypatch):
    """Unattended-safe: a dead :8082 (MLX not started) → ollama small model, never a hang/co-load."""
    monkeypatch.setattr(lc, "_endpoint_reachable", lambda *a, **k: False)
    p = lc.agent_client_params(_MLX_AGENT)
    assert p["base_url"] == "http://localhost:11434/v1"
    assert p["model"] == "qwen3:8b" and p["api_key"] == "ollama"


def test_agent_params_remote_role_used_as_is_no_ping(monkeypatch):
    """A REMOTE agent role (any non-localhost URL) is never probed — no co-load/headroom concern."""
    monkeypatch.setattr(lc, "_endpoint_reachable",
                        lambda *a, **k: pytest.fail("must not ping a remote agent role"))
    cfg = {"llm": {"roles": {"agent": {"client": "openai", "model": "sota",
            "base_url": "https://api.example.com/v1", "api_key": "k"}}}}
    p = lc.agent_client_params(cfg)
    assert p["base_url"] == "https://api.example.com/v1" and p["model"] == "sota"


def test_agent_params_ollama_default_used_as_is_no_ping(monkeypatch):
    """The default ollama agent (:11434) is not probed — assumed locally available."""
    monkeypatch.setattr(lc, "_endpoint_reachable",
                        lambda *a, **k: pytest.fail("must not ping the ollama default port"))
    cfg = {"llm": {"judge_model": "qwen3:8b", "roles": {"agent": {
        "client": "ollama", "model": "qwen3:8b", "provider": "openai",
        "base_url": "http://localhost:11434/v1", "api_key": "ollama"}}}}
    assert lc.agent_client_params(cfg)["base_url"] == "http://localhost:11434/v1"


def test_agent_params_no_role_is_ollama_fallback():
    assert lc.agent_client_params({"llm": {"judge_model": "qwen3:8b"}})["model"] == "qwen3:8b"
