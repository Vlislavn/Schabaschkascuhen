"""Cascade LLM clients — OpenAI-compatible endpoints alongside the frozen ollama client.

`schabasch/llm.py` is a FROZEN contract (its public signatures must not change), so this module
is ADDITIVE: it reuses `OllamaClient` / `LLMError` / `with_retry` as-is and adds

  * ``OpenAIClient`` — a requests-based client for ANY OpenAI-compatible ``/chat/completions``
    endpoint (mlx_lm.server for the local Qwen3.6-35B-OptiQ-4bit, api.kather.ai `sota`), mirroring
    ``OllamaClient.chat_json(system, user) -> dict`` / ``model_digest() -> str | None`` so callers
    are client-agnostic.
  * ``make_llm_client(cfg, role)`` — a config-driven router (``cfg["llm"]["roles"][role]``) that
    returns the right client per role (normalizer / judge / candidate / agent / deep_reasoning /
    sota). Defaults keep every role on ollama qwen3:8b, so existing behaviour is unchanged unless
    a role is explicitly opted in.

Request shape mirrors prototype-internal-KL ``src/kl_agent_builder/llm.py`` (LLMClient): POST
``{base_url}/chat/completions`` with ``Authorization: Bearer``; the answer is
``choices[0].message.content`` with a ``reasoning_content`` fallback (reasoning models emit the
final JSON in content but can leave it empty at low max_tokens — global gotcha). Secrets are read
from an env var (``api_key_env``) at call-build time; they are NEVER stored in the repo or config.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .llm import LLMError, OllamaClient, with_retry
from .models import ErrorClass

# Roles whose ollama fallback model comes from a specific legacy llm.* key (back-compat: when a role
# is not configured under llm.roles we behave exactly like before — ollama with the old model knob).
_ROLE_FALLBACK_KEY = {
    "normalizer": "normalizer_model",
    "candidate": "normalizer_model",
    "judge": "judge_model",
    "deep_reasoning": "judge_model",
    "sota": "judge_model",
    "agent": "judge_model",
}

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_json_obj(text: str) -> dict | None:
    """Parse a JSON object from model output, tolerating <think> blocks / ``` fences / prose."""
    s = (text or "").strip()
    s = _THINK_RE.sub("", s)
    s = re.sub(r"<think>.*$", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"^```(?:json)?\s*", "", s.strip())
    s = re.sub(r"\s*```\s*$", "", s).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJ_RE.search(s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


@dataclass
class OpenAIClient:
    """OpenAI-compatible chat client (mlx_lm.server / api.kather.ai). Interface-compatible with
    ``OllamaClient``: ``chat_json(system, user) -> dict`` raising classified ``LLMError``."""

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = field(default="", repr=False)
    num_ctx: int = 8192          # accepted for parity; not part of the OpenAI request body
    temperature: float = 0.1
    timeout_s: int = 300         # remote SOTA / a 35B local model can be slow on first load
    max_tokens: int = 16000      # large: reasoning models burn tokens before the final JSON
    label: str = "openai"        # provenance tag surfaced as model_used on the card

    def model_digest(self) -> str | None:
        # OpenAI-compatible endpoints don't expose a content digest; the grader-tuple pins the
        # model NAME separately, so None is the honest value (never fabricate a digest).
        return None

    def _post(self, body: dict) -> requests.Response:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            return requests.post(f"{self.base_url}/chat/completions", json=body,
                                 headers=headers, timeout=self.timeout_s)
        except requests.Timeout as e:
            raise LLMError(ErrorClass.TIMEOUT, str(e)) from e
        except requests.ConnectionError as e:
            raise LLMError(ErrorClass.ENDPOINT_DOWN, str(e)) from e

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        def _call() -> dict[str, Any]:
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            }
            r = self._post(body)
            # Some local servers (older mlx_lm) reject response_format → retry once without it.
            if r.status_code in (400, 422) and "response_format" in (r.text or "").lower():
                body.pop("response_format", None)
                r = self._post(body)
            if r.status_code != 200:
                raise LLMError(ErrorClass.HTTP_ERROR, f"HTTP {r.status_code}: {r.text[:500]}")
            try:
                data = r.json()
            except json.JSONDecodeError as e:
                raise LLMError(ErrorClass.INVALID_JSON, str(e)) from e
            msg = ((data.get("choices") or [{}])[0] or {}).get("message") or {}
            content = (msg.get("content") or msg.get("reasoning_content") or "").strip()
            if not content:
                raise LLMError(ErrorClass.EMPTY_OUTPUT, "empty content")
            obj = _extract_json_obj(content)
            if obj is None:
                raise LLMError(ErrorClass.INVALID_JSON, content[:500])
            return obj

        return with_retry(_call)


def _read_env_file_key(path: str, var: str) -> str | None:
    """Read ONE key from a .env file, stripping an inline ` # comment` from an unquoted value.

    Deliberately minimal (no full dotenv parsing): the user points at prototype-internal-KL/.env
    so the kather key resolves without exporting it. Quoted values keep their content verbatim.
    """
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.strip() != var:
                continue
            val = val.strip()
            if val[:1] in {'"', "'"}:
                quote = val[0]
                end = val.find(quote, 1)
                return (val[1:end] if end > 0 else val[1:]) or None
            for sep in ("  #", " #", "\t#"):
                idx = val.find(sep)
                if idx >= 0:
                    val = val[:idx]
            return val.strip() or None
    except OSError:
        return None
    return None


def _resolve_key(cfg: dict, role_cfg: dict) -> str:
    """Resolve an API key for an openai role: explicit api_key, else env var, else an env_file."""
    if role_cfg.get("api_key"):
        return str(role_cfg["api_key"])
    env_var = role_cfg.get("api_key_env")
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val
        env_file = role_cfg.get("env_file") or (cfg.get("llm") or {}).get("env_file")
        if env_file:
            from_file = _read_env_file_key(str(env_file), env_var)
            if from_file:
                return from_file
    return ""


def _role_cfg(cfg: dict, role: str) -> dict:
    return ((cfg.get("llm") or {}).get("roles") or {}).get(role) or {}


def role_available(cfg: dict, role: str) -> bool:
    """True if a role is usable: ollama roles always (local), openai roles need a reachable base
    (localhost → assume up, checked at call time) or a resolvable key (remote). Lets the cascade
    skip an unconfigured/keyless `sota` cleanly instead of failing per-card."""
    rc = _role_cfg(cfg, role)
    if not rc:
        return False
    if (rc.get("client") or "ollama").lower() != "openai":
        return True
    base = (rc.get("base_url") or "").lower()
    if "localhost" in base or "127.0.0.1" in base:
        return True
    return bool(_resolve_key(cfg, rc))


def make_llm_client(cfg: dict, role: str):
    """Return the configured client for a role. Falls back to ollama (legacy llm.* model knob)
    when the role is not declared under llm.roles — so default behaviour is byte-for-byte unchanged.
    """
    llm = cfg.get("llm") or {}
    rc = _role_cfg(cfg, role)
    num_ctx = int(rc.get("num_ctx", llm.get("num_ctx", 8192)))
    temperature = float(rc.get("temperature", llm.get("temperature", 0.1)))
    if (rc.get("client") or "ollama").lower() == "openai":
        return OpenAIClient(
            model=rc.get("model") or "sota",
            base_url=(rc.get("base_url") or "http://localhost:8000/v1").rstrip("/"),
            api_key=_resolve_key(cfg, rc),
            num_ctx=num_ctx,
            temperature=temperature,
            timeout_s=int(rc.get("timeout_s", 300)),
            max_tokens=int(rc.get("max_tokens", 16000)),
            label=role,
        )
    fallback_model = llm.get(_ROLE_FALLBACK_KEY.get(role, "normalizer_model"), "qwen3:8b")
    return OllamaClient(model=rc.get("model") or fallback_model, num_ctx=num_ctx,
                        temperature=temperature)


def _endpoint_reachable(base_url: str, *, timeout: float = 2.0) -> bool:
    """Quick liveness probe for a local model server (GET ``{base_url}/models``).

    Used to fall the agent back to ollama when the opt-in MLX 35B server (:8082) isn't running, so an
    unattended tick degrades gracefully instead of hanging/erroring on a dead endpoint. Any non-5xx
    response (or even a 4xx) means "a server answered" → reachable; only a connection/timeout error is
    treated as down.
    """
    try:
        r = requests.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        return r.status_code < 500
    except requests.RequestException:
        return False


def agent_client_params(cfg: dict) -> dict:
    """Resolve {provider, base_url, model, api_key} for the ReAct agent role (investigate/discover).

    The ``agent`` role may point at the local **MLX 35B** (client openai, localhost:8082) — the
    strongest LOCAL agent (measured sota-grade), but supervised/opt-in (never auto-served). When that
    localhost server is **not reachable** (the common unattended case), fall back to the ollama small
    model (``judge_model``) so a tick still investigates AND never co-loads the 22GB 35B with the bulk
    ollama pool / hangs on a dead :8082. A REMOTE agent role (e.g. api.kather.ai) is used as-is — no
    ping (it's not a co-load/headroom concern and a probe would just add latency).
    """
    llm = cfg.get("llm") or {}
    rc = _role_cfg(cfg, "agent")
    fallback = {
        "provider": "openai",
        "base_url": "http://localhost:11434/v1",
        "model": llm.get("judge_model", "qwen3:8b"),
        "api_key": "ollama",
    }
    if not rc:
        return fallback
    base_url = rc.get("base_url") or "http://localhost:11434/v1"
    params = {
        "provider": rc.get("provider", "openai"),
        "base_url": base_url,
        "model": rc.get("model") or llm.get("judge_model", "qwen3:8b"),
        "api_key": rc.get("api_key") or _resolve_key(cfg, rc) or "ollama",
    }
    # A localhost openai endpoint OTHER than ollama's default port is a supervised model server (the
    # 35B MLX on :8082). If it's down, degrade to the ollama small model rather than erroring/hanging.
    base_low = base_url.lower()
    is_local = ("localhost" in base_low) or ("127.0.0.1" in base_low)
    is_ollama_default = "11434" in base_low
    if (rc.get("client") or "ollama").lower() == "openai" and is_local and not is_ollama_default:
        if not _endpoint_reachable(base_url):
            import logging
            logging.getLogger(__name__).info(
                "agent endpoint %s unreachable — falling back to ollama %s (start the MLX server to "
                "use the 35B agent)", base_url, fallback["model"])
            return fallback
    return params


def client_label(client) -> str:
    """Provenance label for the card's `model_used` field."""
    if isinstance(client, OpenAIClient):
        return client.model
    return getattr(client, "model", "ollama")
