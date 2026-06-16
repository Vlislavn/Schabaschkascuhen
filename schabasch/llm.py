"""Локальный LLM-клиент (ollama) + центральный retry-враппер.

Паттерны: claude-code/retry-backoff (слоистый retry: классификация → backoff с
джиттером), claude-code/error-envelope (закрытый enum + lossless details).
Один клиент на Normalizer и Judge; ручки моделей раздельные (profile.yaml).
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests

from .models import ErrorClass

OLLAMA_URL = "http://localhost:11434"


class LLMError(Exception):
    def __init__(self, error_class: ErrorClass, details: str):
        self.error_class = error_class
        self.details = details
        super().__init__(f"{error_class.value}: {details[:200]}")


_RETRYABLE = {ErrorClass.TIMEOUT, ErrorClass.ENDPOINT_DOWN, ErrorClass.EMPTY_OUTPUT,
              ErrorClass.INVALID_JSON}


def with_retry(fn: Callable[[], Any], *, attempts: int = 3, base_delay: float = 2.0) -> Any:
    """Капнутый экспоненциальный backoff + 0–25% джиттер; ретраим только retryable-классы."""
    last: LLMError | None = None
    for i in range(attempts):
        try:
            return fn()
        except LLMError as e:
            last = e
            if e.error_class not in _RETRYABLE or i == attempts - 1:
                raise
            delay = min(base_delay * (2 ** i), 30.0)
            time.sleep(delay * (1 + random.random() * 0.25))
    raise last  # pragma: no cover


@dataclass
class OllamaClient:
    model: str
    num_ctx: int = 8192
    temperature: float = 0.1
    timeout_s: int = 120

    def model_digest(self) -> str | None:
        """Digest модели для пиннинга grader-tuple (are/pinned-judge-model)."""
        try:
            r = requests.post(f"{OLLAMA_URL}/api/show", json={"model": self.model}, timeout=10)
            r.raise_for_status()
            d = r.json()
            return (d.get("details", {}) or {}).get("parent_model") or d.get("digest") or None
        except Exception:
            return None

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        """Один вызов → валидный JSON-объект. Бросает LLMError с классом."""

        def _call() -> dict[str, Any]:
            try:
                r = requests.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": self.model,
                        "stream": False,
                        "format": "json",
                        "think": False,
                        "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    },
                    timeout=self.timeout_s,
                )
            except requests.Timeout as e:
                raise LLMError(ErrorClass.TIMEOUT, str(e)) from e
            except requests.ConnectionError as e:
                raise LLMError(ErrorClass.ENDPOINT_DOWN, str(e)) from e
            if r.status_code != 200:
                raise LLMError(ErrorClass.HTTP_ERROR, f"HTTP {r.status_code}: {r.text[:500]}")
            content = (r.json().get("message") or {}).get("content") or ""
            if not content.strip():
                raise LLMError(ErrorClass.EMPTY_OUTPUT, "empty content")
            try:
                obj = json.loads(content)
            except json.JSONDecodeError as e:
                raise LLMError(ErrorClass.INVALID_JSON, f"{e}: {content[:500]}") from e
            if not isinstance(obj, dict):
                raise LLMError(ErrorClass.SCHEMA_VIOLATION, f"not an object: {content[:200]}")
            return obj

        return with_retry(_call)


def http_get_json(url: str, *, headers: dict | None = None, params: dict | None = None,
                  timeout_s: int = 30, attempts: int = 3) -> dict | list:
    """Общий сетевой GET c тем же retry-враппером (для Arbeitsagentur и пр.)."""

    def _call():
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout_s)
        except requests.Timeout as e:
            raise LLMError(ErrorClass.TIMEOUT, str(e)) from e
        except requests.ConnectionError as e:
            raise LLMError(ErrorClass.ENDPOINT_DOWN, str(e)) from e
        if r.status_code in (404, 410):
            raise LLMError(ErrorClass.HTTP_ERROR, f"HTTP {r.status_code} (permanent)")
        if r.status_code != 200:
            raise LLMError(ErrorClass.HTTP_ERROR, f"HTTP {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except json.JSONDecodeError as e:
            raise LLMError(ErrorClass.INVALID_JSON, str(e)) from e

    return with_retry(_call, attempts=attempts)


_BROWSER_UA = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")}


def http_get_status(url: str, *, timeout_s: int = 10, headers: dict | None = None) -> int | None:
    """Raw HTTP status for a deterministic liveness check (no JSON parse, no retry). Returns the
    status code, or None on timeout / connection error — i.e. 'could NOT verify', which the caller
    must treat as UNKNOWN, never as 'closed'. A browser-like UA reduces trivial bot-walls; redirects
    are followed so a removed posting surfaces its real 404/410 instead of a redirect shell."""
    try:
        r = requests.get(url, headers=headers or _BROWSER_UA, timeout=timeout_s, allow_redirects=True)
        return r.status_code
    except requests.RequestException:
        return None
