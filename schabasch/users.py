"""Multi-user isolation layer: one overlay YAML + one SQLite DB per user.

The user "default" is config/profile.yaml unchanged (100% backward compatible).
Any other user is a thin overlay at config/users/<key>.yaml merged over the base config;
its mutable paths (db, slate_dir, model_dir, golden_csv) are FORCED to data/users/<key>/…
unless the overlay overrides them — a shared model_dir would silently overwrite another
user's trained triage model.

Identity comes from Telegram: by_telegram_id() maps a chat id to the user key. The bot
fetches the registry via GET /users.json and passes user=<key> on every call.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from . import config

DEFAULT_USER = "default"
USERS_DIR = config.ROOT / "config" / "users"
MAX_USERS = 2   # registration cap ("for now") — the one knob to raise later
_RESERVED_KEYS = {DEFAULT_USER, "example"}

# per-user mutable artifacts — forced under data/users/<key>/ for non-default users
# (agent_workdir stays shared: scratch only, nothing user-owned lands there)
_PER_USER_PATHS = {
    "db": "{key}.sqlite3",
    "golden_csv": "golden/golden.csv",
    "slate_dir": "slates",
    "model_dir": "models",
}

_USER_KEY_RE_HELP = "letters/digits/dash/underscore"


def _valid_key(key: str) -> bool:
    return bool(key) and key.replace("-", "").replace("_", "").isalnum()


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge — overlay wins; nested dicts merge instead of replace."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def user_file(key: str) -> Path:
    return USERS_DIR / f"{key}.yaml"


def list_users() -> list[str]:
    """All user keys: 'default' (the base config) + every config/users/*.yaml stem."""
    extra = sorted(p.stem for p in USERS_DIR.glob("*.yaml")) if USERS_DIR.is_dir() else []
    return [DEFAULT_USER] + [k for k in extra if k != DEFAULT_USER and k != "example"]


def load(key: str | None = None) -> dict[str, Any]:
    """Per-user cfg. 'default'/None → config.load() verbatim; else overlay-merge + forced paths."""
    if key in (None, "", DEFAULT_USER):
        return config.load()
    if not _valid_key(key):
        raise ValueError(f"invalid user key {key!r} (use {_USER_KEY_RE_HELP})")
    path = user_file(key)
    if not path.is_file():
        raise FileNotFoundError(f"unknown user {key!r}: {path} does not exist")
    with path.open("r", encoding="utf-8") as f:
        overlay = yaml.safe_load(f) or {}
    if not isinstance(overlay, dict):
        raise ValueError(f"user overlay {path} must be a YAML mapping")
    cfg = _deep_merge(config.load(), overlay)
    # Force per-user artifact paths (overlay paths.* still wins). config.load() already
    # absolutized base paths, so only rewrite the ones the overlay did NOT set.
    overlay_paths = overlay.get("paths") or {}
    for name, rel in _PER_USER_PATHS.items():
        if name not in overlay_paths:
            cfg["paths"][name] = str(config.ROOT / "data" / "users" / key / rel.format(key=key))
        else:
            cfg["paths"][name] = str(config.ROOT / overlay_paths[name])
    # re-absolutize any other overlay-relative paths the same way config.load() does
    for name, v in overlay_paths.items():
        if name not in _PER_USER_PATHS:
            cfg["paths"][name] = str(config.ROOT / v)
    return cfg


def by_telegram_id(chat_id: int) -> str | None:
    """User key whose telegram.chat_id matches, else None. chat_id 0/None never matches
    (0 is the base config's auto-lock sentinel, not an identity)."""
    if not chat_id:
        return None
    for key in list_users():
        try:
            cfg = load(key)
        except (FileNotFoundError, ValueError):
            continue
        if int((cfg.get("telegram") or {}).get("chat_id") or 0) == int(chat_id):
            return key
    return None


def registry() -> list[dict[str, Any]]:
    """[{user, chat_id}] for GET /users.json — the bot's telegram_id→user map."""
    out = []
    for key in list_users():
        try:
            cfg = load(key)
        except (FileNotFoundError, ValueError):
            continue
        out.append({"user": key, "chat_id": int((cfg.get("telegram") or {}).get("chat_id") or 0)})
    return out


# ── programmatic user creation (Telegram self-registration) ───────────────────────────────────

def derive_key(name: str, chat_id: int) -> str:
    """A valid, unique user key from a Telegram display name: 'Маша К.' → 'masha-k' style ASCII
    slug; non-ASCII/empty names fall back to 'u<chat_id>'; collisions get a numeric suffix."""
    slug = re.sub(r"[^a-z0-9-_]+", "-", (name or "").lower()).strip("-")[:24]
    if not _valid_key(slug) or slug in _RESERVED_KEYS:
        slug = f"u{chat_id}"
    taken = set(list_users())
    key, n = slug, 2
    while key in taken or key in _RESERVED_KEYS:
        key, n = f"{slug}{n}", n + 1
    return key


def create_user(key: str, overlay: dict) -> Path:
    """Write config/users/<key>.yaml + materialize the user's DB. Fail-loud on: invalid/reserved
    key, existing user, duplicate telegram.chat_id, or the MAX_USERS cap. Returns the yaml path."""
    from . import db   # local import — users.py stays importable without sqlite side effects
    if not _valid_key(key) or key in _RESERVED_KEYS:
        raise ValueError(f"invalid user key {key!r} (use {_USER_KEY_RE_HELP})")
    if user_file(key).is_file():
        raise FileExistsError(f"user {key!r} already exists")
    if len(list_users()) >= MAX_USERS:
        raise PermissionError(f"user cap reached ({MAX_USERS})")
    chat_id = int((overlay.get("telegram") or {}).get("chat_id") or 0)
    if not chat_id:
        raise ValueError("overlay must carry a non-zero telegram.chat_id (the identity key)")
    existing = by_telegram_id(chat_id)
    if existing:
        raise FileExistsError(f"chat_id {chat_id} is already registered as {existing!r}")
    USERS_DIR.mkdir(parents=True, exist_ok=True)
    path = user_file(key)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(overlay, f, allow_unicode=True, sort_keys=False)
    db.connect(load(key)["paths"]["db"]).close()   # mkdirs data/users/<key>/ + full schema
    return path


def update_overlay(key: str, patch: dict) -> None:
    """Deep-merge `patch` into an existing overlay yaml (registration's post-extraction rewrite)."""
    path = user_file(key)
    if not path.is_file():
        raise FileNotFoundError(f"unknown user {key!r}")
    with path.open("r", encoding="utf-8") as f:
        overlay = yaml.safe_load(f) or {}
    overlay = _deep_merge(overlay, patch)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(overlay, f, allow_unicode=True, sort_keys=False)


def delete_user(key: str) -> None:
    """Remove a user's overlay yaml + data/users/<key>/ dir (registration rollback). The default
    user is never deletable."""
    if key in _RESERVED_KEYS or not _valid_key(key):
        raise ValueError(f"refusing to delete {key!r}")
    user_file(key).unlink(missing_ok=True)
    data_dir = config.ROOT / "data" / "users" / key
    if data_dir.is_dir():
        shutil.rmtree(data_dir)
