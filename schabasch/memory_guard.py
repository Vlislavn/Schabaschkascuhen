"""Memory safeguards — headroom gates + a runtime watchdog (Apple-silicon swap safety).

VENDORED from the user's IVAI repo, verbatim in spirit:
  /Users/vladnikulin/code/personal/IVAI/src/modules/core/memory_guard.py
Two real incidents motivated the original (a 47 GB zombie that froze the machine; a kernel panic
from swap exhaustion) — both UNGUARDED heavyweight loads (a local model server ballooning 20–30 GB
on first request). This is the same production-grade guard, adapted for schabasch's UI `/fetch`
worker which loads qwen3:8b (normalize/judge) + bge models (rerank) + an optional 35B MLX server.

Changes from the source: stdlib ``logging`` (no IVAI logging_config dep); env prefix
``SCHABASCH_MEMORY_*``; ``configure_from_cfg(cfg)`` maps the ``memory:`` block in profile.yaml onto
the env-tunable thresholds before ``start_watchdog()``.

  * ``require_headroom(context)`` — call BEFORE loading anything heavy. Raises
    ``MemoryHeadroomError`` below the hard floor (default 10% free RAM).
  * ``start_watchdog()`` — daemon sampler; below the soft floor it warns, below the hard floor it
    sets ``memory_critical`` so heavy call sites can refuse new work with a clear error instead of
    diving into the macOS compressor death spiral. The flag clears automatically on recovery.

Thresholds use free-RAM PERCENTAGE (absolute swap is NOT a usable macOS signal). Stdlib-only by
design. A guard must never break the app: when the platform probe is unavailable (non-darwin/linux,
sandboxed subprocess) every gate degrades to a no-op after one warning log — deliberate, load-bearing.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger("schabasch.memory_guard")

_SOFT_FLOOR_PCT_DEFAULT = 20  # warn: heavy work will start failing soon
_HARD_FLOOR_PCT_DEFAULT = 10  # refuse new heavy work (bench kill floor is 12)
_WATCHDOG_INTERVAL_SECONDS_DEFAULT = 15.0
# Swap GROWTH-delta floor (MB gained between two watchdog samples) that flags memory pressure.
# CAPA: the features stage deadlocked at 86% swap while macOS free-RAM% read a healthy 84% — the
# free%-only signal is blind to compressor/swap saturation. Per the project doctrine ("free% +
# growth-delta, never ABSOLUTE swap") a rapidly-GROWING swap is the leading indicator the free%
# floor misses. Env SCHABASCH_MEMORY_SWAP_GROWTH_MB; config memory.swap_growth_floor_mb; 0 = off.
_SWAP_GROWTH_FLOOR_MB_DEFAULT = 512

_MEMORY_PRESSURE_RE = re.compile(r"free percentage: (\d+)%")
_SWAP_USED_RE = re.compile(r"used\s*=\s*([\d.]+)M")

_state_lock = threading.Lock()
_memory_critical = False
_memory_pressured = False  # free RAM below the SOFT floor (background-work backoff)
_watchdog_thread: threading.Thread | None = None
_probe_warned = False


class MemoryHeadroomError(RuntimeError):
    """Raised when a heavyweight load is requested without memory headroom."""


@dataclass(frozen=True)
class MemorySnapshot:
    free_pct: float | None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw.isdigit() else default


def _guard_enabled() -> bool:
    return os.environ.get("SCHABASCH_MEMORY_GUARD", "1").strip().lower() not in {"0", "false", "off", "no"}


def configure_from_cfg(cfg: dict) -> None:
    """Map the ``memory:`` block in profile.yaml onto the env-tunable thresholds (env wins if set).

    Lets the user tune floors per-machine via config without exporting env vars; an explicit env
    var still overrides (operator escape hatch). Call once before ``start_watchdog()``.
    """
    mem = cfg.get("memory") or {}
    pairs = {
        "SCHABASCH_MEMORY_HARD_FLOOR_PCT": mem.get("hard_floor_pct"),
        "SCHABASCH_MEMORY_SOFT_FLOOR_PCT": mem.get("soft_floor_pct"),
        "SCHABASCH_MEMORY_GUARD_INTERVAL_SECONDS": mem.get("watchdog_interval_seconds"),
        "SCHABASCH_MEMORY_SWAP_GROWTH_MB": mem.get("swap_growth_floor_mb"),
    }
    for env_name, value in pairs.items():
        if value is not None and env_name not in os.environ:
            os.environ[env_name] = str(value)
    if mem.get("guard_enabled") is False and "SCHABASCH_MEMORY_GUARD" not in os.environ:
        os.environ["SCHABASCH_MEMORY_GUARD"] = "0"


def _system_free_pct() -> float | None:
    """System free-RAM percentage, or ``None`` when the platform probe fails.

    darwin: ``memory_pressure -Q``; linux: ``MemAvailable/MemTotal`` from /proc/meminfo.
    """
    global _probe_warned
    try:
        if os.uname().sysname == "Darwin":
            out = subprocess.run(
                ["memory_pressure", "-Q"], capture_output=True, text=True, timeout=5, check=True
            ).stdout
            match = _MEMORY_PRESSURE_RE.search(out)
            return float(match.group(1)) if match else None
        meminfo = {}
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                meminfo[key] = int(rest.split()[0])
        return 100.0 * meminfo["MemAvailable"] / meminfo["MemTotal"]
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, AttributeError) as exc:
        # Authorized boundary: production safeguards must not themselves break the app. Narrow:
        # tool missing / non-zero exit / timeout, /proc parse drift, no os.uname on non-unix. Once.
        if not _probe_warned:
            _probe_warned = True
            log.warning("memory_guard probe unavailable (%s: %s); guards degrade to no-ops",
                        type(exc).__name__, exc)
        return None


def _swap_used_mb() -> float | None:
    """Swap currently in use (MB) on darwin via ``sysctl -n vm.swapusage``; ``None`` elsewhere/on
    probe failure. Used ONLY for a growth-delta (see ``_swap_growth_exceeded``) — never as an
    absolute floor (absolute swap is not a trustworthy macOS signal)."""
    try:
        if os.uname().sysname != "Darwin":
            return None
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5, check=True
        ).stdout
        match = _SWAP_USED_RE.search(out)
        return float(match.group(1)) if match else None
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def _swap_growth_exceeded(prev_mb: float | None, cur_mb: float | None, floor_mb: int) -> bool:
    """True when swap GREW by more than ``floor_mb`` since the previous sample (``floor_mb`` <= 0 =
    off). Growth-delta — not the absolute level — is the trustworthy macOS swap-pressure signal: the
    OS keeps free-RAM% high while compressing/swapping hard, so a rapidly-growing swap is the leading
    indicator the free% floor misses (CAPA: the features deadlock fired at 86% swap / 84% free)."""
    if floor_mb <= 0 or prev_mb is None or cur_mb is None:
        return False
    return (cur_mb - prev_mb) > floor_mb


def snapshot() -> MemorySnapshot:
    return MemorySnapshot(free_pct=_system_free_pct())


def memory_critical() -> bool:
    """True while the watchdog sees free RAM below the hard floor."""
    with _state_lock:
        return _memory_critical


def memory_under_pressure(*, probe: bool = False) -> bool:
    """True when free RAM is below the SOFT floor — the background-work backoff.

    ``probe=False`` (default) reads the cheap watchdog-maintained flag — safe per-iteration in a
    tight loop (no subprocess). ``probe=True`` takes a fresh sample (one ``memory_pressure``
    subprocess) for a single pre-flight check. Probe unavailable → ``False`` (no-op).
    """
    if not _guard_enabled():
        return False
    if probe:
        free = _system_free_pct()
        if free is None:
            return False
        soft_floor = _env_int("SCHABASCH_MEMORY_SOFT_FLOOR_PCT", _SOFT_FLOOR_PCT_DEFAULT)
        return free < soft_floor
    with _state_lock:
        return _memory_pressured


def require_headroom(context: str) -> None:
    """Gate a heavyweight load on system free-RAM headroom.

    Raises ``MemoryHeadroomError`` below the hard floor — the caller turns that into a clear
    user-facing error instead of letting the load start a swap death spiral.
    """
    if not _guard_enabled():
        return
    hard_floor = _env_int("SCHABASCH_MEMORY_HARD_FLOOR_PCT", _HARD_FLOOR_PCT_DEFAULT)
    free = _system_free_pct()
    if free is None:
        return
    if free < hard_floor:
        raise MemoryHeadroomError(
            f"Refusing to load {context}: only {free:.0f}% RAM free (hard floor {hard_floor}%). "
            f"Close memory-heavy apps or unload resident model servers, then retry. "
            f"(Override floor via SCHABASCH_MEMORY_HARD_FLOOR_PCT; disable via SCHABASCH_MEMORY_GUARD=0.)"
        )
    soft_floor = _env_int("SCHABASCH_MEMORY_SOFT_FLOOR_PCT", _SOFT_FLOOR_PCT_DEFAULT)
    if free < soft_floor:
        log.warning("memory_guard: loading %s with low headroom (%.0f%% free, soft floor %d%%)",
                    context, free, soft_floor)


def _watchdog_loop(interval_seconds: float, on_critical: "Callable[[MemorySnapshot], None] | None") -> None:
    global _memory_critical, _memory_pressured
    hard_floor = _env_int("SCHABASCH_MEMORY_HARD_FLOOR_PCT", _HARD_FLOOR_PCT_DEFAULT)
    soft_floor = _env_int("SCHABASCH_MEMORY_SOFT_FLOOR_PCT", _SOFT_FLOOR_PCT_DEFAULT)
    swap_floor = _env_int("SCHABASCH_MEMORY_SWAP_GROWTH_MB", _SWAP_GROWTH_FLOOR_MB_DEFAULT)
    soft_warned = False
    prev_swap: float | None = None
    while True:
        free = _system_free_pct()
        cur_swap = _swap_used_mb()
        swap_growing = _swap_growth_exceeded(prev_swap, cur_swap, swap_floor)
        prev_swap = cur_swap
        if free is not None:
            with _state_lock:
                _memory_pressured = (free < soft_floor) or swap_growing   # CAPA: free% is swap-blind
            if swap_growing:
                log.warning("memory_guard: swap grew >%dMB/sample (now %.0fMB) — flagging memory "
                            "pressure; macOS free-RAM%% (%.0f%%) is blind to swap saturation",
                            swap_floor, cur_swap or 0.0, free)
            if free < hard_floor:
                with _state_lock:
                    entered_critical = not _memory_critical
                    _memory_critical = True
                if entered_critical:
                    log.critical(
                        "memory_guard: %.0f%% RAM free (< hard floor %d%%) — refusing new heavy "
                        "work until memory recovers", free, hard_floor)
                    if on_critical is not None:
                        on_critical(MemorySnapshot(free_pct=free))
            else:
                with _state_lock:
                    recovered = _memory_critical
                    _memory_critical = False
                if recovered:
                    log.warning("memory_guard: recovered (%.0f%% RAM free) — heavy work re-enabled", free)
                if free < soft_floor and not soft_warned:
                    soft_warned = True
                    log.warning("memory_guard: %.0f%% RAM free (< soft floor %d%%)", free, soft_floor)
                elif free >= soft_floor:
                    soft_warned = False
        threading.Event().wait(interval_seconds)


def start_watchdog(on_critical: "Callable[[MemorySnapshot], None] | None" = None) -> bool:
    """Start the daemon watchdog (idempotent). Returns True when running."""
    global _watchdog_thread
    if not _guard_enabled():
        return False
    if _system_free_pct() is None:
        return False
    with _state_lock:
        if _watchdog_thread is not None and _watchdog_thread.is_alive():
            return True
        interval_raw = os.environ.get("SCHABASCH_MEMORY_GUARD_INTERVAL_SECONDS", "").strip()
        try:
            interval = float(interval_raw) if interval_raw else _WATCHDOG_INTERVAL_SECONDS_DEFAULT
        except ValueError:
            interval = _WATCHDOG_INTERVAL_SECONDS_DEFAULT
        _watchdog_thread = threading.Thread(
            target=_watchdog_loop, args=(interval, on_critical),
            name="schabasch-memory-guard", daemon=True,  # never block interpreter teardown
        )
        _watchdog_thread.start()
    log.info("memory_guard watchdog started (interval %.0fs)", interval)
    return True


def _reset_for_tests() -> None:
    global _memory_critical, _memory_pressured, _watchdog_thread, _probe_warned
    with _state_lock:
        _memory_critical = False
        _memory_pressured = False
        _watchdog_thread = None
        _probe_warned = False
