#!/bin/bash
# Foreground, single-instance launcher for the local 35B MLX reasoning model (Tier-1 of the cascade).
# Thin, memory-gated wrapper around the user's existing ~/models/mlx/start-mlx.sh (reuse-first):
# it adds a free-RAM headroom pre-check and a single-instance guard, then execs the real launcher.
#
# Schabasch's `deep_reasoning` role (config/profile.yaml llm.roles.deep_reasoning) targets
# http://localhost:8082/v1 — the port this script serves. NEVER run two instances (double model-load
# on a constrained Mac); NEVER auto-spawn from the app. Run it yourself, foreground, single-instance.
#
#   bash scripts/serve_mlx.sh          # serves on :8082 (matches profile.yaml)
#   PORT=8083 bash scripts/serve_mlx.sh
set -euo pipefail

PORT="${PORT:-8082}"
LAUNCHER="$HOME/models/mlx/start-mlx.sh"
HARD_FLOOR="${SCHABASCH_MEMORY_HARD_FLOOR_PCT:-10}"

if [[ ! -x "$LAUNCHER" && ! -f "$LAUNCHER" ]]; then
  echo "ERROR: launcher not found at $LAUNCHER (expected the user's MLX start script)." >&2
  exit 1
fi

# Single-instance: refuse if something already listens on the port.
if lsof -i ":$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "ERROR: port $PORT already in use — an MLX server (or another process) is already running." >&2
  echo "       Reuse it, or pick a free PORT. Refusing to double-load the model." >&2
  exit 1
fi

# Memory headroom pre-check (macOS): the 35B model balloons ~10-12 GB on first request. Refuse below
# the hard floor so we never start a swap death spiral (mirrors schabasch/memory_guard.require_headroom).
if [[ "$(uname -s)" == "Darwin" ]]; then
  FREE_PCT="$(memory_pressure -Q 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p' | head -1 || true)"
  if [[ -n "${FREE_PCT:-}" && "$FREE_PCT" -lt "$HARD_FLOOR" ]]; then
    echo "ERROR: only ${FREE_PCT}% RAM free (hard floor ${HARD_FLOOR}%). Close heavy apps and retry." >&2
    exit 1
  fi
  echo "memory headroom OK (${FREE_PCT:-?}% free); loading 35B MLX on :$PORT (foreground, single-instance)…"
fi

PORT="$PORT" exec bash "$LAUNCHER"
