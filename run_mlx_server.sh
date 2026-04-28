#!/usr/bin/env bash
# Launch mlx-lm's mlx_lm.server with a Qwen3.6-27B MLX quant.
#
# Sibling of run_llama_server.sh — switches the inference backend to MLX
# (Apple-native, generally faster prefill on M-series than llama.cpp).
#
# Pull weights first:
#   micromamba run -n internet hf download Brooooooklyn/Qwen3.6-27B-UD-Q5_K_XL-mlx
#
# Run from this directory:  ./run_mlx_server.sh
#
# Tuning knobs (override via env or .env):
#   MLX_MODEL          — HF repo or local path of the MLX model to serve
#   PORT               — server port (must match LLAMA_SERVER_URL in .env)
#   MLX_MAX_TOKENS     — default max_tokens cap (server default 512 is too low)
#   MLX_PREFILL_STEP   — prefill step size (default 2048; lower for less memory pressure)
#   MLX_PROMPT_CACHE_MB — prompt-cache budget in MB (re-uses KV across same-prefix calls)

set -euo pipefail

ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  set +u
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set -u
  set +a
fi

MLX_MODEL="${MLX_MODEL:-Brooooooklyn/Qwen3.6-27B-UD-Q5_K_XL-mlx}"
PORT="${PORT:-8081}"
MLX_MAX_TOKENS="${MLX_MAX_TOKENS:-4096}"
MLX_PREFILL_STEP="${MLX_PREFILL_STEP:-2048}"
MLX_PROMPT_CACHE_MB="${MLX_PROMPT_CACHE_MB:-8192}"
MLX_PROMPT_CACHE_BYTES="$((MLX_PROMPT_CACHE_MB * 1024 * 1024))"
MICROMAMBA_ENV="${MICROMAMBA_ENV:-internet}"

# --- AgentSearch (https://github.com/brcrusoe72/agent-search) -------------
# Same autostart / cleanup as run_llama_server.sh.
AGENT_SEARCH_DIR="${AGENT_SEARCH_DIR:-$HOME/agent-search}"
AGENT_SEARCH_URL="${AGENT_SEARCH_URL:-http://localhost:3939}"
AGENT_SEARCH_AUTOSTART="${AGENT_SEARCH_AUTOSTART:-1}"
_AS_STARTED_BY_US=0

stop_agent_search() {
  if [[ "$_AS_STARTED_BY_US" == "1" ]]; then
    echo "[agent-search] stopping…"
    (cd "$AGENT_SEARCH_DIR" && docker compose stop) || true
  fi
}
trap stop_agent_search EXIT INT TERM

if [[ "$AGENT_SEARCH_AUTOSTART" != "0" ]]; then
  if curl -fsS -o /dev/null --max-time 2 "$AGENT_SEARCH_URL/health"; then
    echo "[agent-search] already running at $AGENT_SEARCH_URL"
  elif [[ ! -d "$AGENT_SEARCH_DIR" ]]; then
    echo "[agent-search] $AGENT_SEARCH_DIR not found — skipping." >&2
  elif ! command -v docker >/dev/null 2>&1; then
    echo "[agent-search] docker not found — skipping." >&2
  else
    echo "[agent-search] starting in $AGENT_SEARCH_DIR …"
    (cd "$AGENT_SEARCH_DIR" && docker compose up -d)
    _AS_STARTED_BY_US=1
    for _ in $(seq 1 30); do
      if curl -fsS -o /dev/null --max-time 2 "$AGENT_SEARCH_URL/health"; then
        echo "[agent-search] ready at $AGENT_SEARCH_URL"
        break
      fi
      sleep 1
    done
  fi
fi

echo "[mlx-server] model=$MLX_MODEL port=$PORT max_tokens=$MLX_MAX_TOKENS"
exec micromamba run -n "$MICROMAMBA_ENV" python -m mlx_lm server \
  --model "$MLX_MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --max-tokens "$MLX_MAX_TOKENS" \
  --prefill-step-size "$MLX_PREFILL_STEP" \
  --prompt-cache-bytes "$MLX_PROMPT_CACHE_BYTES" \
  --temp 0.0
