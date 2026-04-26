#!/usr/bin/env bash
# Launch ik_llama.cpp's llama-server with speculative decoding for qwen3.6:27b.
#
# Reuses Ollama's GGUF blobs directly (no re-download).
# Run from this directory:  ./run_llama_server.sh
#
# Tweak knobs:
#   CTX        — runtime context length (KV is allocated up front; bigger = slower attention)
#   CTX_DRAFT  — draft model context (smaller is fine; speculative bursts are short)
#   PORT       — server port (must match LLAMA_SERVER_URL in .env)

set -euo pipefail

# Load .env (next to this script) if present, so knobs like SPEC, CTX, PORT,
# MAIN_GGUF, etc. can live alongside the rest of the project config.
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  set +u
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set -u
  set +a
fi

LLAMA_BIN="${LLAMA_BIN:-$HOME/llama_cpp/build/bin/llama-server}"
MODELS="${MODELS:-$HOME/llama_models}"

# Main: unsloth Qwen3.6-27B Q4_K_M (16 GB) — must be the unsloth release;
# the Ollama-shipped qwen3.6:27b blob has incompatible tensor naming.
MAIN_GGUF="${MAIN_GGUF:-$MODELS/Qwen3.6-27B-Q4_K_M.gguf}"
# Draft: unsloth Qwen3.5-0.8B Q4_K_M (~500 MB).
# NOTE: speculative decoding is *disabled* by default below — measured 10.17 tok/s
# vs 12.03 tok/s baseline on M4 Pro, even with 100% draft acceptance. On Apple
# Silicon, draft inference serializes on the same GPU as the main model rather
# than overlapping (as it would on CUDA), so the draft cost isn't amortized.
# Re-enable by adding `-md "$DRAFT_GGUF" -ngld 99 -cd 8192 --draft-max 12 --draft-min 3 --draft-p-min 0.6` below.
DRAFT_GGUF="${DRAFT_GGUF:-$MODELS/Qwen3.5-0.8B-Q4_K_M.gguf}"

CTX="${CTX:-32768}"
PORT="${PORT:-8081}"

# SPEC: speculative-decoding strategy with no draft model.
#   ""         (default) — no speculation
#   "default"  — --spec-default preset (ngram-mod, n=24, draft 48..64). PR #19164.
#   "ngram-mod","ngram-cache","ngram-simple","ngram-map-k","ngram-map-k4v"
#              — passed through as --spec-type <value>
SPEC="${SPEC:-}"
SPEC_ARGS=()
case "$SPEC" in
  "")            ;;
  default)       SPEC_ARGS+=(--spec-default) ;;
  ngram-mod|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v)
                 SPEC_ARGS+=(--spec-type "$SPEC") ;;
  *) echo "Unknown SPEC=$SPEC" >&2; exit 1 ;;
esac

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "llama-server not found at $LLAMA_BIN — build ik_llama.cpp first." >&2
  exit 1
fi
if [[ ! -f "$MAIN_GGUF" ]]; then
  echo "Main GGUF missing: $MAIN_GGUF" >&2
  exit 1
fi

# --- AgentSearch (https://github.com/brcrusoe72/agent-search) -------------
# Bring up the dockerized search service alongside llama-server. Skip if the
# repo isn't cloned, if AGENT_SEARCH_AUTOSTART=0, or if it's already running.
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
    echo "[agent-search] $AGENT_SEARCH_DIR not found — skipping. Clone with:" >&2
    echo "    git clone https://github.com/brcrusoe72/agent-search.git \"$AGENT_SEARCH_DIR\"" >&2
  elif ! command -v docker >/dev/null 2>&1; then
    echo "[agent-search] docker not found — skipping. Install OrbStack or Docker Desktop." >&2
  else
    echo "[agent-search] starting in $AGENT_SEARCH_DIR …"
    (cd "$AGENT_SEARCH_DIR" && docker compose up -d)
    _AS_STARTED_BY_US=1
    # Wait up to ~30s for /health.
    for _ in $(seq 1 30); do
      if curl -fsS -o /dev/null --max-time 2 "$AGENT_SEARCH_URL/health"; then
        echo "[agent-search] ready at $AGENT_SEARCH_URL"
        break
      fi
      sleep 1
    done
  fi
fi

# Run llama-server in foreground (not exec) so the EXIT trap fires.
"$LLAMA_BIN" \
  -m  "$MAIN_GGUF" \
  -ngl 99 \
  -c "$CTX" \
  -fa on -ctk q8_0 -ctv q8_0 \
  --jinja \
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
  --host 127.0.0.1 --port "$PORT"
