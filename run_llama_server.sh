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

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "llama-server not found at $LLAMA_BIN — build ik_llama.cpp first." >&2
  exit 1
fi
if [[ ! -f "$MAIN_GGUF" ]]; then
  echo "Main GGUF missing: $MAIN_GGUF" >&2
  exit 1
fi
exec "$LLAMA_BIN" \
  -m  "$MAIN_GGUF" \
  -ngl 99 \
  -c "$CTX" \
  -fa on -ctk q8_0 -ctv q8_0 \
  --jinja \
  --host 127.0.0.1 --port "$PORT"
