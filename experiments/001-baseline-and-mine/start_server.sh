#!/bin/bash
# Experiment inference server.
# Context 20480, max 4 sequences, no tool-call parser. Thinking stays on.
# The binary is the uv environment at /content/.venv. A logits processor is
# added only when VLLM_LOGITS_PROCESSORS is set, so rounds that do not set
# it keep the same decoding path as 001.
set -euo pipefail
cd /content
export PYTHONPATH="/content/cut-swift-rev-crack${PYTHONPATH:+:$PYTHONPATH}"
# FlashInfer JIT calls `ninja` and `nvcc` by name. Both exist, but neither
# is on the default PATH of a fresh shell.
export PATH="/content/.venv/bin:/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
VLLM_BIN="${VLLM_BIN:-/content/.venv/bin/vllm}"
if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vLLM binary not found: $VLLM_BIN" >&2
  exit 1
fi
args=(
  serve /content/models/ornith-1.5-9B
  --served-model-name Ornith-1.5-9B
  --host 0.0.0.0
  --port 8000
  --max-model-len 20480
  --gpu-memory-utilization 0.90
  --max-num-seqs 4
  --enable-prefix-caching
  --reasoning-parser qwen3
  --trust-remote-code
  --default-chat-template-kwargs '{"enable_thinking": true}'
)
if [[ -n "${VLLM_LOGITS_PROCESSORS:-}" ]]; then
  args+=(--logits-processors "$VLLM_LOGITS_PROCESSORS")
fi
exec "$VLLM_BIN" "${args[@]}"
