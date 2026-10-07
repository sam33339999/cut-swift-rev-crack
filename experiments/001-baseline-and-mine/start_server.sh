#!/bin/bash
# Experiment 001 inference server.
# Differs from the server that was already running: context 20480 instead of
# 262144, max 4 concurrent sequences, no tool-call parser.
# Thinking stays on. The original command is recorded in PROTOCOL.md.
set -euo pipefail
cd /content
exec /usr/bin/python3 /usr/local/bin/vllm serve /content/models/ornith-1.5-9B \
  --served-model-name Ornith-1.5-9B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 20480 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": true}'
