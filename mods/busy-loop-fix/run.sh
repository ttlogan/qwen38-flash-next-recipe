#!/bin/bash
# busy-loop-fix mod: reduce vLLM shm_broadcast busy_loop_s from 1s to 0.002s.
# Fixes the CPU spin-wait that heats the GB10 SoC under TP>=2 decode.
# Runs inside the container each launch (survives recreation). Idempotent.
set -euo pipefail
PY=/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/shm_broadcast.py
if grep -q "busy_loop_s: float = 0.002" "$PY"; then
  echo "[busy-loop-fix] already 0.002, skipping"
else
  sed -i "s/busy_loop_s: float = 1,/busy_loop_s: float = 0.002,/" "$PY"
  echo "[busy-loop-fix] patched busy_loop_s -> 0.002 in $PY"
fi
grep -n "busy_loop_s: float" "$PY" | head -1
