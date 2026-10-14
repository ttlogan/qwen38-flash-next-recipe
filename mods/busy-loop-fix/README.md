# busy-loop-fix mod

Container-level thermal fix for the GB10 SoC during TP>=2 vLLM decode.

## Problem

Under TP>=2, vLLM's inter-process message queue (`shm_broadcast.py`) **spins for
up to 1 second** (`busy_loop_s: float = 1`) instead of sleeping, holding 3-4
performance cores at 3.9GHz continuously. On the GB10 the CPU and GPU share one
die/heatsink, so this spin-wait is roughly **half the thermal budget inches from
the GPU** — it heats the board (acpitz) to ~85°C even when the GPU is cool.

## Fix

Patch `busy_loop_s` from `1` to `0.002` in the container's vLLM site-packages:

```python
busy_loop_s: float = 1,  ->  busy_loop_s: float = 0.002,
```

## Why 0.002, not 0

`0` forces idle-poll / wakeups and can *raise* CPU. `0.002` is the measured
sweet spot.

## How it's applied

`launch-cluster.sh` / `run-recipe.py` apply a recipe's `mods:` by copying the mod
dir into the container and running its `run.sh` **inside** the container (via
`--apply-mod <path>`). Because it runs on every launch, it survives container
recreation without rebuilding the image. Idempotent: `run.sh` checks whether the
patch is already present and skips if so.

## Measured effect

| Metric | Before | After |
|---|---|---|
| vLLM idle CPU | ~57-79% | **~4%** |
| Board (acpitz) temp | ~85°C | **~51°C** |
| GPU temp | ~70°C | **~48°C** |

## Host-side complement

This fixes the *container* spin-wait. For the host-level CPU governor +
performance-core cap (a separate, boot-persistent systemd fix), see
[`host-cpu-thermal/`](../host-cpu-thermal/README.md). Both are documented in the
repo root [README](../README.md).
