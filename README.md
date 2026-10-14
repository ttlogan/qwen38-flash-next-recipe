# Qwen3.8-Flash-Next (Uncensored) — 2-node GX10 Serve Recipe

Production recipe for serving **`orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4`**
(native 262K context, NVFP4 experts, MTP speculative decoding) across a **2-node
NVIDIA GB10 / DGX Spark cluster** with **tensor-parallel 2 + expert-parallel**, a
**boot autostart** launcher, and a **bot-autostart / swap** orchestrator.

This repo is the **sanitized configuration + documentation** for a working setup.
No API keys, tokens, passwords, real hostnames, or internal IPs are included
(only the ConnectX fabric interface/ROCE hostnames are kept as the internal
topology — all host/credential values are placeholders).

---

## What's in this repo

| Path | What it is | Source |
|---|---|---|
| `recipes/qwen38-nvfp4-262k.yaml` | The **tuned recipe**: native 262K ctx, TP=2, EP, MTP3, fp8 KV, `qwen3_coder` tool parser | **Own work** (tuned from the b12x + MiaAI patterns) |
| `scripts/qwen38-vllm.sh` | **Per-node launcher** (rank 0/1): direct `podman run -d`, HEAD/worker-first, `/health` poll | **Own work** (adapted from GLM lane) |
| `scripts/swap-model.sh` | **Swap orchestrator**: `--swap <deepseek\|glm\|qwen38>`, idempotent ensure, `/etc/motd` | **Own work** |
| `launcher/.env.example` | Cluster fabric + podman config template (placeholders) | **Own work** |
| `mods/busy-loop-fix/` | Apply-on-launch **thermal fix**: `busy_loop_s` 1 → 0.002 in vLLM `shm_broadcast.py` | **Own work** (same as DS4) |
| `host-cpu-thermal/` | **Host-level** thermal fix: `schedutil` governor + cap X925 P-cores to 2.8GHz | **Own work** |

---

## The model

- **`orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4`** — Qwen3.8-Flash-Next,
  abliterated (uncensored) + NVFP4 quantized, with the **full vision + video
  tower** and **MTP** speculative head preserved. This is the **NVFP4 parent**
  (170.93 GiB on disk, PLE n-gram table kept **bf16**).
  **HF repo:** [https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4)
- **Served name** (API `id`) = `qwen3.8-flash-next-uncensored`.
- Loaded with **`--load-format safetensors`** (not the b12x loader — this is a
  NVFP4/compressed-tensors checkpoint, not the b12x B12xModelLoader path).

> **Why the NVFP4 parent (170.93 GiB) and not the smaller FP8PLE (123 GiB)?**
> The NVFP4 parent keeps the 102 GB PLE table in **bf16** (best quality), and the
> community's own data (`veloGB10`, NVIDIA forum) confirms a 2× GB10 **fits** it
> with the right KV/memory setup. The FP8PLE derivative (PLE → FP8, −47 GiB) is
> the fallback if 170 GiB + 262K context won't fit at your gpu-mem target.
> Both derive from the same uncensored weights; pick based on your memory budget.

---

## Cluster topology (sanitized)

| Role | Host (placeholder) | RoCE IP (placeholder) | Rank |
|---|---|---|---|
| Head (server) | `<HEAD_IP>` | `<HEAD_ROCE_IP>` | 0 |
| Worker | `<WORKER_IP>` | `<WORKER_ROCE_IP>` | 1 |

- 2 nodes, TP=2 (1 GPU per Spark), expert-parallel, API exposed on the **head**
  only (`:8000`); `ENABLE_EXPERT_PARALLEL=true` is **required** for NVFP4.
- The RoCE fabric is the **ConnectX x0** interface (`IB_IF`: the ConnectX-7 port,
  e.g. `rocep1s0f0`), not the management LAN. `MASTER_PORT` on this lane was
  chosen distinct from the DeepSeek lane to allow co-existence if both stacks
  are configured.
- `CONTAINER_RT=podman` (rootless, owned by the deploying user).

---

## Context: why 262K (not 500K / 1M)

This lane is deliberately served at the **native 262,144-token** context. The
community data is unambiguous that stretching past native hurts:

| Context | YaRN factor | Verdict |
|---|---|---|
| **262K (native)** | — | **Default. Vetted, quality-preserving, memory-feasible.** |
| ~390K | 1.5× | Max "workable" stretch per community testing; already degraded. |
| 500K | 2.0× | **Severe degradation** (community-measured), and the MTP draft head **stays at 262K** → Mamba/cache geometry validation failures at 524K+ (reported vLLM bug). |
| 1M | 4.0× | **Not recommended.** Static YaRN degrades short-context quality too (this checkpoint ships `rope_type: default`, never trained on a YaRN basis); the recipe authors' own 256K test hit a token-0 loop; 20-min cold prefill at 1M makes agentic work unusable. |

**Practical rule:** run **262K per thread** and use concurrent subagents /
session compaction for long-horizon work instead of forcing one giant context.
That matches the model's training and avoids the MTP long-context bug entirely.

---

## Tuning (why these values)

Serving parameters for the 180B (6B active) MoE at 262K ctx on 2× GB10, TP=2+EP:

| Config | max-num-seqs | max-num-batched-tokens | gpu-mem | KV dtype | Spec (MTP) |
|---|---|---|---|---|---|
| baseline | 8 | 8192 | 0.80 | fp8 | 3 |

- **`max-num-seqs 8`** — the agentic/tool-calling concurrency sweet spot
  (community reports 8 concurrent streams show no degradation on this model's
  cluster).
- **`max-num-batched-tokens 8192`** — vLLM's common sweet spot.
- **`gpu-memory-utilization 0.80`** — the MiaAI dual-Spark measured value; at
  `0.835` the box had <1 GiB MemAvailable per node under load (OOM risk). `0.80`
  at 262K gave ~32 GiB KV / ~3.65M fp8 cache tokens (13.9× the 262K window) on
  the community kit.
- **`kv-cache-dtype fp8`** — 1.70× the cache tokens of bf16 per GiB; no measured
  reasoning/needle regression on the MiaAI kit (12/12 both). Requires the QSA
  fp8-KV patch (see below).
- **`speculative-config {mtp, num_speculative_tokens:3}`** — the model's built-in
  MTP head; 3 is the tuned depth.

---

## The QSA fp8-KV patch (required for fp8 KV)

The stock QSA sparse-attention kernels **refuse fp8 KV** (`kv-cache-dtype fp8`).
The community recipe carries a patch (`files/patch_qsa_fp8_kv.py`, vendored from
the single-Spark recipe, AGPL-3.0-or-later) that applies the fp8 QSA gather. Apply
it as a bind-mounted overlay (no image rebuild), and set the bf16 SSM state dtype
to halve the mamba page. **If you run `kv-cache-dtype bf16` you can skip this
patch.**
- `MAMBA_SSM_CACHE_DTYPE=bfloat16` (GDN recurrent state; checkpoint asks for
  float32, the fused kernel also takes bf16, +6.8–8.5% aggregate decode and a
  smaller attention block on the community kit).
- `MM_ENCODER_TP_MODE=data` (vision MLP intermediate 4304 is not divisible by 16
  after TP=2; `data` replicates the encoder per GPU, `weights` would crash).

---

## The thermal fix (why `busy_loop_s`)

On GB10 the CPU and GPU share one die/heatsink. Under TP≥2 decode, vLLM's
inter-process queue (`shm_broadcast.py`) **spins for up to 1 second**
(`busy_loop_s: float = 1`) instead of sleeping, holding 3-4 P-cores at 3.9 GHz
continuously — roughly **half the thermal budget millimeters from the GPU**.

`mods/busy-loop-fix/run.sh` patches it to **0.002** on every container launch
(idempotent):
```
busy_loop_s: float = 1,  ->  busy_loop_s: float = 0.002,
```
**Measured effect (DS4 lane, same hardware):** vLLM idle CPU ~57-79% → **4%**;
board (acpitz) temp ~85°C → **~51°C**; GPU ~70°C → **~48°C**. Use 0.002, not 0.

### Host-level: governor + performance-core cap (`host-cpu-thermal/`)

Same as the DS4 lane. The 10 Cortex-X925 P-cores boost to 3.9GHz and push the SoC
ACPI zones to ~90°C. `cpu-governor.service` + `gx10-cpu-cap.sh`:
- Governor `performance` → `schedutil` (idle cores drop to ~338MHz).
- Cap the 10 X925 P-cores to **2.8GHz** (the single biggest lever).
- **Measured effect:** SoC −9°C, P-core cluster −14°C, GPU −1.7°C; single-stream
  token rate unchanged, 8 concurrent streams ≈ −5%.
- `nvidia-smi -pl` is a **NO-OP** on GB10 — do not use a power-limit service.

See `host-cpu-thermal/README.md` for install/apply/verify/revert.

---

## Launch-orchestration (the non-obvious parts)

These are the traps that break a fresh launch or reboot. Documented so a rebuild
works first try.

### 1. Worker-first (rank 1 before rank 0)
The head (`rank 0`) owns the TCPStore rendezvous port. Launching the worker first
avoids the Gloo "Connection reset by peer" init race. `scripts/qwen38-vllm.sh 1`
(worker) then `scripts/qwen38-vllm.sh 0` (head), each with `--nnodes 2`.

### 2. podman, rootless, `CONTAINER_RT=podman`
This lane uses **podman** (rootless, owned by the deploying user), not docker.
`launcher/.env.example` sets `CONTAINER_RT=podman`. All `docker run` in the
upstream docs must be `podman run`.

### 3. `/models` mount → HF re-download hang
The HF cache snapshot weight shards are **symlinks into `/models`**. If the
container does **not** mount `/models`, those symlinks dangle → `snapshot_download`
re-downloads from HF Hub and **stalls**. Mount the model path read-only:
```
-v "$MODEL_PATH:$MODEL_PATH:ro"
```
With the mount, the model loads from local disk in minutes, not a network stall.

### 4. `VLLM_ALLOW_LONG_MAX_MODEL_LEN` only >262K
At native 262K this is **not** set. Only raise it (plus a YaRN `rope_parameters`
override) if you deliberately stretch past native — which this repo does **not**
recommend (see the context table).

---

## Boot autostart + swap

`scripts/swap-model.sh`:
- Reads the **last-used model** from a state file and brings it up on boot
  (idempotent — no-op if already serving).
- `--swap <deepseek|glm|qwen38>`: stops the current stack on both nodes (bounded
  :8000 free wait), launches the target lane, updates `/etc/motd`.
- Writes the accurate live model + ctx + stack into `/etc/motd`.

**Boot timer** (systemd, oneshot):
```ini
[Unit]
Description=Bring up the default vLLM model on the 2-node GX10 cluster
After=network.target systemd-user-sessions.service
Wants=network.target

[Service]
Type=oneshot
ExecStartPre=/bin/sleep 30
ExecStart=/usr/local/bin/swap-model.sh
TimeoutStartSec=1200
User=root

[Install]
WantedBy=multi-user.target
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Engine dies past ~24K ctx | Missing top-k patch for the sparse-attention indexer | Install `sparse_attn_indexer_kpool.py` overlay (same as GLM lane) |
| OOM on boot (171 GiB NVFP4 + KV) | `gpu-memory-utilization` too high, or PLE bf16 too big | Lower to 0.78–0.80; or use the FP8PLE derivative (−47 GiB) |
| MTP geometry validation failure at 524K+ | MTP draft head stays at 262K while target extended | **Don't go past 262K** — stay native |
| `kv-cache-dtype fp8` refused | Stock QSA kernel rejects fp8 | Apply the QSA fp8-KV patch, or use `bf16` |
| Head never reaches `/health` | Worker not launched first, or `:8000`/TCPStore race | Launch rank 1 before rank 0 |
| Model re-downloads and stalls | `/models` not mounted in the container | Add `-v "$MODEL_PATH:$MODEL_PATH:ro"` |
| CPU spinning, SoC hot | `busy_loop_s` not patched | Ensure `mods/busy-loop-fix` applied (idempotent) |

---

## Attribution / provenance

- **Container image** `vllm/vllm-openai:qwen38-flash-next` — the vLLM
  `qwen4_exp` runtime (day-0 image for the Qwen4 architecture preview).
- **Model** `orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` — community
  uncensored (abliterated) NVFP4 build of `Qwen/Qwen3.8-Flash-Next`.
- The **QSA fp8-KV patch** and **`busy_loop_s` fix** follow community reports
  (NVIDIA dev forums, MiaAI-Lab dual-Spark recipe, vLLM recipes page).
- The **launcher/orchestrator** in this repo is the author's own work, adapted
  from the GLM + DS4 lanes.

> No API keys, tokens, passwords, real hostnames, or internal IPs are present in
> this repo. Replace the `<HEAD_IP>`/`<WORKER_IP>`/`<HEAD_ROCE_IP>` placeholders
> in `launcher/.env.example` and set the model download token only where you
> deploy it to reproduce.

---

## Quick reproduce (docs)

1. Confirm the model is on both nodes at `/models/<repo>` (download once, rsync
   to the worker — or use the launcher's `--sync-model`).
2. Configure `launcher/.env` (copy from `.env.example`): cluster nodes, RoCE
   interface, `CONTAINER_RT=podman`, HF token.
3. Apply `host-cpu-thermal/` and the `mods/busy-loop-fix/` (both idempotent).
4. Launch worker then head:
   ```bash
   ./scripts/qwen38-vllm.sh 1   # on the worker (rank 1) — FIRST
   ./scripts/qwen38-vllm.sh 0   # on the head (rank 0)
   ```
5. Wait for `/health` → 200, then sanity-check tool calling + vision, and verify
   thermals (`nvidia-smi` + `cat /sys/class/thermal/thermal_zone*/temp`).
