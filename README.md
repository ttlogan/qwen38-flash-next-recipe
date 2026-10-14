# Qwen3.8-Flash-Next (Uncensored) — 2-node GX10 Serve Recipe

Production recipe for serving **`jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4`**
(the OrcaRouter abliterated/uncensored weights, repackaged into the **ModelOpt NVFP4** format
with the **PLE table in FP8**) across a **2-node NVIDIA GB10 / DGX Spark cluster**
with **tensor-parallel 2**, served via **SGLang** (`--tp 2 --nnodes 2`) with
**NEXTN/MTP** speculative decoding, a **boot autostart** launcher, and a
**bot-autostart / swap** orchestrator.

This repo is the **sanitized configuration + documentation** for a working setup.
No API keys, tokens, passwords, real hostnames, or internal IPs are included
(only the ConnectX fabric interface/ROCE hostnames are kept as the internal
topology — all host/credential values are placeholders).

---

## What's in this repo

| Path | What it is | Source |
|---|---|---|
| `sglang/scripts/sglang-qwen38.sh` | **Per-node launcher** (rank 0/1): SGLang `podman run -d`, HEAD/worker-first, SM121 QSA/ token-0 patches | **Own work** (adapted from PixelML 2x-DGX-Spark) |
| `sglang/patches/` | **SM121 QSA + token-0 guard patches** (required on GB10 for SGLang, from PixelML/MiaAI) | **PixelML** (MIT) |
| `sglang/scripts/apply-sm121-qsa-patch.sh` + `launch_secure.py` + `redact_stream.py` | **Patch-staging + secure SGLang launch** wrappers | **Own work** (adapted from PixelML) |
| `scripts/swap-model.sh` | **Swap orchestrator**: `--swap <deepseek\|glm\|qwen38>`, idempotent ensure, `/etc/motd` (dispatches to the SGLang Qwen lane) | **Own work** |
| `launcher/.env.example` | Cluster fabric + podman config template (placeholders) | **Own work** |
| `mods/busy-loop-fix/` | **vLLM-only** thermal fix (`busy_loop_s` 1 → 0.002 in vLLM `shm_broadcast.py`); not used by the SGLang Qwen lane | **Own work** (same as DS4) |
| `host-cpu-thermal/` | **Host-level** thermal fix (works for vLLM + SGLang): `schedutil` governor + cap X925 P-cores to 2.8GHz | **Own work** |

---

## The model

- **`jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4`** — the
  **OrcaRouter abliterated (uncensored)** Qwen3.8-Flash-Next weights, repackaged
  into the **ModelOpt NVFP4** format with the **PLE n-gram table in FP8** (so it
  fits a 2× GB10 at TP2). ~126 GiB on disk. Full vision + video tower and **MTP**
  speculative head preserved. `hf_quant_config.json` present (`quant_algo: NVFP4`).
  **HF repo:** [https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4)
- **Served name** (API `id`) = `qwen3.8-flash-next-uncensored`.
- Loaded via **`--quantization modelopt_fp4`** (SGLang) / `--quantization modelopt` (vLLM).
  The orca-native `-NVFP4` (RadixArk layout, 95 GiB **bf16** PLE) does **NOT** fit a
  2× GB10 at TP2 and its PLE tensor is misnamed for the loader — the ModelOpt repack is
  required.

> **Why the ModelOpt NVFP4 repack, not the orca-native NVFP4?** The orca-native
> `-NVFP4` keeps the 102 GB PLE n-gram table in **bf16** (95 GiB), which at TP2 on a
> 121 GiB unified pool OOMs alongside the weights + KV, and its `ple_embedding.shard_*`
> tensor name can't be bound by the vLLM/SGLang loader. The `jpezzulli`/`gorbatjovy`
> ModelOpt repacks move the PLE to **FP8 E4M3** (~24 GiB/rank at TP2) and name the
> shards `ngram_embedding.shard_*` + a global `weight_scale` — the form the runtimes
> expect. Same uncensored weights, same serving behavior, fits the hardware.

---

## Why SGLang (not vLLM)

**This lane is served on SGLang, deliberately.** The question "should Qwen run on
vLLM instead?" was researched (SGLang vs vLLM head-to-head on this exact model +
2× GB10 hardware) and the answer is **no** — we stay on SGLang. Rationale:

| Path | c=1 | c=8 agg | Notes |
|---|---|---|---|
| **SGLang (this lane)** | ~40 | ~120 | stable; soak-clean (2400+ req / 0 fail / health 200) |
| vLLM **fast path** (local-inference-lab **b12x-fork** + MTP4 + MXFP8 lm_head + fp8 KV + GDN-MSE requant; `ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2` b1.4) | ~62 | ~196 | **1.5-1.6x faster**, but a large fork + re-quantized-checkpoint build with its own soak |
| vLLM **stock/nightly** (blazux + Ray, MTP3, bf16 KV) | ~31 | ~74 | **slower than SGLang** |

Key facts:
- **vLLM only beats SGLang with the full b12x-fork + MTP4 + GDN-MSE re-quantized
  checkpoint stack** (a substantial, ongoing build). Stock/nightly vLLM (including
  any `cu129-nightly` lane) is *slower* than SGLang on the same hardware.
- **The SM121 device-side crash is NOT SGLang-specific.** vLLM has an equally
  severe fault class on this model: vllm#54199 (CUDA IMA in `precopy_mamba_align_fused_kernel`
  on a prefix-cache hit, 100% reproducible) and vllm#54629 (>100K-token prefills hang
  a worker stream on GB10 sm_121). Swapping to vLLM buys **no reliability**.
- Therefore: **stay on SGLang.** We keep the stable, soak-tested SGLang baseline and
  stabilize it (rather than swap runtimes) if throughput headroom is needed. Only a
  committed b12x-fork project would justify moving off SGLang.

*Sources: SGLang#37052/#37111 (SM121 hybrid fault), vllm#54199/#54629 (vLLM's
own equal severity), ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2 (b1.4), NVIDIA
forum "Qwen3.8-Flash-Next NVFP4 on 2x DGX Spark".*

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
|| **262K (native)** | — | **Default. Validated.** MTP/quality preserved, memory-feasible. Run concurrently for throughput. |
|| 512K (YaRN ×2) | 2.0× | **Validated** (SGLang, model-card point). Good middle ground for long-context agentic work. |
|| 1M (YaRN ×4) | 4.0× | **Validated on 2× Spark via SGLang** (maci0/MiaAI: ~40-44 t/s single, ~148-155 agg at 24 conc, KV pool >1M). Premium: SGLang **clamps concurrency** (GDN/mamba state pool, not KV, is the limit ~4-10 conc) and prefill is slower. Use for deep-context, not throughput. |

**Practical rule (locked 2026-10-14):** run **262K native at max-running-requests 4**.
On 2× GX10 the measured optimum is c=4 (~117 tok/s agg, 29.3 tok/s stream); above 4
extra requests merely queue (no throughput gain) and sit closer to the spec-decode
stability edge. For long-horizon work use concurrent subagents/session compaction.
**1M is NOT the default.** 1M is *supported* via YaRN ×4 but on this cluster it
clamps concurrency (~4-10), slows prefill, and has a demonstrated single-request
memory wall (~294-390K) that can wedge the node (MiaAI-Lab #11). Keep 262K native
for serving; only run 1M behind a separate served name/env gate if a real >262K
single-request workload appears.

---

## Tuning (SGLang values)

Serving parameters for the 180B (6B active) MoE at 262K ctx on 2× GB10, TP=2 (SGLang):

| Config | max-running-requests | chunked-prefill | mem-fraction | KV dtype | Spec (NEXTN) |
|---|---|---|---|---|---|
| **locked** | **4** | 4096 | 0.80 | fp8_e4m3 | 3 |

- **`max-running-requests 4`** — **measured optimum on this 2× GX10 (locked 2026-10-14).** c=4 = ~117 tok/s agg / 29.3 tok/s stream; c=6 and c=8 merely queue (no gain, worse latency) and sit closer to the spec-decode edge. Matches the DS4 convention (never >4 on 2× GX10).
- **`chunked-prefill-size 4096`** — the PixelML/MiaAI measured sweet spot on GB10.
- **`mem-fraction-static 0.80`** — PixelML-verified (keep ≤0.78 if you see NVRM-OOM).
- **`kv-cache-dtype fp8_e4m3`** — 1.70× the cache tokens of bf16; SGLang handles the FP8 PLE + KV.
- **`speculative-algorithm NEXTN (3/1/4)`** — the model's built-in MTP head; 3 steps is the tuned depth.
- **`--quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass`** — required for the ModelOpt NVFP4 + the GB10 (SM12x) MoE path.
- **`--stream-interval 32`** — GB10 detokenizer-stall mitigation.

> **Tuning battery:** the repo's `tuning/` benchmarks ladder `gpu_mem`/`max-num-seqs`/
> MTP/context to lock the best values for SGLang on this cluster (mirrors the DS4 T4
> lock). Results recorded in `tuning/RESULTS.md`.

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

## The thermal fix (host-level, runtime-agnostic)

On GB10 the CPU and GPU share one die/heatsink. Under TP≥2 decode the 10
Cortex-X925 P-cores boost to ~3.9GHz and push the SoC ACPI zones hot, heating the
board even when the GPU is cool.

**`host-cpu-thermal/`** (systemd, persistent, works for **both vLLM and SGLang**):
- Governor `performance` → `schedutil` (idle cores drop to ~338MHz).
- Cap the 10 X925 P-cores to **2.8GHz** (the single biggest lever).
- **Measured effect:** SoC −9°C, P-core cluster −14°C, GPU −1.7°C; single-stream
  token rate unchanged, 8 concurrent streams ≈ −5%.

> **vLLM note:** the DS4 lane's `mods/busy-loop-fix` (patches `busy_loop_s 1→0.002`
> in vLLM's `shm_broadcast.py`) is **vLLM-specific** — SGLang's scheduler uses a
> different IPC path, so that patch does **not** apply to the SGLang Qwen lane.
> The **host-cpu-thermal** fix above is the transferable lever and is already
> applied on both nodes.
- `nvidia-smi -pl` is a **NO-OP** on GB10 — do not use a power-limit service.

See `host-cpu-thermal/README.md` for install/apply/verify/revert.

---

## Launch-orchestration (the non-obvious parts)

These are the traps that break a fresh launch or reboot. Documented so a rebuild
works first try.

### 1. Head-first (rank 0 before rank 1)
The head (`rank 0`) owns the TCPStore rendezvous port `--dist-init-addr`. Launching
the head first lets it bind that port; the worker (`rank 1`) then joins ~25s later
and clears `Init torch distributed begin`. Launching the worker first deadlocks it
at `Init torch distributed begin` (it blocks on rank 0's port which isn't bound yet).
`sglang/scripts/sglang-qwen38.sh 0` (head) then `sglang/scripts/sglang-qwen38.sh 1`
(worker), each with `--tp 2 --nnodes 2`, ~25s apart. With `--weight-loader-disable-mmap`
each rank frees its shards as it reads them, so head+worker loading together does NOT
deadlock the unified pool.

### 2. podman, rootless
This lane uses **podman** (rootless, `sudo -u wfoster`), not docker. All `docker run`
in the upstream SGLang docs must be `podman run`. The PixelML scripts are converted
in `sglang/scripts/`.

### 3. SM121 QSA + token-0 patches (required on GB10)
SGLang on GB10/SM121 hits a **FlashInfer sparse-decode bug** that emits token id 0 (`!`)
at long context (sglang#36806, #36537). `sglang/patches/` carries the SM121 QSA fallback
+ token-0 guard (from PixelML/MiaAI, `apply-sm121-qsa-patch.sh`), bind-mounted into the
running container. **Without these, long-context output corrupts to `!`.**

### 4. `/models` mount + PLE FP8
The container mounts the model dir read-only; the ModelOpt NVFP4 checkpoint has the
PLE n-gram table in **FP8** shards (`ngram_embedding.shard_*`), handled by SGLang. The
`--quantization modelopt_fp4` path is required — a stock runtime won't bind the FP8 PLE
under an NVFP4 checkpoint.

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
Description=Bring up the last-used model (via swap-model.sh) on the 2-node GX10 cluster
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
| Long-context output = all `!` (token id 0) | Missing SM121 QSA fallback / token-0 guard (GB10 FlashInfer sparse-decode bug) | Ensure `sglang/patches/` applied (bind-mounted in the container) |
| Engine dies past ~24K ctx | Missing top-k patch for the sparse-attention indexer | Install `sparse_attn_indexer_kpool.py` overlay (same as GLM lane) |
| **Rank-0 head hangs during `load_weights`** (worker finishes `Load weight end`, head stuck at `qwen4_exp.py:2096` / MoE `copy_`, head host RAM 0 free + swapping, GPU 1%) | mmap loader accumulates the checkpoint's cold pages on top of the GPU-resident model, exhausting the GB10 unified pool; the H2D `copy_` page-fault stalls under the CUDA driver lock on the head only | Add `--weight-loader-disable-mmap` (load eagerly, free each shard). Inject via `launch_secure.py` argv — the `--model-loader-extra-config` JSON gets mangled by shell quoting. NOT NCCL, NOT `/dev/shm`, NOT loader threading, NOT extra flags (all ruled out with live tests) |
| OOM on boot (NVFP4 + KV, bf16 PLE) | `mem-fraction-static` too high, or abliterated/NVFP4 bf16 PLE too big | Lower to 0.78-0.80; use the ModelOpt FP8-PLE repack (this repo) |
| Head never reaches `/health` | Worker not launched first, or `:8000`/TCPStore race | Launch rank 1 before rank 0 |
| `modelopt_fp4` refused / PLE not bound | Stock runtime can't bind the FP8 PLE under an NVFP4 checkpoint | Use the pinned SGLang image + `--quantization modelopt_fp4` (and the SM121 patches) |
| CPU spinning, SoC hot | `busy_loop_s` not patched | Ensure `mods/busy-loop-fix` applied (idempotent) |

---

## Attribution / provenance

- **Container image** `lmsysorg/sglang@sha256:12d3392bdc...` — the SGLang
  `qwen4_exp` runtime pinned by the PixelML 2x-DGX-Spark recipe.
- **Model** `jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4` —
  the **OrcaRouter** uncensored (abliterated) Qwen3.8-Flash-Next weights,
  repackaged into **ModelOpt NVFP4 + FP8 PLE** (fits 2x GB10 at TP2).
  Original abliteration is OrcaRouter's; the ModelOpt repack + runtime packaging
  is jpezzulli's. Base: `Qwen/Qwen3.8-Flash-Next`.
- The **SM121 QSA + token-0 patches** are from PixelML/MiaAI (`qwen3-8-flash-next-sglang-2x-dgx-spark`,
  MIT) — required on GB10/SM121.
- The **launcher/orchestrator** in this repo is the author's own work, adapted
  from the GLM + DS4 lanes and the PixelML 2x-DGX-Spark SGLang recipe.

> No API keys, tokens, passwords, real hostnames, or internal IPs are present in
> this repo. Replace the `<HEAD_IP>`/`<WORKER_IP>`/`<HEAD_ROCE_IP>` placeholders
> in `launcher/.env.example` and set the model download token only where you
> deploy it to reproduce.

---

## Quick reproduce (docs)

1. Confirm the model is on both nodes at `/models/<repo>` (download once, rsync
   to the worker).
2. Configure `launcher/.env` (copy from `.env.example`): cluster nodes, RoCE
   interface, podman, HF token.
3. Apply `host-cpu-thermal/` and the `mods/busy-loop-fix/` (both idempotent).
4. Ensure the SGLang image is pulled and the SM121 patches are staged, then
   launch head then worker (~25s apart):
   ```bash
   ./sglang/scripts/sglang-qwen38.sh 0   # on the head (rank 0) — TCPStore master — FIRST
   ./sglang/scripts/sglang-qwen38.sh 1   # on the worker (rank 1) — joins ~25s later
   ```
5. Wait for `/health` → 200, then sanity-check tool calling + vision, and verify
   thermals (`nvidia-smi` + `cat /sys/class/thermal/thermal_zone*/temp`).

## Client / consumer config

Set the client's `contextWindow` / `max_model_len` to **262144** (native), NOT
1M. The server serves 262K native; a 1M client context will error or truncate on
long requests.

```yaml
  - id: dgx_hobo_default
    name: Qwen-3.8-Flash-Next (remote SGLang)
    input: [text, image]
    reasoning: true
    contextWindow: 262144      # native context; do NOT set 1M
    maxTokens: 131072
```

The SGLang API is bearer-gated: send `Authorization: Bearer <key>` (the lane's
`.sglang-api-key`). A client must set `authHeader: true`. The upstream vLLM /
SGLang endpoint is `http://<head-mgmt-ip>:8000/v1`; if routed via a proxy (e.g.
liteLLM on `:4000`), point `baseUrl` at the proxy and set the served model id to
`qwen3.8-flash-next-uncensored`.
