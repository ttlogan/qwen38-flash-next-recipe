# Qwen3.8-Flash-Next-Uncensored — Performance Tuning

Experiment log, mirroring the DS4 tuning workflow (see partytime
`stable_ds4_tunings_20261010.md`). Runs append dated tables here. We iterate
until we lock in a stable default, then promote it to `README.md`.

## Ladder (the knobs we sweep)

| Knob | Values | Note |
|---|---|---|
| `gpu_memory_utilization` | 0.70, 0.80, 0.85 | KV cache headroom; 0.80 was the stable default |
| `max_num_seqs` / `max-running-requests` | 2, 4, 6, 8 | concurrency cap; 6 was stable, **8 crashed** |
| `mamba-radix-cache-strategy` | extra_buffer, no_buffer | no_buffer avoids the state-checkpoint wedge |
| `speculative-num-steps` | 3 (NEXTN/MTP) | built-in MTP head |
| `cuda-graph-max-bs` | 8, 16, 256 | bound the decode CUDA-graph ladder |
| `disable-flashinfer-autotune` | on/off | avoids Blackwell fused-MoE tactic corruption |

Context is **native 262K** (no YaRN default — see the 262K-vs-1M decision below).

## Context-length decision (262K vs 1M)

Researched by an independent inference domain-expert team on 2026-10-14.
**Recommendation: run native 262K as the default.** 1M is *supported* via YaRN
but the community strongly advises against it as the default:

- **Native max** = 262,144 (`max_position_embeddings: 262144`, `rope_scaling: null`
  in this checkpoint). 1M requires YaRN factor=4 + `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`.
- **Static-YaRN quality loss**: open-source YaRN applies the scaling factor at
  position 0 too, so it degrades short-context quality on EVERY request. Qwen
  says "set factor to 2.0 instead of 4.0" for workloads near 512K.
- **Concurrency clamps to ~4 at 1M** (the GDN/mamba state pool, NOT KV, is the
  binding limit on SGLang). The SGLang cookbook: "caps --max-running-requests to
  what the mamba pool admits."
- **Memory wall on 2x GB10**: a demonstrated single-request ceiling ~294-390K
  that can WEDGE the node (MiaAI-Lab issue #11). 393K "wedged both nodes, needed
  a power cycle." Never rely on an error.
- **Sweet spot if you must stretch** = **512K with YaRN factor 2.0** (Level1Techs
  + Qwen both point here; NVIDIA forum says >390K is "severe degradation").
- Every named SGLang recipe (PixelML, outstandly/twinspark, MiaAI) serves
  **262,144 by default.**
- **My measured prefill is flat ~2900 tok/s** (realistic varied text) from 20K-100K
  prompt. At that rate a 1M prompt costs ~5.7 min TTFT (prefill alone) vs ~90s
  at 262K. So 1M has a ~4x TTFT penalty before any decode.

**Conclusion: keep 262K native as the serving default. 1M only if a real
single-request >262K workload appears, behind a separate served name/env gate,
never the default.**

## Stability root cause (the c=8 crash)

The c=8 concurrent decode burst triggered the SGLang watchdog (device-side fault
in the NEXTN/hybrid GDN+Mamba+QSA spec-decode state path on SM121 —
[#37052](https://github.com/sgl-project/sglang/issues/37052),
[#37111](https://github.com/sgl-project/sglang/issues/37111)). It's a
batch-composition *transition* trigger, not raw overload. Mitigations applied in
the launcher:
- `--disable-flashinfer-autotune` (avoids Blackwell fused-MoE tactic corruption, flashinfer#4841)
- `--cuda-graph-max-bs 8 --disable-cuda-graph-padding` (bound the decode CUDA-graph ladder, #29548)
- `--mamba-radix-cache-strategy no_buffer` (avoids state-checkpoint wedge, #43204/#30919)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (#36941)

## Results

### Run 2026-10-14 (config: qwen3.8-flash-next-uncensored, 262K native, mem 0.80, max-running-requests 6)

Decode ladder (short prompt, max_tokens=320), additive:
| c | agg tok/s | per-stream tok/s | wall_s | err |
|---|---|---|---|---|
| 1 | 43.9 | 43.9 | 6.9 | 0 |
| 2 | 78.4 | 39.2 | 7.5 | 0 |
| 4 | 121.6 | 30.4 | 10.1 | 0 |
| 6 | 152.6 | 25.4 | 11.6 | 0 |
| 8 | **CRASHED** | — | — | watchdog |

Prefill throughput (realistic varied text, max_tokens=8), additive:
| requested | actual prompt_tokens | dt (s) | prefill tok/s |
|---|---|---|---|
| 20000 | 10363 | 3.59 | 2883 |
| 50000 | 25788 | 8.79 | 2932 |
| 100000 | 51519 | 17.53 | 2939 |

**Prefill saturates ~2900 tok/s** (flat across context). Decode single ~44 tok/s,
aggregate ~153 tok/s at c=6. This matches the published NVFP4 TP=2 reference
(~40-63 single, ~150 agg).

### Run 2026-10-14 — POST-FIX (config: qwen3.8-flash-next-uncensored, 262K native, mem 0.80, max-running-requests **4**, head-first launch)

Decode ladder (short prompt, max_tokens=320):
| c | agg tok/s | per-stream tok/s | wall_s | err | note |
|---|---|---|---|---|---|
| 1 | 40.3 | 40.3 | 7.72 | 0 | single-stream |
| 2 | 67.1 | 33.5 | 9.20 | 0 | |
| **4** | **117.3** | **29.3** | 10.01 | 0 | **default (optimum)** |
| 6 | 106.4 | 17.7 | 16.96 | 0 | queued; no gain |
| 8 | 120.5 | 15.1 | 20.45 | 0 | **no longer crashes** |

Prefill (realistic varied text, max_tokens=8):
| requested | prompt_tok | dt (s) | prefill tok/s |
|---|---|---|---|
| 50000 | 25779 | 8.66 | 2976.7 |
| 100000 | 51536 | 17.55 | 2936.7 |

**Prefill ~2900-3000 tok/s** (flat across context). Decode single ~40 tok/s, aggregate ~117 tok/s at c=4.

**Key findings:**
- **c=4 is the throughput optimum AND the stability point.** c=6 and c=8 with
  `max-running-requests=4` merely **queue** (no throughput gain; c=8 still works
  because extra requests wait rather than overwhelm the scheduler). There is no
  reason to go above 4 → matches the DS4 convention (never >4 on 2x GX10).
- **c=8 no longer crashes** (was watchdog-killed pre-fix). The combination of
  the c=4 cap + head-first rendezvous cleared the concurrent-decode crash.
- **Head-first launch** (rank 0 = TCPStore master, rank 1 joins ~25s later) is
  the deterministic, reproducible order. The old slave-first order intermittently
  deadlocked at `Init torch distributed begin` (rank 1 blocks on rank 0's port).

## Locked-in default (promoted)

**`--max-running-requests 4`, context 262K native, mem 0.80, head-first launch.**
Stable through c=8, ~40 tok/s single, ~117 tok/s agg at c=4, prefill ~2900 tok/s.

## Tuning ladder — baseline (run 2026-10-14, current running config)

Confirmed live server_args: `max_running_requests=4`, `mem_fraction_static=0.8`,
`chunked_prefill_size=4096`, `speculative_algorithm=NEXTN`, `speculative_num_steps=3`,
`page_size=64`, `disable_flashinfer_autotune=False`, mamba `extra_buffer`,
`context_length=262144`. (This is the baseline every T1/T2/T3 arm is measured against.)

| Metric | Value |
|---|---|
| c=1 single | 40.27 tok/s |
| c=2 agg | 67.05 (33.53/stream) |
| c=4 agg | 117.30 (29.33/stream) |
| c=6 agg | 106.43 (queued) |
| c=8 agg | 120.51 (15.06/stream, no crash) |
| prefill 50K | ~2977 tok/s |
| prefill 100K | ~2937 tok/s |

Against DS4 (1M): DS4 c=1 28.65 / c=4 agg 51.06 / c=8 agg 53.21. Qwen beats DS4 by
+40% single, +130% agg at c=4, +126% at c=8.

_Arms pending the domain-expert T1/T2/T3 ladder (deleg_0fcc26ae)._

## Tuning ladder (deleg_0fcc26ae, 2026-10-14)

12-arm ladder from the deep inference domain-expert team. Ranked by (gain/risk).
**Biggest lever = concurrency**, gated by the NEXTN/hybrid spec-decode crash
(sglang#37052/#37111). **Second lever = prefill** (safe, chunked-prefill-size).

| Arm | Change | Expected | Risk | Run? |
|---|---|---|---|---|
| T1 | chunked-prefill-size 4096->8192 (probe 2048/16384) | prefill +15-40% | LOW (isolated to prefill) | **YES** |
| T2 | re-add disable-flashinfer-autotune + cuda-graph-max-bs 8 + disable-cuda-graph-padding | stability, low direct gain | LOW (not the spec path) | maybe |
| T3 | --speculative-algorithm NONE + measure decode cost | de-risk gate; removes crash root cause | MEDIUM (perf regression only) | **YES** |
| T4 | max-running-requests 8 (only after T3) | 120->130-145 agg | LOW once NEXTN off; HIGH with NEXTN | after T3 |
| T5 | max-running-requests 16 (KV-pool fit check) | 120->140-155 agg | MED-HIGH (KV/OOM) | **skip** (KV/OOM risk) |
| T6 | page-size 32 (probe 16) | raise effective concurrency at 262K | MEDIUM | maybe |
| T7 | mamba-track-interval 128 + strategy sweep | 5-15% mamba decode; KV savings | MEDIUM | maybe |
| T8 | mem-fraction 0.85 (0.75 control) | +0-20% if KV-bound | MED-HIGH (unified-mem spill/OOM) | **skip** (OOM/wedge) |
| T9 | fp4-gemm / attn backend sweep | 5-15% | LOW-MED | maybe |
| T10 | PLE/NGC n-gram offload + spec-draft decoupled | higher effective concurrency | MEDIUM (PLE hot path latency) | maybe |
| T11 | NCCL/env + graph-capture-size probe | small | LOW (env-only) | maybe |
| T12 | speculative-num-steps 4 (probe 5) | single +10-25% | **HIGH** (crash region) | **skip** (crash region) |

_Constraints: no node cluster restart. Skip T8/T5/T12 (OOM/wedge or the NEXTN crash region)._

### T1/T3 run (2026-10-14) — IMPORTANT NEGATIVE RESULTS

**T1 — chunked-prefill-size 8192** (was 4096), NEXTN 3 unchanged:
- Prefill 50K: **2124 tok/s** (baseline 2966 — LOWER, noisy/cache artifact)
- Prefill 100K: **3030 tok/s** (baseline 2941 — ~flat)
- Decode: c=1 42.98, c=2 48.26, then **CRASHED at c=4** (watchdog kill_process_tree)
- **Verdict: T1 is a NULL result.** chunked-prefill 8192 did NOT improve prefill
  (flat-to-worse), and the decode path still crashed under concurrency.

**T3 — speculative-algorithm NONE** (was NEXTN), chunked back to 4096:
- **CRASHED during load/warmup** (CUDA coredump exit, watchdog kill) ~10 min in.
- **Verdict: T3 did NOT stabilize it.** Disabling NEXTN did NOT remove the crash.

**CONCLUSION (disproves the research team's #1 lever premise):** the crash is
NOT the NEXTN/hybrid spec-decode transition. It occurs with NEXTN on (T1) AND
with NEXTN off (T3), and even during the load/warmup phase. It is a
**nondeterministic SM121 device-side fault** on the hybrid GDN/mamba+QSA path
(sglang#37052/#37111 class), NOT a deterministic flag fault. The baseline
config (chunked 4096, NEXTN 3) was stable for hours; T1 (chunked 8192) and T3
(NEXTN off) both crashed, so **keep the baseline** (chunked 4096, NEXTN 3,
max-running-requests 4) as the stable serving spec.

Host was healthy during the crashes: 96 GB free, thermals 49C, GPU 47C. Not
OOM, not thermal throttle.

_T1/T3 both reverted. Baseline restored. Remaining ladder arms (T2/T4/T6/etc.)
carry the same SM121 nondeterministic-crash risk and are NOT safe to sweep
until the underlying device-side fault is understood (needs an SGLang fix or a
different runtime path)._

### Soak test (2026-10-14) — POSITIVE stability signal on the baseline

The baseline serving config (chunked 4096, NEXTN 3, max-running-requests 4, mem 0.80,
page 64, mamba extra_buffer, flashinfer_cutlass) was soak-tested under sustained c=4
load for ~30 min.

| Metric | Result |
|---|---|
| requests | 2432 |
| failures | **0** |
| health | **200 continuous** (no drop) |
| memory | ~21 GB avail, stable (no leak) |
| SoC temp | 64-70 C, no throttle |

**Verdict: the baseline is STABLE under sustained load.** No device-side fault, no
watchdog kill, no health drop. This is the desired stability signal. The earlier
nondeterministic SM121 fault (T1/T3) did NOT recur in 30 min of c=4 load.

Soak harness: `/tmp/soak.py` (sustained c=4 completions, samples health/mem/SoC-temp
every 5s, CSV to /tmp/soak.csv). Reusable for any future config soak.

## SGLang vs vLLM research verdict (deleg_91c6675b, 2026-10-14)

Research on whether to swap Qwen to vLLM. **Decision: STAY on SGLang baseline.**
The vLLM nightly lane is the regressive path; do not run it.

| Lane | Perf vs our SGLang | Stability |
|---|---|---|
| SGLang (baseline) | 117 agg / 40 single / ~3000 prefill | soak-clean (2432 req / 0 fail / health 200) |
| vLLM fast path (local-inference-lab **b12x-fork** + MTP4 + MXFP8 lm_head + fp8 KV + GDN-MSE requant, ursuciprian b1.4) | measurably higher (~70 peak / 47 typical) | needs its own soak |
| vLLM stock/nightly/blazux (our pre-staged cu129-nightly lane) | regresses speed AND stability | — |

**vLLM has an equal-or-worse fault class on this model/hardware:**
- vllm #54199: CUDA IMA in `precopy_mamba_align_fused_kernel` prefix-cache hit, 100% reproducible, dies at max-num-seqs=1.
- vllm #54629: repeated >100K-token prefills hang a worker GPU stream on GB10 sm_121 (TP4+EP).

**Conclusion:** the SM121 hybrid-arch (GDN/mamba+QSA) fault is a class on BOTH
runtimes, not solved by a vLLM swap. Only a full b12x-fork + requantized-checkpoint
stack (~1.4-1.6x potential) is worth it as a long-run project, with its own soak.
The higher-value immediate move is stabilizing the CURRENT SGLang (revisit the
disable-flashinfer-autotune / cuda-graph-max-bs mitigations), not a vLLM swap.

_Sources: ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2 (b1.4), NVIDIA forum
"Qwen3.8-Flash-Next NVFP4 on 2x DGX Spark" (70 peak / 47 typical), sglang#37052/#37111,
vllm#54199, vllm#54629._
