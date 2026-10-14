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

## Locked-in default (promote to README.md once stable)

_Pending the post-fix re-benchmark._ Target: stable through c=6 (was crashing at c=8
pre-fix), ~44 tok/s single, ~153 tok/s agg at c=6, prefill ~2900 tok/s.
