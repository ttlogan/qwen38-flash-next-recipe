# Qwen3.8-Flash-Next-Uncensored — Performance Tuning

Experiment log, mirroring the DS4 tuning workflow (see partytime
`stable_ds4_tunings_20261010.md`). Runs append dated tables here. We iterate
until we lock in a stable default, then promote it to `README.md`.

## Ladder (the knobs we sweep)

| Knob | Values | Note |
|---|---|---|
| `gpu_memory_utilization` | 0.80, 0.835, 0.85 | KV cache headroom; 0.85 was DS4 sweet spot |
| `max_num_seqs` | 2, 4, 6, 8 | concurrency cap; DS4 landed on 6 |
| `num_speculative_tokens` | 2, 3, 4, 6 | MTP draft depth; DS4 used 6 |
| RoCE all-reduce | A/B (off/on) | maintenance window |

Context is **native 262K** (no YaRN — quality degrades hard past ~390K; MTP
draft head stays at 262K). We do NOT sweep 500K/1M arms (known failure).

Reasonable target: single-stream ≥ ~55-64 tok/s; aggregate ~117-135 with
spec-decode (published NVFP4 TP=2 reference).

## Results

<!-- New runs append below this line -->

## Locked-in default (promote to README.md once stable)

_(pending — not yet locked)_
