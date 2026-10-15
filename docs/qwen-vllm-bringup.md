# Qwen3.8 Flash Next vLLM Bring-Up — Troubleshooting

The one page to read when the qwen38 lane (MiaAI native vLLM runbook) won't come
up. All three failures below were hit during the first bring-up on the 2-node GX10
and are **hand-staging artifacts**, not bugs in the vetted runbook.

## 0. The runbook expects a standard HF snapshot layout

`start.sh` step 2 (`resolve_snapshot.py`) wants `models--org--name/snapshots/<hash>/`.
A checkpoint staged by extracting files straight into `models--org--name/` has no
`snapshots/` tree and the launch dies with `No snapshot under ...`.

**Fix (both nodes):** create `snapshots/<hash>/` and put the files there, with
`refs/main` = that hash. The `config.json`, tokenizer files, **and every weight
shard** must be **real files or hardlinks**, not symlinks to the host path
(see #3).

## 1. `Value error, Couldn't instantiate the backend tokenizer ... need sentencepiece`

The checkpoint's `tokenizer_config.json` declares `tokenizer_class: Qwen2Tokenizer`
(sentencepiece), but this Flash-Next export ships a **BPE `tokenizer.json`** (no
`spiece.model`). vLLM's `get_tokenizer` reads the class and tries to build a
slow sentencepiece tokenizer, which fails.

**Root cause:** the MiaAI vLLM fork's `vllm/tokenizers/registry.py` has a
`_MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS` set that reroutes known-wrong model
types to `TokenizersBackend` (the generic fast tokenizer). `qwen4_exp` was **not**
in that set.

**Fix:** add `"qwen4_exp"` to that set (see `mods/tokenizer-registry-overlay/registry.py`),
and bind-mount the patched file over the image's copy via the runbook's
`add_overlay` mechanism. In `start.sh` add (after the Step 4d QSA block):

```bash
OV="$SCRIPT_DIR/files/overlay"
if $DO_LAUNCH && [[ -f "$OV/registry.py" ]]; then
    info "=== tokenizer registry overlay (qwen4_exp -> TokenizersBackend) ==="
    add_overlay "$OV/registry.py" "$VLLM_PKG/tokenizers/registry.py"
fi
```

## 2. `FileNotFoundError` on `model-0000N-of-00010.safetensors` (getsize)

`weight_utils._get_checkpoints_size_bytes` calls `os.path.getsize()` on the shard
paths in the snapshot. If the shards are **symlinks to the host path**
(`/home/wfoster/...`), they break **inside the container**, which mounts the cache
at `/root/.cache/...` — so `/home/wfoster/...` doesn't exist there.

**Fix:** make the shards **hardlinks** (`ln real_file snapshot_link`, not `ln -s`).
Hardlinks share the inode and resolve at both the host path and the container mount
with no data copy (same filesystem). `os.path.getsize` then returns the real byte
count on both. Do this for every file in the snapshot.

## 3. Symlinks resolve in host but not container

Generalizes #2. Any symlink whose target is an absolute host path
(`/home/wfoster/...`) breaks when the container mounts that dir elsewhere
(`/root/.cache/...`). The lazy fix is hardlinks for same-filesystem data, or real
copies for the small config/tokenizer text files.

## Confirmed-working bring-up recipe

- Checkpoint: `drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-dual-ablit-house-qsa-L3-47`
  (124 GiB, staged on both nodes, `ABLIT=1`).
- Snapshot dir `snapshots/a393318fb56d9aedc56d91b6f4962d9af26d2fe7/` with **real
  files** for config/tokenizer and **hardlinks** for the `.safetensors` shards.
- `registry.py` overlay adding `qwen4_exp` to the incorrect-tokenizer-class set.
- Served name stays `qwen3.8-flash-next-uncensored` → litellm `dgx_hobo_default`
  unchanged.

**Boot/warmup time ~880 s** (about 15 min) for first cold start on 2-node TP=2 at
262K ctx; the weight load + KV pool + kernel compile dominates.
