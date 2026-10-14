"""Launch SGLang while loading its API key from a mounted secret file."""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

from sglang.launch_server import run_server
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def main() -> None:
    load_plugins()
    secret_path = Path(
        os.environ.get("SGLANG_API_KEY_FILE", "/run/secrets/sglang-api-key")
    )
    api_key = secret_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError(f"SGLang API key file is empty: {secret_path}")

    # ServerArgs becomes immutable after resolution. Add the key to the
    # in-memory argument list before parsing so it never appears in host argv.
    # This Qwen4Exp NVFP4 checkpoint hangs on GB10 unified memory with the
    # default multithreaded mmap safetensors loader (CUDA driver rwlock held
    # across cold mmap page-fault -> rank-0 stall). Inject single-thread load.
    _loader_extra = json.dumps({"enable_multithread_load": False})
    server_args = prepare_server_args(
        [*sys.argv[1:], "--model-loader-extra-config", _loader_extra, "--api-key", api_key]
    )

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
