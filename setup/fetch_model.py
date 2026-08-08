"""Pre-download model weights with progress and a stall detector.

`hf_xet` stalled at 0 B/s on this machine (Windows). This script disables it and
uses hf_transfer's parallel chunked downloads instead, and — importantly —
reports throughput so a stall is visible in seconds rather than after twenty
minutes of a silent progress bar.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

# Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download  # noqa: E402

MB = 1024**2
GB = 1024**3


def cache_bytes(repo_id: str) -> int:
    from huggingface_hub.constants import HF_HUB_CACHE

    folder = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    if not folder.exists():
        return 0
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())


def watch(repo_id: str, stop: threading.Event, interval: int = 15) -> None:
    last = cache_bytes(repo_id)
    start = time.time()
    while not stop.wait(interval):
        now = cache_bytes(repo_id)
        rate = (now - last) / MB / interval
        elapsed = time.time() - start
        flag = "  <-- STALLED" if rate < 0.01 else ""
        print(f"  [{elapsed:6.0f}s] {now / GB:6.2f} GB  {rate:6.2f} MB/s{flag}", flush=True)
        last = now


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-VL-4B-Instruct")
    args = parser.parse_args()

    print(f"fetching {args.repo}")
    print(f"  xet disabled       : {os.environ['HF_HUB_DISABLE_XET']}")
    print(f"  hf_transfer enabled: {os.environ['HF_HUB_ENABLE_HF_TRANSFER']}")
    print(f"  already cached     : {cache_bytes(args.repo) / GB:.2f} GB")

    stop = threading.Event()
    t = threading.Thread(target=watch, args=(args.repo, stop), daemon=True)
    t.start()

    began = time.time()
    path = snapshot_download(args.repo, max_workers=8)
    stop.set()

    total = cache_bytes(args.repo)
    took = time.time() - began
    print(f"\ndone: {total / GB:.2f} GB in {took / 60:.1f} min "
          f"({total / MB / max(took, 1):.2f} MB/s avg)")
    print(f"path: {path}")


if __name__ == "__main__":
    main()
