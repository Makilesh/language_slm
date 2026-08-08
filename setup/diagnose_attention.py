"""Can we get off the SDPA math backend on Windows?

`diagnose_backward.py` showed every attention call falling through to the math
backend, which materialises a [heads, seq, seq] score matrix per layer and keeps
it for backward. Torch's own warnings gave the three reasons:

  * flash      -> "Torch was not compiled with flash attention" (Windows wheels)
  * mem-efficient -> rejected because GQA leaves Q at 32 heads and K/V at 8
  * cuDNN      -> "runtime disabled"

Only the last one is a switch rather than a wall, and cuDNN's attention does
support GQA. This measures whether flipping it changes the backward footprint,
against eager and the default as controls.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from setup.verify_vlm import GIB, attach_lora, build_batch, load_model, resize_long_edge  # noqa: E402


def reset() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def backend_flags() -> dict:
    b = torch.backends.cuda
    return {
        "flash": b.flash_sdp_enabled(),
        "mem_efficient": b.mem_efficient_sdp_enabled(),
        "math": b.math_sdp_enabled(),
        "cudnn": b.cudnn_sdp_enabled(),
    }


def force_repeat_kv(enabled: bool) -> None:
    """Make transformers expand GQA K/V heads instead of passing enable_gqa=True.

    transformers takes the `enable_gqa=True` path whenever attention_mask is
    None (sdpa_attention.py::use_gqa_in_sdpa). The Windows torch build's fused
    kernels reject that -- "both fused kernels require query, key and value to
    have the same num_heads" -- and silently drop to the math backend, which
    materialises a [heads, seq, seq] matrix per layer. Forcing the repeat_kv
    branch costs a K/V broadcast but keeps the memory-efficient kernel usable.
    """
    from transformers.integrations import sdpa_attention as sa

    if not hasattr(sa, "_orig_use_gqa_in_sdpa"):
        sa._orig_use_gqa_in_sdpa = sa.use_gqa_in_sdpa
    sa.use_gqa_in_sdpa = (lambda *a, **k: False) if enabled else sa._orig_use_gqa_in_sdpa


def trial(
    name: str,
    attn_impl: str,
    image,
    enable_cudnn: bool,
    long_edge: int,
    repeat_kv: bool = False,
) -> dict:
    print(f"\n--- {name} ---")
    torch.backends.cuda.enable_cudnn_sdp(enable_cudnn)
    force_repeat_kv(repeat_kv)
    print(f"  sdp backends: {backend_flags()}  force_repeat_kv={repeat_kv}")

    model, processor = load_model(attn_impl, quantize_vision=False)
    model = attach_lora(model, 32, 64, use_prepare=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    inputs, image_tokens, _ = build_batch(processor, image)
    reset()

    row = {
        "name": name,
        "attn_impl": attn_impl,
        "cudnn_sdp": enable_cudnn,
        "force_repeat_kv": repeat_kv,
        "long_edge": long_edge,
        "image_tokens": image_tokens,
        "seq_len": int(inputs["input_ids"].shape[1]),
        "backends": backend_flags(),
    }

    try:
        out = model(**inputs)          # warmup
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        del out
        reset()

        t0 = time.perf_counter()
        out = model(**inputs)
        out.loss.backward()
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / GIB
        model.zero_grad(set_to_none=True)
        del out

        row["peak_gib"] = round(peak, 3)
        row["ms"] = round(secs * 1000)
        total = torch.cuda.get_device_properties(0).total_memory / GIB
        flag = "  <-- EXCEEDS VRAM" if peak > total else "  <-- FITS"
        print(f"  backward peak {peak:6.2f} GiB  {secs * 1000:8.0f} ms{flag}")
    except Exception as exc:  # noqa: BLE001 - a failure is a result
        row["peak_gib"] = None
        row["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        print(f"  FAILED: {row['error']}")

    del model, processor, inputs
    reset()
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long-edge", type=int, default=768)
    ap.add_argument("--image", default=str(REPO / "assets" / "demo" / "probe_chart.png"))
    args = ap.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(f"card: {props.name}  {props.total_memory / GIB:.2f} GiB")
    print(f"torch {torch.__version__}")
    print(f"default sdp backends: {backend_flags()}")

    source = Image.open(args.image).convert("RGB")
    image = resize_long_edge(source, args.long_edge)

    rows = [
        trial("sdpa, stock (enable_gqa -> math fallback)", "sdpa", image, True, args.long_edge),
        trial("sdpa + force repeat_kv", "sdpa", image, True, args.long_edge, repeat_kv=True),
        trial("sdpa + force repeat_kv, cuDNN off", "sdpa", image, False, args.long_edge,
              repeat_kv=True),
        trial("eager", "eager", image, False, args.long_edge),
    ]

    out = REPO / "results" / f"phase0_attention_backends_{args.long_edge}px.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"card_gib": round(props.total_memory / GIB, 3),
                               "torch": torch.__version__, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
