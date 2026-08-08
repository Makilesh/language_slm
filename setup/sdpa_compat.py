"""Keep SDPA off the math backend on Windows.

Measured on this machine (Qwen3-VL-4B, 768px probe, batch 1, backward pass):

    sdpa, stock                16.26 GiB   14417 ms
    sdpa + force repeat_kv     12.58 GiB    1189 ms
    eager                      17.09 GiB   11570 ms

Why the stock path is so bad here. Qwen3-VL uses grouped-query attention: 32
query heads against 8 key/value heads. transformers decides between two ways of
reconciling that (`integrations/sdpa_attention.py::use_gqa_in_sdpa`):

  * `attention_mask is None`  -> pass `enable_gqa=True` to SDPA
  * otherwise                 -> `repeat_kv`, broadcasting K/V up to 32 heads

Training with batch 1 and no padding means no attention mask, so it takes the
`enable_gqa` branch. On the Windows CUDA build that branch has nowhere to
dispatch:

    Torch was not compiled with flash attention
    For dense input, both fused kernels require query, key and value to have
    the same num_heads. Query.sizes(): [1, 32, 857, 128], Key sizes(): [1, 8,
    857, 128]
    cuDNN attention has been runtime disabled

so SDPA silently falls back to the math backend, which materialises a
[32, seq, seq] score matrix per layer and keeps it for backward. At 36 layers
that is where the missing gigabytes go.

Forcing the `repeat_kv` branch costs one K/V broadcast and lets the
memory-efficient kernel take the call. Toggling cuDNN makes no difference
either way, so the switch is not the fix -- the head-count mismatch is.

This is platform-specific: on Linux with flash-attn built for sm_120 the stock
path is fine, and this patch should be unnecessary. Verify before assuming.
"""

from __future__ import annotations


def force_repeat_kv(enabled: bool = True) -> None:
    """Force transformers to broadcast GQA K/V heads instead of using enable_gqa.

    Idempotent, and reversible by calling with ``enabled=False``.
    """
    from transformers.integrations import sdpa_attention as sa

    if not hasattr(sa, "_orig_use_gqa_in_sdpa"):
        sa._orig_use_gqa_in_sdpa = sa.use_gqa_in_sdpa

    sa.use_gqa_in_sdpa = (
        (lambda *args, **kwargs: False) if enabled else sa._orig_use_gqa_in_sdpa
    )


def sdpa_backend_flags() -> dict[str, bool]:
    import torch

    b = torch.backends.cuda
    return {
        "flash": b.flash_sdp_enabled(),
        "mem_efficient": b.mem_efficient_sdp_enabled(),
        "math": b.math_sdp_enabled(),
        "cudnn": b.cudnn_sdp_enabled(),
    }
