"""Find what actually dominates backward memory.

Phase 0's first table said a backward pass does not fit in 11.94 GiB at any
tested resolution, and that gradient checkpointing barely helps. Both claims
contradict ordinary QLoRA practice on a 4B VLM, so before that conclusion goes
into the budget it gets tested against the obvious suspects:

  A. loss over the full 151,936-wide vocab for every position
     (logits + log-softmax + grad, all seq x vocab)
  B. SDPA falling back to the `math` backend because of the 4D attention mask,
     which materialises a [heads, seq, seq] score matrix per layer
  C. activations, i.e. the thing gradient checkpointing is supposed to fix

Each row isolates one variable against the same baseline. Peak allocated is
reported, plus wall-clock, because on Windows WDDM the allocator spills to host
memory rather than raising OOM.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from setup.verify_vlm import (  # noqa: E402
    GIB,
    MODEL_ID,
    attach_lora,
    build_batch,
    load_model,
    resize_long_edge,
)


def reset() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def measure(name: str, fn) -> dict:
    reset()
    t0 = time.perf_counter()
    try:
        fn()
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / GIB
        print(f"  {name:44s} {peak:6.2f} GiB  {secs * 1000:8.0f} ms")
        return {"name": name, "peak_gib": round(peak, 3), "ms": round(secs * 1000)}
    except Exception as exc:  # noqa: BLE001 - a failure is a result here
        print(f"  {name:44s} FAILED: {type(exc).__name__}")
        return {"name": name, "peak_gib": None, "ms": None, "error": type(exc).__name__}
    finally:
        reset()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long-edge", type=int, default=768)
    ap.add_argument("--image", default=str(REPO / "assets" / "demo" / "probe_chart.png"))
    args = ap.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(f"card: {props.name}  {props.total_memory / GIB:.2f} GiB")
    print(f"probe: {args.long_edge}px long edge\n")

    source = Image.open(args.image).convert("RGB")
    image = resize_long_edge(source, args.long_edge)

    model, processor = load_model("sdpa", quantize_vision=False)
    model = attach_lora(model, 32, 64, use_prepare=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    inputs, image_tokens, prompt_len = build_batch(processor, image)
    seq = int(inputs["input_ids"].shape[1])
    vocab = model.config.get_text_config().vocab_size
    reset()
    resident = torch.cuda.memory_allocated() / GIB

    print(f"seq len {seq}, image tokens {image_tokens}, vocab {vocab}")
    print(f"resident weights+adapters: {resident:.2f} GiB")
    print(f"one full-sequence logits tensor: fp32 {seq * vocab * 4 / GIB:.2f} GiB, "
          f"bf16 {seq * vocab * 2 / GIB:.2f} GiB\n")

    rows = []

    # --- baseline -----------------------------------------------------------
    def full_backward():
        out = model(**inputs)
        out.loss.backward()
        model.zero_grad(set_to_none=True)

    rows.append(measure("backward, full loss (baseline)", full_backward))

    # --- A: is it the vocab projection? -------------------------------------
    # Same graph, but the loss touches a single position instead of all of them.
    # If this collapses, the seq x vocab logits path is the driver.
    no_label = {k: v for k, v in inputs.items() if k != "labels"}

    def backward_last_token_only():
        out = model(**no_label)
        loss = out.logits[:, -1, :].float().log_softmax(-1).mean()
        loss.backward()
        model.zero_grad(set_to_none=True)

    rows.append(measure("A. backward, loss on last token only", backward_last_token_only))

    def forward_logits_only():
        with torch.no_grad():
            model(**no_label)

    rows.append(measure("A. forward only, no labels", forward_logits_only))

    # --- B: is SDPA falling back to the math backend? -----------------------
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def backward_flash_or_mem_efficient():
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            out = model(**inputs)
            out.loss.backward()
        model.zero_grad(set_to_none=True)

    rows.append(measure("B. backward, math backend disabled", backward_flash_or_mem_efficient))

    def backward_math_only():
        with sdpa_kernel([SDPBackend.MATH]):
            out = model(**inputs)
            out.loss.backward()
        model.zero_grad(set_to_none=True)

    rows.append(measure("B. backward, math backend forced", backward_math_only))

    # --- C: how much do activations actually cost? --------------------------
    model.gradient_checkpointing_disable()
    rows.append(measure("C. backward, checkpointing OFF", full_backward))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    rows.append(measure("C. backward, checkpointing ON", full_backward))

    print("\n" + "-" * 70)
    base = next((r for r in rows if r["name"].startswith("backward, full")), None)
    if base and base["peak_gib"]:
        print(f"baseline peak {base['peak_gib']:.2f} GiB, of which "
              f"{resident:.2f} GiB is resident weights "
              f"=> {base['peak_gib'] - resident:.2f} GiB transient")

    out_path = REPO / "results" / f"phase0_backward_diagnosis_{args.long_edge}px.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    out_path.write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "long_edge": args.long_edge,
                "seq_len": seq,
                "image_tokens": image_tokens,
                "vocab_size": vocab,
                "resident_gib": round(resident, 3),
                "card_gib": round(props.total_memory / GIB, 3),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
