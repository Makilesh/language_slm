"""Phase 0 acceptance: resolution -> image tokens -> peak VRAM.

Loads Qwen3-VL-4B-Instruct in 4-bit, attaches LoRA to the LLM decoder only
(vision tower and merger frozen, per standing rule 2), and for each target
resolution measures:

  * the actual number of image tokens the processor emits
  * total sequence length of a realistic training sample
  * peak VRAM for a forward pass
  * peak VRAM for a forward+backward pass, with and without gradient
    checkpointing

The resulting table is the token budget that governs every later design
decision, so nothing here is estimated -- every number is read back off the
device after the op has actually run.

Usage:
    python setup/verify_vlm.py
    python setup/verify_vlm.py --resolutions 448 768 1280 1600 --out setup/VRAM_BUDGET.md
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
GIB = 1024**3

# The instruction the real task will use. Kept here so the sequence lengths
# measured in Phase 0 match what training actually sees in Phase 2.
INSTRUCTION = (
    "Extract the underlying data from this chart as JSON with keys: "
    "chart_type, title, x_label, y_label, series. Each series has a name and "
    "a list of {x, y} points. Respond with JSON only."
)

# A representative gold answer, so the label span in the backward pass is a
# realistic length rather than a token or two.
SAMPLE_ANSWER = json.dumps(
    {
        "chart_type": "bar",
        "title": "Quarterly Revenue by Region",
        "x_label": "Quarter",
        "y_label": "Revenue (thousands USD)",
        "series": [
            {
                "name": name,
                "data": [
                    {"x": f"Q{q} 202{y}", "y": round(100 + 7 * i + 11 * q + 20 * y, 1)}
                    for y in (3, 4)
                    for q in range(1, 5)
                ],
            }
            for i, name in enumerate(["North America", "EMEA", "APAC"])
        ],
    },
    separators=(",", ":"),
)


# --------------------------------------------------------------------------- #
# measurement helpers
# --------------------------------------------------------------------------- #


def cuda_reset() -> None:
    """Drop cached blocks and zero the peak counters before a measurement."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_gib() -> tuple[float, float]:
    """(allocated, reserved) peak since the last reset, in GiB.

    Reserved is the number that matters: it is what the allocator is actually
    holding from the driver, so it is what nvidia-smi shows and what OOMs.
    """
    return (
        torch.cuda.max_memory_allocated() / GIB,
        torch.cuda.max_memory_reserved() / GIB,
    )


@dataclass
class ResolutionResult:
    long_edge: int
    resized: tuple[int, int] = (0, 0)
    grid: tuple[int, int] = (0, 0)
    image_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    fwd_peak: float | None = None
    bwd_peak_nockpt: float | None = None
    bwd_peak_ckpt: float | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #


def env_report() -> dict:
    import bitsandbytes
    import transformers

    props = torch.cuda.get_device_properties(0)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "gpu": props.name,
        "vram_total_gib": round(props.total_memory / GIB, 2),
        "compute_capability": f"sm_{props.major}{props.minor}",
        "torch_arch_list": ",".join(torch.cuda.get_arch_list()),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


def load_model(attn_impl: str, quantize_vision: bool):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    # The frozen ViT is only ~0.44B params; leaving it in bf16 costs well under
    # a gigabyte and avoids pushing NF4 error through visual features we never
    # train. `--quantize-vision` exists so the B1 ablation can revisit this.
    skip = ["lm_head"] if quantize_vision else ["visual", "lm_head"]

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip,
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_cfg,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map={"": 0},
    )
    model.config.use_cache = False
    return model, processor


def attach_lora(model, rank: int, alpha: int):
    """LoRA on the LLM decoder linear layers only.

    Anchored on `language_model` so the pattern cannot accidentally reach into
    the vision tower or the merger -- this is standing rule 2, enforced in code
    rather than trusted to naming conventions.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
        ),
    )
    return get_peft_model(model, cfg)


def trainable_breakdown(model) -> dict:
    """Trainable params split by component -- the proof the ViT is frozen."""
    buckets = {"vision": 0, "llm": 0, "other": 0}
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if not param.requires_grad:
            continue
        if "visual" in name:
            buckets["vision"] += param.numel()
        elif "language_model" in name:
            buckets["llm"] += param.numel()
        else:
            buckets["other"] += param.numel()
    return {
        "total_params": total,
        "trainable_vision": buckets["vision"],
        "trainable_llm": buckets["llm"],
        "trainable_other": buckets["other"],
        "trainable_total": sum(buckets.values()),
    }


# --------------------------------------------------------------------------- #
# sample construction
# --------------------------------------------------------------------------- #


def resize_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    w, h = image.size
    scale = long_edge / max(w, h)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def build_batch(processor, image: Image.Image, device: str = "cuda"):
    """Tokenize one training-shaped sample and mask the prompt out of the labels.

    Returns (inputs, image_token_count, prompt_len). The prompt length is taken
    from a separate prompt-only encoding: the image placeholder expands to the
    same token count in both, so the prefix lengths line up exactly.
    """
    user_turn = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}],
        }
    ]
    full_turn = user_turn + [
        {"role": "assistant", "content": [{"type": "text", "text": SAMPLE_ANSWER}]}
    ]

    prompt_text = processor.apply_chat_template(
        user_turn, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(full_turn, tokenize=False)

    prompt_only = processor(text=[prompt_text], images=[image], return_tensors="pt")
    inputs = processor(text=[full_text], images=[image], return_tensors="pt")

    prompt_len = int(prompt_only["input_ids"].shape[1])
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    image_tokens = int((inputs["input_ids"] == image_token_id).sum())

    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels

    inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()
    }
    return inputs, image_tokens, prompt_len


# --------------------------------------------------------------------------- #
# passes
# --------------------------------------------------------------------------- #


def measure_forward(model, inputs) -> tuple[float, float]:
    cuda_reset()
    with torch.no_grad():
        out = model(**inputs)
    loss = float(out.loss)
    torch.cuda.synchronize()
    _, reserved = peak_gib()
    del out
    return reserved, loss


def measure_backward(model, inputs, grad_checkpointing: bool) -> float:
    if grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_disable()

    cuda_reset()
    out = model(**inputs)
    out.loss.backward()
    torch.cuda.synchronize()
    _, reserved = peak_gib()

    model.zero_grad(set_to_none=True)
    del out
    cuda_reset()
    return reserved


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def run(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable -- Phase 0 cannot produce a VRAM budget.")

    env = env_report()
    print("=" * 78)
    print("ENVIRONMENT")
    print("=" * 78)
    for k, v in env.items():
        print(f"  {k:22s} {v}")

    image_path = Path(args.image)
    if not image_path.exists():
        from setup.make_probe_chart import render

        render(image_path)
    source = Image.open(image_path).convert("RGB")
    print(f"\n  probe image            {image_path}  {source.width}x{source.height}")

    print("\n" + "=" * 78)
    print(f"LOADING {MODEL_ID} in 4-bit NF4 (attn={args.attn})")
    print("=" * 78)
    cuda_reset()
    model, processor = load_model(args.attn, args.quantize_vision)
    torch.cuda.synchronize()
    _, weights_reserved = peak_gib()
    print(f"  weights resident       {weights_reserved:.2f} GiB reserved")

    model = attach_lora(model, args.lora_r, args.lora_alpha)
    breakdown = trainable_breakdown(model)
    print(
        f"  trainable params       {breakdown['trainable_total']:,} / "
        f"{breakdown['total_params']:,} "
        f"({100 * breakdown['trainable_total'] / breakdown['total_params']:.3f}%)"
    )
    print(f"    in vision tower      {breakdown['trainable_vision']:,}   <- must be 0")
    print(f"    in LLM decoder       {breakdown['trainable_llm']:,}")
    print(f"    elsewhere            {breakdown['trainable_other']:,}")
    if breakdown["trainable_vision"] != 0:
        raise SystemExit("vision tower is not frozen -- violates standing rule 2")

    torch.cuda.synchronize()
    _, after_lora = peak_gib()

    results: list[ResolutionResult] = []
    merge_px = processor.image_processor.patch_size * processor.image_processor.merge_size

    for long_edge in args.resolutions:
        res = ResolutionResult(long_edge=long_edge)
        print("\n" + "-" * 78)
        print(f"RESOLUTION {long_edge}px long edge")
        print("-" * 78)

        image = resize_long_edge(source, long_edge)
        res.resized = image.size

        try:
            inputs, image_tokens, prompt_len = build_batch(processor, image)
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            res.notes.append(f"tokenization failed: {exc}")
            results.append(res)
            print(f"  FAILED: {exc}")
            continue

        thw = inputs.get("image_grid_thw")
        if thw is not None:
            t, gh, gw = (int(v) for v in thw[0])
            res.grid = (gh // processor.image_processor.merge_size,
                        gw // processor.image_processor.merge_size)

        res.image_tokens = image_tokens
        res.prompt_tokens = prompt_len
        res.total_tokens = int(inputs["input_ids"].shape[1])

        print(f"  resized to             {res.resized[0]}x{res.resized[1]}")
        print(f"  token grid             {res.grid[0]}x{res.grid[1]} "
              f"({merge_px}px per token)")
        print(f"  IMAGE TOKENS           {res.image_tokens}")
        print(f"  prompt tokens (total)  {res.prompt_tokens}")
        print(f"  full seq len           {res.total_tokens}")
        print(f"  image share of seq     "
              f"{100 * res.image_tokens / max(1, res.total_tokens):.1f}%")

        for label, fn in (
            ("forward", lambda: measure_forward(model, inputs)),
            ("backward (no ckpt)", lambda: measure_backward(model, inputs, False)),
            ("backward (ckpt)", lambda: measure_backward(model, inputs, True)),
        ):
            try:
                out = fn()
                peak, loss = out if isinstance(out, tuple) else (out, None)
                if label == "forward":
                    res.fwd_peak = peak
                    print(f"  {label:22s} peak {peak:6.2f} GiB   loss {loss:.4f}")
                else:
                    if "no ckpt" in label:
                        res.bwd_peak_nockpt = peak
                    else:
                        res.bwd_peak_ckpt = peak
                    print(f"  {label:22s} peak {peak:6.2f} GiB")
            except torch.OutOfMemoryError as exc:
                res.notes.append(f"{label}: OOM")
                print(f"  {label:22s} OOM")
                model.zero_grad(set_to_none=True)
                cuda_reset()
                del exc

        del inputs
        cuda_reset()
        results.append(res)

    write_report(args.out, env, results, breakdown, weights_reserved, after_lora, args)
    print("\n" + "=" * 78)
    print(f"wrote {args.out}")
    print("=" * 78)


def write_report(out, env, results, breakdown, weights_reserved, after_lora, args) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def cell(v: float | None) -> str:
        return "OOM" if v is None else f"{v:.2f}"

    lines = [
        "# VRAM budget — Phase 0 measurements",
        "",
        "Produced by `setup/verify_vlm.py`. Every number below was read off the",
        "device after the operation actually ran; nothing is estimated.",
        "",
        "## Environment",
        "",
        "| key | value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in env.items()]

    lines += [
        "",
        "## Configuration",
        "",
        "| key | value |",
        "|---|---|",
        f"| model | `{MODEL_ID}` |",
        "| quantization | 4-bit NF4, double quant, bf16 compute |",
        f"| vision tower quantized | {args.quantize_vision} |",
        f"| attention impl | `{args.attn}` |",
        f"| LoRA | r={args.lora_r}, alpha={args.lora_alpha}, dropout=0.05 |",
        "| LoRA targets | LLM decoder q/k/v/o/gate/up/down only |",
        "| batch size | 1 |",
        f"| px per image token | {args.px_per_token} |",
        "",
        "## Trainable parameters",
        "",
        "| bucket | params |",
        "|---|---|",
        f"| total (base + adapters) | {breakdown['total_params']:,} |",
        f"| trainable — vision tower | {breakdown['trainable_vision']:,} |",
        f"| trainable — LLM decoder | {breakdown['trainable_llm']:,} |",
        f"| trainable — elsewhere | {breakdown['trainable_other']:,} |",
        f"| trainable — total | {breakdown['trainable_total']:,} "
        f"({100 * breakdown['trainable_total'] / breakdown['total_params']:.3f}%) |",
        "",
        f"Weights resident after 4-bit load: **{weights_reserved:.2f} GiB** reserved.",
        f"After attaching LoRA: **{after_lora:.2f} GiB**.",
        "",
        "## THE TABLE — resolution vs image tokens vs peak VRAM",
        "",
        "| long edge | resized | token grid | image tokens | full seq len | "
        "fwd peak (GiB) | bwd peak, no ckpt (GiB) | bwd peak, ckpt (GiB) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        lines.append(
            f"| {r.long_edge}px | {r.resized[0]}x{r.resized[1]} | "
            f"{r.grid[0]}x{r.grid[1]} | **{r.image_tokens}** | {r.total_tokens} | "
            f"{cell(r.fwd_peak)} | {cell(r.bwd_peak_nockpt)} | {cell(r.bwd_peak_ckpt)} |"
        )

    notes = [(r.long_edge, n) for r in results for n in r.notes]
    if notes:
        lines += ["", "### Notes", ""]
        lines += [f"- {le}px: {n}" for le, n in notes]

    lines += [
        "",
        "## Machine-readable",
        "",
        "```json",
        json.dumps(
            {
                "env": env,
                "config": {
                    "model": MODEL_ID,
                    "attn": args.attn,
                    "quantize_vision": args.quantize_vision,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "px_per_image_token": args.px_per_token,
                },
                "weights_reserved_gib": round(weights_reserved, 3),
                "rows": [
                    {
                        "long_edge": r.long_edge,
                        "resized": list(r.resized),
                        "grid": list(r.grid),
                        "image_tokens": r.image_tokens,
                        "total_tokens": r.total_tokens,
                        "fwd_peak_gib": r.fwd_peak,
                        "bwd_peak_nockpt_gib": r.bwd_peak_nockpt,
                        "bwd_peak_ckpt_gib": r.bwd_peak_ckpt,
                        "notes": r.notes,
                    }
                    for r in results
                ],
            },
            indent=2,
        ),
        "```",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[448, 768, 1280])
    parser.add_argument("--image", default=str(REPO / "assets" / "demo" / "probe_chart.png"))
    parser.add_argument("--out", default=str(REPO / "setup" / "VRAM_BUDGET.md"))
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument(
        "--quantize-vision",
        action="store_true",
        help="also NF4-quantize the frozen vision tower (default: keep it bf16)",
    )
    args = parser.parse_args()
    args.px_per_token = 32  # patch_size 16 * spatial_merge_size 2
    run(args)


if __name__ == "__main__":
    main()
