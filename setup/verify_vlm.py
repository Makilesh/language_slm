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
import time
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


# Torch raises at least three different things for an allocation failure. The
# clean `OutOfMemoryError` only appears when the caching allocator itself
# refuses; once Windows WDDM has already spilled into host memory the failure
# surfaces from the driver as AcceleratorError instead, which is what killed the
# first Phase 0 run.
OOM_ERRORS = tuple(
    e
    for e in (
        getattr(torch, "OutOfMemoryError", None),
        getattr(torch, "AcceleratorError", None),
        torch.cuda.OutOfMemoryError,
        RuntimeError,
    )
    if e is not None
)


def oom_reason(exc: BaseException) -> str:
    text = str(exc).lower()
    if "out of memory" in text or "outofmemory" in text:
        return "OOM"
    return f"FAILED: {type(exc).__name__}"


def recover(model) -> bool:
    """Try to make the CUDA context usable again after an allocation failure.

    A driver-level `AcceleratorError` poisons the context badly enough that
    even `empty_cache()` raises, so every step here is individually guarded --
    an unguarded cleanup call turns a recorded OOM into a crashed run, which is
    exactly what it did on the first attempt. Returns whether the device still
    works.
    """
    for step in (
        lambda: model.zero_grad(set_to_none=True),
        gc.collect,
        torch.cuda.empty_cache,
        torch.cuda.reset_peak_memory_stats,
    ):
        try:
            step()
        except Exception:  # noqa: BLE001 - already in a failure path
            pass
    try:
        torch.zeros(8, device="cuda").sum().item()
        return True
    except Exception:  # noqa: BLE001
        return False


def spill_flag(peak: float | None) -> str:
    """Mark measurements that exceeded physical VRAM.

    Anything above the card's real capacity is being served from host memory
    over PCIe. It did not OOM, but it is not a usable operating point either.
    """
    if peak is None:
        return ""
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    return "  <-- EXCEEDS VRAM (host spill)" if peak > total else ""


def fmt_pass(label: str, s: "PassStats", extra: str = "") -> str:
    return (
        f"  {label:22s} alloc {s.allocated:6.2f} / rsvd {s.reserved:6.2f} GiB"
        f"  {s.seconds * 1000:8.0f} ms{spill_flag(s.allocated)}{extra}"
    )


@dataclass
class ResolutionResult:
    long_edge: int
    resized: tuple[int, int] = (0, 0)
    grid: tuple[int, int] = (0, 0)
    image_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    fwd: "PassStats | None" = None
    bwd_nockpt: "PassStats | None" = None
    bwd_ckpt: "PassStats | None" = None
    ckpt_active: bool | None = None
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


def load_model(attn_impl: str, quantize_vision: bool, repeat_kv: bool = True):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    from setup.sdpa_compat import force_repeat_kv

    # Without this, SDPA drops to the math backend on Windows and backward costs
    # ~4 GiB more and runs ~12x slower. See setup/sdpa_compat.py for the data.
    force_repeat_kv(repeat_kv)

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


def dtype_census(model) -> dict[str, float]:
    """GiB of parameter memory per dtype.

    `prepare_model_for_kbit_training` silently upcasts every non-quantized
    module -- embeddings, lm_head, and the entire vision tower -- to fp32. On a
    4B VLM that is several gigabytes and it also drags the compute path out of
    bf16. This census is how we catch it instead of trusting the recipe.
    """
    out: dict[str, float] = {}
    for _, p in model.named_parameters():
        key = str(p.dtype).replace("torch.", "")
        out[key] = out.get(key, 0.0) + p.numel() * p.element_size() / GIB
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def attach_lora(model, rank: int, alpha: int, use_prepare: bool):
    """LoRA on the LLM decoder linear layers only.

    Anchored on `language_model` so the pattern cannot accidentally reach into
    the vision tower or the merger -- this is standing rule 2, enforced in code
    rather than trusted to naming conventions.

    `use_prepare` selects between peft's `prepare_model_for_kbit_training` and
    doing by hand only the two things we actually need from it (freeze the base,
    let gradients flow back through the frozen embedding into checkpointed
    blocks). The stock helper's fp32 upcast is measured in Phase 0 rather than
    assumed harmless -- see results/phase0_fp32_upcast.md.
    """
    from peft import LoraConfig, get_peft_model

    if use_prepare:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        # Gradient checkpointing recomputes each block from its inputs, so the
        # block input must carry grad_fn. With a frozen embedding it does not,
        # and every checkpointed segment silently detaches.
        model.enable_input_require_grads()

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


def checkpointing_active(model) -> bool:
    """Whether any submodule actually has gradient checkpointing switched on.

    Calling `gradient_checkpointing_enable()` on a PEFT wrapper does not always
    reach the inner decoder, and the failure is silent -- memory just never
    drops. Read the flag back rather than trusting the call.
    """
    return any(getattr(m, "gradient_checkpointing", False) for m in model.modules())


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


@dataclass
class PassStats:
    """One measured pass.

    `allocated` is the real requirement -- the high-water mark of live tensors.
    `reserved` is what the caching allocator held from the driver; it is sticky
    across measurements and inflated by fragmentation, so it consistently
    overstates need. Report both, size the budget on allocated, and use
    `seconds` to catch host spill, which neither memory number reveals on
    Windows.
    """

    allocated: float
    reserved: float
    seconds: float


def timed(fn, max_iters: int = 3, slow_threshold: float = 3.0) -> PassStats:
    # One untimed warmup: the first call pays cuDNN/cuBLAS autotuning and
    # allocator growth that would otherwise land in the median.
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    warmup_secs = time.perf_counter() - t0

    # A pass that is already spilling to host memory takes tens of seconds.
    # Repeating it three times buys precision nobody needs on a number whose
    # only job is to say "this does not fit".
    iters = 1 if warmup_secs > slow_threshold else max_iters

    cuda_reset()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    allocated, reserved = peak_gib()
    times.sort()
    return PassStats(allocated, reserved, times[len(times) // 2])


def measure_forward(model, inputs) -> tuple[PassStats, float]:
    holder = {}

    def once():
        with torch.no_grad():
            out = model(**inputs)
        holder["loss"] = float(out.loss)
        del out

    stats = timed(once)
    return stats, holder["loss"]


def measure_backward(model, inputs, grad_checkpointing: bool) -> tuple[PassStats, bool]:
    if grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_disable()

    active = checkpointing_active(model)

    def once():
        out = model(**inputs)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        del out

    stats = timed(once)
    model.zero_grad(set_to_none=True)
    cuda_reset()
    return stats, active


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
    model, processor = load_model(args.attn, args.quantize_vision, not args.no_repeat_kv)
    from setup.sdpa_compat import sdpa_backend_flags

    print(f"  sdp backends           {sdpa_backend_flags()}")
    print(f"  force repeat_kv        {not args.no_repeat_kv}")
    torch.cuda.synchronize()
    _, weights_reserved = peak_gib()
    print(f"  weights resident       {weights_reserved:.2f} GiB reserved")

    model = attach_lora(model, args.lora_r, args.lora_alpha, args.use_prepare)
    census = dtype_census(model)
    print(f"  prepare_for_kbit       {args.use_prepare}")
    print("  param memory by dtype  " + ", ".join(f"{k} {v:.2f} GiB" for k, v in census.items()))

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
    free, total = torch.cuda.mem_get_info()
    print(f"  resident after LoRA    {after_lora:.2f} GiB reserved  "
          f"({(total - free) / GIB:.2f} GiB used on device, {free / GIB:.2f} GiB free)")

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

        try:
            stats, loss = measure_forward(model, inputs)
            res.fwd = stats
            print(fmt_pass("forward", stats, f"  loss {loss:.4f}"))
        except OOM_ERRORS as exc:
            res.notes.append(f"forward: {oom_reason(exc)}")
            print(f"  {'forward':22s} {oom_reason(exc)}")
            recover(model)

        for label, ckpt in (("backward (no ckpt)", False), ("backward (ckpt)", True)):
            try:
                stats, active = measure_backward(model, inputs, ckpt)
                if ckpt:
                    res.bwd_ckpt = stats
                    res.ckpt_active = active
                    if not active:
                        res.notes.append(
                            "gradient checkpointing requested but not active on any module"
                        )
                else:
                    res.bwd_nockpt = stats
                flag = "" if not ckpt else ("" if active else "  <-- CKPT NOT ACTIVE")
                print(fmt_pass(label, stats, flag))
            except OOM_ERRORS as exc:
                reason = oom_reason(exc)
                res.notes.append(f"{label}: {reason}")
                print(f"  {label:22s} {reason}")
                if not recover(model):
                    res.notes.append("CUDA context unusable after OOM; aborting run")
                    print("  CUDA context unusable after OOM -- aborting remaining passes")
                    results.append(res)
                    write_report(args.out, env, results, breakdown,
                                 weights_reserved, after_lora, census, args)
                    print(f"\nwrote {args.out} (partial)")
                    return

        del inputs
        cuda_reset()
        results.append(res)
        # Written after every resolution so a hard CUDA abort still leaves the
        # rows we already paid for on disk.
        write_report(args.out, env, results, breakdown, weights_reserved,
                     after_lora, census, args)

    print("\n" + "=" * 78)
    print(f"wrote {args.out}")
    print("=" * 78)


def write_report(out, env, results, breakdown, weights_reserved, after_lora, census, args) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vram_total = torch.cuda.get_device_properties(0).total_memory / GIB

    def gib(s: "PassStats | None") -> str:
        if s is None:
            return "OOM"
        # Bold-flag anything the card cannot actually hold: it ran, but only by
        # spilling to host memory over PCIe.
        return f"**{s.allocated:.2f}**" if s.allocated > vram_total else f"{s.allocated:.2f}"

    def ms(s: "PassStats | None") -> str:
        return "—" if s is None else f"{s.seconds * 1000:.0f}"

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
        f"| prepare_model_for_kbit_training | {args.use_prepare} |",
        f"| force GQA repeat_kv | {not args.no_repeat_kv} |",
        "",
        "### Parameter memory by dtype",
        "",
        "| dtype | GiB |",
        "|---|---|",
        *[f"| {k} | {v:.2f} |" for k, v in census.items()],
        "",
        "`uint8` is NF4-packed weight storage (two 4-bit params per byte), so",
        "parameter *counts* below read low against the model's nominal 4B.",
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
        f"Card capacity is **{vram_total:.2f} GiB**. GiB columns are peak *allocated*",
        "(live tensors), not reserved — the caching allocator's reserved pool is sticky",
        "across measurements and overstates the requirement. Values in **bold** exceed",
        "card capacity: on Windows the WDDM driver serves the overflow from host RAM",
        "over PCIe instead of raising OOM, so those rows *ran* but are not usable",
        "operating points. The millisecond columns are what expose that — read them.",
        "",
        "| long edge | resized | grid | image tokens | seq len | "
        "fwd GiB | fwd ms | bwd GiB (no ckpt) | bwd ms | bwd GiB (ckpt) | bwd ms | ckpt active |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        lines.append(
            f"| {r.long_edge}px | {r.resized[0]}x{r.resized[1]} | "
            f"{r.grid[0]}x{r.grid[1]} | **{r.image_tokens}** | {r.total_tokens} | "
            f"{gib(r.fwd)} | {ms(r.fwd)} | "
            f"{gib(r.bwd_nockpt)} | {ms(r.bwd_nockpt)} | "
            f"{gib(r.bwd_ckpt)} | {ms(r.bwd_ckpt)} | "
            f"{'—' if r.ckpt_active is None else ('yes' if r.ckpt_active else 'NO')} |"
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
                    "prepare_for_kbit": args.use_prepare,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "px_per_image_token": args.px_per_token,
                },
                "vram_total_gib": round(vram_total, 3),
                "weights_reserved_gib": round(weights_reserved, 3),
                "param_gib_by_dtype": {k: round(v, 3) for k, v in census.items()},
                "rows": [
                    {
                        "long_edge": r.long_edge,
                        "resized": list(r.resized),
                        "grid": list(r.grid),
                        "image_tokens": r.image_tokens,
                        "total_tokens": r.total_tokens,
                        **{
                            f"{name}_{stat}": val
                            for name, s in (
                                ("fwd", r.fwd),
                                ("bwd_nockpt", r.bwd_nockpt),
                                ("bwd_ckpt", r.bwd_ckpt),
                            )
                            for stat, val in (
                                ("alloc_gib", None if s is None else round(s.allocated, 3)),
                                ("rsvd_gib", None if s is None else round(s.reserved, 3)),
                                ("ms", None if s is None else round(s.seconds * 1000)),
                            )
                        },
                        "ckpt_active": r.ckpt_active,
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
    parser.add_argument(
        "--no-repeat-kv",
        action="store_true",
        help="disable the GQA repeat_kv patch and let SDPA fall back to the math "
        "backend (much slower on Windows; see setup/sdpa_compat.py)",
    )
    parser.add_argument(
        "--use-prepare",
        action="store_true",
        help="use peft.prepare_model_for_kbit_training, which upcasts non-quantized "
        "modules to fp32 (default: off; freeze + enable_input_require_grads by hand)",
    )
    args = parser.parse_args()
    args.px_per_token = 32  # patch_size 16 * spatial_merge_size 2
    run(args)


if __name__ == "__main__":
    main()
