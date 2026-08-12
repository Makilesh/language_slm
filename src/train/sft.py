"""QLoRA supervised fine-tuning for chart-to-data extraction.

Config-driven so that every run is reproducible from a committed YAML, and so
that B1 — which part of the model to train — is a one-line change rather than
an edit to this file.

Everything Phase 0 learned the hard way is applied here, and each is load-bearing:

* `force_repeat_kv()` before the model is built. Without it SDPA falls back to
  the math backend on Windows and backward costs ~4 GiB more and runs ~12x
  slower. See `setup/sdpa_compat.py`.
* **No** `prepare_model_for_kbit_training`. It upcasts every non-quantized
  module to fp32, which drags the compute path out of bf16 for no benefit here.
  The two things it is actually needed for — freezing the base and letting
  gradients reach checkpointed blocks — are done explicitly below.
* Gradient checkpointing with `use_reentrant=False`, plus
  `enable_input_require_grads()`, without which the checkpointed segments
  silently detach from the graph.
* Peak VRAM and wall-clock are recorded, because on Windows an oversized config
  does not OOM — it spills to host RAM and merely gets 10-100x slower, so time
  is the fit signal.

Label masking: only the assistant JSON contributes loss. The prompt half is
tokenised separately and asserted to be a true prefix of the full sequence, so
a template change cannot quietly shift the mask boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from setup.sdpa_compat import force_repeat_kv
from src.data.build_dataset import image_tokens_for
from src.data.format import render_text, verify_template_matches_inference

GIB = 1024**3


# --------------------------------------------------------------------------- #
# B1: which parameters to train
# --------------------------------------------------------------------------- #

# Anchored patterns, not substring matches. The vision blocks and the merger
# BOTH expose `linear_fc1` / `linear_fc2`, so a pattern like `.*linear_fc1$`
# would attach LoRA to the vision tower while claiming to train only the
# projector -- silently invalidating the entire B1 comparison.
DECODER = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
PROJECTOR = r"model\.visual\.(merger|deepstack_merger_list\.\d+)\.linear_fc[12]$"
VISION = r"model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])$"

TRAIN_MODES: dict[str, str] = {
    # B1(a) -- the recommended default: language decoder only, vision frozen.
    "decoder": DECODER,
    # B1(b) -- decoder plus the vision-language projection (main merger and the
    # deepstack mergers, which are part of the same projection path).
    "decoder_projector": f"({DECODER})|({PROJECTOR})",
    # B1(c) -- also LoRA the vision encoder itself.
    "decoder_projector_vision": f"({DECODER})|({PROJECTOR})|({VISION})",
}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    model: str = "Qwen/Qwen3-VL-4B-Instruct"
    dataset: str = "data/train/train.jsonl"
    output_dir: str = "outputs/b1_decoder"
    run_name: str = "b1_decoder"

    train_mode: str = "decoder"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    lr: float = 1e-4
    epochs: float = 2.0
    max_steps: int = -1
    batch_size: int = 1
    grad_accum: int = 16
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    long_edge: int = 448
    max_seq_len: int = 2048
    prompt_style: str = "engineered"

    quantize_vision: bool = False
    attn: str = "sdpa"
    seed: int = 42

    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    limit: int | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    def hash(self) -> str:
        """Short digest of everything that changes what is trained.

        Excludes paths and logging cadence so the same experiment run to two
        different output directories hashes identically.
        """
        payload = {
            k: v for k, v in self.__dict__.items()
            if k not in {"output_dir", "run_name", "dataset", "logging_steps",
                         "save_steps", "save_total_limit", "extra"}
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:12]


def load_config(path: Path, overrides: list[str]) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f for f in Config.__dataclass_fields__}
    extra = {k: v for k, v in raw.items() if k not in known}
    cfg = Config(**{k: v for k, v in raw.items() if k in known})
    cfg.extra = extra

    for item in overrides:
        key, _, val = item.partition("=")
        if key not in known:
            raise SystemExit(f"unknown override: {key}")
        setattr(cfg, key, _coerce(val, getattr(cfg, key)))
    return cfg


def _coerce(val: str, current: Any) -> Any:
    """Parse a CLI override to match the field's type.

    Fields whose default is None (`limit`, `max_steps` overrides) carry no type
    information at runtime, so the value is inferred from its own text. Without
    this, `--set limit=64` silently assigns the *string* "64" and fails later
    with an unhelpful slice error.
    """
    if isinstance(current, bool):
        return val.lower() in ("1", "true", "yes")
    if isinstance(current, int):
        return int(val)
    if isinstance(current, float):
        return float(val)
    if isinstance(current, str):
        return val

    if val.lower() in ("none", "null", ""):
        return None
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


class ChartDataset(Dataset):
    """Training samples, filtered to those that fit the sequence budget.

    Oversized samples are **dropped, not truncated**. Truncating a target
    teaches the model to stop mid-JSON, which is exactly the degenerate
    behaviour Phase 1 observed at inference. With max_seq_len 2048 and the
    measured max of 1,928 this should drop nothing; it exists so that lowering
    the budget fails loudly instead of quietly corrupting the targets.
    """

    def __init__(self, path: str, tokenizer, cfg: Config):
        rows = [json.loads(x) for x in Path(path).open(encoding="utf-8") if x.strip()]
        if cfg.limit:
            rows = rows[: cfg.limit]

        self.rows: list[dict] = []
        self.dropped = 0
        longest = 0

        for r in rows:
            target = r["messages"][-1]["content"][0]["text"]
            n_target = len(tokenizer(target, add_special_tokens=False)["input_ids"])
            n_image = self._image_tokens(r, cfg.long_edge)
            # +~330 for the engineered prompt and chat scaffolding; measured in
            # configs/RESOLUTION_POLICY.md.
            total = n_target + n_image + 340
            longest = max(longest, total)
            if total > cfg.max_seq_len:
                self.dropped += 1
                continue
            self.rows.append(r)

        self.longest = longest

    @staticmethod
    def _image_tokens(row: dict, long_edge: int) -> int:
        try:
            with Image.open(Path(row.get("image_root", ".")) / row["image"]) as im:
                return image_tokens_for(im.width, im.height, long_edge)
        except OSError:
            return 196  # worst case measured at 448px

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


@dataclass
class Collator:
    processor: Any
    long_edge: int
    max_seq_len: int

    def _load(self, row: dict) -> Image.Image:
        img = Image.open(Path(row.get("image_root", ".")) / row["image"]).convert("RGB")
        scale = self.long_edge / max(img.size)
        return img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.LANCZOS,
        )

    def __call__(self, features: list[dict]) -> dict:
        images = [self._load(f) for f in features]
        prompts, fulls = [], []
        for f in features:
            p, t = render_text(self.processor, f)
            prompts.append(p)
            fulls.append(t)

        batch = self.processor(
            text=fulls, images=images, return_tensors="pt", padding=True
        )
        # Second pass on the prompt half only. The image placeholder expands to
        # the same token count in both, so prompt length is a true prefix
        # length -- verified below rather than assumed.
        prompt_only = self.processor(
            text=prompts, images=images, return_tensors="pt", padding=True
        )

        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        for i in range(len(features)):
            n_prompt = int((prompt_only["attention_mask"][i] == 1).sum())
            if not torch.equal(
                prompt_only["input_ids"][i][:n_prompt], batch["input_ids"][i][:n_prompt]
            ):
                raise AssertionError(
                    f"prompt is not a prefix of the full sequence for {features[i]['id']}; "
                    "the chat template changed between the two encodings and the "
                    "label mask would be misaligned"
                )
            labels[i, :n_prompt] = -100
        if pad_id is not None:
            labels[labels == pad_id] = -100

        batch["labels"] = labels
        return dict(batch)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


def build_model(cfg: Config):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    # Must run before the model is constructed.
    force_repeat_kv(True)

    skip = ["lm_head"] if cfg.quantize_vision else ["visual", "lm_head"]
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip,
    )

    processor = AutoProcessor.from_pretrained(cfg.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model,
        quantization_config=quant,
        dtype=torch.bfloat16,
        attn_implementation=cfg.attn,
        device_map={"": 0},
    )
    model.config.use_cache = False

    # What prepare_model_for_kbit_training would do, minus the fp32 upcast.
    for p in model.parameters():
        p.requires_grad_(False)
    model.enable_input_require_grads()

    if cfg.train_mode not in TRAIN_MODES:
        raise SystemExit(
            f"unknown train_mode {cfg.train_mode!r}; choose from {sorted(TRAIN_MODES)}"
        )

    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TRAIN_MODES[cfg.train_mode],
        ),
    )
    return model, processor


def trainable_breakdown(model) -> dict[str, int]:
    buckets = {"vision_tower": 0, "projector": 0, "llm": 0, "other": 0}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "visual.blocks" in name:
            buckets["vision_tower"] += p.numel()
        elif "merger" in name:
            buckets["projector"] += p.numel()
        elif "language_model" in name:
            buckets["llm"] += p.numel()
        else:
            buckets["other"] += p.numel()
    return buckets


def assert_mode_matches_params(mode: str, b: dict[str, int]) -> None:
    """The B1 ablation is only meaningful if each arm trains what it claims.

    A regex that silently matched nothing, or matched the vision tower through
    the shared `linear_fc*` naming, would produce three runs that differ by
    less than their labels say.
    """
    if b["llm"] == 0:
        raise SystemExit(f"train_mode={mode}: no decoder params trainable")
    want_proj = mode in ("decoder_projector", "decoder_projector_vision")
    want_vis = mode == "decoder_projector_vision"

    if want_proj and b["projector"] == 0:
        raise SystemExit(f"train_mode={mode}: projector requested but nothing matched")
    if not want_proj and b["projector"]:
        raise SystemExit(f"train_mode={mode}: projector must be frozen, got {b['projector']:,}")
    if want_vis and b["vision_tower"] == 0:
        raise SystemExit(f"train_mode={mode}: vision LoRA requested but nothing matched")
    if not want_vis and b["vision_tower"]:
        raise SystemExit(
            f"train_mode={mode}: vision tower must be frozen, got {b['vision_tower']:,} "
            "(standing rule 2)"
        )


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--dry-run", action="store_true",
                    help="build everything and report, but do not train")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"RUN {cfg.run_name}   config hash {cfg.hash()}   seed {cfg.seed}")
    print("=" * 70)
    print(f"  train_mode      {cfg.train_mode}")
    print(f"  lora            r={cfg.lora_r} alpha={cfg.lora_alpha} drop={cfg.lora_dropout}")
    print(f"  optim           lr={cfg.lr} {cfg.scheduler} warmup={cfg.warmup_ratio}")
    print(f"  batch           {cfg.batch_size} x grad_accum {cfg.grad_accum} "
          f"= effective {cfg.batch_size * cfg.grad_accum}")
    print(f"  resolution      {cfg.long_edge}px, max_seq_len {cfg.max_seq_len}")

    model, processor = build_model(cfg)
    torch.cuda.synchronize()
    weights_gib = torch.cuda.max_memory_allocated() / GIB

    breakdown = trainable_breakdown(model)
    total_trainable = sum(breakdown.values())
    print(f"\n  trainable params {total_trainable:,}")
    for k, v in breakdown.items():
        print(f"    {k:14s} {v:,}")
    assert_mode_matches_params(cfg.train_mode, breakdown)
    print(f"  weights resident {weights_gib:.2f} GiB")

    dataset = ChartDataset(cfg.dataset, processor.tokenizer, cfg)
    print(f"\n  dataset          {len(dataset)} samples "
          f"(dropped {dataset.dropped} over budget, longest {dataset.longest} tokens)")

    # The check that makes the whole run trustworthy.
    verify_template_matches_inference(processor, dataset[0], cfg.prompt_style)
    print("  template         matches inference exactly")

    collator = Collator(processor, cfg.long_edge, cfg.max_seq_len)

    from transformers import Trainer, TrainingArguments

    # transformers v5 deprecates warmup_ratio in favour of warmup_steps. The
    # brief specifies a ratio, so convert rather than change the recipe, and
    # log both so the config and the run agree.
    effective_batch = cfg.batch_size * cfg.grad_accum
    steps_per_epoch = max(1, -(-len(dataset) // effective_batch))
    total_steps = cfg.max_steps if cfg.max_steps > 0 else int(steps_per_epoch * cfg.epochs)
    warmup_steps = max(1, round(total_steps * cfg.warmup_ratio))
    print(f"  schedule         {total_steps} steps "
          f"({steps_per_epoch}/epoch), warmup {warmup_steps}")

    targs = TrainingArguments(
        output_dir=str(out),
        run_name=cfg.run_name,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.lr,
        lr_scheduler_type=cfg.scheduler,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,   # our collator needs the raw columns
        dataloader_num_workers=0,      # image decode is not the bottleneck here
        report_to=[],
        seed=cfg.seed,
        # `save_safetensors` was removed in transformers v5 -- safetensors is
        # now the only serialisation path, so the flag is implicit.
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=collator,
    )

    if args.dry_run:
        batch = collator([dataset[i] for i in range(cfg.batch_size)])
        n_sup = int((batch["labels"] != -100).sum())
        print(f"\n  dry-run batch    seq={batch['input_ids'].shape}  "
              f"supervised tokens={n_sup}")
        print("  dry run complete -- not training")
        return

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    result = trainer.train()
    wall = time.time() - started
    peak = torch.cuda.max_memory_allocated() / GIB

    trainer.save_model(str(out / "adapter"))
    processor.save_pretrained(str(out / "adapter"))

    summary = {
        "run_name": cfg.run_name,
        "config_hash": cfg.hash(),
        "config": {k: v for k, v in cfg.__dict__.items() if k != "extra"},
        "trainable_params": breakdown,
        "trainable_total": total_trainable,
        "dataset": {
            "n": len(dataset),
            "dropped_over_budget": dataset.dropped,
            "longest_tokens": dataset.longest,
        },
        "train_runtime_s": round(wall, 1),
        "train_runtime_h": round(wall / 3600, 2),
        "peak_vram_gib": round(peak, 2),
        "weights_gib": round(weights_gib, 2),
        "final_loss": result.metrics.get("train_loss"),
        "steps": result.global_step,
        "vram_total_gib": round(torch.cuda.get_device_properties(0).total_memory / GIB, 2),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"  loss        {summary['final_loss']}")
    print(f"  steps       {summary['steps']}")
    print(f"  wall clock  {summary['train_runtime_h']} h")
    print(f"  peak VRAM   {summary['peak_vram_gib']} GiB "
          f"of {summary['vram_total_gib']} GiB")
    if peak > torch.cuda.get_device_properties(0).total_memory / GIB:
        print("  WARNING: exceeded physical VRAM -- host spill, timings not comparable")
    print(f"  adapter     {out / 'adapter'}")
    print(f"  summary     {out / 'run_summary.json'}")


if __name__ == "__main__":
    main()
