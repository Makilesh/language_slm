"""Run a model over a chart manifest and score the predictions.

One entry point for every baseline and every fine-tuned checkpoint, so numbers
stay comparable. Everything that could change a score is a recorded flag:
prompt style, constrained decoding, image resolution, dtype, adapter path,
decoding params, seed.

Constrained decoding is a flag rather than a default because Phase 3's B8
ablation exists to separate "the model learned the format" from "the grammar
enforced the format". Runs with and without it are both first-class.

Predictions are written incrementally and the run is resumable: a 150-chart
sweep at ~1 minute per chart is long enough that losing it to a crash matters.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from src.eval.metrics import aggregate, score_sample
from src.eval.prompts import PROMPT_VERSION, build_messages
from src.eval.schema import ChartData, parse_prediction

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
GIB = 1024**3


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


def load(model_id: str, dtype: str, adapter: str | None, attn: str = "sdpa"):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    from setup.sdpa_compat import force_repeat_kv

    # Without this SDPA drops to the math backend on Windows. See
    # setup/sdpa_compat.py -- it costs 3.7 GiB and 12x throughput.
    force_repeat_kv(True)

    kwargs: dict = {"dtype": torch.bfloat16, "attn_implementation": attn, "device_map": {"": 0}}
    if dtype == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "lm_head"],
        )

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return model, processor


def compile_grammar(processor, model):
    """Compile the ChartData JSON schema into an XGrammar grammar.

    Returns the *compiled grammar*, not a logits processor. xgrammar's
    `LogitsProcessor` carries per-sequence matcher state and is explicitly
    single-use ("For each generate() call, instantiate a new one") -- it has no
    reset. Compilation is the expensive step and is safe to cache; building a
    processor around an already-compiled grammar is cheap, so the split here is
    what makes per-sample construction affordable.
    """
    import xgrammar as xgr

    # vocab_size must be the model's logits width, not len(tokenizer): Qwen pads
    # the embedding table past the tokenizer's vocabulary, and a mismatch here
    # silently misaligns the bitmask.
    vocab_size = model.config.get_text_config().vocab_size
    info = xgr.TokenizerInfo.from_huggingface(processor.tokenizer, vocab_size=vocab_size)
    compiler = xgr.GrammarCompiler(info)
    return compiler.compile_json_schema(ChartData)


def make_grammar_processor(compiled_grammar):
    """Fresh single-use logits processor for one generate() call.

    xgrammar applies its token bitmask with a Triton kernel whenever `scores`
    is on CUDA, and Triton has no Windows build -- the stock processor raises
    `ImportError: Triton is not installed` on every token. Its own code already
    has a CPU branch, so handing it CPU scores and moving the result back takes
    that branch instead.

    Cost is one vocab-wide round trip per generated token (151,936 floats,
    ~0.6 MB), which is noise next to a 4B forward pass. Correctness is
    unaffected: the mask is identical, only where it is applied changes.
    """
    from xgrammar.contrib.hf import LogitsProcessor

    class CpuBitmaskLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            device = scores.device
            masked = super().__call__(input_ids, scores.to("cpu"))
            return masked.to(device)

    return CpuBitmaskLogitsProcessor(compiled_grammar)


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #


def resize_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    w, h = image.size
    scale = long_edge / max(w, h)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


@torch.no_grad()
def generate_one(model, processor, image, prompt_style, max_new_tokens, compiled_grammar):
    messages = build_messages(prompt_style)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

    kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,          # greedy, so reruns are reproducible
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }
    if compiled_grammar is not None:
        # Single-use by design: a new processor for every generate() call.
        kwargs["logits_processor"] = [make_grammar_processor(compiled_grammar)]

    out = model.generate(**inputs, **kwargs)
    generated = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True), int(generated.shape[0])


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def load_predictions(path: Path, drop_errors: bool = False) -> dict[str, dict]:
    """Predictions keyed by sample id; later rows win over earlier ones.

    The file is append-only, so a retried sample appears twice and the newer
    row must be the one that survives.

    `drop_errors` distinguishes the two callers, which want opposite things:

    * **resume** (True) — a row that recorded a `gen_error` is work still owed,
      not work done. Counting it as done freezes a transient fault, or a bug
      like the single-use grammar processor, permanently into the results.
    * **scoring** (False) — a generation failure is a real failure and has to
      stay in the denominator. Dropping it would quietly shrink n and turn a
      150/150 wipeout into a clean-looking n=0.
    """
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if drop_errors and row.get("gen_error"):
            out.pop(row["id"], None)
            continue
        out[row["id"]] = row
    return out


def run(args) -> None:
    manifest_dir = args.manifest.parent
    rows = [json.loads(x) for x in args.manifest.open(encoding="utf-8") if x.strip()]
    if args.limit:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_predictions(args.out, drop_errors=True) if args.resume else {}
    if done:
        print(f"resuming: {len(done)} predictions already present")

    model, processor = load(args.model, args.dtype, args.adapter, args.attn)
    compiled_grammar = compile_grammar(processor, model) if args.constrained else None

    print(f"model={args.model} dtype={args.dtype} adapter={args.adapter}")
    print(f"prompt={args.prompt} (v{PROMPT_VERSION}) constrained={args.constrained} "
          f"long_edge={args.long_edge} max_new_tokens={args.max_new_tokens}")
    print(f"scoring {len(rows)} charts\n")

    started = time.time()
    fh = args.out.open("a", encoding="utf-8")
    try:
        for i, row in enumerate(rows, 1):
            if row["id"] in done:
                continue

            image = resize_long_edge(
                Image.open(manifest_dir / row["image"]).convert("RGB"), args.long_edge
            )
            t0 = time.time()
            try:
                text, n_tokens = generate_one(
                    model, processor, image, args.prompt, args.max_new_tokens, compiled_grammar
                )
                error = None
            except Exception as exc:  # noqa: BLE001 - a generation failure is a result
                text, n_tokens, error = "", 0, f"{type(exc).__name__}: {str(exc)[:200]}"
            secs = time.time() - t0

            record = {
                "id": row["id"],
                "raw": text,
                "n_tokens": n_tokens,
                "seconds": round(secs, 2),
                "gen_error": error,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

            rate = (time.time() - started) / i
            eta = rate * (len(rows) - i) / 60
            print(f"  [{i:4d}/{len(rows)}] {row['id']}  {n_tokens:4d} tok  "
                  f"{secs:6.1f}s  eta {eta:5.1f} min"
                  + (f"  ERROR {error}" if error else ""))
    finally:
        fh.close()

    score(args.manifest, args.out, args.report, vars(args))


def score(manifest: Path, preds_path: Path, report: Path | None, config: dict) -> dict:
    """Score a prediction file against its manifest."""
    gold_by_id = {
        r["id"]: r for r in (json.loads(x) for x in manifest.open(encoding="utf-8") if x.strip())
    }
    preds = load_predictions(preds_path)

    samples = []
    for sid, row in preds.items():
        if sid not in gold_by_id:
            continue
        grow = gold_by_id[sid]
        gold = ChartData.model_validate(grow["gold"])
        parsed, err, repairs = parse_prediction(row["raw"])
        if row.get("gen_error"):
            parsed, err = None, row["gen_error"]
        samples.append(
            score_sample(
                gold, parsed, sid,
                parse_error=err,
                scored_fields=set(grow["scored_fields"]) if "scored_fields" in grow else None,
                repairs=repairs,
            )
        )

    agg = aggregate(samples)
    keys = {"model", "dtype", "adapter", "prompt", "constrained", "long_edge",
            "max_new_tokens", "attn", "manifest", "out"}
    agg["config"] = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in config.items() if k in keys
    }
    agg["config"]["prompt_version"] = PROMPT_VERSION

    # `--score-only` re-scores existing predictions, so its argv describes the
    # *scoring* run, not the generation that produced them. Writing it verbatim
    # silently rewrote `constrained: true` to false on a re-score, mislabelling
    # how the predictions were made. Generation settings are carried over from
    # the previous report instead; only scoring-time fields may change.
    if config.get("score_only") and report and Path(report).exists():
        try:
            prev = json.loads(Path(report).read_text(encoding="utf-8")).get("config", {})
        except (OSError, json.JSONDecodeError):
            prev = {}
        generation_keys = keys - {"manifest", "out"}
        for k in generation_keys:
            if k in prev:
                agg["config"][k] = prev[k]
        agg["config"]["rescored"] = True
    agg["timing"] = {
        "median_seconds": sorted(r["seconds"] for r in preds.values())[len(preds) // 2]
        if preds else None,
        "mean_tokens": sum(r["n_tokens"] for r in preds.values()) / len(preds) if preds else None,
    }

    def show(label: str, value: float | None, fmt: str = "{:.1%}") -> None:
        print(f"  {label:16s}{'n/a' if value is None else fmt.format(value)}")

    o = agg["overall"]
    print(f"\n{'=' * 60}\nn={o['n']}")
    show("valid JSON", o["valid_json_rate"])
    show("schema-exact", o["schema_conformance_rate"])
    print(f"  {'repaired':16s}{o['n_repaired']}")
    show("chart type", o["chart_type_accuracy"])
    show("series names", o["series_name_accuracy"])
    show("structural", o["exact_structural_match"])
    show("point recall", o["point_recall"])
    for tol, v in o["value_accuracy"].items():
        show(f"value @{tol}", v)
    show("median APE", o["median_ape"], "{:.1f}%")
    show("MAPE", o["mape"], "{:.1f}%")
    if agg.get("repairs"):
        print(f"  repairs: {dict(list(agg['repairs'].items())[:5])}")

    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print(f"\nwrote {report}")
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="predictions JSONL")
    ap.add_argument("--report", type=Path, default=None, help="scored metrics JSON")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dtype", default="4bit", choices=["4bit", "bf16"])
    ap.add_argument("--prompt", default="minimal", choices=["minimal", "engineered"])
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--long-edge", type=int, default=448)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    if args.score_only:
        score(args.manifest, args.out, args.report, vars(args))
    else:
        run(args)


if __name__ == "__main__":
    main()
