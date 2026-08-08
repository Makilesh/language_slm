"""General-capability probe — the catastrophic-forgetting detector.

Fine-tuning a VLM on one narrow task degrades general visual ability, and most
portfolio projects never check. This runs a 200-question DocVQA subset and
scores it with ANLS, the standard DocVQA metric, so every headline chart run can
report what it cost in general capability.

Run it on the base model to get the "before" number, then on each fine-tuned
checkpoint with **identical settings**. Only the delta is meaningful: absolute
ANLS here is not comparable to published DocVQA numbers, because this is a
200-question subset at a fixed resolution with a fixed prompt.

ANLS, per the DocVQA paper: for each question take the best normalised
Levenshtein similarity across the accepted answers, and zero it if it falls
below 0.5 — a wrong answer should score nothing, not partial credit for
coincidental character overlap.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein

ANLS_THRESHOLD = 0.5

PROMPT = (
    "Answer the question using a single word or phrase taken from the document. "
    "Give only the answer, with no explanation.\n\nQuestion: {question}"
)

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """DocVQA's comparison normalisation: casefold and collapse whitespace."""
    return _WS.sub(" ", (text or "").strip()).casefold()


def anls_score(prediction: str, answers: list[str]) -> float:
    pred = normalize(prediction)
    best = 0.0
    for answer in answers:
        gold = normalize(answer)
        if not gold and not pred:
            best = max(best, 1.0)
            continue
        denom = max(len(gold), len(pred))
        if denom == 0:
            continue
        similarity = 1.0 - Levenshtein.distance(gold, pred) / denom
        best = max(best, similarity)
    return best if best >= ANLS_THRESHOLD else 0.0


def resize_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    w, h = image.size
    if max(w, h) <= long_edge:
        return image
    scale = long_edge / max(w, h)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


@torch.no_grad()
def answer_one(model, processor, image, question, long_edge, max_new_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT.format(question=question)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[resize_long_edge(image, long_edge)], return_tensors="pt"
    ).to("cuda")

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def run(args) -> dict:
    from datasets import load_dataset

    from src.eval.run_eval import load

    ds = load_dataset(args.dataset, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    model, processor = load(args.model, args.dtype, args.adapter)

    print(f"model={args.model} adapter={args.adapter} dtype={args.dtype}")
    print(f"probe={args.dataset}:{args.split} n={len(ds)} long_edge={args.long_edge}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows, scores = [], []
    started = time.time()

    with args.out.open("w", encoding="utf-8") as fh:
        for i, ex in enumerate(ds, 1):
            question = ex["query"]["en"] if isinstance(ex["query"], dict) else ex["query"]
            answers = ex["answers"] if isinstance(ex["answers"], list) else [ex["answers"]]

            try:
                pred = answer_one(
                    model, processor, ex["image"].convert("RGB"),
                    question, args.long_edge, args.max_new_tokens,
                )
                err = None
            except Exception as exc:  # noqa: BLE001 - a failure scores zero, not crashes
                pred, err = "", f"{type(exc).__name__}: {str(exc)[:150]}"

            score = anls_score(pred, answers)
            scores.append(score)
            row = {
                "id": ex.get("id", str(i)),
                "question": question,
                "answers": answers,
                "prediction": pred,
                "anls": round(score, 4),
                "error": err,
            }
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

            if i % 10 == 0 or i == len(ds):
                eta = (time.time() - started) / i * (len(ds) - i) / 60
                print(f"  [{i:4d}/{len(ds)}] running ANLS {sum(scores) / len(scores):.4f}"
                      f"  eta {eta:5.1f} min")

    anls = sum(scores) / len(scores) if scores else 0.0
    exact = sum(1 for r in rows if normalize(r["prediction"]) in
                [normalize(a) for a in r["answers"]]) / len(rows) if rows else 0.0

    summary = {
        "anls": round(anls, 4),
        "exact_match": round(exact, 4),
        "n": len(rows),
        "n_errors": sum(1 for r in rows if r["error"]),
        "config": {
            "model": args.model,
            "adapter": args.adapter,
            "dtype": args.dtype,
            "dataset": args.dataset,
            "split": args.split,
            "long_edge": args.long_edge,
            "max_new_tokens": args.max_new_tokens,
            "anls_threshold": ANLS_THRESHOLD,
        },
    }

    print(f"\n{'=' * 50}")
    print(f"  ANLS         {summary['anls']:.4f}")
    print(f"  exact match  {summary['exact_match']:.4f}")
    print(f"  n            {summary['n']}  (errors: {summary['n_errors']})")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {args.report}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dtype", default="4bit", choices=["4bit", "bf16"])
    ap.add_argument("--dataset", default="nielsr/docvqa_1200_examples")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("results/preds/docvqa_base.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("results/scores/docvqa_base.json"))
    ap.add_argument("--long-edge", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
