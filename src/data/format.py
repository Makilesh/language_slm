"""Render training samples in the model's multimodal chat format.

This module exists to make one specific failure impossible. A fine-tune that
trains on a slightly different template than inference uses looks exactly like
a fine-tune that did not work: loss falls, eval does not move. It is the single
most common silent failure in VLM fine-tuning, and it is invisible unless you
compare the two strings.

The defence here is structural, not diligence: the user turn is built by
importing `build_messages` from `src.eval.prompts` — the *same function the
evaluation harness calls*. There is no second copy of the prompt to drift out
of sync. `verify_template_matches_inference()` then asserts that the training
prompt prefix is byte-identical to what `run_eval` would send, and the builder
runs that check on every dataset build.

What a training sample looks like end to end:

    <|im_start|>user
    <|vision_start|><|image_pad|>...<|vision_end|>{instruction}<|im_end|>
    <|im_start|>assistant
    {compact JSON}<|im_end|>
                  ^^^^^^^^^^ loss is computed only from here

Label masking. Everything up to and including the assistant header is masked to
-100, so the model is never trained to predict the image tokens or the
instruction — only the JSON answer. Getting this wrong wastes most of the
gradient on reproducing a constant prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.eval.prompts import PROMPT_VERSION, build_messages
from src.eval.schema import ChartData

# Compact separators: every space in the target is a token the model must spend
# capacity predicting, and at ~350 output tokens per chart that is a real
# fraction of the sequence budget measured in Phase 0. Pretty-printing the gold
# would inflate targets by roughly 2x for zero information gain.
JSON_SEPARATORS = (",", ":")


def target_json(gold: dict | ChartData) -> str:
    """The assistant turn: gold data as compact, key-ordered JSON.

    Key order follows the schema declaration rather than sorting, so the model
    learns one consistent field order. Sorted order would put `chart_type`
    after `series`, which is both unnatural and harder to constrain-decode.
    """
    data = gold if isinstance(gold, ChartData) else ChartData.model_validate(gold)
    return json.dumps(
        data.model_dump(mode="json"), separators=JSON_SEPARATORS, ensure_ascii=False
    )


def build_sample(row: dict, prompt_style: str = "engineered") -> dict:
    """One manifest row -> one training sample.

    The image is carried as a path, not pixels: the collator loads and resizes
    at the configured resolution, so a resolution ablation does not require
    rebuilding the dataset.
    """
    return {
        "id": row["id"],
        "image": row["image"],
        "messages": build_messages(prompt_style)
        + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_json(row["gold"])}],
            }
        ],
        "prompt_style": prompt_style,
        "prompt_version": PROMPT_VERSION,
        "source": row.get("source", "synth"),
        "degradations": row.get("degradations", []),
        "properties": row.get("properties", {}),
        "scored_fields": row.get("scored_fields"),
    }


def render_text(processor, sample: dict) -> tuple[str, str]:
    """(prompt_text, full_text) as the model will actually see them.

    Returned separately because the prompt half is exactly what must be masked
    out of the labels, and computing that boundary from a re-tokenisation is
    how off-by-one label bugs happen.
    """
    messages = sample["messages"]
    user_only = [m for m in messages if m["role"] == "user"]
    prompt_text = processor.apply_chat_template(
        user_only, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(messages, tokenize=False)
    return prompt_text, full_text


def verify_template_matches_inference(processor, sample: dict, prompt_style: str) -> None:
    """Assert the training prompt is byte-identical to the inference prompt.

    `run_eval.generate_one` applies the chat template to `build_messages(style)`
    with `add_generation_prompt=True`. A training sample's prompt half must
    equal that string exactly. Raises rather than warns: a mismatch here
    silently wastes an entire training run.
    """
    prompt_text, _ = render_text(processor, sample)
    expected = processor.apply_chat_template(
        build_messages(prompt_style), tokenize=False, add_generation_prompt=True
    )
    if prompt_text != expected:
        raise AssertionError(
            "training prompt differs from inference prompt.\n"
            f"--- training ({len(prompt_text)} chars) ---\n{prompt_text!r}\n"
            f"--- inference ({len(expected)} chars) ---\n{expected!r}"
        )


def build_labels(input_ids, prompt_len: int, pad_token_id: int | None):
    """Mask the prompt out of the labels.

    `prompt_len` comes from tokenising the prompt half separately. The image
    placeholder expands to the same number of tokens in both encodings, so the
    prefix lengths line up exactly — this is the same technique Phase 0's
    `verify_vlm.build_batch` used and verified.
    """
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    return labels


def write_samples(rows: list[dict], out: Path, prompt_style: str = "engineered") -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(build_sample(row, prompt_style), ensure_ascii=False) + "\n")
    return out


def dump_rendered(processor, samples: list[dict], out: Path) -> Path:
    """Write fully-rendered training samples for manual inspection.

    The brief asks for five samples to be eyeballed, and this is the artifact
    to eyeball: the literal strings the model trains on, with the masked span
    marked, not a summary of them.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    for i, s in enumerate(samples, 1):
        prompt_text, full_text = render_text(processor, s)
        answer = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else "<PREFIX MISMATCH>"
        chunks.append(
            f"{'=' * 78}\nSAMPLE {i}  id={s['id']}  source={s['source']}  "
            f"degradations={s['degradations']}\n{'=' * 78}\n"
            f"image: {s['image']}\n\n"
            f"--- MASKED (labels = -100) ---\n{prompt_text}\n"
            f"--- TRAINED ON (labels = token ids) ---\n{answer}\n"
        )
    out.write_text("\n".join(chunks), encoding="utf-8")
    return out


def stats(samples: list[dict]) -> dict[str, Any]:
    """Dataset composition, for the Phase 2 acceptance criteria."""
    from collections import Counter

    by_source = Counter(s["source"] for s in samples)
    by_type = Counter(
        (s.get("properties") or {}).get("chart_type", "unknown") for s in samples
    )
    by_series = Counter(
        (s.get("properties") or {}).get("n_series", "unknown") for s in samples
    )
    degraded = sum(1 for s in samples if s["degradations"])
    deg_kinds = Counter(d for s in samples for d in s["degradations"])

    return {
        "n": len(samples),
        "by_source": dict(by_source),
        "synthetic_fraction": by_source.get("synth", 0) / max(1, len(samples)),
        "by_chart_type": dict(sorted(by_type.items())),
        "by_series_count": dict(sorted(by_series.items(), key=lambda kv: str(kv[0]))),
        "n_degraded": degraded,
        "degraded_fraction": degraded / max(1, len(samples)),
        "degradation_kinds": dict(deg_kinds.most_common()),
    }
