# Chart2Data-VLM

Fine-tuning a small vision-language model to read a chart image and emit the
underlying data as structured JSON — series names, axis labels, and numeric
values — accurately enough to reconstruct the chart.

**Positioning:** chart-to-table is an established task with existing benchmarks.
The contribution here is not the task. It is *the best chart-to-data extraction
achievable in 12 GB, with the architectural ablation showing what actually
matters* — which of the vision tower / projector / language decoder to train,
and where the image-resolution knee sits.

## Status

| Phase | State |
|---|---|
| 0 — Environment delta | **done** — see [`setup/VRAM_BUDGET.md`](setup/VRAM_BUDGET.md) |
| 1 — Evaluation harness | **done** — see [`results/baselines.md`](results/baselines.md) |
| 2 — Training data | **done** — see [`results/dataset_stats.json`](results/dataset_stats.json) |
| 3 — Fine-tuning + ablations | not started |
| 4 — Error analysis | not started |
| 5 — Quantization, deploy, release | not started |

No trained model exists yet. Every number below came out of the Phase 1 harness.

## Base-model baselines (no training)

150 synthetic charts, Qwen3-VL-4B-Instruct in 4-bit at 448px, greedy decoding.
Full table and caveats: [`results/baselines.md`](results/baselines.md).

| arm | valid JSON | chart type | structural | **value@5%** | median APE |
|---|---|---|---|---|---|
| A. minimal prompt | 94.7% | 78.0% | 14.0% | **24.7%** | 28% |
| B. engineered prompt | 96.0% | 94.0% | 28.0% | **23.9%** | 30% |
| C. + constrained decoding | 96.0% | 94.0% | 28.0% | **23.9%** | 30% |

Forgetting probe, base model: **ANLS 0.9237 / EM 0.86** on 200 DocVQA samples.
That is the "before" number every headline run gets compared against.

## Training corpus (Phase 2)

10,000 samples, contamination-verified against both eval splits.
Full stats: [`results/dataset_stats.json`](results/dataset_stats.json).
Resolution reasoning: [`configs/RESOLUTION_POLICY.md`](configs/RESOLUTION_POLICY.md).

| | |
|---|---|
| synthetic / real | 7,000 / 3,000 (70%) |
| chart types | bar 2,202 · line 1,797 · stacked_bar 1,217 · scatter 897 · pie 887 · unannotated 3,000 |
| series counts | 1: 4,876 · 2: 2,235 · 3: 2,327 · 4: 378 · 5: 184 |
| degraded (JPEG, blur, rotation, screenshot border) | 1,382 (20% of synthetic) |
| sequence length @448px | p50 601 · p99 1,280 · max 1,928 → `max_seq_len 2048` |

The image is only ~19% of the median sequence; the target JSON dominates the
budget, which is why compact separators are a real saving rather than a
micro-optimisation.

**Contamination is filtered before mixing, and the build aborts if anything
survives.** Id-based exclusion alone was not enough — the real corpus stores
some charts twice under consecutive sample ids, so five reached the training
pool with different ids and identical content. Only the content hash caught it.

Three things the baselines already settled:

- **Constrained decoding changed nothing** — arms B and C are byte-identical on
  150/150 charts. Under greedy decoding the argmax was already
  grammar-conformant, so none of the base model's JSON validity is attributable
  to grammar enforcement. That is the Phase 3 B8 question, answered up front.
- **Prompt engineering moved structure, not numbers.** A → B gained 16 points of
  chart-type accuracy and 14 of structural match while value@5% went *down* 0.8.
  Prompting is not the lever on the metric that matters.
- **The residual 4% invalid JSON is two distinct bugs.** Four are plain
  truncation at `max_new_tokens=1024` (they complete at ~1.2–1.4k). Two are
  greedy degeneration — one scatter chart emitted the same point 90 times, one
  unique point out of 90 — which no token budget fixes and which constrained
  decoding **cannot** catch, because a repetition loop is grammatically legal.

## Hardware

RTX 5070 Ti Laptop, 11.94 GiB usable VRAM, Blackwell sm_120, Windows 11.
Setup and Blackwell-specific gotchas: [`setup/ENVIRONMENT.md`](setup/ENVIRONMENT.md).

## The constraint that drives everything

Image tokens, not parameter count, are what OOM this project. Qwen3-VL uses
patch size 16 with a spatial merge of 2, so **one image token covers a 32×32
pixel region** — image token count scales with the *area* of the input, and the
processor's default `max_pixels` allows ~16k image tokens if you let it.

Measured on an RTX 5070 Ti Laptop (11.94 GiB), Qwen3-VL-4B in 4-bit NF4, LoRA
r=32 on the decoder, batch 1:

| long edge | image tokens | seq len | fwd GiB | bwd GiB | bwd ms |
|---|---|---|---|---|---|
| **448** | **126** | 623 | 3.96 | **10.03** ✅ | **528** |
| 768 | 360 | 857 | 4.34 | 12.59 ❌ | 9,933 |
| 1280 | 960 | 1457 | 5.28 | 19.21 ❌ | 118,388 |

**448px is the training operating point.** Forward-only fits at every
resolution, so evaluation is unconstrained — only training is pinned.

Two Phase 0 findings worth more than the table itself:

1. **On Windows, exceeding VRAM does not raise OOM.** WDDM spills to host RAM,
   so an oversized config runs correctly and merely gets 10–100× slower. Fit has
   to be measured in milliseconds, not caught as an exception.
2. **SDPA silently falls back to the math backend.** Qwen3-VL's grouped-query
   attention (32 Q heads, 8 KV heads) plus a Windows torch build with no
   flash-attention kernel means every attention call materialises a
   `[32, seq, seq]` matrix per layer. Forcing transformers' `repeat_kv` branch
   ([`setup/sdpa_compat.py`](setup/sdpa_compat.py)) recovers **3.7 GiB and 12×
   throughput** and emits no error either way.

Full write-up: [`results/phase0_findings.md`](results/phase0_findings.md).
Mechanical table: [`setup/VRAM_BUDGET.md`](setup/VRAM_BUDGET.md).

## Reproducing Phase 0

```bash
.venv/Scripts/python.exe setup/make_probe_chart.py
.venv/Scripts/python.exe setup/verify_vlm.py --resolutions 448 768 1280
```

## Layout

```
setup/     verify_vlm.py, VRAM_BUDGET.md, ENVIRONMENT.md, requirements
configs/   one committed YAML per experiment
src/data/  synth.py  format.py  build_dataset.py  augment.py
src/eval/  metrics.py  run_eval.py  forgetting_probe.py  report.py
src/train/ sft.py
src/serve/ api.py  demo.py  Dockerfile
results/   baselines.md  ablations.md  error_analysis.md  final.md
```
