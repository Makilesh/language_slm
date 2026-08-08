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
| 1 — Evaluation harness | not started |
| 2 — Training data | not started |
| 3 — Fine-tuning + ablations | not started |
| 4 — Error analysis | not started |
| 5 — Quantization, deploy, release | not started |

No trained model exists yet. No accuracy number appears anywhere in this repo
until the Phase 1 harness produces it.

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
