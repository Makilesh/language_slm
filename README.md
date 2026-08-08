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

Every config in this repo therefore states its resolution, expected image
tokens, and max sequence length explicitly. The measured table lives in
[`setup/VRAM_BUDGET.md`](setup/VRAM_BUDGET.md) and governs the rest of the work.

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
