# Phase 1 baselines - Qwen3-VL-4B-Instruct, no training

Base-model numbers on the 150-chart synthetic eval set, before any training.
Every number here was produced by `src/eval/run_eval.py`; nothing is estimated.

Reproduce with:

```powershell
powershell -NoProfile -File scripts/run_baselines.ps1
```

## What the arms isolate

| arm | change from previous | question it answers |
|---|---|---|
| `base_minimal` | — | what does the model do when simply asked? |
| `base_engineered` | prompt only | how much is prompt engineering worth? |
| `base_constrained` | + grammar enforcement | how much is grammar enforcement worth? |

Holding everything else fixed (same model, 4-bit, 448px, same 150 charts, same
seed) is what makes the deltas between rows attributable.

## Reading the two JSON columns

`schema-exact` is the fraction of outputs matching the schema with **zero** key
substitutions. `valid JSON` is the fraction usable after aliasing well-known key
variants (`points`→`data` and similar); every substitution is recorded in the
`repairs` block of each score file.

They are reported separately because collapsing them destroys information. The
first `base_minimal` run scored **0% valid JSON on 150/150 charts** — not
because the model failed, but because prompt v1.0 said "a list of {x, y} points"
without ever naming the key `data`, so the model emitted `points`. The values
underneath were largely correct. Rescored with alias-aware parsing, the same
predictions gave 94.7% valid JSON and 23.7% value@5%.

That run is kept at `results/preds/base_minimal_promptv1.0.jsonl` and its score
at `results/scores/base_minimal_promptv1.0.json`. Prompt v1.1 names `data`
explicitly. The lesson is in the harness now: a metric that cannot tell
"malformed" from "correctly extracted, differently named" will silently report
a working model as a broken one.

## MAPE vs median APE

MAPE is unbounded and a single scale misread dominates it — one chart with a
gold value of 0.036 read as 36000 pushed MAPE past 10,000,000%. It is retained
because a scale blow-up is genuine information, but **median APE is the column
to compare runs on**.

## Findings

### Constrained decoding changed nothing — literally

Arms B and C produced **byte-identical output on 150/150 charts**. Not similar
scores; the same bytes. Under greedy decoding the model's argmax was already
grammar-conformant at every token, so the mask never altered a choice.

This answers the Phase 3 **B8** question ahead of time, for the base model:
none of the base-model JSON validity is attributable to grammar enforcement.
Any later claim that constrained decoding drove a gain has to be measured
against this, not assumed.

Two caveats it does not cover: this is greedy decoding (`do_sample=False`), so a
sampled decode would tell a different story, and a fine-tuned model's output
distribution may drift somewhere the grammar does bite.

### Prompt engineering was worth ~0 on extraction

A → B moved chart-type accuracy (72.7% → 88.7%), structural match (14.0% →
28.0%) and point recall (71.6% → 77.9%), but value@5% went **24.7% → 23.9%** —
down 0.8 points, inside noise at n=150.

So the engineered prompt makes the model describe the chart's *shape* better
while reading the *numbers* no better at all. Since value accuracy is the
headline metric, prompting is not the lever on this task. That is a useful thing
to know before spending a week on prompt variants.

### The 4% invalid JSON is two different bugs, not one

All 6 failures across every arm are the same 6 dense charts (36–55 points), and
all hit `max_new_tokens=1024` exactly. Re-running just those at 2048
(`results/preds/diag_truncation_2048.jsonl`) splits them cleanly:

| cause | n | fixed by a bigger budget? |
|---|---|---|
| genuine truncation — completed at 1112–1420 tokens | 4 | yes |
| degenerate repetition loop | 2 | **no** |

The two loopers are both **scatter** charts, and they are stuck:
`synth_00023` emitted `{"x":"2.5","y":0.0}` **90 times — one unique point out of
90**; `synth_00149` did the same 86 times. They exhaust any budget you give them.

Two consequences:

1. **`max_new_tokens=1024` is too small** for dense charts and understates valid
   JSON. With 2048 the ceiling is 148/150 = **98.7%**, which clears the brief's
   98% minimum. The baseline table below is at 1024 and is honest about that;
   the budget should be raised before Phase 3, and raising it changes all arms
   equally so the comparisons here stay valid.
2. **Constrained decoding cannot catch degeneration.** The grammar permits
   arbitrarily many well-formed points, so a repetition loop is *grammatically
   legal*. Grammar enforcement guarantees syntax, not sanity — worth stating
   plainly, since "constrained decoding makes malformed output impossible" is
   exactly the kind of claim this project is supposed to test rather than repeat.

Repetition penalty / no-repeat-ngram is the obvious mitigation and is untested
here. Logged for Phase 4's failure taxonomy, where "degenerate output" now has
two confirmed instances and a reproduction.

## Baseline D (frontier VLM ceiling) — not run

The brief asks for a frontier VLM as the ceiling reference. It is deliberately
absent from this pass: it needs an external API, and it was descoped. Nothing
in this file is a substitute for it, so the "how far from the ceiling" question
stays open until it is run. Recorded here rather than left as a silent gap.


| run | n | valid JSON | schema-exact | chart type | series names | structural | point recall | val@1% | val@5% | val@10% | median APE | res | constrained |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A. zero-shot, minimal | 150 | 94.7% | 94.7% | 72.7% | 51.4% | 14.0% | 71.6% | 9.6% | **24.7%** | 30.5% | 28% | 448px | no |
| B. engineered prompt | 150 | 96.0% | 96.0% | 88.7% | 51.1% | 28.0% | 77.9% | 9.1% | **23.9%** | 30.4% | 30% | 448px | no |
| C. engineered + constrained | 150 | 96.0% | 96.0% | 88.7% | 51.1% | 28.0% | 77.9% | 9.1% | **23.9%** | 30.4% | 30% | 448px | yes |

## A. zero-shot, minimal — by chart type

| chart type | n | val@5% | structural | point recall |
|---|---|---|---|---|
| bar | 43 | 32.4% | 23.3% | 69.0% |
| line | 39 | 21.0% | 20.5% | 83.3% |
| pie | 12 | 0.0% | 0.0% | 21.8% |
| scatter | 19 | 29.2% | 10.5% | 69.5% |
| stacked_bar | 37 | 19.9% | 2.7% | 79.7% |

## A. zero-shot, minimal — by series count

| series | n | val@5% | structural |
|---|---|---|---|
| 1 | 58 | 40.9% | 0.0% |
| 2-3 | 79 | 26.2% | 25.3% |
| 4-5 | 13 | 5.0% | 7.7% |

## B. engineered prompt — by chart type

| chart type | n | val@5% | structural | point recall |
|---|---|---|---|---|
| bar | 43 | 32.7% | 30.2% | 80.5% |
| line | 39 | 20.7% | 33.3% | 86.7% |
| pie | 12 | 0.0% | 0.0% | 21.8% |
| scatter | 19 | 29.1% | 10.5% | 69.6% |
| stacked_bar | 37 | 16.9% | 37.8% | 87.8% |

## B. engineered prompt — by series count

| series | n | val@5% | structural |
|---|---|---|---|
| 1 | 58 | 42.3% | 0.0% |
| 2-3 | 79 | 24.5% | 46.8% |
| 4-5 | 13 | 2.9% | 38.5% |

## C. engineered + constrained — by chart type

| chart type | n | val@5% | structural | point recall |
|---|---|---|---|---|
| bar | 43 | 32.7% | 30.2% | 80.5% |
| line | 39 | 20.7% | 33.3% | 86.7% |
| pie | 12 | 0.0% | 0.0% | 21.8% |
| scatter | 19 | 29.1% | 10.5% | 69.6% |
| stacked_bar | 37 | 16.9% | 37.8% | 87.8% |

## C. engineered + constrained — by series count

| series | n | val@5% | structural |
|---|---|---|---|
| 1 | 58 | 42.3% | 0.0% |
| 2-3 | 79 | 24.5% | 46.8% |
| 4-5 | 13 | 2.9% | 38.5% |

## Run configs

- **A. zero-shot, minimal** — `Qwen/Qwen3-VL-4B-Instruct`, no adapter, 4bit, prompt `minimal` (v1.1), constrained=False, 448px, max_new_tokens=1024, median 17.9s/chart, mean 405 tokens
- **B. engineered prompt** — `Qwen/Qwen3-VL-4B-Instruct`, no adapter, 4bit, prompt `engineered` (v1.1), constrained=False, 448px, max_new_tokens=1024, median 15.6s/chart, mean 396 tokens
- **C. engineered + constrained** — `Qwen/Qwen3-VL-4B-Instruct`, no adapter, 4bit, prompt `engineered` (v1.1), constrained=True, 448px, max_new_tokens=1024, median 17.2s/chart, mean 396 tokens
