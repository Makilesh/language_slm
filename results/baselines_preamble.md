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

## Baseline D (frontier VLM ceiling) — not run

The brief asks for a frontier VLM as the ceiling reference. It is deliberately
absent from this pass: it needs an external API, and it was descoped. Nothing
in this file is a substitute for it, so the "how far from the ceiling" question
stays open until it is run. Recorded here rather than left as a silent gap.
