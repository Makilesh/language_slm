# Resolution policy

**Decision: 448 px long edge for training. 768 px available for inference.**

This is the Phase 2 deliverable that turns Phase 0's measurements into a config
number. It is a hardware decision, not a quality decision, and the difference
matters when reading the results.

## The measurements it follows from

From [`setup/VRAM_BUDGET.md`](../setup/VRAM_BUDGET.md), Qwen3-VL-4B in 4-bit
NF4, LoRA r=32 on the decoder, batch 1, on an 11.94 GiB card:

| long edge | image tokens | seq len | fwd GiB | bwd GiB | bwd ms |
|---|---|---|---|---|---|
| **448** | **126** | 623 | 3.96 | **10.03** ✅ | **528** |
| 768 | 360 | 857 | 4.34 | 12.59 ❌ | 9,933 |
| 1280 | 960 | 1457 | 5.28 | 19.21 ❌ | 118,388 |

One image token covers a 32×32 px region (patch 16 × spatial merge 2), so token
count grows with **area**: 126 → 360 → 960 across those three rows.

## Why 448

448 is the only resolution whose **backward** pass fits. The 768 row is not a
fit-with-less-headroom; it is 0.65 GiB over, and on Windows that does not raise
OOM — WDDM serves the overflow from host RAM and the step time collapses from
0.53 s to 9.9 s. A 10k-sample epoch at 768 px would take ~27 hours against
~1.5 hours at 448.

That is the whole argument. 448 is not chosen because it is sufficient for
reading charts; it is chosen because it is what the card can train.

## What this costs

At 448 px the probe chart renders to 448×274. Axis tick labels land at roughly
6–8 px tall. That is legible to the model, but it is the regime where dense
scatter plots and rotated labels get hard, and the Phase 1 baselines already
show value@5% at 23.9% with series-name accuracy at 51% — some of which is
plausibly resolution-bound rather than capability-bound.

**This is a confound to state, not to hide.** Any claim of the form "the model
struggles with dense charts" is, at 448 px, partly a claim about 448 px.

## Training vs inference are decoupled deliberately

Forward-only fits at every resolution tested (3.96 / 4.34 / 5.28 GiB), so
evaluation and serving are *not* constrained to 448. The dataset stores image
**paths**, not pixels, and the collator resizes at load time — so changing
resolution is a config flag, not a dataset rebuild. That is what makes the B2
ablation cheap.

Two consequences worth planning around:

- A model trained at 448 and evaluated at 768 is a train/test resolution
  mismatch. It may help (more detail) or hurt (unfamiliar token counts). It is
  a measurable question and belongs in B2, not an assumption.
- The demo can serve at 768 or 1280 for free. Whether that is *better* is
  exactly what B2 answers.

## Sequence budget

At 448 px, with the engineered prompt:

| component | tokens |
|---|---|
| image | 126 |
| instruction (engineered prompt) | ~290 |
| chat template scaffolding | ~30 |
| target JSON | median ~215, p99 ~700 |

`max_seq_len: 1024` covers the corpus comfortably; the p99 target sits well
inside it. Note this is the *training* budget — Phase 1 separately found that
`max_new_tokens=1024` at **inference** truncates 6/150 dense charts, and should
be raised to 2048 there. The two numbers are unrelated and it is easy to
conflate them.

## Revisit conditions

Raise resolution if any of these change:

1. The ~6 GiB backward overhead in [`results/phase0_findings.md`](../results/phase0_findings.md) §4
   turns out to be NF4 dequantization with a fix — that is the single biggest
   lever available and would put 768 in range.
2. Training moves to the Kaggle 2×T4 (fp16), where the memory budget differs.
3. Error analysis (Phase 4) attributes a large share of failures to unreadable
   labels rather than misread values.
