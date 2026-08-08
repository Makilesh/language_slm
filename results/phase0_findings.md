# Phase 0 findings

Everything here was measured on this machine by `setup/verify_vlm.py`,
`setup/diagnose_backward.py`, and `setup/diagnose_attention.py`. Raw logs are in
this directory; the mechanical table is `setup/VRAM_BUDGET.md`.

Model: `Qwen/Qwen3-VL-4B-Instruct`, 4-bit NF4, LoRA r=32 on the LLM decoder only,
vision tower frozen and left in bf16, batch 1, SDPA.
Card: RTX 5070 Ti Laptop, **11.94 GiB**.

---

## 1. The operating point is 448px

| long edge | image tokens | seq len | fwd GiB | fwd ms | bwd GiB | bwd ms |
|---|---|---|---|---|---|---|
| **448** | **126** | 623 | 3.96 | 256 | **10.03** | **528** |
| 768 | 360 | 857 | 4.34 | 358 | 12.59 | 9,933 |
| 1280 | 960 | 1457 | 5.28 | 903 | 19.21 | 118,388 |

448px is the only resolution whose **backward** pass fits. 768px misses by
0.65 GiB and pays for it with a 19× throughput collapse; 1280px is not close.

Image tokens scale with *area*, so the token count roughly quadruples for each
near-doubling of the long edge: 126 → 360 → 960. Sequence length follows, and
backward memory follows that.

**Forward-only fits at every resolution** (3.96 / 4.34 / 5.28 GiB). Evaluation
and inference are therefore unconstrained up to 1280px — only training is
pinned to 448px. That asymmetry matters for Phase 1, which is inference-only.

## 2. On Windows, exceeding VRAM does not raise OOM — it just gets slow

The WDDM driver serves overflow from host RAM over PCIe. A config that does not
fit keeps running and returns correct numbers, at 10–100× the wall-clock. Only
past roughly 1.5× card capacity does a real `torch.AcceleratorError: CUDA error:
out of memory` arrive.

Consequences, both of which cost time here before being understood:

- **The §5 Phase 3 OOM playbook will not trigger on this machine.** There is no
  exception to catch. Throughput is the fit signal, so every config must be
  timed, not just memory-profiled.
- **`torch.OutOfMemoryError` is the wrong thing to catch.** The failure arrives
  as `torch.AcceleratorError`, and once it does, even `torch.cuda.empty_cache()`
  raises — so cleanup paths need individual guards or a recorded OOM turns into
  a crashed run.

## 3. The big one: SDPA was silently falling back to the math backend

Qwen3-VL uses grouped-query attention — 32 query heads, 8 KV heads. transformers
picks between two ways to reconcile that
(`integrations/sdpa_attention.py::use_gqa_in_sdpa`): pass `enable_gqa=True` when
there is no attention mask, or `repeat_kv` to broadcast K/V up to 32 heads.
Training at batch 1 with no padding means no mask, so it takes `enable_gqa`.

On the Windows CUDA build that branch has nowhere to dispatch. Torch says so:

```
Torch was not compiled with flash attention
For dense input, both fused kernels require query, key and value to have the
  same num_heads. Query.sizes(): [1, 32, 857, 128], Key sizes(): [1, 8, 857, 128]
cuDNN attention has been runtime disabled
```

So every attention call fell to the **math** backend, which materialises a
`[32, seq, seq]` score matrix per layer and keeps it for backward. Across 36
layers that is where the missing gigabytes went.

Forcing the `repeat_kv` branch (`setup/sdpa_compat.py`) fixes it. Measured at
768px, backward:

| config | peak | time |
|---|---|---|
| sdpa, stock | 16.26 GiB | 14,417 ms |
| **sdpa + force repeat_kv** | **12.58 GiB** | **1,189 ms** |
| sdpa + repeat_kv, cuDNN off | 12.58 GiB | 1,168 ms |
| eager | 17.09 GiB | 11,570 ms |

**3.7 GiB and 12× throughput, from one boolean.** Toggling cuDNN changes
nothing in either direction, so the "cuDNN runtime disabled" warning is a red
herring — the head-count mismatch is the actual cause.

This is platform-specific. On Linux with flash-attn built for sm_120 the stock
path should be fine; re-measure rather than porting the patch blindly.

## 4. Gradient checkpointing buys nothing here — and that is informative

Peak allocated is *identical* with checkpointing on and off, at every
resolution (10.03/10.03, 12.59/12.59, 19.21/19.21). The flag is genuinely
active — `verify_vlm.py` reads it back off the modules rather than trusting the
call.

If activations dominated the peak, checkpointing would cut it. It does not, so
they do not. Backward costs 6.07 GiB more than forward at 448px, and that
delta is not activation memory.

**Leading hypothesis, not yet isolated:** NF4 weight dequantization in
bitsandbytes' `Linear4bit` backward. Computing the input gradient requires
dequantizing each weight to bf16, and that cost is invisible to checkpointing,
which only replays activations. This is consistent with what was measured but
has not been proven — it wants a controlled comparison against an unquantized
baseline before it goes in the README.

Practical effect: checkpointing is still worth keeping on (it is free), but it
is not the lever the brief's OOM playbook assumes. **Resolution is.**

## 5. `prepare_model_for_kbit_training` upcasts to fp32

peft's helper casts every non-quantized module — embeddings, `lm_head`, and the
whole vision tower — to fp32, which drags the compute path out of bf16. Doing by
hand only what it is actually needed for (freeze the base, then
`enable_input_require_grads()` so gradients reach checkpointed blocks) keeps
parameter memory at uint8 1.88 / bf16 0.73 / fp32 0.25 GiB, the fp32 being just
the LoRA adapters.

`--use-prepare` keeps the stock path available for comparison.

## 6. Vision tower: frozen, and verified as frozen

`verify_vlm.py` hard-fails if any vision parameter has `requires_grad`. LoRA
targeting is anchored on a `language_model`-scoped regex rather than trusting
module names not to collide. Trainable: 66,060,288 params, 0 of them in the ViT.

The ViT is left unquantized in bf16 — it is ~0.44B params, so bf16 costs well
under a gigabyte versus NF4, and there is no reason to push quantization error
through features that are never trained. `--quantize-vision` flips this for the
B1 ablation.

---

## What this means for the plan

**Phase 1 (eval harness) is unblocked.** It is inference-only, and forward fits
at every resolution tested.

**Phase 3's B2 ablation needs rescoping.** The brief asks for training runs at
448 / 768 / 1280. Only 448 trains on this card. Options, in preference order:

1. Run the 768/1280 arms on the Kaggle 2×T4 (fp16, no bf16) and report the
   hardware split honestly in the results table.
2. Chase the 6 GiB backward overhead in §4 first — if it is the NF4
   dequantization path, there may be a real fix, and it would be worth more
   than any other single change here.
3. Reduce the label span so the loss does not run over all 151,936 vocab
   entries for every position.

Dropping resolution silently to make 768 fit is exactly the anti-pattern the
brief names, so it is recorded here instead.

**The headline number for the README** is not the resolution knee the brief
anticipated. It is that the naive Windows configuration wastes 3.7 GiB and 12×
throughput on an attention-dispatch fallback that produces no error message.
