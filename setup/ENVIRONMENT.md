# Environment

## What this machine actually is

The project brief assumed WSL2 + Ubuntu with Project A's environment already
working. Neither held. This is a **Windows-native** build from scratch.

| Brief assumed | Actual |
|---|---|
| WSL2 + Ubuntu, CUDA 12.8+ | Windows 11, driver 592.01 (CUDA 13.1 runtime) |
| Project A env already working | nothing pre-existing; `.venv` built from zero |
| flash-attn "if it builds on sm_120" | not installed — see below |

Everything else in the hardware section of the brief is correct: RTX 5070 Ti
Laptop, 11.94 GiB usable VRAM, Blackwell sm_120.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip setuptools wheel
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r setup/requirements.txt
```

## Blackwell notes

**The cu128 index is not optional.** `pip install torch` from default PyPI gives
wheels without sm_120 kernels; they import fine, report `cuda.is_available()`
true, and then fail at the first real kernel launch. Verify with:

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.get_arch_list())"
```

`sm_120` must appear in that list. On this machine torch 2.11.0+cu128 gives
`sm_75, sm_80, sm_86, sm_90, sm_100, sm_120`.

**bf16 is supported** (Blackwell), so unlike the Kaggle T4 fallback path we are
not forced into fp16. Use bf16 as the compute dtype throughout.

**bitsandbytes 0.50.0 works on sm_120 under Windows** — verified with a real
`Linear4bit` forward pass, not just an import.

**flash-attn is not installed.** There is no prebuilt wheel for sm_120 on
Windows, and a source build requires MSVC plus a multi-hour nvcc compile that
commonly fails on Blackwell. We use PyTorch SDPA (`attn_implementation="sdpa"`).
This is a recorded decision, not a silent fallback: if a later phase needs the
memory profile of flash-attn, the gap is documented here rather than assumed
away.

**The Windows torch build has no flash-attention SDPA backend either** — not
just the separate `flash-attn` package. `torch.nn.functional.
scaled_dot_product_attention` reports "Torch was not compiled with flash
attention". Combined with Qwen3-VL's grouped-query attention this pushes SDPA
onto the math backend, costing 3.7 GiB and 12× throughput. `setup/sdpa_compat.py`
patches it; **import and call `force_repeat_kv()` before any training run.**
Full measurements in `results/phase0_findings.md` §3.

**Exceeding VRAM does not raise OOM on Windows.** WDDM spills to host RAM over
PCIe, so an oversized config runs correctly and merely becomes 10–100× slower.
Time every config; do not rely on catching an exception. When a real failure
does arrive it is `torch.AcceleratorError`, not `torch.OutOfMemoryError`, and
`empty_cache()` will raise too.

**triton is unavailable on Windows.** Harmless — it only disables FLOP counting
in `torch.utils.flop_counter` and any triton-backed fused kernels. Torch prints
a warning at import; ignore it.

## Vision quantization choice

`verify_vlm.py` keeps the vision tower and `lm_head` in bf16 and NF4-quantizes
only the language decoder (`llm_int8_skip_modules=["visual", "lm_head"]`).

Rationale: the ViT is ~0.44B params, so bf16 costs well under a gigabyte over
NF4, and we never train it — pushing NF4 quantization error through frozen
visual features buys nothing. `--quantize-vision` flips this so the B1 ablation
can measure the claim instead of inheriting it.

## Reproducing the Phase 0 table

```bash
.venv/Scripts/python.exe setup/verify_vlm.py --resolutions 448 768 1280
```

Writes `setup/VRAM_BUDGET.md`.
