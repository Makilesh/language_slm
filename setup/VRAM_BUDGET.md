# VRAM budget — Phase 0 measurements

Produced by `setup/verify_vlm.py`. Every number below was read off the
device after the operation actually ran; nothing is estimated.

## Environment

| key | value |
|---|---|
| timestamp | 2026-08-08T04:45:24+00:00 |
| platform | Windows 11 |
| python | 3.12.10 |
| torch | 2.11.0+cu128 |
| transformers | 5.14.1 |
| bitsandbytes | 0.50.0 |
| gpu | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| vram_total_gib | 11.94 |
| compute_capability | sm_120 |
| torch_arch_list | sm_75,sm_80,sm_86,sm_90,sm_100,sm_120 |
| bf16_supported | True |

## Configuration

| key | value |
|---|---|
| model | `Qwen/Qwen3-VL-4B-Instruct` |
| quantization | 4-bit NF4, double quant, bf16 compute |
| vision tower quantized | False |
| attention impl | `sdpa` |
| LoRA | r=32, alpha=64, dropout=0.05 |
| LoRA targets | LLM decoder q/k/v/o/gate/up/down only |
| batch size | 1 |
| px per image token | 32 |
| prepare_model_for_kbit_training | False |
| force GQA repeat_kv | True |

### Parameter memory by dtype

| dtype | GiB |
|---|---|
| uint8 | 1.88 |
| bfloat16 | 0.73 |
| float32 | 0.25 |

`uint8` is NF4-packed weight storage (two 4-bit params per byte), so
parameter *counts* below read low against the model's nominal 4B.

## Trainable parameters

| bucket | params |
|---|---|
| total (base + adapters) | 2,481,697,280 |
| trainable — vision tower | 0 |
| trainable — LLM decoder | 66,060,288 |
| trainable — elsewhere | 0 |
| trainable — total | 66,060,288 (2.662%) |

Weights resident after 4-bit load: **2.76 GiB** reserved.
After attaching LoRA: **2.98 GiB**.

## THE TABLE — resolution vs image tokens vs peak VRAM

Card capacity is **11.94 GiB**. GiB columns are peak *allocated*
(live tensors), not reserved — the caching allocator's reserved pool is sticky
across measurements and overstates the requirement. Values in **bold** exceed
card capacity: on Windows the WDDM driver serves the overflow from host RAM
over PCIe instead of raising OOM, so those rows *ran* but are not usable
operating points. The millisecond columns are what expose that — read them.

| long edge | resized | grid | image tokens | seq len | fwd GiB | fwd ms | bwd GiB (no ckpt) | bwd ms | bwd GiB (ckpt) | bwd ms | ckpt active |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 448px | 448x274 | 9x14 | **126** | 623 | 3.96 | 256 | 10.03 | 528 | 10.03 | 538 | yes |
| 768px | 768x469 | 15x24 | **360** | 857 | 4.34 | 358 | **12.59** | 9933 | **12.59** | 23886 | yes |
| 1280px | 1280x782 | 24x40 | **960** | 1457 | 5.28 | 903 | **19.21** | 118388 | **19.21** | 25044 | yes |

## Machine-readable

```json
{
  "env": {
    "timestamp": "2026-08-08T04:45:24+00:00",
    "platform": "Windows 11",
    "python": "3.12.10",
    "torch": "2.11.0+cu128",
    "transformers": "5.14.1",
    "bitsandbytes": "0.50.0",
    "gpu": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
    "vram_total_gib": 11.94,
    "compute_capability": "sm_120",
    "torch_arch_list": "sm_75,sm_80,sm_86,sm_90,sm_100,sm_120",
    "bf16_supported": true
  },
  "config": {
    "model": "Qwen/Qwen3-VL-4B-Instruct",
    "attn": "sdpa",
    "quantize_vision": false,
    "prepare_for_kbit": false,
    "lora_r": 32,
    "lora_alpha": 64,
    "px_per_image_token": 32
  },
  "vram_total_gib": 11.94,
  "weights_reserved_gib": 2.764,
  "param_gib_by_dtype": {
    "uint8": 1.883,
    "bfloat16": 0.733,
    "float32": 0.246
  },
  "rows": [
    {
      "long_edge": 448,
      "resized": [
        448,
        274
      ],
      "grid": [
        9,
        14
      ],
      "image_tokens": 126,
      "total_tokens": 623,
      "fwd_alloc_gib": 3.962,
      "fwd_rsvd_gib": 4.09,
      "fwd_ms": 256,
      "bwd_nockpt_alloc_gib": 10.028,
      "bwd_nockpt_rsvd_gib": 10.578,
      "bwd_nockpt_ms": 528,
      "bwd_ckpt_alloc_gib": 10.028,
      "bwd_ckpt_rsvd_gib": 10.578,
      "bwd_ckpt_ms": 538,
      "ckpt_active": true,
      "notes": []
    },
    {
      "long_edge": 768,
      "resized": [
        768,
        469
      ],
      "grid": [
        15,
        24
      ],
      "image_tokens": 360,
      "total_tokens": 857,
      "fwd_alloc_gib": 4.341,
      "fwd_rsvd_gib": 4.525,
      "fwd_ms": 358,
      "bwd_nockpt_alloc_gib": 12.589,
      "bwd_nockpt_rsvd_gib": 12.98,
      "bwd_nockpt_ms": 9933,
      "bwd_ckpt_alloc_gib": 12.589,
      "bwd_ckpt_rsvd_gib": 12.98,
      "bwd_ckpt_ms": 23886,
      "ckpt_active": true,
      "notes": []
    },
    {
      "long_edge": 1280,
      "resized": [
        1280,
        782
      ],
      "grid": [
        24,
        40
      ],
      "image_tokens": 960,
      "total_tokens": 1457,
      "fwd_alloc_gib": 5.276,
      "fwd_rsvd_gib": 5.682,
      "fwd_ms": 903,
      "bwd_nockpt_alloc_gib": 19.215,
      "bwd_nockpt_rsvd_gib": 20.182,
      "bwd_nockpt_ms": 118388,
      "bwd_ckpt_alloc_gib": 19.215,
      "bwd_ckpt_rsvd_gib": 20.182,
      "bwd_ckpt_ms": 25044,
      "ckpt_active": true,
      "notes": []
    }
  ]
}
```
