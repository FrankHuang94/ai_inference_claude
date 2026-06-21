# FP8 Inference on H100/H200/Blackwell

> **Section:** 05_quantization
> **Last Updated:** June 2026
> **Related Files:** [03_activation_quantization_challenges.md](03_activation_quantization_challenges.md), [../01_hardware/02_nvidia_h100_b200_architecture.md](../01_hardware/02_nvidia_h100_b200_architecture.md), [00_quantization_fundamentals.md](00_quantization_fundamentals.md)
> **Must-Read Papers:** Micikevicius et al. (2022) "FP8 Formats for Deep Learning"; NVIDIA Transformer Engine docs; NVIDIA Hopper/Blackwell whitepapers
> **Estimated Study Time:** 35 minutes

***

## TL;DR
- FP8 is the **production-default** quantization on Hopper/Blackwell: ~2× Tensor-Core throughput vs FP16 *and* half the bytes — speeds **both** prefill and decode.
- Two formats: **E4M3** (4 exp, 3 mantissa, range ±448) for weights/activations forward; **E5M2** (range ±57344) for wide dynamic range (gradients).
- The **Transformer Engine** does FP8 GEMM with **FP16/BF16 accumulation** and manages **scaling** (per-tensor, per-row) including **delayed scaling** for stability.
- Near-lossless for most models; the exponent absorbs activation outliers that wreck INT8 ([§03](03_activation_quantization_challenges.md)).
- Blackwell adds **FP4** (E2M1 + micro-scaling) for another ~2× — quality still being characterized.

***

## Overview
FP8 is the quantization format that changed production inference economics on Hopper. Unlike weight-only INT4 (which helps only decode) or INT8 activations (which need outlier gymnastics), FP8 is a **floating-point** format whose exponent natively represents the wide dynamic ranges of LLM activations, so both weights and activations quantize to 8 bits with **near-lossless** quality and minimal preprocessing. Because the H100 Transformer Engine has native FP8 Tensor Cores at ~1979 TFLOP/s (2× the FP16 989), FP8 **doubles compute throughput** (helping compute-bound prefill) *and* **halves bytes moved** (helping bandwidth-bound decode). It is the rare optimization that improves both phases, which is why it became the default precision for serving on Hopper/Blackwell.

The two FP8 formats encode the range/precision tradeoff. **E4M3** (4 exponent, 3 mantissa bits, max ±448) gives more precision and is used for **forward-pass weights and activations** in inference. **E5M2** (5 exponent, 2 mantissa, max ±57344) trades precision for range and is used where dynamic range is extreme (e.g., gradients in training). For inference you mostly care about E4M3. The challenge with any 8-bit float is **scaling**: you must map each tensor's values into FP8's representable range without overflow (clipping to ±448) or underflow. The **Transformer Engine** handles this with **per-tensor scaling factors** (or finer per-row/per-block on Blackwell) and **delayed scaling** — using scale factors computed from recent history (an exponential moving max) to avoid an expensive synchronous max-reduction every step, trading a small staleness for stability and speed.

The result is the 2025–2026 production consensus: serve in FP8 on Hopper/Blackwell unless a specific model/task shows degradation, in which case keep sensitive layers higher precision. Blackwell extends the idea to **FP4 (E2M1) with micro-scaling (MX)** block formats for another ~2× throughput/byte reduction, but FP4 quality is still being characterized and is not yet a universal default ([§01](../01_hardware/02_nvidia_h100_b200_architecture.md)). This file covers FP8 formats, scaling, the Transformer Engine, and the gotchas.

***

## Core Concepts & Mechanics

### Formats
| Format | Exp/Mant | Max | Use |
|---|---|---|---|
| E4M3 | 4 / 3 | ±448 | inference weights & activations (forward) |
| E5M2 | 5 / 2 | ±57344 | wide range (gradients) |
📐 More exponent bits → wider range, fewer mantissa bits → less precision. E4M3's ±448 comfortably covers activation outliers that overflow INT8's effective range without a giant scale.

### FP8 GEMM with high-precision accumulation
The Tensor Core multiplies FP8×FP8 but **accumulates in FP16/BF16/FP32**, so the dot-product sum doesn't lose precision catastrophically. Inputs (weights, activations) are FP8; the matmul output is higher precision, then re-quantized to FP8 for the next op if desired.

### Scaling
- **Per-tensor scaling**: one scale per tensor maps its range into FP8. Simple, hardware-friendly.
- **Per-row / per-block (Blackwell, MX)**: finer scales for better accuracy.
- 📐 `x_fp8 = round_to_fp8(x / scale)`, `scale ≈ amax(x)/448`. Wrong scale → overflow (clip to 448, lose large values) or underflow (small values → 0).

### Delayed scaling
Computing `amax` synchronously each step is costly. **Delayed scaling** uses an EMA/history of recent `amax` to set the current scale, avoiding a per-step reduction. Trade-off: a sudden range change can briefly clip until the scale catches up. Transformer Engine manages this.

### Blackwell FP4 (preview)
E2M1 (2 exp, 1 mantissa) with **micro-scaling**: a shared scale per small block (e.g., 32 elements) retains accuracy at 4 bits. ~2× over FP8; quality being characterized — validate per task ([§01](../01_hardware/02_nvidia_h100_b200_architecture.md)).

***

## Key Challenges
1. **Scale selection / overflow.** Wrong per-tensor scale clips outliers (quality loss) or underflows small values; delayed scaling can lag sudden range shifts.
2. **Per-tensor coarseness.** Single scale may be too coarse for some layers; finer granularity (per-row/block) costs metadata/kernels.
3. **Mixed precision bookkeeping.** Some layers (LM head, layernorm, sensitive attention) may need higher precision; managing which stays FP16 adds complexity.
4. **FP4 quality.** Aggressive 4-bit float degrades on sensitive tasks; not yet a universal default.

***

## Solutions & Current Best Practices
- **FP8 E4M3 weights+activations** as the default on Hopper/Blackwell; near-lossless, ~2× both phases.
- **Transformer Engine** for scaling + delayed scaling + FP8 GEMM; or framework-native FP8 (vLLM/SGLang/TRT-LLM).
- **Keep sensitive layers FP16** (e.g., final LM head, certain norms) if needed.
- **Validate FP4 per task** before adopting; keep fallback to FP8.

***

## Implementation Notes
- vLLM/SGLang/TRT-LLM accept FP8 checkpoints and do FP8 GEMM; enable per-tensor (or per-channel) scaling and calibrate amax on representative data.
- Watch for **amax spikes** on rare inputs; delayed scaling may briefly clip — monitor for quality blips.
- Combine FP8 with continuous batching, paged KV, and CUDA graphs for full effect.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **FP8 doubled prefill but decode gained less** — decode is bandwidth-bound; the win is the halved bytes (~up to 2× if weights dominate), not the FLOP doubling.
- **Rare prompt produced garbage** — an activation amax spike overflowed FP8 before delayed scaling adjusted; tighten scaling or use per-row.
- **LM head in FP8 hurt quality** — the output projection can be sensitive; keep it higher precision.
- **FP4 passed perplexity, failed reasoning** — 4-bit float damage is task-specific; validate beyond PPL.

***

## Performance Numbers & Benchmarks
| Hardware | Precision | TFLOP/s | vs FP16 |
|---|---|---|---|
| H100 | FP16 | 989 | 1× |
| H100 | FP8 | 1979 | 2× |
| B200 | FP8 | ~4.5 PF | — |
| B200 | FP4 | ~9 PF | ~2× over FP8 |
| FP8 serving (typical) | — | — | ~near-lossless, ~2× prefill, decode ↓bytes |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why does FP8 speed up both prefill and decode while INT4 weight-only only speeds decode?"* — Expected: FP8 GEMM doubles compute (prefill) and halves bytes (decode); W4A16 only halves bytes.
- *"E4M3 vs E5M2 — when each?"* — Expected: E4M3 precision (forward), E5M2 range (gradients).
- *"What is delayed scaling and why use it?"* — Expected: avoid per-step amax reduction via history; small staleness risk.
- *"Why does FP8 handle activation outliers better than INT8?"* — Expected: exponent represents wide range without a giant scale.

***

## Open Problems & Active Research (2025–2026)
- **FP4/MX quality at scale** and per-layer FP4/FP8 mixing.
- **Robust dynamic scaling** that never clips on amax spikes.
- **FP8 for KV cache and attention** (FP8 attention kernels) end-to-end ([§05](05_kv_cache_quantization.md)).

***

## References
- Micikevicius, P., et al. (2022). "FP8 Formats for Deep Learning." arXiv:2209.05433.
- NVIDIA. "Transformer Engine" documentation and Hopper/Blackwell whitepapers.
- Open Compute Project (2023). "Microscaling (MX) Formats."
