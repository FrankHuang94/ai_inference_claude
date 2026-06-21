# Activation Quantization Challenges

> **Section:** 05_quantization
> **Last Updated:** June 2026
> **Related Files:** [01_post_training_quantization.md](01_post_training_quantization.md), [04_fp8_inference_h100.md](04_fp8_inference_h100.md), [00_quantization_fundamentals.md](00_quantization_fundamentals.md)
> **Must-Read Papers:** Dettmers et al. (2022, NeurIPS) "LLM.int8()"; Xiao et al. (2023, ICML) "SmoothQuant"; Ashkboos et al. (2024) "QuaRot"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- Activations are harder to quantize than weights because of **outliers**: a few channels have values 10–100× larger, forcing a huge scale that crushes precision for everyone else.
- Outliers are **systematic** (specific channels/dimensions), emerge with scale, and concentrate in certain layers — not random noise.
- Solutions: **mixed-precision** (LLM.int8(): keep outlier channels in FP16), **migration** (SmoothQuant: move range to weights), **rotation** (QuaRot/Hadamard: spread outliers across channels), or **FP8** (float range absorbs outliers).
- Activation quantization is what enables **W8A8 compute speedup** (prefill + decode), unlike weight-only.
- FP8 (E4M3) is the production answer on Hopper/Blackwell — its exponent handles outliers natively.

***

## Overview
Weight distributions in LLMs are well-behaved and quantize to INT4 readily ([§02](02_weight_only_quantization.md)), but **activations** are the hard case. Dettmers et al. (LLM.int8(), 2022) showed that as models scale past ~6.7B parameters, **emergent outlier features** appear: a small number of activation channels take on magnitudes 10–100× the rest, and these outliers are *critical* for model quality (zeroing them destroys accuracy). Because integer quantization uses a single scale across the quantized range, one giant outlier forces a large scale, which maps all the normal-magnitude values onto just a few integer levels — catastrophic precision loss. This is why naive per-tensor INT8 activations wreck accuracy and why activation quantization, not weight quantization, is the binding obstacle to full low-precision compute.

Why bother with activations at all? Because **weight-only quantization doesn't speed up compute-bound prefill** — the matmul still runs in FP16. To get a true compute speedup (INT8/FP8 GEMM on Tensor Cores, benefiting both prefill and decode), you must quantize the activations too (W8A8). So handling activation outliers is the price of admission for compute-side gains. The field has produced four families of solutions: **mixed-precision decomposition** (LLM.int8() keeps the ~0.1% outlier channels in FP16 and quantizes the rest to INT8), **range migration** (SmoothQuant shifts the dynamic-range burden from activations into weights via a per-channel transform, [§01](01_post_training_quantization.md)), **rotation** (QuaRot/Hadamard transforms rotate the activation space so outlier energy spreads across channels, removing the spikes), and **floating-point** (FP8's exponent natively represents wide ranges, sidestepping the problem — the production winner on Hopper, [§04](04_fp8_inference_h100.md)).

The practical 2025–2026 consensus: on Hopper/Blackwell, **FP8 activations** are the default because the hardware and format handle outliers gracefully with near-lossless quality and full compute speedup. INT8 activations remain relevant where FP8 isn't available, using SmoothQuant or rotation. W4A4 (4-bit activations) is the research frontier, where rotation methods (QuaRot, SpinQuant) are most promising. This file explains the outlier phenomenon and the toolkit.

***

## Core Concepts & Mechanics

### The outlier phenomenon
- **Systematic**: outliers occur in specific feature dimensions/channels, consistently across tokens, and emerge with scale (Dettmers et al.).
- **Critical**: they carry important information; removing them collapses quality.
- 📐 With outlier magnitude `M` and normal magnitude `m`, per-tensor INT8 scale ≈ `M/127`; normal values occupy only `±m/(M/127) = ±127·m/M` levels — if `M/m = 100`, normals use ~1 level → destroyed.

### Solution families
1. **Mixed-precision (LLM.int8())**: detect outlier channels (threshold), compute them in FP16, the rest in INT8, recombine. Preserves quality; some kernel complexity and overhead.
2. **Migration (SmoothQuant)**: per-channel `X→X/s`, `W→W·s` so activations lose their range; both quantize to INT8. Equivalent math, full W8A8 GEMM. ([§01](01_post_training_quantization.md))
3. **Rotation (QuaRot/SpinQuant)**: apply an orthogonal (Hadamard) transform that spreads outlier energy across channels, eliminating spikes; invert appropriately. Enables INT4 activations (W4A4) far better than naive.
4. **Floating-point (FP8)**: E4M3's exponent represents 10–100× ranges directly; per-tensor/row scaling suffices. Near-lossless, full speedup, hardware-native on Hopper+. ([§04](04_fp8_inference_h100.md))

### Granularity
Per-token (dynamic) activation scaling helps (each token's range), as does per-channel; but per-channel activation scaling is awkward for GEMM (scale on the reduction dim) — SmoothQuant/rotation exist partly to avoid it.

***

## Key Challenges
1. **Outliers force precision collapse.** A single large channel destroys INT8 precision for the rest; the fundamental difficulty.
2. **Per-channel activation scaling is GEMM-unfriendly.** Scaling along the contraction dimension breaks the clean matmul; hence migration/rotation tricks.
3. **Dynamic ranges per token.** Activation ranges vary by input; static calibration can mis-estimate, needing dynamic/per-token scales.
4. **W4A4 is still hard.** Even with rotation, 4-bit activations lose accuracy on sensitive tasks; an open problem.

***

## Solutions & Current Best Practices
- **FP8 (E4M3) activations** on Hopper/Blackwell — default, near-lossless, full speedup ([§04](04_fp8_inference_h100.md)).
- **SmoothQuant** for INT8 activations where FP8 unavailable.
- **Rotation (QuaRot/SpinQuant)** for aggressive W4A4 research/deployment.
- **Per-token dynamic scaling** to handle input-dependent ranges.

***

## Implementation Notes
- For FP8, use per-tensor or per-row scaling with (delayed) scale tracking via Transformer Engine ([§04](04_fp8_inference_h100.md)).
- For INT8, apply SmoothQuant offline (fold smoothing into weights) and use per-token activation scales at runtime.
- Validate on outlier-sensitive tasks; activation quant damage shows up on specific benchmarks.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Per-tensor INT8 activations destroyed accuracy** — outliers forced a huge scale; use FP8/SmoothQuant/rotation.
- **W8A8 helped prefill but quality dropped on one task** — activation quant damage is task-specific; eval broadly.
- **Static activation calibration mismatched production** — input-dependent ranges; use per-token dynamic scaling.
- **Tried W4A4 naively** — collapsed; needs rotation (QuaRot) and careful eval.

***

## Performance Numbers & Benchmarks
| Method | Activations | Quality | Speedup |
|---|---|---|---|
| LLM.int8() | INT8 + FP16 outliers | lossless | modest (mixed path) |
| SmoothQuant | INT8 | small loss | full W8A8 |
| FP8 (E4M3) | FP8 | near-lossless | full, hardware-native |
| QuaRot | INT4 | moderate loss | full W4A4 (research) |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why are activations harder to quantize than weights?"* — Expected: systematic emergent outliers force large scale → precision collapse.
- *"How does SmoothQuant vs LLM.int8() vs FP8 handle outliers?"* — Expected: migrate range / mixed-precision / float exponent.
- *"Why do you need activation quantization at all if weight-only speeds decode?"* — Expected: compute (prefill) speedup needs low-precision GEMM (W8A8).
- *"What's the state of 4-bit activations?"* — Expected: rotation methods (QuaRot/SpinQuant); still hard.

***

## Open Problems & Active Research (2025–2026)
- **Robust W4A4** without task degradation (rotation + learned transforms).
- **Outlier-free pretraining** (architectures/normalizations that avoid emergent outliers).
- **Dynamic per-token FP8/INT8** scaling at minimal kernel cost.

***

## References
- Dettmers, T., et al. (2022). "LLM.int8()." *NeurIPS 2022*. arXiv:2208.07339.
- Xiao, G., et al. (2023). "SmoothQuant." *ICML 2023*. arXiv:2211.10438.
- Ashkboos, S., et al. (2024). "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs." arXiv:2404.00456.
- Micikevicius, P., et al. (2022). "FP8 Formats." arXiv:2209.05433.
