# Post-Training Quantization: GPTQ, AWQ, SmoothQuant

> **Section:** 05_quantization
> **Last Updated:** June 2026
> **Related Files:** [00_quantization_fundamentals.md](00_quantization_fundamentals.md), [02_weight_only_quantization.md](02_weight_only_quantization.md), [03_activation_quantization_challenges.md](03_activation_quantization_challenges.md)
> **Must-Read Papers:** Frantar et al. (2023, ICLR) "GPTQ"; Lin et al. (2024, MLSys) "AWQ"; Xiao et al. (2023, ICML) "SmoothQuant"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- PTQ quantizes a **trained** model with only a small **calibration set** — no retraining — making it the default for deployment.
- **GPTQ** (Frantar et al. 2023): layer-wise weight quantization minimizing output error using approximate second-order (Hessian) information → accurate INT4 weights.
- **AWQ** (Lin et al. 2024): **activation-aware** weight quantization — protect the ~1% of weight channels that matter most (by activation magnitude) via per-channel scaling → INT4 with minimal loss, fast kernels.
- **SmoothQuant** (Xiao et al. 2023): migrate activation outliers into weights via a smoothing transform → enables **INT8 weights+activations** (W8A8) with full matmul speedup.
- Choose: AWQ/GPTQ for memory-bound decode (W4A16); SmoothQuant/FP8 for compute speedup (W8A8).

***

## Overview
Post-training quantization compresses an already-trained model without the cost of retraining, using a small calibration dataset to estimate scales and corrections. It is the workhorse of deployment because it's cheap, fast, and — with modern methods — nearly lossless at INT4 weights or INT8/FP8 weights+activations. The three landmark methods address different parts of the problem: **GPTQ** and **AWQ** focus on **weight-only** quantization (W4A16: 4-bit weights, 16-bit activations), which targets the memory-bound decode bottleneck; **SmoothQuant** enables **weight+activation** quantization (W8A8), which additionally speeds up compute-bound prefill by running the matmul itself in INT8.

**GPTQ** frames weight quantization as minimizing the layer's output reconstruction error and solves it greedily using approximate **second-order (Hessian) information** about how each weight affects the loss, quantizing weights one column at a time and updating the rest to compensate. This yields highly accurate INT4 weights but requires the Hessian computation per layer. **AWQ** takes a complementary, simpler insight: not all weights matter equally — a small fraction of channels, identifiable by the **magnitude of the activations** that multiply them, dominate output quality. AWQ scales those salient channels to protect their precision before quantizing, achieving INT4 with minimal loss and, importantly, **hardware-friendly kernels** (no per-weight Hessian, fast dequant). AWQ is widely used in production for its accuracy/speed/simplicity balance.

**SmoothQuant** attacks the activation-outlier problem ([§03](03_activation_quantization_challenges.md)) that blocks INT8 activations: it applies a per-channel **smoothing** transform `X → X·diag(s)⁻¹`, `W → diag(s)·W`, mathematically equivalent but shifting the "difficulty" (dynamic range) from the outlier-heavy activations into the better-behaved weights, so both can be quantized to INT8. This unlocks W8A8 GEMM with full Tensor-Core speedup on prefill *and* decode. The practical decision is: if your bottleneck is decode memory bandwidth, weight-only INT4 (AWQ/GPTQ) is simplest and most effective; if you also need prefill/compute speedup, go W8A8 (SmoothQuant) or FP8 ([§04](04_fp8_inference_h100.md)).

***

## Core Concepts & Mechanics

### GPTQ
- Objective: per layer, find quantized `Ŵ` minimizing `||WX − ŴX||²` over calibration activations X.
- Uses the (approximate) **Hessian** `H = 2XXᵀ` to order and compensate: quantize column by column, updating remaining columns to absorb the introduced error (an OBS/OBQ-style update). 📐 Greedy second-order error correction.
- Result: accurate INT3/INT4 weights; per-group scales (e.g., 128). Calibration ~128 samples.

### AWQ (Activation-aware Weight Quantization)
- Observation: scaling salient weight channels (those multiplied by large-magnitude activations) before quantization reduces their relative error.
- Per-channel scale `s` chosen to minimize quantization error on salient channels; `W' = W·diag(s)`, activations divided by `s` (folded). Then quantize `W'` to INT4.
- No Hessian; simple, fast **W4A16 kernels** with per-group dequant. Strong accuracy, production-popular.

### SmoothQuant
- Activation outliers (per-channel) make INT8 activations lossy. Define per-channel smoothing factor `s_j = max(|X_j|)^α / max(|W_j|)^{1−α}`.
- Transform: `X̂ = X·diag(s)⁻¹`, `Ŵ = diag(s)·W` — identical product `X̂Ŵ = XW`, but now both have manageable ranges → quantize both to INT8.
- Enables **W8A8** GEMM: full compute speedup (prefill + decode). α (migration strength) ~0.5 typical.

### Choosing
| Method | Scheme | Speeds up | Notes |
|---|---|---|---|
| GPTQ | W4A16 | decode | accurate, Hessian cost |
| AWQ | W4A16 | decode | simple, fast kernels, popular |
| SmoothQuant | W8A8 | prefill + decode | activation enabler |
| FP8 (TE) | W8A8 (FP8) | prefill + decode | Hopper+, near-lossless ([§04](04_fp8_inference_h100.md)) |

***

## Key Challenges
1. **Calibration sensitivity.** Scales/Hessians depend on calibration data; unrepresentative data → quality loss, especially for activations/SmoothQuant.
2. **W4A16 doesn't speed prefill.** Weight-only INT4 dequantizes to FP16 for the matmul, so compute-bound prefill barely benefits; only W8A8/FP8 speed compute.
3. **Per-group kernel complexity.** Fine-grained INT4 needs efficient fused dequant kernels; naive implementations lose the bandwidth savings to overhead.
4. **Task-specific degradation.** INT4 may pass perplexity but hurt reasoning/long-context; evaluation must be downstream-task aware.

***

## Solutions & Current Best Practices
- **AWQ or GPTQ (W4A16)** for decode-bound serving; AWQ favored for kernel speed/simplicity.
- **SmoothQuant (W8A8) or FP8** when prefill/compute speedup is needed.
- **Per-group (128) scales** for 4-bit; representative calibration; downstream eval.
- **Fused dequant + GEMM kernels** (Marlin, AWQ kernels) to realize the bandwidth win ([§06](../06_kernel_optimization/03_custom_cuda_kernels_gemm.md)).

***

## Implementation Notes
- vLLM/SGLang/TRT-LLM support AWQ/GPTQ/SmoothQuant/FP8 checkpoints; use optimized kernels (e.g., **Marlin** for INT4 W4A16) for real speedups.
- Calibrate on data resembling production prompts; ~128–512 samples usually suffice.
- Verify the kernel path actually runs low-precision GEMM (W8A8) vs dequant-to-FP16 (W4A16) — determines prefill speedup.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **INT4 model no faster on prefill** — W4A16 dequantizes to FP16 for matmul; expected. Use W8A8/FP8 for compute.
- **Slow INT4 despite small weights** — unfused/naive dequant kernel ate the bandwidth savings; use Marlin/AWQ kernels.
- **SmoothQuant degraded with wrong α or calibration** — migration strength and calibration data matter; tune α (~0.5), calibrate well.
- **Quantized model fails a specific task** — downstream degradation hidden by PPL; always eval on target tasks.

***

## Performance Numbers & Benchmarks
| Method | Model | Result |
|---|---|---|
| AWQ INT4 | Llama-70B | ~near-FP16 accuracy, ~3–4× smaller weights, faster decode |
| GPTQ INT4 | OPT/Llama | minimal PPL increase at 4-bit |
| SmoothQuant W8A8 | OPT-175B | ~lossless, ~1.5–2× speedup incl. prefill |
| Marlin INT4 kernel | — | near-roofline W4A16 GEMM throughput |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Contrast GPTQ, AWQ, and SmoothQuant."* — Expected: Hessian weight-only vs activation-aware weight-only vs activation-outlier migration for W8A8.
- *"Why does SmoothQuant enable INT8 activations?"* — Expected: migrate outlier range from activations to weights via per-channel smoothing.
- *"When does INT4 weight-only fail to speed up your workload?"* — Expected: prefill/compute-bound; dequant to FP16.
- *"What makes AWQ production-friendly?"* — Expected: no Hessian, fast fused kernels, strong accuracy.

***

## Open Problems & Active Research (2025–2026)
- **W4A4 PTQ** (4-bit weights+activations) without quality loss.
- **Outlier handling beyond SmoothQuant** (rotation/Hadamard methods, QuaRot).
- **PTQ for MoE and long-context** sensitivity.

***

## References
- Frantar, E., et al. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023*. arXiv:2210.17323.
- Lin, J., et al. (2024). "AWQ: Activation-aware Weight Quantization." *MLSys 2024*. arXiv:2306.00978.
- Xiao, G., et al. (2023). "SmoothQuant." *ICML 2023*. arXiv:2211.10438.
- Frantar, E., Alistarh, D. (2024). "Marlin: a Mixed-Precision Inference Kernel."
