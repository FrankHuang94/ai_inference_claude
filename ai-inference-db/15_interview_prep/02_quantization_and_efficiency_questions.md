# Quantization and Efficiency Questions (Q&A)

> **Section:** 15_interview_prep
> **Last Updated:** June 2026
> **Related Files:** [../05_quantization/00_quantization_fundamentals.md](../05_quantization/00_quantization_fundamentals.md), [../05_quantization/04_fp8_inference_h100.md](../05_quantization/04_fp8_inference_h100.md), [01_kernel_and_gpu_technical_questions.md](01_kernel_and_gpu_technical_questions.md)
> **Must-Read Papers:** Frantar et al. (2023) "GPTQ"; Lin et al. (2024) "AWQ"; Xiao et al. (2023) "SmoothQuant"; Dettmers et al. (2022) "LLM.int8()"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- Anchor answers in: **decode is bandwidth-bound → weight quantization helps decode (bytes); compute speedup (prefill) needs low-precision GEMM (W8A8/FP8)**.
- Know the methods cold: GPTQ (Hessian), AWQ (activation-aware), SmoothQuant (outlier migration), FP8 (float range), KV quant (per-channel keys/per-token values).
- Be ready to explain **why activations are harder than weights** (outliers) and the four fixes.
- Quantify byte/speedup: INT4 weights → 4× bytes → up to ~4× decode, ~0 prefill.

***

## Q1: Why does weight-only INT4 speed up decode ~4× but barely help prefill?
Decode is **bandwidth-bound**: the dominant cost is reading weights from HBM. INT4 weights are 4× smaller → ~4× less weight traffic → ~4× faster decode (if the kernel fuses dequant; else no win). Prefill is **compute-bound**: the matmul runs in FP16 after dequant, so FLOPs are unchanged — no speedup, and dequant even adds a little work. To speed prefill you need low-precision **compute** (W8A8/FP8 GEMM on Tensor Cores). ([§05](../05_quantization/02_weight_only_quantization.md))

## Q2: Why are activations harder to quantize than weights?
LLM activations have **systematic emergent outliers** (Dettmers et al.): a few channels are 10–100× larger and are critical for quality. Integer quantization uses one scale across the range, so a giant outlier forces a large scale that maps normal values onto ~1 level → precision collapse. Weights are well-behaved (quantize to INT4 easily); activations need: **mixed-precision** (LLM.int8() keeps outliers in FP16), **migration** (SmoothQuant shifts range to weights), **rotation** (QuaRot spreads outliers), or **float format** (FP8's exponent handles range). ([§05](../05_quantization/03_activation_quantization_challenges.md))

## Q3: Compare GPTQ, AWQ, and SmoothQuant.
- **GPTQ** (W4A16): layer-wise weight quantization minimizing output error via approximate **Hessian** (second-order) compensation. Accurate INT4 weights; Hessian cost.
- **AWQ** (W4A16): **activation-aware** — scale up the ~1% salient weight channels (by activation magnitude) before quantizing; no Hessian, fast kernels, production-popular.
- **SmoothQuant** (W8A8): **migrate** activation outlier range into weights via per-channel smoothing → both quantize to INT8 → full compute speedup (prefill+decode).
Choose W4A16 (AWQ/GPTQ) for decode-bound; W8A8/FP8 for compute speedup. ([§05](../05_quantization/01_post_training_quantization.md))

## Q4: Why is FP8 the production default on Hopper/Blackwell?
FP8 is a **float** format whose exponent natively represents activation outlier ranges (no INT8 outlier gymnastics), and H100 has FP8 Tensor Cores at 2× FP16. So FP8 weights+activations are **near-lossless** *and* speed **both** prefill (2× FLOPs) and decode (half bytes). E4M3 (range ±448) for forward; E5M2 (±57344) for wide range. Transformer Engine handles per-tensor scaling + delayed scaling. ([§05](../05_quantization/04_fp8_inference_h100.md))

## Q5: When does KV-cache quantization matter more than weight quantization?
At **long context / high batch**, the KV cache exceeds the weights (KV grows linearly with batch×seq; crossover ~batch×seq > weights/KV_per_token). Then quantizing KV (INT4/FP8) frees the binding memory and enables bigger batches. Use **per-channel keys, per-token values** (KIVI) — keys have channel outliers. Validate **long-context recall** (not just perplexity), and watch the **prefix-caching interaction** (quantized blocks may not match). ([§05](../05_quantization/05_kv_cache_quantization.md), [§02](../02_kv_cache/00_kv_cache_fundamentals.md))

## Q6: When is PTQ insufficient and you need QAT?
PTQ suffices to ~INT4 weights / FP8. Below that (**W4A4, INT3/INT2**) or for sensitive models, PTQ degrades because weights weren't trained to tolerate quantization error. **QAT** simulates quantization in the forward pass (fake-quant) and trains through it via the **straight-through estimator** (identity gradient through rounding), so weights adapt — recovering accuracy at the cost of training compute. For LLMs, QAT-fine-tuning + distillation (LLM-QAT). ([§05](../05_quantization/06_quantization_aware_training.md))

## Q7: Estimate the throughput gain from FP8 on a decode-heavy 70B workload.
Weights 140GB FP16 → 70GB FP8. Decode ≈ bandwidth/weight_bytes → ~2× faster (bandwidth-bound, halved bytes). Prefill ~2× from FP8 GEMM. Net depends on prefill/decode mix; for decode-heavy, ~close to 2× on the decode portion, plus larger feasible batch (freed HBM). ([§05](../05_quantization/04_fp8_inference_h100.md))

***

## Interview Angles
> 💡 **Delivery tips:**
- Always tie quantization to **which phase** it helps (decode=bytes, prefill=compute).
- For "why activations hard," lead with **outliers** and name the four fixes.
- Quantify byte reduction → speedup.
- Mention **validation** (downstream tasks, long-context recall), not just perplexity.

***

## Complexity Gotchas
> ⚠️ **Common mistakes:**
- Claiming INT4 weight-only speeds prefill (it doesn't).
- Forgetting the **fused dequant kernel** requirement for W4A16 (else no win).
- Per-token quantizing keys (they need per-channel).
- Gating quantization on perplexity alone (misses task/recall damage).

***

## References
- Frantar et al. (2023) "GPTQ" arXiv:2210.17323; Lin et al. (2024) "AWQ" arXiv:2306.00978.
- Xiao et al. (2023) "SmoothQuant" arXiv:2211.10438; Dettmers et al. (2022) "LLM.int8()" arXiv:2208.07339.
- Micikevicius et al. (2022) "FP8 Formats" arXiv:2209.05433.
