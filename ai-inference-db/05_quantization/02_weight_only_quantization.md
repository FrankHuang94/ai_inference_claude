# Weight-Only Quantization (INT4/INT3)

> **Section:** 05_quantization
> **Last Updated:** June 2026
> **Related Files:** [01_post_training_quantization.md](01_post_training_quantization.md), [00_quantization_fundamentals.md](00_quantization_fundamentals.md), [../06_kernel_optimization/03_custom_cuda_kernels_gemm.md](../06_kernel_optimization/03_custom_cuda_kernels_gemm.md)
> **Must-Read Papers:** Frantar et al. (2023) "GPTQ"; Lin et al. (2024) "AWQ"; Frantar & Alistarh (2024) "Marlin kernel"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- Weight-only quantization (W4A16/W3A16) stores weights in INT4/INT3 but **computes in FP16/BF16** after on-the-fly dequantization.
- It targets **memory-bound decode**: weights are the dominant HBM read, so 4× smaller weights → up to ~4× decode throughput.
- It does **not** speed up compute-bound prefill (matmul still FP16) and adds a **dequant cost** that must be fused into the GEMM.
- Specialized kernels (**Marlin**, AWQ/Exllama kernels) make W4A16 GEMM near-roofline; naive dequant loses the benefit.
- The default choice when the goal is **lower decode latency / more batch capacity** at minimal quality loss.

***

## Overview
Weight-only quantization is the most popular production quantization for LLM serving because it directly and cheaply attacks the decode bottleneck. In decode, the dominant HBM traffic is **reading the model weights** ([§00](../00_fundamentals/05_transformer_inference_walkthrough.md)); the activations are tiny (one token × batch). So if weights are stored at 4 bits instead of 16, you move ~4× fewer weight bytes per token, and since decode is memory-bandwidth-bound, throughput scales nearly with the byte reduction — up to ~4× faster decode and ~4× more model/batch capacity in the same HBM. Activations stay in FP16 (hence "W4A16"), avoiding the activation-outlier problem entirely.

The mechanism is **dequantize-then-matmul**: weights are loaded from HBM as INT4 (small), dequantized to FP16 in-register/SRAM using per-group scales, and fed to the FP16 Tensor Cores. Because the matmul runs in FP16, **prefill (compute-bound) sees little speedup** — its bottleneck is FLOPs, not weight bytes, and the dequant even adds a bit of work. This is the key conceptual point candidates must articulate: weight-only quantization is a *bandwidth* optimization, so it helps the *bandwidth-bound* phase (decode) and not the *compute-bound* phase (prefill).

Realizing the benefit requires good kernels. A naive implementation that dequantizes weights to a full FP16 buffer in HBM and then does a standard GEMM **defeats the purpose** — you've re-created the FP16 bandwidth cost. The win comes only from **fused dequant+GEMM** kernels that load INT4 weights, dequantize in fast memory, and multiply, never materializing FP16 weights in HBM. Kernels like **Marlin** (Frantar & Alistarh 2024) and AWQ/ExLlama kernels achieve near-roofline W4A16 throughput and are why weight-only quantization is practical. This file focuses on the mechanics, kernel requirements, and limits of W4A16/W3A16.

***

## Core Concepts & Mechanics

### What's quantized
- **Weights**: INT4 (or INT3) with per-group scales (group size 64–128). Optionally zero-points (asymmetric).
- **Activations**: stay FP16/BF16. KV cache separate ([§05](05_kv_cache_quantization.md)).
- **Compute**: FP16 GEMM after dequant.

### Why decode speeds up, prefill doesn't
📐 Decode/token weight bytes: FP16 = `2·N`, INT4 = `0.5·N` → ~4× less HBM traffic → ~4× decode (bandwidth-bound). Prefill FLOPs unchanged (FP16 matmul) → ~no speedup; dequant adds minor overhead. (See roofline regions, [§00](../00_fundamentals/04_roofline_model_for_llm.md).)

### Fused dequant+GEMM
- Load INT4 weight tiles → dequantize in registers/SRAM with group scales → MMA against FP16 activations → accumulate.
- Must **never** write FP16 weights to HBM (that would restore the bandwidth cost).
- Marlin and similar kernels overlap dequant with MMA and use efficient bit-unpacking; achieve near peak W4A16 GEMM.

### Lower bits (INT3, INT2)
INT3 gives more compression but rising quality loss; INT2 is largely research-stage. Group size and outlier handling become critical below 4 bits.

***

## Key Challenges
1. **Kernel quality is everything.** Without fused dequant+GEMM, INT4 weights yield little speedup; the bottleneck moves to a poor kernel.
2. **No prefill benefit.** Compute-bound prefill doesn't speed up; mixed workloads see partial gains only.
3. **Sub-4-bit accuracy.** INT3/INT2 need finer groups, outlier handling, or QAT; PTQ alone degrades.
4. **Group-scale overhead.** Many small groups add metadata bandwidth and kernel complexity; too-coarse groups lose accuracy.

***

## Solutions & Current Best Practices
- **AWQ/GPTQ INT4 + Marlin/AWQ kernels** for decode-bound serving — the standard ([§01](01_post_training_quantization.md)).
- **Group size 128** as a default accuracy/overhead balance.
- For **compute speedup too**, combine/replace with W8A8 or FP8 ([§04](04_fp8_inference_h100.md)).
- Validate the kernel achieves near-roofline W4A16 throughput in profiling ([§06](../06_kernel_optimization/05_kernel_profiling_and_benchmarking.md)).

***

## Implementation Notes
- Use framework-integrated optimized kernels (vLLM Marlin path, AWQ kernels); avoid generic dequant.
- Confirm the path is genuinely fused (no FP16 weight materialization) via memory profiling.
- For latency-critical decode, also enable CUDA graphs to remove launch overhead ([§06](../06_kernel_optimization/02_fused_kernels_and_operator_fusion.md)).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **INT4 weights, FP16-speed decode** — kernel materialized dequantized weights in HBM; no bandwidth saving. Use a fused kernel (Marlin).
- **"Why is prefill the same?"** — expected; W4A16 doesn't touch compute-bound prefill.
- **INT3 broke a reasoning task** — sub-4-bit PTQ degrades; needs finer groups/outlier handling or QAT.
- **Group size 32 was slower** — too many scales added overhead; 128 usually better balance.

***

## Performance Numbers & Benchmarks
| Scheme | Weight bytes | Decode | Prefill | Quality |
|---|---|---|---|---|
| FP16 | 1× | baseline | baseline | ref |
| INT8 W8A16 | 0.5× | ~2× | ~0 | ~lossless |
| INT4 W4A16 (Marlin) | 0.25× | ~3–4× | ~0 | small loss |
| INT3 W3A16 | ~0.19× | higher | ~0 | larger loss |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why does weight-only INT4 speed up decode ~4× but not prefill?"* — Expected: bandwidth-bound decode (weight bytes) vs compute-bound prefill (FP16 matmul).
- *"What must a W4A16 kernel do to realize the benefit?"* — Expected: fused dequant+GEMM, never materialize FP16 weights in HBM.
- *"When would you choose W8A8/FP8 over W4A16?"* — Expected: when prefill/compute speedup matters.
- *"What limits going below 4 bits?"* — Expected: accuracy; needs finer groups/outlier handling/QAT.

***

## Open Problems & Active Research (2025–2026)
- **Accurate sub-4-bit** weight-only (W3/W2) via better rounding/rotation and QAT.
- **Even faster low-bit kernels** on Blackwell (FP4 native).
- **Mixed-precision per-layer** (keep sensitive layers higher bit) automation.

***

## References
- Frantar, E., et al. (2023). "GPTQ." *ICLR 2023*. arXiv:2210.17323.
- Lin, J., et al. (2024). "AWQ." *MLSys 2024*. arXiv:2306.00978.
- Frantar, E., Alistarh, D. (2024). "Marlin: Mixed-Precision Auto-Regressive Inference Kernel."
