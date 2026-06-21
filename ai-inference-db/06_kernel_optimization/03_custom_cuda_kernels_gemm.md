# Custom CUDA Kernels and GEMM Optimization

> **Section:** 06_kernel_optimization
> **Last Updated:** June 2026
> **Related Files:** [00_cuda_kernel_basics_for_inference.md](00_cuda_kernel_basics_for_inference.md), [02_fused_kernels_and_operator_fusion.md](02_fused_kernels_and_operator_fusion.md), [../05_quantization/02_weight_only_quantization.md](../05_quantization/02_weight_only_quantization.md)
> **Must-Read Papers:** NVIDIA CUTLASS docs; Williams et al. (2009) "Roofline"; Frantar & Alistarh (2024) "Marlin"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- GEMM (matrix multiply) is the core compute primitive; fast GEMM = **hierarchical tiling** (block → warp → thread/MMA) staging data through SRAM and registers to maximize reuse.
- **Tensor-Core MMA** instructions (wmma/wgmma) do the actual matmul; CUTLASS provides composable templates so you rarely write raw MMA.
- For inference, custom kernels matter most for **quantized GEMM** (W4A16 dequant+matmul like **Marlin**) and **fused epilogues** libraries don't cover.
- GEMM is compute-bound at large M (prefill); **GEMV-like** decode (M=batch small) is memory-bound — different kernels (and that's why decode needs different optimization).
- Arithmetic intensity of GEMM ≈ tied to the tile/reuse; the goal is to keep Tensor Cores fed from SRAM.

***

## Overview
General matrix-matrix multiply (GEMM) is the computational backbone of transformers: the QKV/output projections, the FFN, and the LM head are all GEMMs. Its performance defines prefill throughput and, in quantized form, much of decode efficiency. Fast GEMM is a textbook exercise in the memory hierarchy: the C = A·B computation has O(MNK) FLOPs but only O(MN+MK+NK) data, so with proper **reuse** it's compute-bound — but only if you stage data through the fast levels (SRAM, registers) so each loaded element is multiplied many times before eviction. The standard structure is **hierarchical tiling**: partition the output into block-tiles (one per thread block, staged in shared memory), warp-tiles (per warp), and thread/MMA-tiles (fed to Tensor Cores), with the loop over K streaming successive tiles through SRAM.

In practice you rarely hand-write raw MMA loops; **CUTLASS** (NVIDIA's open-source template library) provides composable, highly-tuned GEMM building blocks parameterized by tile sizes, data types, and **epilogues** (the post-GEMM fused ops like bias/activation/quantize). cuBLAS provides black-box optimized GEMM for standard cases. So when do you write custom kernels for inference? Primarily for **quantized GEMM** — e.g., W4A16 where you must dequantize INT4 weights in-kernel and feed FP16 Tensor Cores (Marlin, [§05](../05_quantization/02_weight_only_quantization.md)) — and for **fusions** the libraries don't provide (custom epilogues, attention variants, MLA). These are where teams differentiate, and where "wrote a custom CUDA kernel" on a resume gets probed.

The crucial inference nuance: **prefill GEMM (large M = batch×seq) is compute-bound** and rewards classic tiling/Tensor-Core optimization, while **decode is effectively GEMV** (M = batch, often small) — memory-bound, dominated by streaming the weight matrix, with little reuse to exploit. So "GEMM optimization" splits into two regimes: feed-the-Tensor-Cores for prefill, and minimize-weight-bytes (quantization, fused dequant) for decode. Confusing the two is a common error. This file covers GEMM tiling, Tensor-Core MMA, CUTLASS, and quantized-GEMM kernels.

***

## Core Concepts & Mechanics

### Hierarchical tiling
```
for each block-tile of C (in shared memory):
  for k-tile in K:
     load A,B k-tiles → shared memory (coalesced, async/TMA)
     for each warp-tile:
        for each MMA-tile:
           Tensor-Core MMA accumulate into registers
  epilogue (bias, activation, quantize) → write C
```
📐 Reuse factor = tile size; bigger tiles → each HBM-loaded element reused more → higher arithmetic intensity → closer to compute roof. Limited by SRAM/registers (occupancy tradeoff, [§00](00_cuda_kernel_basics_for_inference.md)).

### Tensor-Core MMA
- `wmma` (Volta+) / **`wgmma`** (Hopper, warp-group async) execute small matrix multiply-accumulates natively (e.g., 16×16×16), accumulating in higher precision.
- Hopper's async wgmma + **TMA** copies enable overlapping load and compute (used by FA-3, [§01](01_flash_attention_deep_dive.md)).

### CUTLASS and epilogues
CUTLASS templates let you compose tile sizes, dtypes, and **epilogue** (fused post-ops) without writing MMA by hand. Epilogue fusion (bias+activation+quantize) avoids extra HBM round-trips ([§02](02_fused_kernels_and_operator_fusion.md)).

### Prefill GEMM vs decode GEMV
📐 Prefill: M=batch×seq large → AI high → compute-bound → tile/Tensor-Core optimization. Decode: M=batch (often 1–64) → little reuse → memory-bound (GEMV-like) → optimize weight bytes (quantization), use fused dequant kernels, batch to raise M.

### Quantized GEMM (Marlin)
W4A16: load INT4 weights (small), dequantize in registers/SRAM with group scales, MMA against FP16 activations, accumulate. Marlin overlaps bit-unpacking/dequant with MMA to hit near-roofline W4A16 throughput. Never materializes FP16 weights in HBM.

***

## Key Challenges
1. **Tile-size / occupancy tuning.** Optimal tiles depend on shapes, dtype, and hardware; auto-tuning (CUTLASS/Triton) needed; wrong tiles waste Tensor Cores or SRAM.
2. **Quantized dequant overlap.** Hiding INT4 unpacking/dequant behind MMA is hard; naive versions are bandwidth-bound on metadata or compute-bound on unpacking.
3. **Decode GEMV is intrinsically memory-bound.** No tiling fixes the lack of reuse at small M; the lever is bytes (quantization) and batch, not GEMM cleverness.
4. **Generation portability.** wgmma/TMA are Hopper-specific; FP4 MMA is Blackwell; kernels need per-generation paths.

***

## Solutions & Current Best Practices
- **Use cuBLAS/CUTLASS** for standard GEMM; **custom kernels** only for quantized GEMM and unusual fusions.
- **Marlin/AWQ kernels** for W4A16 decode; ensure fused dequant ([§05](../05_quantization/02_weight_only_quantization.md)).
- **Auto-tune tiles** (CUTLASS profiler, Triton autotuner) per shape/hardware.
- **Raise M (batch)** to move decode toward compute-bound; quantize weights for the rest.

***

## Implementation Notes
- Prefer CUTLASS epilogue fusion over separate post-op kernels.
- For Hopper, exploit wgmma+TMA async pipelines; for Blackwell, FP4/FP8 MMA paths.
- Benchmark against cuBLAS as a baseline; a custom kernel that doesn't beat cuBLAS isn't worth maintaining (except for fused/quantized cases libraries lack).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Custom GEMM slower than cuBLAS** — hand-rolled kernels rarely beat vendor libraries for standard GEMM; reserve custom work for quantized/fused cases.
- **Optimized prefill GEMM didn't help decode** — decode is GEMV/memory-bound; tiling doesn't address lack of reuse. Quantize + batch instead.
- **W4A16 kernel bandwidth-bound on scales** — too-fine group scales added metadata traffic; balance group size.
- **wgmma path crashed on A100** — Hopper-only instruction; need per-generation fallback.

***

## Performance Numbers & Benchmarks
| Kernel | Regime | Note |
|---|---|---|
| cuBLAS/CUTLASS FP16 GEMM | prefill | near peak with good tiles |
| FP8 GEMM (TE/CUTLASS) | prefill | ~2× FP16 |
| Marlin W4A16 | decode | near-roofline, fused dequant |
| decode GEMV | decode | memory-bound; batch/quantize |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Describe hierarchical GEMM tiling and why it makes GEMM compute-bound."* — Expected: block/warp/MMA tiles, SRAM reuse, AI ∝ tile.
- *"Why is decode GEMV memory-bound and what do you do about it?"* — Expected: small M, no reuse; quantize weights, raise batch.
- *"What does a W4A16 kernel like Marlin do?"* — Expected: load INT4, dequant in-register overlapped with MMA, FP16 accumulate.
- *"When write a custom kernel vs use cuBLAS/CUTLASS?"* — Expected: quantized GEMM, custom fusions, attention variants; not standard GEMM.

***

## Open Problems & Active Research (2025–2026)
- **Optimal low-bit (FP4/INT4) GEMM kernels** on Blackwell.
- **Compiler-generated GEMM** (Triton/Mojo) matching CUTLASS across shapes.
- **Fused quantized GEMM + epilogue + attention** mega-kernels.

***

## References
- NVIDIA. "CUTLASS" library documentation.
- Williams, S., et al. (2009). "Roofline." *CACM* 52(4).
- Frantar, E., Alistarh, D. (2024). "Marlin: Mixed-Precision Auto-Regressive Inference Kernel."
