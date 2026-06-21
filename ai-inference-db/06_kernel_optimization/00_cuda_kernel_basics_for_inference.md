# CUDA Kernel Basics for Inference

> **Section:** 06_kernel_optimization
> **Last Updated:** June 2026
> **Related Files:** [01_flash_attention_deep_dive.md](01_flash_attention_deep_dive.md), [05_kernel_profiling_and_benchmarking.md](05_kernel_profiling_and_benchmarking.md), [../01_hardware/00_gpu_architecture_for_inference.md](../01_hardware/00_gpu_architecture_for_inference.md)
> **Must-Read Papers:** NVIDIA CUDA C++ Programming Guide; Williams et al. (2009) "Roofline"; Dao et al. (2022) "FlashAttention"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- CUDA executes a **grid of thread blocks**; each block runs on one SM as **warps of 32 threads** in SIMT lockstep.
- Performance hinges on: **memory coalescing** (contiguous warp accesses), **shared-memory tiling** (reuse data in SRAM), **occupancy** (hide latency), and avoiding **warp divergence**.
- For inference, classify every kernel as **memory-bound or compute-bound** via the roofline; optimize accordingly ([§00](../00_fundamentals/04_roofline_model_for_llm.md)).
- **Nsight Compute** is the tool: read Memory vs SM throughput, L2 hit rate, achieved occupancy, and the built-in roofline.
- Decode kernels are memory-bound and small; **CUDA graphs** and fusion fight launch overhead ([§02](02_fused_kernels_and_operator_fusion.md)).

***

## Overview
Writing or reasoning about GPU kernels requires a working model of the CUDA execution and memory hierarchy. A kernel launch creates a **grid** of **thread blocks**; the hardware assigns each block to a **streaming multiprocessor (SM)**, which executes the block's threads in **warps** of 32 under the SIMT model — all 32 threads issue the same instruction per step, operating on different data. Threads within a block can cooperate via fast **shared memory (SRAM)** and synchronize with barriers; threads in different blocks cannot (cheaply). This structure dictates the optimization levers: arrange memory accesses so each warp reads contiguous addresses (**coalescing**, to use HBM efficiently), stage reused data in shared memory (**tiling**, to avoid repeated HBM reads), keep enough warps resident to **hide memory latency** (occupancy), and avoid branches that split a warp (**divergence**).

For inference, the single most useful habit is to classify each kernel on the **roofline** ([§00](../00_fundamentals/04_roofline_model_for_llm.md)) before optimizing. Compute-bound kernels (prefill GEMMs) want better tiling, larger MMA tiles, and FP8 Tensor Cores; memory-bound kernels (decode matmuls, attention KV reads) want fewer bytes moved (fusion, quantization), coalesced access, and SRAM reuse. Optimizing a memory-bound kernel for FLOPs, or vice versa, wastes effort — a frequent novice mistake the roofline prevents.

The tooling reality is **Nsight Compute** (kernel-level) and **Nsight Systems** (timeline-level). Nsight Compute reports whether a kernel is memory- or compute-bound (Memory vs SM throughput sections), L2 hit rate, achieved vs theoretical occupancy, warp stall reasons, and overlays the kernel on a roofline. Reading these is a directly-tested interview skill for inference engineering roles. This file establishes the execution model, the optimization levers, and how to diagnose a kernel — the foundation for the FlashAttention, fusion, GEMM, and Triton files that follow.

***

## Core Concepts & Mechanics

### Execution model
- **Thread → warp (32) → block → grid.** Blocks map to SMs; warps are the scheduling unit. Many resident warps per SM hide latency by switching on stalls.
- **Shared memory / L1** (up to 228 KB/SM on H100): software-managed scratchpad for tiling; ~100× faster than HBM.
- **Registers**: per-thread, fastest; heavy register use lowers occupancy.

### Memory coalescing
📐 When the 32 threads of a warp access **contiguous, aligned** addresses, the hardware merges them into the minimum number of memory transactions (e.g., one 128-byte transaction). Strided/scattered access → many transactions → wasted bandwidth. Paged-KV attention kernels must lay out blocks to keep reads coalesced ([§02](../02_kv_cache/01_paged_attention_vllm.md)).

### Shared-memory tiling
Load a tile of the operands into shared memory once, reuse it across many threads/iterations, then move on — turning repeated HBM reads into one HBM read + many SRAM reads. The basis of fast GEMM and FlashAttention ([§01](01_flash_attention_deep_dive.md)).

### Occupancy and latency hiding
📐 Occupancy = resident warps / max warps per SM. Higher occupancy hides memory latency (more warps to switch to). But large register/shared-memory tiles (good for compute-bound GEMM) reduce occupancy — so "maximize occupancy" is *not* universally right; compute-bound kernels may prefer fat tiles at lower occupancy.

### Warp divergence
Branches where warp threads take different paths serialize the paths. Sources in inference: attention masking, variable-length handling, top-p sampling. Minimize data-dependent branching in hot kernels.

### Launch overhead
Each kernel launch costs ~µs of CPU/GPU overhead. At batch=1 decode (sub-ms steps with dozens of kernels), launch overhead is a real fraction of runtime → **CUDA graphs** capture the sequence as one launch ([§02](02_fused_kernels_and_operator_fusion.md)).

***

## Key Challenges
1. **Memory vs compute mismatch in optimization.** Optimizing the wrong dimension (FLOPs for a memory-bound kernel) wastes effort; the roofline must guide the work.
2. **Occupancy/register tension.** Fast GEMM wants big tiles (low occupancy); naive occupancy maximization can slow compute-bound kernels.
3. **Coalescing with dynamic layouts.** Paged/quantized KV and ragged batches make coalesced access harder; layout matters.
4. **Launch overhead at small batch.** Decode's many tiny kernels make per-launch overhead significant without graphs/fusion.

***

## Solutions & Current Best Practices
- **Classify on the roofline first**, then optimize the binding resource ([§00](../00_fundamentals/04_roofline_model_for_llm.md)).
- **Tile to SRAM** and **coalesce** all HBM access; verify in Nsight ([§05](05_kernel_profiling_and_benchmarking.md)).
- **CUDA graphs + fusion** for small-batch decode ([§02](02_fused_kernels_and_operator_fusion.md)).
- **Use TMA/async copies (Hopper)** to overlap HBM→SRAM with compute.

***

## Implementation Notes
- Prefer library kernels (cuBLAS/CUTLASS for GEMM, FlashAttention for attention) before hand-writing; reserve custom kernels for fusions libraries don't cover ([§03](03_custom_cuda_kernels_gemm.md)).
- Check **shared-memory bank conflicts** (a frequent hidden 2× loss) and alignment for coalescing.
- Capture the decode step in a CUDA graph once shapes are stable.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Maximizing occupancy slowed a GEMM** — spilled registers / smaller tiles; compute-bound kernels prefer fat tiles over max occupancy.
- **Strided KV access halved bandwidth** — non-coalesced reads from a bad paged layout; fix block/stride layout.
- **Bank conflicts silently cost 2×** — shared-memory access pattern conflicts; pad/stride to avoid.
- **Decode dominated by launch overhead** — dozens of tiny kernels/step; use CUDA graphs.

***

## Performance Numbers & Benchmarks
| Lever | Effect |
|---|---|
| Coalesced vs strided HBM | up to several× bandwidth |
| Shared-mem tiling (GEMM) | turns O(N³) HBM into O(N²) |
| CUDA graphs (decode) | removes µs×kernels launch overhead |
| Bank-conflict-free SRAM | up to 2× shared-mem throughput |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Explain the CUDA execution model and memory coalescing."* — Expected: grid/block/warp/SIMT; contiguous warp access → merged transactions.
- *"How do you tell if a kernel is memory- or compute-bound in Nsight?"* — Expected: Memory vs SM throughput, roofline overlay.
- *"Why isn't maximizing occupancy always right?"* — Expected: register/tile tradeoff; compute-bound prefers fat tiles.
- *"Why do CUDA graphs help decode?"* — Expected: many tiny kernels/step; launch overhead removed.

***

## Open Problems & Active Research (2025–2026)
- **Mega-kernels** fusing whole transformer layers to minimize launches/HBM round-trips for decode.
- **Auto-tuning** kernels across precisions/generations (FP8/FP4) and dynamic shapes.
- **Compiler-generated** optimal kernels (Triton/Mojo) closing the gap to hand-written CUDA.

***

## References
- NVIDIA. "CUDA C++ Programming Guide" and "Nsight Compute" documentation.
- Williams, S., et al. (2009). "Roofline." *CACM* 52(4).
- Dao, T., et al. (2022). "FlashAttention." *NeurIPS 2022*. arXiv:2205.14135.
