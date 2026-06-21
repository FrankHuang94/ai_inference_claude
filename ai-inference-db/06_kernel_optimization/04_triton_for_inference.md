# Triton for Inference Kernels

> **Section:** 06_kernel_optimization
> **Last Updated:** June 2026
> **Related Files:** [00_cuda_kernel_basics_for_inference.md](00_cuda_kernel_basics_for_inference.md), [03_custom_cuda_kernels_gemm.md](03_custom_cuda_kernels_gemm.md), [02_fused_kernels_and_operator_fusion.md](02_fused_kernels_and_operator_fusion.md)
> **Must-Read Papers:** Tillet et al. (2019, MAPL) "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"; OpenAI Triton docs
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- **Triton** (Tillet et al. 2019, OpenAI) is a Python-embedded DSL for GPU kernels: you write **tile-level** code; the compiler handles coalescing, shared-memory management, and scheduling.
- It hits a sweet spot: **much easier than CUDA**, often within ~10–20% of hand-tuned performance, with a built-in **autotuner**.
- Heavily used in inference: FlashAttention reference impls, **quantization kernels**, SGLang's kernel library, fused norm/activation kernels.
- Limits (historically): weaker on async copies (TMA), irregular access, and squeezing the last % vs CUTLASS/wgmma — gaps narrowing each release.
- Great for **rapid custom fused kernels** (dequant+matmul, fused epilogues) without CUDA expertise.

***

## Overview
Triton is a domain-specific language and compiler that lets you write high-performance GPU kernels in Python at the **tile** abstraction level, leaving low-level concerns — thread/warp mapping, memory coalescing, shared-memory allocation, and instruction scheduling — to the compiler. Instead of managing 32-thread warps and bank conflicts by hand (as in CUDA), you express computation over blocks of data (`tl.load`/`tl.store` on pointer ranges, `tl.dot` for matmul), and Triton compiles it to efficient PTX. This dramatically lowers the barrier to writing custom kernels: a fused layernorm+matmul or a quantized GEMM that would be hundreds of lines of intricate CUDA becomes a readable Python function, while still reaching a large fraction of hand-tuned performance.

This productivity is why Triton became ubiquitous in the inference stack. The widely-used **FlashAttention-2** reference implementation is in Triton; many **quantization kernels** (dequant+GEMM), fused activations/norms, and SGLang's custom kernel library are Triton-based; and PyTorch's `torch.compile` uses Triton as its GPU backend for generated fused kernels. The **autotuner** is a key feature: you annotate a kernel with candidate configurations (tile sizes, num_warps, num_stages) and Triton benchmarks them to pick the best for the given shapes/hardware, automating the tile-tuning that's painful in raw CUDA.

The tradeoffs: Triton historically lagged hand-written CUDA/CUTLASS on the most demanding kernels — particularly those needing fine control over **asynchronous memory copies** (Hopper TMA), warp specialization, or highly irregular access patterns — so the absolute peak (e.g., FA-3's H100-specific scheduling) still comes from expert CUDA. But the gap narrows every release, and for the vast majority of inference kernels (fused elementwise, quantized GEMM, attention variants) Triton delivers near-peak performance at a fraction of the development cost. This file covers the programming model, autotuning, where Triton excels, and its limits.

***

## Core Concepts & Mechanics

### Programming model
- A Triton kernel is a Python function decorated with `@triton.jit`, launched over a **grid** of program instances (like thread blocks).
- Each instance operates on **tiles** via `tl.load(ptr + offsets, mask=...)`, computes (e.g., `tl.dot(a, b)`), and `tl.store`s results.
- The compiler maps tiles to warps/threads, allocates shared memory, vectorizes, and **coalesces** loads automatically. You reason about *blocks of data*, not threads.

### Example: fused RMSNorm + matmul (sketch)
```python
@triton.jit
def fused_kernel(x_ptr, w_ptr, out_ptr, ...):
    row = tl.program_id(0)
    x = tl.load(x_ptr + row*D + tl.arange(0, D))      # load a row tile
    rms = tl.sqrt(tl.sum(x*x)/D + eps)
    xn = x / rms * g
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):                     # tiled matmul
        w = tl.load(w_ptr + ...)
        acc += tl.dot(xn_tile, w)
    tl.store(out_ptr + ..., acc)
```
The compiler handles coalescing, shared memory, and scheduling — you wrote no thread/warp code.

### Autotuning
```python
@triton.autotune(configs=[triton.Config({'BLOCK_M':128,'BLOCK_K':64}, num_warps=4, num_stages=3), ...],
                 key=['M','N','K'])
```
Triton benchmarks configs per shape (`key`) and caches the best — automating tile/warp/stage tuning.

### Where Triton shines
- **Fused custom kernels**: dequant+GEMM, fused norm/activation/residual, quantization.
- **Attention variants**: FlashAttention-2 Triton ref, custom masking, MLA experiments.
- **Rapid iteration**: prototype a kernel in minutes vs days of CUDA.

***

## Key Challenges
1. **Peak-performance gap.** For the most demanding kernels (Hopper async TMA/wgmma, warp specialization), expert CUDA/CUTLASS still wins the last 10–30%.
2. **Irregular/async patterns.** Historically limited support for fine-grained async copies and very irregular access; improving but not on par with CUDA.
3. **Autotuning cost / variance.** Tuning many configs takes time; cache invalidation across shapes/hardware needs care.
4. **Debugging.** Higher abstraction can make low-level correctness/perf bugs harder to pinpoint than explicit CUDA.

***

## Solutions & Current Best Practices
- **Use Triton for custom fused/quantized kernels** and rapid prototyping; reach for CUDA/CUTLASS only for the most demanding peak-perf kernels.
- **Leverage the autotuner** with a sensible config set keyed on shapes.
- **Adopt existing Triton kernels** (FlashAttention-2, framework libraries) rather than reinventing.
- **Profile Triton kernels in Nsight** like any CUDA kernel ([§05](05_kernel_profiling_and_benchmarking.md)).

***

## Implementation Notes
- `torch.compile` emits Triton automatically for fused elementwise chains — often the easiest win.
- Pick `BLOCK_*`, `num_warps`, `num_stages` via autotune; defaults are rarely optimal.
- Mask loads/stores for boundary tiles; mismatched masks cause subtle correctness bugs.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Triton kernel ~20% below CUTLASS on H100** — missing async TMA/wgmma exploitation; for peak prefill GEMM, CUTLASS/FA-3 may win.
- **Autotune picked a config that's wrong for a new shape** — `key` didn't capture the shape; ensure tuning keys cover your shape space.
- **Boundary mask bug produced NaNs** — unmasked out-of-range loads; always mask edge tiles.
- **torch.compile recompiled per shape** — dynamic shapes triggered recompilation; use dynamic shapes/padding settings.

***

## Performance Numbers & Benchmarks
| Kernel | Triton vs CUDA/CUTLASS |
|---|---|
| Fused elementwise/norm | ~on par, far less code |
| FlashAttention-2 (Triton ref) | near hand-tuned |
| Quantized dequant+GEMM | competitive; Marlin (CUDA) may edge it |
| Hopper peak GEMM (wgmma/TMA) | CUDA/CUTLASS ahead |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"What is Triton and why is it used in inference frameworks?"* — Expected: tile-level Python DSL, compiler handles low-level; fused/quantized/attention kernels.
- *"How would you write a fused layernorm+linear kernel in Triton?"* — Expected: load row tile, compute norm, tiled `tl.dot`, store; autotune.
- *"Where does Triton fall short of hand-written CUDA?"* — Expected: async TMA/wgmma, irregular access, last % of peak.
- *"How does Triton's autotuner work?"* — Expected: benchmark candidate configs keyed on shapes, cache best.

***

## Open Problems & Active Research (2025–2026)
- **Closing the Hopper/Blackwell async gap** (TMA/wgmma, warp specialization) in Triton.
- **Better compiler heuristics** to reduce reliance on autotuning.
- **Triton for FP4/MX** and emerging formats.

***

## References
- Tillet, P., Kung, H.T., Cox, D. (2019). "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." *MAPL 2019*.
- OpenAI. "Triton" documentation and tutorials.
- Dao, T. (2023). "FlashAttention-2" (Triton reference). arXiv:2307.08691.
