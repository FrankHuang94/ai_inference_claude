# Fused Kernels, Operator Fusion, and CUDA Graphs

> **Section:** 06_kernel_optimization
> **Last Updated:** June 2026
> **Related Files:** [00_cuda_kernel_basics_for_inference.md](00_cuda_kernel_basics_for_inference.md), [01_flash_attention_deep_dive.md](01_flash_attention_deep_dive.md), [../00_fundamentals/01_autoregressive_decoding.md](../00_fundamentals/01_autoregressive_decoding.md)
> **Must-Read Papers:** Dao et al. (2022) "FlashAttention" (fusion exemplar); NVIDIA CUDA Graphs docs; Chen et al. (2018, OSDI) "TVM"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- **Operator fusion** combines multiple ops into one kernel to **eliminate intermediate HBM round-trips** and kernel launches — key for memory-bound decode.
- Common fusions: **LayerNorm+QKV**, **attention+residual**, **SwiGLU FFN** (gate·up·activation), **dequant+GEMM**.
- **CUDA graphs** capture a sequence of kernels as one replayable graph, removing per-launch CPU overhead — critical at small-batch decode where launches are a real fraction of step time.
- Fusion helps most when ops are **memory-bound and small** (decode glue); large compute-bound GEMMs benefit less.
- Frameworks rely on fused kernels (xformers, TensorRT-LLM plugins, Triton) + CUDA graphs as standard.

***

## Overview
Many transformer operations are small, memory-bound, and elementwise — LayerNorm/RMSNorm, RoPE, residual adds, activations, dequantization. Run as separate kernels, each reads its input from HBM and writes its output back, so a chain of N such ops incurs N HBM round-trips plus N kernel launches, even though the actual arithmetic is trivial. **Operator fusion** merges these into a single kernel that reads once, does all the work in registers/SRAM, and writes once — collapsing N HBM round-trips into one and N launches into one. For the bandwidth-bound decode path, where HBM traffic and launch overhead dominate, fusion is a major win; FlashAttention is the extreme example (fusing the entire attention computation, [§01](01_flash_attention_deep_dive.md)).

The complementary technique is **CUDA graphs**. A single decode step issues dozens of kernels (per layer: norms, QKV, attention, output proj, FFN, etc.), each with ~µs of launch overhead. At batch=1 a decode step may be only a few milliseconds, so dozens of µs-scale launches are a meaningful fraction — and worse, the CPU may not issue them fast enough to keep the GPU fed (the scheduler-overhead problem, [§03](../03_batching_and_scheduling/01_iteration_level_scheduling.md)). CUDA graphs **capture** the whole sequence of kernels once and **replay** it as a single launch, eliminating per-kernel CPU overhead and GPU idle gaps. This is why frameworks capture the (fixed-shape) decode step in a CUDA graph and replay it every iteration.

The art is knowing where fusion and graphs pay off. Fusion helps **memory-bound, small** ops (decode glue, dequant+GEMM); large compute-bound prefill GEMMs are already efficient and gain little from fusing surrounding elementwise ops. CUDA graphs require **stable shapes** (they capture a specific execution), so they fit decode (fixed batch/shape per captured graph) better than dynamic prefill — frameworks capture graphs for a set of batch sizes and pad to them. This file covers the fusion patterns, CUDA graphs, and their limits.

***

## Core Concepts & Mechanics

### Why fusion helps memory-bound ops
📐 N separate elementwise ops on a tensor of B bytes: HBM traffic ≈ `2NB` (read+write each), launches = N. Fused: HBM ≈ `2B`, launches = 1. For memory-bound ops this is ~N× less HBM traffic and N× fewer launches. For compute-bound ops, the matmul dominates and fusion of surrounding glue helps little.

### Common inference fusions
- **RMSNorm + QKV projection**: norm output feeds projection without HBM round-trip.
- **Attention + residual add** / **bias + activation**.
- **SwiGLU FFN**: `down(SiLU(gate) ⊙ up)` fused (or gate/up GEMMs + fused activation).
- **Dequant + GEMM** (W4A16): dequantize weights in-kernel, never materialize FP16 weights ([§05](../05_quantization/02_weight_only_quantization.md)).
- **FlashAttention**: the whole attention block fused ([§01](01_flash_attention_deep_dive.md)).

### CUDA graphs
- **Capture**: record the stream of kernel launches (for a fixed shape) into a graph.
- **Replay**: launch the entire graph with one call — no per-kernel CPU launch cost, no inter-kernel gaps.
- Requires **static shapes**; frameworks capture graphs for several batch sizes and pad. Re-capture on shape change.

### Where each applies
| Technique | Best for | Limited for |
|---|---|---|
| Fusion | memory-bound small ops, decode glue, dequant | large compute-bound GEMM |
| CUDA graphs | fixed-shape decode | dynamic prefill, variable shapes |

***

## Key Challenges
1. **Shape stability for graphs.** CUDA graphs need fixed shapes; dynamic batch/sequence requires capturing multiple graphs and padding, adding memory/complexity.
2. **Fusion engineering cost.** Hand-fusing kernels is error-prone; correctness across precisions/edge cases is hard. Compilers (Triton, TVM, torch.compile) help but aren't perfect.
3. **Diminishing returns on compute-bound ops.** Fusing glue around a big GEMM barely helps; effort should target memory-bound paths.
4. **Capture/replay correctness.** Stateful ops (RNG, dynamic control flow) complicate graph capture; bugs cause subtle wrong outputs.

***

## Solutions & Current Best Practices
- **Use framework-fused kernels** (xformers, FlashAttention, TRT-LLM plugins, Triton kernels) rather than hand-writing.
- **Capture decode in CUDA graphs** for a set of batch sizes; pad to the nearest captured size ([§03](../03_batching_and_scheduling/01_iteration_level_scheduling.md)).
- **Fuse dequant+GEMM** for quantized weights (Marlin/AWQ) ([§05](../05_quantization/02_weight_only_quantization.md)).
- **torch.compile / Triton** for automatic fusion of elementwise chains.

***

## Implementation Notes
- vLLM/SGLang/TRT-LLM enable CUDA graphs for decode by default for common batch sizes; verify your batch sizes are captured (else fallback to eager = slower).
- Ensure RNG/sampling is graph-compatible or handled outside the captured region.
- Profile to confirm HBM round-trips and launch gaps actually dropped after fusion/graphs ([§05](05_kernel_profiling_and_benchmarking.md)).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Batch size not captured → eager fallback** — an uncaptured batch size silently runs without CUDA graphs, much slower; capture a range and pad.
- **Fused everything, prefill unchanged** — fusing glue around compute-bound GEMMs gives little; target memory-bound paths.
- **CUDA graph captured stale data/RNG** — sampling RNG inside the captured graph broke per-request randomness; keep RNG/sampling correct under replay.
- **Dynamic shapes thrashed graph re-capture** — frequent shape changes re-capture graphs, adding overhead; pad to a fixed set.

***

## Performance Numbers & Benchmarks
| Technique | Decode effect |
|---|---|
| Elementwise fusion | fewer HBM round-trips for glue ops |
| Dequant+GEMM fusion | realizes W4A16 bandwidth win |
| CUDA graphs | removes µs×kernels/step launch overhead, fills GPU gaps |
| FlashAttention (fused) | major attention speedup, O(N) memory |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why does operator fusion help decode but not prefill much?"* — Expected: memory-bound glue (HBM round-trips) vs compute-bound GEMM.
- *"What do CUDA graphs do and when are they applicable?"* — Expected: capture/replay to kill launch overhead; need static shapes (decode).
- *"How would you fuse a dequant+GEMM kernel and why?"* — Expected: dequant in-register, never materialize FP16 weights; realizes bandwidth win.
- *"What breaks CUDA graph capture?"* — Expected: dynamic shapes, RNG/control flow; pad and handle RNG.

***

## Open Problems & Active Research (2025–2026)
- **Mega-kernels** fusing entire transformer layers/decode steps to one launch.
- **Better compilers** (Triton, Mojo, torch.compile) auto-generating fused, graph-friendly kernels.
- **Dynamic-shape CUDA graphs** to reduce padding overhead.

***

## References
- Dao, T., et al. (2022). "FlashAttention." *NeurIPS 2022*. arXiv:2205.14135.
- NVIDIA. "CUDA Graphs" programming guide.
- Chen, T., et al. (2018). "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." *OSDI 2018*. arXiv:1802.04799.
