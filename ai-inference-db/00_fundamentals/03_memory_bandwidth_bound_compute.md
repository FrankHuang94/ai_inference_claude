# Memory-Bandwidth-Bound vs Compute-Bound Inference

> **Section:** 00_fundamentals
> **Last Updated:** June 2026
> **Related Files:** [04_roofline_model_for_llm.md](04_roofline_model_for_llm.md), [02_prefill_vs_decode_phases.md](02_prefill_vs_decode_phases.md), [../01_hardware/01_memory_hierarchy_HBM_SRAM.md](../01_hardware/01_memory_hierarchy_HBM_SRAM.md)
> **Must-Read Papers:** Williams et al. (2009, CACM) "Roofline"; Dao et al. (2022, NeurIPS) "FlashAttention"; Pope et al. (2022, MLSys) "Efficiently Scaling Transformer Inference"
> **Estimated Study Time:** 35 minutes

***

## TL;DR
- A kernel is **memory-bound** when it spends more time moving bytes than doing math: `AI < AI_ridge = peak_FLOPs / peak_bandwidth`.
- The GPU memory hierarchy spans ~5 orders of magnitude in bandwidth: **registers/SRAM (TB/s–PB/s) → L2 → HBM (~2–4.8 TB/s) → PCIe/NVLink (off-chip)**. Decode is bottlenecked by **HBM**.
- HBM bandwidths to memorize: **A100 ~2.0 TB/s (HBM2e), H100 ~3.35 TB/s (HBM3), H200 ~4.8 TB/s (HBM3e), B200 ~8 TB/s (HBM3e)**.
- The **batch-size breakeven point** (memory→compute transition) is roughly `batch ≈ AI_ridge` for the dense matmuls — ~hundreds on H100 FP16, lower in FP8.
- Weight loading dominates decode at small batch; this is why **quantization and batching** are the two primary decode levers.

***

## Overview
"Memory-bound vs compute-bound" is the most useful binary classification in performance engineering. Every kernel has a quantity of arithmetic to perform and a quantity of bytes to move; whichever takes longer is the bound. The crossover is the **roofline ridge point**, `AI_ridge = peak_compute / peak_bandwidth`, expressed in FLOPs/byte. Operations with arithmetic intensity above the ridge are limited by the compute units (Tensor Cores); below it, by memory bandwidth (almost always HBM for LLM inference).

For LLM inference the practical consequence is stark: **decode at the batch sizes that fit in memory is memory-bandwidth-bound**, because each step reads the entire weight matrix to produce very few output elements. The compute units are idle waiting on HBM. This is not a tuning failure — it is the fundamental arithmetic of streaming a multi-gigabyte weight set to compute a handful of tokens. Prefill, by contrast, reuses each weight across hundreds of prompt positions and is compute-bound.

Understanding *where in the hierarchy* the bottleneck sits also explains why FlashAttention is fast (it keeps the attention working set in SRAM, avoiding O(N²) HBM traffic — see [§06](../06_kernel_optimization/01_flash_attention_deep_dive.md)) and why off-chip transfers (PCIe, cross-node) are catastrophic for latency. The hierarchy and its bandwidths are the physical substrate every later optimization manipulates.

***

## Core Concepts & Mechanics

### The memory hierarchy and bandwidths
| Level | H100 capacity | Bandwidth (approx) | Role in inference |
|---|---|---|---|
| Registers | 256 KB/SM | ~tens of PB/s aggregate | per-thread operands |
| L1 / Shared (SRAM) | 256 KB/SM | ~tens of TB/s aggregate | FlashAttention tiles, GEMM tiles |
| L2 cache | 50 MB | ~several TB/s | weight/activation reuse |
| HBM (global) | 80 GB | **3.35 TB/s** | weights + KV cache — the decode bottleneck |
| NVLink 4 | — | 900 GB/s/GPU | TP all-reduce, KV transfer |
| PCIe 5 | — | ~64 GB/s | host↔device, offload |

Each step down is ~10× slower. Decode performance is set by the HBM row because weights and KV live in HBM.

### Arithmetic intensity and the ridge
📐
```
AI = useful_FLOPs / bytes_moved
AI_ridge = peak_compute / peak_bandwidth
memory-bound  ⇔  AI < AI_ridge
compute-bound ⇔  AI > AI_ridge
```
Ridge points (compute/bandwidth):
- A100 FP16: 312 TFLOP/s ÷ 2.0 TB/s ≈ **156 FLOPs/byte**
- H100 FP16: 989 TFLOP/s ÷ 3.35 TB/s ≈ **295 FLOPs/byte**
- H100 FP8: 1979 TFLOP/s ÷ 3.35 TB/s ≈ **591 FLOPs/byte**

Higher ridge means you need *more* arithmetic intensity to be compute-bound — FP8 doubles compute but not bandwidth, so the ridge moves right and more workloads become memory-bound. This is a subtle but important point.

### Why decode is weight-load-bound
A linear layer `y = Wx` with `W ∈ R^{m×k}` reads `m·k` weight elements and does `2·m·k·B` FLOPs for batch B. AI = `2·m·k·B / (m·k·bytes) = 2B/bytes`. In FP16 (2 bytes): **AI = B**. So a single dense matmul has arithmetic intensity equal to the batch size. To reach the H100 FP16 ridge (~295) you need batch ≈ 295 *per concurrent matmul* — far above what KV memory usually permits for large models. Hence decode is memory-bound in practice.

### The batch-size breakeven point
📐 The memory→compute transition occurs when `AI(B) ≈ AI_ridge`, i.e. `B ≈ AI_ridge`. So:
- H100 FP16: breakeven batch ≈ ~150–300 (depends on KV/attention contribution).
- Smaller models reach it at smaller batches (less KV per token); larger models often cannot reach it before exhausting HBM. This is why **70B+ serving is essentially always memory-bound** in production and why MoE (few active params) and quantization help so much.

***

## Key Challenges
1. **You cannot batch your way out for large models.** KV-cache memory exhausts HBM before batch reaches the compute ridge, so big dense models stay memory-bound regardless.
2. **FP8/FP4 move the ridge rightward.** Doubling compute without doubling bandwidth means more kernels become memory-bound, so low-precision compute gains are capped by bandwidth unless you also cut bytes moved.
3. **KV reads scale with sequence length.** As context grows, attention's HBM traffic grows, adding a second memory-bound term on top of weight loading and slowing long-context decode.
4. **Off-chip transfers wreck latency.** Anything that spills weights/KV to PCIe or another node pays ~50× the HBM time; offloading must be carefully overlapped (see [§02](../02_kv_cache/04_kv_cache_offloading.md)).

***

## Solutions & Current Best Practices
- **Quantization** to cut bytes-per-weight (INT4/FP8) — directly attacks the memory bound, near-linear decode speedup ([§05](../05_quantization/00_quantization_fundamentals.md)).
- **Continuous batching** to raise effective AI toward the ridge ([§03](../03_batching_and_scheduling/00_static_vs_continuous_batching.md)).
- **FlashAttention** to keep attention working set in SRAM and avoid O(N²) HBM traffic ([§06](../06_kernel_optimization/01_flash_attention_deep_dive.md)).
- **MQA/GQA and KV quantization** to shrink KV HBM traffic ([§02](../02_kv_cache/00_kv_cache_fundamentals.md)).
- **MoE** to reduce active-parameter bytes per token ([§04](../04_parallelism/04_expert_parallelism_MoE.md)).

***

## Implementation Notes
- In **Nsight Compute**, classify a kernel via the "Memory Throughput" vs "Compute (SM) Throughput" — whichever is near 100% is the bound. For decode kernels expect memory ~high, SM ~low.
- The effective HBM bandwidth you achieve is ~70–90% of peak; use the *achieved* figure when modeling decode throughput, not the datasheet peak.
- Watch L2 reuse: with a 50 MB L2 on H100, parts of activations/KV can be served from L2, slightly beating naive HBM-only models — but weights (GBs) never fit in L2.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **FP8 didn't double decode throughput.** Expected — FP8 doubles *compute* and the ridge moves right; decode is bandwidth-bound, so the win comes only from the halved *weight bytes*, not the FLOPs. The matmul speedup mostly helps prefill.
- **Long context slowed decode even at fixed batch.** KV-read HBM traffic grew with sequence length, adding a memory-bound term independent of weights. Mitigate with GQA/KV quant.
- **Achieved bandwidth << peak.** Poor access patterns (non-coalesced, small transfers, fragmented paged KV) leave bandwidth on the table; the bound is real but you may be 50% below the *achievable* peak — a tuning opportunity, not a hardware limit.

***

## Performance Numbers & Benchmarks
| Hardware | Peak FP16 | HBM BW | FP16 ridge | Notes |
|---|---|---|---|---|
| A100 80GB | 312 TFLOP/s | 2.0 TB/s | ~156 | HBM2e |
| H100 SXM5 | 989 TFLOP/s | 3.35 TB/s | ~295 | HBM3; FP8 ridge ~591 |
| H200 | 989 TFLOP/s | 4.8 TB/s | ~206 | same compute, more BW → better decode |
| B200 | ~2.2–2.5 PFLOP/s FP8 (sparse higher) | ~8 TB/s | high | HBM3e; FP4 support |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Compute the ridge point for H100 in FP16 and FP8. What does the shift imply?"* — Expected: ~295 and ~591; FP8 makes more workloads memory-bound.
- *"At what batch size does a dense matmul become compute-bound, and why can't a 70B model usually get there?"* — Expected: B≈AI_ridge; KV memory exhausts HBM first.
- *"Why is H200 better than H100 for decode despite identical FLOPs?"* — Expected: decode tracks bandwidth (4.8 vs 3.35 TB/s).
- *"Where in the memory hierarchy does FlashAttention win?"* — Expected: keeps tiles in SRAM, avoids O(N²) HBM round-trips.

***

## Open Problems & Active Research (2025–2026)
- **Bandwidth scaling vs SRAM-centric designs** — whether HBM4 or wafer-scale/SRAM accelerators (Groq, Cerebras) better address the decode bound ([§01](../01_hardware/04_alternative_accelerators.md)).
- **Reducing KV HBM traffic at long context** — sparse/linear attention and aggressive KV compression remain accuracy-limited ([§10](../10_long_context/03_sparse_and_linear_attention.md)).
- **Co-designing precision and bandwidth** so FP4/FP8 gains aren't capped by the rightward ridge shift.

***

## References
- Williams, S., Waterman, A., Patterson, D. (2009). "Roofline." *CACM* 52(4).
- Dao, T., et al. (2022). "FlashAttention." *NeurIPS 2022*. arXiv:2205.14135.
- Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference." *MLSys 2023*. arXiv:2211.05102.
- NVIDIA (2022–2024). H100/H200/Blackwell architecture whitepapers.
