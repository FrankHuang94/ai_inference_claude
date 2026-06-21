# FlashAttention Deep Dive (FA-1/2/3 and MLA)

> **Section:** 06_kernel_optimization
> **Last Updated:** June 2026
> **Related Files:** [00_cuda_kernel_basics_for_inference.md](00_cuda_kernel_basics_for_inference.md), [../01_hardware/01_memory_hierarchy_HBM_SRAM.md](../01_hardware/01_memory_hierarchy_HBM_SRAM.md), [../10_long_context/00_long_context_challenges.md](../10_long_context/00_long_context_challenges.md)
> **Must-Read Papers:** Dao et al. (2022, NeurIPS) "FlashAttention"; Dao (2023) "FlashAttention-2"; Shah et al. (2024) "FlashAttention-3"; DeepSeek-AI (2024) "MLA"
> **Estimated Study Time:** 40 minutes

***

## TL;DR
- Standard attention materializes the **N×N** score matrix in HBM → O(N²) memory and O(N²) HBM traffic. FlashAttention **never materializes it**: it tiles Q,K,V into SRAM and computes softmax **online**.
- 📐 IO complexity drops to **O(N²d²/M)** HBM accesses (M = SRAM size) — a large constant-factor and asymptotic reduction; memory drops to O(N).
- **FA-2** (Dao 2023): better work partitioning (parallelize over sequence, fewer non-matmul FLOPs) → ~2× FA-1.
- **FA-3** (Shah et al. 2024): Hopper-specific — async **wgmma**/**TMA**, FP8, ping-pong/warp-specialized pipelines → ~1.5–2× FA-2 on H100.
- **MLA** (DeepSeek-V2): low-rank latent KV changes attention compute, slashing KV memory; needs specialized kernels.

***

## Overview
FlashAttention is, alongside PagedAttention, one of the two most important inference-systems papers. The problem it solves: standard attention computes `S = QKᵀ` (an N×N matrix for sequence length N), applies softmax, then `O = softmax(S)V`. Materializing S in HBM costs **O(N²) memory** and, worse, **O(N²) HBM reads/writes** (write S, read it back for softmax, read again for ·V). Since attention is bandwidth-bound and HBM is the bottleneck ([§00](../00_fundamentals/03_memory_bandwidth_bound_compute.md)), this HBM traffic — not the FLOPs — dominates runtime, and the O(N²) memory makes long sequences infeasible.

FlashAttention (Dao et al. 2022) eliminates the N×N HBM round-trip via **tiling + online softmax**. It loads blocks of Q, K, V into **SRAM**, computes the partial scores for that block, and updates a running softmax (maintaining running max `m` and sum `ℓ` to rescale previous partial outputs as new blocks arrive) — so the full softmax is computed incrementally without ever storing S in HBM. Only the O(N·d) output is written to HBM. The result is exact attention (no approximation) with **O(N) memory** and dramatically fewer HBM accesses, giving large speedups and enabling long context. The online-softmax recurrence is the mathematical heart of the method and a classic interview derivation.

The successors optimize the implementation. **FA-2** improves parallelism (parallelize across the sequence/query dimension, not just batch/heads) and reduces non-matmul FLOPs (the expensive rescaling), roughly doubling FA-1. **FA-3** targets Hopper specifically: it uses asynchronous **wgmma** Tensor-Core instructions and **TMA** (Tensor Memory Accelerator) async copies, overlaps GEMM and softmax via warp specialization / ping-pong scheduling, and supports FP8 — reaching a high fraction of H100 peak. Separately, **MLA** (Multi-head Latent Attention, DeepSeek-V2) is an *architectural* change that compresses K,V into a shared low-rank latent (huge KV-memory savings, [§02](../02_kv_cache/00_kv_cache_fundamentals.md)) and changes the attention compute pattern, requiring its own optimized kernels. This file covers the algorithm, the version differences, and MLA.

***

## Core Concepts & Mechanics

### Standard attention cost
📐 `S=QKᵀ` (N×N×d FLOPs, N² memory), softmax (N²), `O=PV` (N²×d). HBM traffic ≈ O(N²) (write/read S). Memory O(N²). Bandwidth-bound.

### Online softmax (the key trick)
Process K,V in blocks. Maintain running max `m_i` and normalizer `ℓ_i` and accumulated output `O_i`. For each new block with local max `m̃`:
```
m_new = max(m_i, m̃)
ℓ_new = e^{m_i − m_new}·ℓ_i + Σ e^{s − m_new}
O_new = e^{m_i − m_new}·O_i + Σ e^{s − m_new}·v
```
This rescales prior partial results so the final result equals exact softmax-attention, computed block-by-block in SRAM. 📐 No N×N HBM materialization.

### IO complexity
📐 FlashAttention HBM accesses ≈ `Θ(N²d²/M)` (M = SRAM size in elements), vs `Θ(Nd + N²)` for standard. For large N and realistic M, this is a major reduction; memory O(N) vs O(N²).

### Version differences
| Version | Key change | Gain |
|---|---|---|
| FA-1 (2022) | tiling + online softmax, SRAM | exact, O(N) mem, fewer HBM |
| FA-2 (2023) | parallelize over seq, fewer non-matmul FLOPs, better work partition | ~2× FA-1 |
| FA-3 (2024) | Hopper async wgmma+TMA, FP8, warp-specialized ping-pong | ~1.5–2× FA-2 on H100 |

### MLA (Multi-head Latent Attention)
DeepSeek-V2: project K,V into a shared **low-rank latent** `c` cached per token (instead of full per-head K,V); reconstruct per-head on the fly. KV cache shrinks dramatically (a small latent vector/token) while preserving quality. Changes the attention math/kernel; specialized MLA kernels (in SGLang/vLLM) make it efficient.

### Decode vs prefill
FlashAttention's biggest win is **prefill / full attention** (large N²). At **decode** (query length 1), the "N×N" is "1×S", so it's a different kernel (often "FlashDecoding" / paged attention) that parallelizes over the KV sequence to keep the GPU busy for a single query.

***

## Key Challenges
1. **Decode is a different regime.** Single-query decode underutilizes the GPU; FlashDecoding splits the KV dimension across blocks/SMs to parallelize, then combines — a distinct kernel from prefill FA.
2. **Hardware-specific tuning.** FA-3's gains require Hopper async instructions; portable kernels lag peak on each new generation.
3. **Paged/non-contiguous KV.** Integrating FlashAttention with paged KV ([§02](../02_kv_cache/01_paged_attention_vllm.md)) requires gathering scattered blocks while keeping SRAM tiling and coalescing.
4. **MLA kernel complexity.** Low-rank latent attention needs new kernels; immature implementations lose the theoretical benefit.

***

## Solutions & Current Best Practices
- **Use FlashAttention-2/3** (or FlashDecoding for decode) via frameworks; FA-3 on H100, FA-2 elsewhere.
- **Paged + FlashAttention** integration (vLLM/SGLang kernels) for production.
- **MLA + specialized kernels** for KV-constrained long-context MoE (DeepSeek) ([§04](../04_parallelism/04_expert_parallelism_MoE.md)).
- **FP8 attention (FA-3)** on Hopper for further speedup.

***

## Implementation Notes
- For long-context prefill, FA-3 (Hopper) gives the best TTFT; verify the framework uses it.
- Decode uses FlashDecoding / paged-attention kernels — different code path; ensure KV-split parallelism is enabled for single-query efficiency.
- Validate numerics: online softmax must use stable max-subtraction; FP8 attention needs careful scaling.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **FA fast at prefill, decode still slow** — decode needs FlashDecoding (KV-split), not the prefill kernel; single-query FA underutilizes the GPU.
- **Paged KV + FA lost coalescing** — scattered blocks broke contiguous reads; layout/kernel must gather efficiently.
- **FA-3 gains absent on A100** — FA-3 needs Hopper async instructions; A100 falls back to FA-2.
- **MLA kernel slower than expected** — immature implementation didn't realize the low-rank savings; kernel maturity matters.

***

## Performance Numbers & Benchmarks
| Kernel | Hardware | Note |
|---|---|---|
| FA-2 | A100/H100 | ~2× FA-1, standard for prefill |
| FA-3 | H100 | ~1.5–2× FA-2, FP8, ~high % of peak |
| FlashDecoding | decode | KV-split parallelism for single query |
| MLA kernels | DeepSeek serving | large KV reduction, long context |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Explain FlashAttention's IO complexity and the online softmax."* — Expected: no N×N HBM, O(N) memory, running max/sum recurrence.
- *"Why is decode attention a different kernel than prefill?"* — Expected: single query underutilizes; FlashDecoding splits KV.
- *"What does FA-3 exploit on Hopper?"* — Expected: async wgmma/TMA, FP8, warp-specialized overlap.
- *"How does MLA change attention and KV cost?"* — Expected: low-rank latent KV; smaller cache; new kernel.

***

## Open Problems & Active Research (2025–2026)
- **FA-4 / Blackwell-specific** kernels (FP4, new async engines).
- **Unified paged + Flash + quantized-KV** attention kernels.
- **MLA generalization** beyond DeepSeek and its kernel ecosystem.
- **Sparse/linear attention kernels** at FlashAttention efficiency ([§10](../10_long_context/03_sparse_and_linear_attention.md)).

***

## References
- Dao, T., et al. (2022). "FlashAttention." *NeurIPS 2022*. arXiv:2205.14135.
- Dao, T. (2023). "FlashAttention-2." arXiv:2307.08691.
- Shah, J., et al. (2024). "FlashAttention-3." arXiv:2407.08608.
- DeepSeek-AI (2024). "DeepSeek-V2" (MLA). arXiv:2405.04434.
