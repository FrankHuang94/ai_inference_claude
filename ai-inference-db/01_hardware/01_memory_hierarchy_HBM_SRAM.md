# Memory Hierarchy: HBM, SRAM, and the Bandwidth Wall

> **Section:** 01_hardware
> **Last Updated:** June 2026
> **Related Files:** [00_gpu_architecture_for_inference.md](00_gpu_architecture_for_inference.md), [../00_fundamentals/03_memory_bandwidth_bound_compute.md](../00_fundamentals/03_memory_bandwidth_bound_compute.md), [../06_kernel_optimization/01_flash_attention_deep_dive.md](../06_kernel_optimization/01_flash_attention_deep_dive.md)
> **Must-Read Papers:** Dao et al. (2022, NeurIPS) "FlashAttention"; Williams et al. (2009, CACM) "Roofline"; NVIDIA Hopper whitepaper
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- The hierarchy: **registers → L1/shared (SRAM) → L2 → HBM → off-chip (NVLink/PCIe)**, each ~10× slower and ~10× larger than the level above.
- **HBM is the inference bottleneck**: weights and KV live there. H100 HBM3 ≈ 3.35 TB/s; H200 HBM3e ≈ 4.8 TB/s; B200 ≈ 8 TB/s.
- **SRAM (shared memory, ~228 KB/SM on H100)** is tiny but enormously fast; FlashAttention's entire trick is keeping the attention working set in SRAM to avoid HBM round-trips.
- The **bandwidth wall**: FLOPs have grown faster than HBM bandwidth across generations, so inference is increasingly memory-bound.
- **Data movement, not arithmetic, is the dominant energy and time cost** of inference.

***

## Overview
GPU memory is a hierarchy that trades capacity for speed. At the top, each thread has private **registers** (fastest, ~256 KB register file per SM). Each SM has **L1/shared memory** (SRAM, ~228 KB configurable on H100) — software-managed scratchpad with terabytes/sec of aggregate bandwidth. A chip-wide **L2 cache** (50 MB on H100) backs all SMs. Finally, **HBM** (High Bandwidth Memory) is the large off-chip DRAM (80–192 GB) where models and KV caches live, delivering single-digit TB/s. Beyond the package, **NVLink** (~900 GB/s/GPU on H100) connects GPUs and **PCIe** (~64 GB/s) connects to the host — both far slower than HBM.

For LLM inference the hierarchy explains nearly everything. Weights (GBs) only fit in HBM, so every decode step streams them across the HBM interface — the binding constraint. The KV cache also lives in HBM and grows with context, adding traffic. The art of fast kernels is **maximizing reuse at the fastest level that fits the working set**: GEMM tiles and attention blocks are staged in SRAM/registers so each HBM byte is reused many times before eviction. FlashAttention is the canonical example — it never materializes the N×N attention matrix in HBM, keeping tiles in SRAM and computing softmax online ([§06](../06_kernel_optimization/01_flash_attention_deep_dive.md)).

The macro trend is the **bandwidth wall**: across A100→H100→B200, compute (FLOPs) has roughly doubled per generation while HBM bandwidth grew more slowly (2.0→3.35→~8 TB/s, with H200 a bandwidth-focused refresh). Since decode tracks bandwidth, this widening gap makes memory the perennial inference bottleneck and motivates SRAM-centric accelerators (Groq) and aggressive byte-reduction (quantization, KV compression).

***

## Core Concepts & Mechanics

### The levels (H100 reference)
| Level | Size | Aggregate BW (approx) | Latency | Managed by |
|---|---|---|---|---|
| Registers | 256 KB/SM | ~tens of PB/s | ~1 cycle | compiler |
| Shared/L1 (SRAM) | up to 228 KB/SM | ~tens of TB/s | ~20–30 cycles | software (kernel) |
| L2 | 50 MB | ~several TB/s | ~200 cycles | hardware |
| HBM3 | 80 GB | 3.35 TB/s | ~400–600 cycles | hardware |
| NVLink 4 | — | 900 GB/s | µs-scale | explicit |
| PCIe 5 | — | ~64 GB/s | µs-scale | explicit |

### Why SRAM matters: the FlashAttention argument
Standard attention writes the `N×N` scores to HBM, then reads them back for softmax and the `·V` product: `O(N²)` HBM traffic. FlashAttention tiles Q,K,V into blocks that fit in **SRAM**, computes partial softmax statistics online, and only writes the `N×d` output to HBM: `O(N·d + N²/M)` effective HBM accesses (M = SRAM tile size). Since HBM is ~100× slower than SRAM, eliminating the N×N HBM round-trip is the entire speedup — even though FLOPs are essentially unchanged. 📐 IO complexity drops from `Θ(N²)` to `Θ(N²·d / M)` HBM accesses.

### The HBM stack
HBM is stacked DRAM dies connected by through-silicon vias on a silicon interposer next to the GPU die, giving a very wide bus (e.g., 5120-bit) at moderate clock — hence high bandwidth at large capacity. Generations: **HBM2e** (A100, ~2 TB/s), **HBM3** (H100, ~3.35 TB/s), **HBM3e** (H200 ~4.8, B200 ~8 TB/s), with **HBM4** on the roadmap.

### Energy
📐 Data movement dominates energy: moving a word from HBM costs ~100–1000× the energy of a FLOP. This is why reducing bytes moved (quantization, fusion, KV compression) improves both speed *and* energy/cost per token — central to TCO ([§01](06_tco_and_cost_modeling.md)).

***

## Key Challenges
1. **SRAM is tiny.** ~228 KB/SM cannot hold weights or long KV; only small tiles fit, forcing careful blocking and limiting how much HBM traffic can be eliminated.
2. **The bandwidth wall widens.** Each generation's FLOP growth outpaces bandwidth growth, so the same model becomes *relatively* more memory-bound over time.
3. **L2 is shared and contended.** 50 MB helps reuse activations/KV but is thrashed under many concurrent requests; benefits are workload-dependent and hard to predict.
4. **Off-chip cliffs.** Spilling to NVLink (TP, KV transfer) or PCIe (offload) is 4–50× slower than HBM and must be overlapped or it dominates latency.

***

## Solutions & Current Best Practices
- **SRAM-resident kernels** (FlashAttention, fused GEMM epilogues) to minimize HBM traffic ([§06](../06_kernel_optimization/01_flash_attention_deep_dive.md)).
- **Quantization & KV compression** to cut HBM bytes/token ([§05](../05_quantization/00_quantization_fundamentals.md), [§02](../02_kv_cache/03_kv_cache_compression.md)).
- **Bandwidth-first hardware selection** for decode-heavy workloads (H200 over H100) ([§01](05_hardware_selection_decision_framework.md)).
- **Overlap off-chip transfers** with compute (async copies, double-buffering) for offload and KV migration ([§02](../02_kv_cache/04_kv_cache_offloading.md)).

***

## Implementation Notes
- Configure the **shared-memory/L1 split** per kernel (Hopper allows up to 228 KB shared); FlashAttention-3 exploits large shared memory plus TMA async copies.
- Use **`cp.async`/TMA** (Hopper Tensor Memory Accelerator) to overlap HBM→SRAM copies with compute.
- Size **paged-KV blocks** so reads are coalesced and L2-friendly; too-small blocks waste bandwidth.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **A kernel that's "compute-bound" on A100 becomes memory-bound on H100** because compute grew more than bandwidth — re-profile per generation; the bound can move.
- **L2 cache effects make microbenchmarks lie.** A small-model benchmark may fit weights partly in L2 and overstate achievable decode rate vs a large model that always hits HBM.
- **Shared-memory bank conflicts** silently halve SRAM throughput in hand-written attention/GEMM kernels — a classic Nsight finding.
- **Assuming peak HBM bandwidth.** Achieved is ~70–90%; fragmented paged KV or strided access drops it further.

***

## Performance Numbers & Benchmarks
| Memory | Capacity | Bandwidth | Notes |
|---|---|---|---|
| H100 SRAM/SM | 228 KB | ~tens TB/s | FlashAttention tiles |
| H100 L2 | 50 MB | ~several TB/s | activation/KV reuse |
| A100 HBM2e | 80 GB | 2.0 TB/s | |
| H100 HBM3 | 80 GB | 3.35 TB/s | |
| H200 HBM3e | 141 GB | 4.8 TB/s | best decode/$ among Hopper |
| B200 HBM3e | 192 GB | ~8 TB/s | Blackwell |
| MI300X HBM3 | 192 GB | ~5.3 TB/s | AMD |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Walk up the memory hierarchy with bandwidths; where does inference bottleneck?"* — Expected: HBM for weights/KV; SRAM tiny but fast.
- *"Explain FlashAttention purely in memory-hierarchy terms."* — Expected: keep tiles in SRAM, avoid O(N²) HBM, online softmax.
- *"What is the bandwidth wall and why does it make inference harder over time?"* — Expected: FLOPs outgrow bandwidth; decode is bandwidth-bound.
- *"Why does reducing bytes moved help energy as well as latency?"* — Expected: data movement >> FLOP energy.

***

## Open Problems & Active Research (2025–2026)
- **HBM4 and beyond** vs on-package SRAM/processing-in-memory to address the bandwidth wall.
- **Larger software-managed SRAM** and better async-copy engines to extend the FlashAttention strategy to more of the model.
- **Compression in the memory path** (compressed KV transferred/stored, decompressed near compute).

***

## References
- Dao, T., et al. (2022). "FlashAttention." *NeurIPS 2022*. arXiv:2205.14135.
- Williams, S., et al. (2009). "Roofline." *CACM* 52(4).
- NVIDIA (2022/2024). Hopper & Blackwell architecture whitepapers.
- Jouppi, N., et al. (2021). "Ten Lessons From Three Generations of TPUs." *ISCA 2021*.
