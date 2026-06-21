# KV Cache Offloading and Tiered Memory

> **Section:** 02_kv_cache
> **Last Updated:** June 2026
> **Related Files:** [01_paged_attention_vllm.md](01_paged_attention_vllm.md), [../09_distributed_inference/03_kv_cache_migration_and_transfer.md](../09_distributed_inference/03_kv_cache_migration_and_transfer.md), [../01_hardware/01_memory_hierarchy_HBM_SRAM.md](../01_hardware/01_memory_hierarchy_HBM_SRAM.md)
> **Must-Read Papers:** Sheng et al. (2023, ICML) "FlexGen"; Kwon et al. (2023, SOSP) "vLLM" (swap); Lin et al. (2024) "LMCache/CacheGen"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- Offloading moves KV (and sometimes weights) to **CPU RAM or NVMe** to serve more/longer requests than HBM alone allows — trading **bandwidth for capacity**.
- The bandwidth cliff: HBM ~3.35 TB/s → PCIe 5 ~64 GB/s (CPU) → NVMe ~7 GB/s. Offloaded KV must be **prefetched/overlapped** with compute or it dominates latency.
- **FlexGen** (Sheng et al. 2023) pioneered offloading for high-throughput *batch* inference on limited GPUs; not for low-latency serving.
- **LMCache/CacheGen** treat KV as a tiered, transferable cache (compress + store + reload), enabling cross-request and cross-node reuse.
- Best for **throughput-oriented/offline** workloads and **multi-turn reuse**; usually a poor fit for tight-ITL real-time serving.

***

## Overview
HBM capacity is the hard ceiling on how much KV (and how many concurrent requests) a GPU can hold. Offloading relaxes that ceiling by treating CPU RAM and NVMe as additional, slower tiers of a memory hierarchy for KV: blocks not immediately needed are evicted to CPU/NVMe and fetched back when a request becomes active. This buys capacity — far more total KV than HBM — at the cost of the steep bandwidth drop between tiers. The engineering problem is hiding that latency: prefetching KV for a request *before* it's scheduled, overlapping transfers with ongoing compute, and compressing KV to reduce transfer volume.

The canonical research system is **FlexGen** (Sheng et al. 2023, ICML), which showed that by aggressively offloading weights, KV, and activations across GPU/CPU/disk and scheduling transfers cleverly, you can run very large models for **high-throughput, latency-tolerant batch** inference on a single commodity GPU. The key insight is that for offline/batch workloads you can sacrifice latency for throughput, amortizing slow transfers across large batches. This is the opposite regime from real-time chat, where the PCIe/NVMe cliff makes offloaded KV reads show up directly in ITL — so naive offloading is usually unacceptable for low-latency serving.

The modern framing is **tiered KV as a cache**: systems like **LMCache/CacheGen** store KV blocks (often compressed/quantized) in CPU/NVMe/remote stores and reload them to serve cache hits — e.g., reusing a long document's KV across many users, or restoring a paused conversation's KV without re-prefill. Combined with disaggregation ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)), KV becomes a first-class, movable, storable object, and offloading is one tier of its lifecycle.

***

## Core Concepts & Mechanics

### The tier bandwidth cliff
| Tier | Capacity | Bandwidth | Use |
|---|---|---|---|
| HBM | 80–192 GB | ~3.35–8 TB/s | active KV |
| CPU RAM (over PCIe 5) | TBs | ~64 GB/s | warm KV / swap |
| NVMe SSD | TBs–PBs | ~3–7 GB/s/drive | cold KV / persistence |
| Remote/object store | ∞ | network-limited | cross-node KV cache |

📐 Reading a 1.25 GB KV (70B@4k) from HBM ≈ 0.4 ms; from CPU over PCIe ≈ 20 ms; from NVMe ≈ 180 ms. The cliff is why overlap/prefetch is mandatory.

### Overlap and prefetch
Transfers must be **double-buffered**: while the GPU computes layer L for the current batch, asynchronously fetch the KV needed for layer L+1 or for the next request. If transfer time < compute time per overlap unit, latency is hidden; otherwise it's exposed. FlexGen formalizes this as a scheduling/overlap problem and computes near-optimal block movement.

### Swap vs recompute (vLLM)
Under HBM pressure, vLLM can **swap** preempted requests' KV to CPU (then back) or **recompute** (drop KV, re-prefill later). 📐 Swap cost ≈ `KV_bytes / PCIe_BW × 2` (out+in); recompute cost ≈ prefill FLOPs/compute. Recompute wins for short sequences; swap wins for long ones where re-prefill is expensive. See [§01](01_paged_attention_vllm.md).

### KV as a tiered cache (LMCache/CacheGen)
Store KV (compressed/quantized) keyed by token sequence in CPU/NVMe/remote; on a matching request, reload instead of re-prefilling. CacheGen additionally **compresses KV for cheap storage/transfer** and decompresses on load. Enables cross-request and cross-node prefix reuse beyond a single GPU's HBM ([§02](02_prefix_caching_and_radix_attention.md)).

***

## Key Challenges
1. **Latency exposure.** If transfers aren't fully overlapped, offloaded KV reads land on the critical path and wreck ITL — fatal for real-time serving.
2. **Bandwidth contention.** PCIe is shared with other host I/O (weight loading, logging); offload traffic can starve or be starved.
3. **Prefetch prediction.** You must know *which* KV to fetch *before* it's needed; mispredicted prefetch wastes bandwidth or stalls.
4. **Consistency with paging/compression.** Offloaded blocks must integrate with the paged allocator and any quantization/compression, complicating the block lifecycle and correctness.

***

## Solutions & Current Best Practices
- **Use offloading for throughput/offline batch** (FlexGen-style) and **multi-turn/cross-request KV reuse** (LMCache), not tight-ITL real-time chat.
- **Always overlap** transfers with compute (async copies, double buffering); verify transfer < compute per overlap unit.
- **Compress/quantize KV before offload** to cut transfer volume ([§05](../05_quantization/05_kv_cache_quantization.md)).
- **Prefer recompute over swap** for short preempted sequences; swap long ones.

***

## Implementation Notes
- vLLM `swap_space` sets CPU swap size; LMCache integrates as a KV layer for vLLM/SGLang.
- Pin host memory for fast DMA; use GPUDirect Storage for NVMe→GPU where available to bypass host bounce.
- Monitor PCIe/NVMe utilization and **transfer-stall time**; rising stalls mean overlap is failing.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Offloading "worked" in a throughput benchmark, destroyed ITL in prod** — batch amortization hid transfer cost offline; real-time exposes it. Different regime.
- **Swap-to-CPU thrash under load** — repeated swap-out/in over PCIe stalls the whole engine; cap concurrency or add HBM.
- **NVMe offload for low latency is a trap** — ~180 ms reloads can't hide behind a ~5 ms decode step; NVMe is for cold/persistent KV only.
- **Prefetch mispredictions waste scarce PCIe** — fetching wrong KV both wastes bandwidth and stalls the right request.

***

## Performance Numbers & Benchmarks
| Scenario | Tier | Latency to load 1.25 GB KV |
|---|---|---|
| Active | HBM | ~0.4 ms |
| Swap/warm | CPU/PCIe5 | ~20 ms |
| Cold | NVMe | ~180 ms |
| FlexGen single-GPU large-model batch | GPU+CPU+disk | high throughput, high latency |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"When is KV offloading appropriate and when is it a mistake?"* — Expected: throughput/offline & cross-request reuse yes; tight-ITL real-time no.
- *"Compare swap vs recompute under memory pressure."* — Expected: transfer cost vs prefill cost; short→recompute, long→swap.
- *"How do you hide the PCIe/NVMe bandwidth cliff?"* — Expected: prefetch + double-buffer overlap; compress before transfer.
- *"What does treating KV as a tiered cache enable?"* — Expected: cross-request/cross-node reuse, conversation restore without re-prefill.

***

## Open Problems & Active Research (2025–2026)
- **Latency-hiding offload for real-time serving** of very long contexts.
- **Global KV cache tiers** (HBM↔CPU↔NVMe↔remote) with smart prefetch and compression at scale ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).
- **GPUDirect Storage / CXL memory** to flatten the tier cliff.

***

## References
- Sheng, Y., et al. (2023). "FlexGen: High-Throughput Generative Inference of LLMs with a Single GPU." *ICML 2023*. arXiv:2303.06865.
- Kwon, W., et al. (2023). "vLLM/PagedAttention." *SOSP 2023*. arXiv:2309.06180.
- Liu, Y., et al. (2024). "CacheGen: KV Cache Compression and Streaming for Fast LLM Serving." *SIGCOMM 2024*. arXiv:2310.07240.
