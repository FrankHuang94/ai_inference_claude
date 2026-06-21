# PagedAttention and vLLM Memory Management

> **Section:** 02_kv_cache
> **Last Updated:** June 2026
> **Related Files:** [00_kv_cache_fundamentals.md](00_kv_cache_fundamentals.md), [02_prefix_caching_and_radix_attention.md](02_prefix_caching_and_radix_attention.md), [../08_serving_frameworks/01_vllm_deep_dive.md](../08_serving_frameworks/01_vllm_deep_dive.md)
> **Must-Read Papers:** Kwon et al. (2023, SOSP) "Efficient Memory Management for LLM Serving with PagedAttention"; Yu et al. (2022, OSDI) "Orca"
> **Estimated Study Time:** 40 minutes

***

## TL;DR
- **The problem:** contiguous per-request KV pre-allocation wastes **60–80%** of KV memory via internal/external fragmentation and reserved-but-unused slots.
- **The idea (OS analogy):** treat KV like **virtual memory** — logical token blocks mapped to non-contiguous physical blocks via a **block table**; fixed block size (e.g., 16 tokens).
- **Result (Kwon et al. 2023, SOSP):** near-zero KV waste → much larger batches → **2–4× throughput** over naive HF serving.
- Enables **copy-on-write** block sharing for parallel sampling/beam and **prefix sharing** (proto-RadixAttention).
- vLLM's scheduler adds **continuous batching**, **LRU eviction**, and **preemption (swap or recompute)** on top of paged KV.

***

## Overview
PagedAttention is the foundational systems paper of modern LLM serving. Before it, serving systems allocated each request a single contiguous KV buffer sized to the *maximum* possible sequence length, because attention kernels assumed contiguous K,V. With highly variable real sequence lengths, this caused massive waste: **internal fragmentation** (reserved slots never used by short sequences), **external fragmentation** (contiguous free regions too small to reuse), and **reservation waste** (space held for tokens not yet generated). Measured waste was 60–80% of KV memory — meaning the binding constraint on batch size (and thus throughput) was largely squandered.

Kwon et al. (2023, SOSP) borrowed the operating-system solution to exactly this problem: **paging**. KV memory is divided into fixed-size **physical blocks** (e.g., 16 tokens of K,V for all heads/layers). Each request has a **block table** mapping its logical token positions to physical blocks, which need not be contiguous. Blocks are allocated on demand as the sequence grows, so a request consumes only as much KV as it actually uses, rounded up to one block. This drops waste to under 4% (just the last partially-filled block per sequence), so far more requests fit in HBM and batch size — hence throughput — rises 2–4×. The cost is an attention kernel that gathers non-contiguous blocks, which the paper implements efficiently.

Paging also unlocks **sharing**: identical logical content (a shared prompt prefix, or parallel samples of the same request) can map to the *same* physical blocks with **copy-on-write** when they diverge. This is the seed of prefix caching and RadixAttention ([§02](02_prefix_caching_and_radix_attention.md)). Combined with continuous batching (from Orca) and a preemption policy, PagedAttention is the core of vLLM and the template most frameworks now follow.

***

## Core Concepts & Mechanics

### Fragmentation taxonomy
- **Internal fragmentation:** space reserved within a request's buffer but unused (short sequence in a max-length buffer).
- **External fragmentation:** free memory exists but not as a contiguous block large enough for a new request.
- **Reservation waste:** space held for future tokens that may never be generated.
PagedAttention eliminates external fragmentation entirely (any free block is usable) and bounds internal fragmentation to <1 block/sequence.

### The data structures
- **Physical block:** fixed-size KV storage, e.g., `block_size=16` tokens × (per-token KV across all heads/layers). Pool of blocks in HBM.
- **Logical blocks:** a sequence's tokens grouped into block-sized chunks.
- **Block table:** per-sequence array mapping logical block index → physical block id (exactly like a page table).
- **Block allocator:** tracks free blocks, ref-counts shared blocks.

📐 Waste per sequence ≤ `block_size − 1` tokens (the last partial block). Total internal fragmentation ≤ `n_sequences × (block_size−1) × KV_per_token` — tiny vs old reservation waste.

### Modified attention kernel
The PagedAttention kernel takes the block table and gathers K,V from scattered physical blocks during the QKᵀ and ·V steps, so non-contiguity is transparent to attention math. Block size trades off: larger blocks → more coalesced reads but more internal fragmentation; smaller → less waste but more scattered access. 16 is a common sweet spot.

### Sharing and copy-on-write
- **Parallel sampling / beam:** N candidates of one prompt share the prompt's physical blocks (ref-counted); when a candidate writes a divergent token, **copy-on-write** clones just that block. Saves N× the prompt KV.
- **Prefix sharing:** requests with a common prefix map the prefix's logical blocks to shared physical blocks (exact-match), the basis of vLLM prefix caching.

### vLLM scheduler additions
- **Continuous batching** (Orca-style): admit/evict requests at iteration boundaries ([§03](../03_batching_and_scheduling/00_static_vs_continuous_batching.md)).
- **Eviction (LRU)** when blocks run out; **preemption** of running requests via **swap** (copy KV to CPU) or **recompute** (drop KV, redo prefill later). Recompute is often cheaper than swap for short sequences; swap for long.
- **Admission control** so newly admitted requests have blocks for at least one step.

***

## Key Challenges
1. **Non-contiguous attention kernel cost.** Gathering scattered blocks risks uncoalesced reads; the kernel must be carefully written (and block size tuned) to keep bandwidth high.
2. **Preemption policy.** Under memory pressure, choosing swap vs recompute and which requests to preempt affects latency tails; wrong choices spike P99 ([§03](../03_batching_and_scheduling/04_request_scheduling_policies.md)).
3. **Block size tuning.** Couples fragmentation, kernel efficiency, and prefix-sharing granularity; one size isn't optimal for all models/workloads.
4. **Sharing correctness.** Copy-on-write ref-counting bugs cause data corruption across requests — a serious correctness hazard.

***

## Solutions & Current Best Practices
- **Adopt paged KV** (vLLM, SGLang, TensorRT-LLM in-flight batching all use paged/block KV). It is the de facto standard.
- **Tune block size** (8–32) per model/hardware; verify coalesced reads in Nsight ([§06](../06_kernel_optimization/05_kernel_profiling_and_benchmarking.md)).
- **Prefer recompute over swap** for short preempted sequences; swap for long ones to avoid re-prefill cost.
- **Layer prefix caching** on top for shared-prompt workloads ([§02](02_prefix_caching_and_radix_attention.md)).

***

## Implementation Notes
- vLLM exposes `gpu_memory_utilization` (fraction of HBM for KV blocks), `block_size`, and `swap_space` (CPU swap size). Set utilization ~0.9 leaving headroom.
- The free-block watermark triggers preemption; tune to balance admitting new requests vs protecting running ones.
- Monitor **KV cache utilization** and **preemption/eviction counts** — rising preemptions signal under-provisioned KV or too-aggressive admission.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Throughput tanks under memory pressure due to preemption thrashing** — requests repeatedly preempted and recomputed; symptom of over-admission. Tune watermark/admission.
- **Tiny block size kills bandwidth** — too-small blocks scatter reads; large blocks waste memory. Tune, don't default blindly.
- **Copy-on-write ref-count bug → cross-request corruption** — shared prefix/sampling blocks mutated without cloning; rare, severe, hard to reproduce.
- **Swap to CPU over PCIe stalls everything** — swapping long-context KV across PCIe is slow; prefer recompute or more HBM for long-context workloads.

***

## Performance Numbers & Benchmarks
| Metric | Naive (HF) | PagedAttention (vLLM) |
|---|---|---|
| KV memory waste | 60–80% | <4% |
| Achievable batch | low | 2–4× higher |
| Throughput (Kwon et al. 2023) | baseline | **2–4× higher** |
| Block size (typical) | — | 16 tokens |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Explain PagedAttention via the OS analogy and quantify the fragmentation savings."* — Expected: page table/block table; waste 60–80%→<4%; 2–4× throughput.
- *"Walk through copy-on-write KV sharing for parallel sampling."* — Expected: ref-counted shared prompt blocks, clone on divergence.
- *"How does the vLLM scheduler handle memory pressure?"* — Expected: LRU eviction, preempt via swap vs recompute, admission control.
- *"What does block size trade off?"* — Expected: fragmentation vs coalesced reads vs prefix-sharing granularity.

***

## Open Problems & Active Research (2025–2026)
- **Cross-request, cross-node paged KV** (disaggregation + paging) and global block management ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).
- **Optimal preemption/eviction policies** under tight SLOs and bursty load.
- **Paged + quantized + compressed KV** interactions (exact prefix matching with quantized blocks) ([§05](../05_quantization/05_kv_cache_quantization.md)).

***

## References
- Kwon, W., et al. (2023). "Efficient Memory Management for LLM Serving with PagedAttention." *SOSP 2023*. arXiv:2309.06180.
- Yu, G.-I., et al. (2022). "Orca." *OSDI 2022*.
- Zheng, L., et al. (2024). "SGLang/RadixAttention." arXiv:2312.07104.
