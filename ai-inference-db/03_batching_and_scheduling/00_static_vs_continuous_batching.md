# Static vs Continuous Batching

> **Section:** 03_batching_and_scheduling
> **Last Updated:** June 2026
> **Related Files:** [01_iteration_level_scheduling.md](01_iteration_level_scheduling.md), [02_chunked_prefill.md](02_chunked_prefill.md), [../00_fundamentals/02_prefill_vs_decode_phases.md](../00_fundamentals/02_prefill_vs_decode_phases.md)
> **Must-Read Papers:** Yu et al. (2022, OSDI) "Orca"; Kwon et al. (2023, SOSP) "vLLM"; Agrawal et al. (2024, OSDI) "Sarathi-Serve"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- **Batching is the primary cure for memory-bound decode**: it shares one weight load across many sequences, raising arithmetic intensity toward the roofline ridge ([§00](../00_fundamentals/04_roofline_model_for_llm.md)).
- **Static batching**: form a batch, run all to completion together, release together. GPU-efficient but **stragglers waste the GPU** and early-finishers wait — terrible latency and utilization.
- **Continuous (iteration-level) batching** (Orca, Yu et al. 2022): admit/evict requests at **every iteration boundary**, so finished sequences leave immediately and new ones join — dramatically higher utilization.
- Continuous batching requires **paged KV** (new requests need blocks on the fly) and careful handling of mixed prefill/decode (→ chunked prefill).
- It is the single highest-leverage serving technique; every modern framework uses it.

***

## Overview
Decode is memory-bandwidth-bound at small batch ([§00](../00_fundamentals/00_inference_vs_training.md)), so the dominant way to make serving efficient is to **batch many independent sequences** and amortize each weight load across them. The question is *how* to batch when requests arrive at different times and finish after different numbers of tokens. **Static batching** is the naive answer: collect a fixed batch, run it through generation until all sequences finish, then return everyone and start the next batch. Its fatal flaw is heterogeneity — sequences finish at wildly different lengths, so the batch runs at the pace of its **longest** member while finished sequences occupy slots doing nothing (the GPU computes padding/EOS). Utilization and latency both suffer: a request that needed 10 tokens waits for a batchmate generating 1000.

**Continuous batching** (a.k.a. iteration-level scheduling), introduced by **Orca** (Yu et al. 2022, OSDI), fixes this by making the batch *dynamic at the granularity of a single decode iteration*. After each forward pass, the scheduler removes any sequence that finished (freeing its KV) and admits waiting requests into the freed slots. The batch composition changes every iteration; no sequence waits for another to finish. This keeps the GPU busy with useful work continuously and is why throughput jumps 2–4× (combined with PagedAttention's memory efficiency). It is the foundational scheduling idea of vLLM, SGLang, TensorRT-LLM (as "in-flight batching"), and TGI.

Continuous batching is not free: it demands **dynamic KV allocation** (a newly admitted request needs KV blocks immediately — solved by paging, [§02](../02_kv_cache/01_paged_attention_vllm.md)), variable batch shapes each iteration (kernels must handle ragged batches), and a policy for **mixing prefill and decode** (a newly admitted request must be prefilled, which is compute-heavy and can stall decodes — the prefill-decode interference problem that motivates chunked prefill, [§02](02_chunked_prefill.md)). Understanding this technique and its complications is essential for any inference role.

***

## Core Concepts & Mechanics

### Static batching cost
📐 With batch B and output lengths `L_1..L_B`, static batch runtime ∝ `max_i L_i`, but useful work ∝ `mean_i L_i`. Efficiency = `mean/max`, which for skewed length distributions can be <30%. Slots of finished sequences are wasted for `max − L_i` iterations each.

### Continuous batching mechanics
- Maintain a **running batch** and a **waiting queue**.
- Each iteration: run one forward pass over the running batch (one new token per active sequence); sample; for any sequence hitting EOS/length, **evict** and free its KV blocks; **admit** waiting requests (subject to KV availability and a token budget) into the running batch; repeat.
- New admits require a **prefill** of their prompt; how prefill is interleaved with ongoing decode is the key design choice (see chunked prefill).

### Why it needs paging
A request admitted mid-stream needs KV space *now*, in arbitrary amounts, without a contiguous pre-reservation. PagedAttention's block allocator provides exactly this — continuous batching and paging are co-dependent ([§02](../02_kv_cache/01_paged_attention_vllm.md)).

### The batch-size-explosion / interference risk
If the scheduler admits many requests at once, their simultaneous prefills (compute-heavy) inflate iteration time and spike ITL for everyone (prefill-decode interference). Mitigations: bound the per-iteration **token budget**, **chunk** prefills, or **disaggregate** ([§02](02_chunked_prefill.md), [§03](03_prefill_decode_disaggregation.md)).

### Throughput/latency effect
Continuous batching raises throughput (more useful tokens/iteration) and *reduces* average latency (no waiting for stragglers), but the chosen max batch / token budget still trades ITL against throughput ([§00](../00_fundamentals/06_key_metrics_latency_throughput_cost.md)).

***

## Key Challenges
1. **Mixing prefill and decode.** Admitting requests injects compute-heavy prefills into a bandwidth-bound decode batch, spiking ITL — the central scheduling tension.
2. **Dynamic KV management.** Admitting/evicting every iteration requires fast, fragmentation-free allocation (paging) and correct freeing on EOS.
3. **Ragged-batch kernels.** Variable sequence lengths and batch composition each iteration require kernels (attention, sampling) that handle non-uniform shapes efficiently.
4. **Admission policy.** Admitting too aggressively causes interference/OOM/preemption; too conservatively wastes capacity. Tuning is workload-dependent.

***

## Solutions & Current Best Practices
- **Continuous batching everywhere** — the default in vLLM/SGLang/TRT-LLM/TGI.
- **Chunked prefill** to bound interference while admitting requests ([§02](02_chunked_prefill.md)).
- **Token-budget-based admission**: cap total prefill+decode tokens per iteration to meet ITL SLO ([§01](01_iteration_level_scheduling.md)).
- **Paged KV** as the memory substrate ([§02](../02_kv_cache/01_paged_attention_vllm.md)).

***

## Implementation Notes
- vLLM: `max_num_seqs` (max running batch) and `max_num_batched_tokens` (per-iteration token budget) are the core knobs; with chunked prefill the token budget governs interference.
- Free KV **at the iteration boundary** on EOS; leaks here silently shrink capacity.
- Monitor running-batch size, queue length, and preemptions to tune admission ([§01](../13_production_systems/01_observability_and_profiling.md)).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Continuous batching without chunked prefill still spikes P99 ITL** — admitted prefills inflate iteration time; you need chunking/disaggregation, not just continuous batching.
- **Aggressive admission → preemption thrashing** — over-admit, run out of KV, preempt, recompute, repeat. Cap admission to KV headroom.
- **EOS-free freeing bug leaks KV** — slots not released at the boundary shrink effective batch over time; throughput slowly degrades.
- **Static batching sneaks back in via padding** — poorly written kernels pad to max length per batch, reintroducing straggler waste even under "continuous" scheduling.

***

## Performance Numbers & Benchmarks
| Approach | GPU utilization | Throughput | Latency |
|---|---|---|---|
| Static batching | low (mean/max) | baseline | poor (straggler wait) |
| Continuous batching (Orca) | high | 2–4× (with paging) | lower avg |
| + chunked prefill | high | high | protected P99 ITL |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why is continuous batching better than static, quantitatively?"* — Expected: static ∝ max length (straggler waste); continuous evicts/admits per iteration → mean, not max.
- *"What does continuous batching require from the KV cache?"* — Expected: dynamic, fragmentation-free allocation → paging.
- *"What new problem does continuous batching create?"* — Expected: prefill-decode interference; fixed by chunked prefill/disaggregation.
- *"How would you set the per-iteration token budget?"* — Expected: cap so iteration time meets ITL SLO; balance throughput.

***

## Open Problems & Active Research (2025–2026)
- **Optimal joint admission + chunking + disaggregation** policies under bursty, heterogeneous load.
- **Length-aware scheduling** using output-length prediction to reduce interference ([§04](04_request_scheduling_policies.md)).
- **Continuous batching for reasoning models** with wildly variable, long outputs ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

***

## References
- Yu, G.-I., et al. (2022). "Orca: A Distributed Serving System for Transformer-Based Generative Models." *OSDI 2022*.
- Kwon, W., et al. (2023). "vLLM/PagedAttention." *SOSP 2023*. arXiv:2309.06180.
- Agrawal, A., et al. (2024). "Sarathi-Serve." *OSDI 2024*. arXiv:2403.02310.
