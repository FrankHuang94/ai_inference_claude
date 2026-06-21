# Chunked Prefill

> **Section:** 03_batching_and_scheduling
> **Last Updated:** June 2026
> **Related Files:** [00_static_vs_continuous_batching.md](00_static_vs_continuous_batching.md), [03_prefill_decode_disaggregation.md](03_prefill_decode_disaggregation.md), [../00_fundamentals/02_prefill_vs_decode_phases.md](../00_fundamentals/02_prefill_vs_decode_phases.md)
> **Must-Read Papers:** Agrawal et al. (2024, OSDI) "Sarathi-Serve: Taming Throughput-Latency Tradeoff"; Agrawal et al. (2023) "Sarathi"; Holmes et al. (2024) "DeepSpeed-FastGen (Dynamic SplitFuse)"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- **Problem:** a long prefill monopolizes the GPU for one long iteration, stalling co-batched decodes → P99 ITL spikes (prefill-decode interference, [§00](../00_fundamentals/02_prefill_vs_decode_phases.md)).
- **Chunked prefill** (Sarathi-Serve, Agrawal et al. 2024): split a long prompt into fixed **chunks** (e.g., 512 tokens) and process one chunk per iteration, **interleaved with decodes** in the same batch.
- Keeps each iteration short (bounding ITL) while still doing prefill efficiently — converts a latency spike into many small, predictable steps.
- **Chunk size** is the key knob: large → better prefill MFU but more ITL inflation; small → smoother ITL, more overhead. Hardware/SLO-dependent.
- Default-on in vLLM/SGLang; DeepSpeed-FastGen's "Dynamic SplitFuse" is the same idea. Often a simpler alternative to disaggregation.

***

## Overview
Continuous batching admits new requests mid-stream, but each admitted request must be **prefilled**, and prefill of a long prompt is a single compute-heavy operation that takes far longer than a decode step. If that whole prefill runs in one iteration alongside decodes, the iteration time balloons to the prefill's duration and every co-batched decode's inter-token latency spikes — the prefill-decode interference problem. The result is the classic "P99 ITL randomly jumps" symptom under mixed traffic.

**Chunked prefill** breaks the long prefill into bounded chunks and processes one chunk per iteration, packing it together with the ongoing decodes up to a per-iteration **token budget**. Because each iteration now does at most `chunk_size` prefill tokens plus the decode tokens, iteration time is bounded and predictable, so ITL stays within SLO even while a long prompt is being ingested over several iterations. Crucially, prefill is still compute-bound and efficient (a 512-token chunk has high arithmetic intensity), so you keep most of prefill's throughput while smoothing latency — directly "taming the throughput-latency tradeoff," as the Sarathi-Serve title puts it. DeepSpeed-FastGen independently introduced the same mechanism as "Dynamic SplitFuse."

This is often the **simpler, single-engine alternative to disaggregation** ([§03](03_prefill_decode_disaggregation.md)): instead of separate prefill/decode clusters with KV transfer, you interleave within one engine. For many workloads (moderate prompt lengths, limited interconnect), chunked prefill captures most of the benefit with far less operational complexity. Disaggregation wins at larger scale and more skewed P/D ratios.

***

## Core Concepts & Mechanics

### The interference math
📐 Decode iteration time `t_d` ≈ a few ms (bandwidth-bound, batch B). A full prefill of P tokens takes `t_p ≈ 2·N·P/(MFU·peak_FLOPs)` — for P=8k this can be tens of ms. Mixed in one iteration, ITL for that step = `t_p` ≫ SLO. Chunked: each iteration does `chunk` prefill tokens, time ≈ `t_d + 2·N·chunk/(MFU·peak)`; choose `chunk` so this ≤ ITL SLO.

### How chunking interleaves
- The scheduler maintains a prefill cursor per admitting request.
- Each iteration: include up to `chunk_size` prefill tokens (from one or more admitting requests) **plus** one token per running decode, capped by `max_num_batched_tokens`.
- The attention kernel handles a **ragged batch** mixing a prefill chunk (many positions, causal within the prompt) and decodes (one position each).
- Repeat until the prompt is fully prefilled, then the request transitions to pure decode.

### Chunk-size tradeoff
- **Large chunk**: fewer iterations to finish prefill, higher prefill MFU (better arithmetic intensity), but larger ITL inflation per step and higher TTFT for co-batched decodes.
- **Small chunk**: smoother ITL, but more iterations, more per-iteration overhead, and prefill runs at lower efficiency (smaller matmuls).
- Optimal depends on model size, hardware, and the ITL SLO; common values 256–1024. Some systems tune it dynamically.

### Relation to token budget
Chunked prefill is enforced via the per-iteration **token budget** ([§01](01_iteration_level_scheduling.md)): the budget = decode tokens + prefill chunk. Setting the budget *is* setting the chunk policy.

***

## Key Challenges
1. **Choosing chunk size.** Couples prefill efficiency, ITL, and TTFT; no universal value, and the optimum shifts with model/hardware/SLO.
2. **TTFT vs ITL retrade.** Chunking lowers ITL for co-batched decodes but can *raise* TTFT for the chunked request (its prefill is spread over iterations). You're rebalancing, not eliminating, the tension.
3. **Ragged-batch kernel efficiency.** Mixing a multi-position prefill chunk with single-position decodes requires attention kernels that stay efficient on heterogeneous shapes.
4. **Very long prefills still cost.** Chunking smooths ITL but the *total* prefill work (and TTFT) for a 1M-token prompt remains large — needs context parallelism/disaggregation ([§10](../10_long_context/02_ring_attention_and_context_parallelism.md)).

***

## Solutions & Current Best Practices
- **Enable chunked prefill by default** (vLLM `enable_chunked_prefill`, SGLang, DeepSpeed-FastGen) for mixed traffic.
- **Tune chunk size / token budget** to the ITL SLO on your model+hardware; start ~512.
- **Combine with prefix caching** so shared prefixes skip chunked prefill entirely ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)).
- **Escalate to disaggregation** when prompts are very long or P/D ratios are skewed ([§03](03_prefill_decode_disaggregation.md)).

***

## Implementation Notes
- Set `max_num_batched_tokens` so the worst-case iteration (chunk + decodes) meets ITL; verify with a P99 ITL benchmark under mixed load.
- Ensure the attention kernel (FlashAttention/paged) supports mixed prefill-chunk + decode batches efficiently.
- Watch TTFT for long-prompt requests; if it regresses too far, increase chunk size or route long prompts to a disaggregated prefill pool.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Chunk size great for 8B, wrong for 70B** — larger models have larger per-token prefill cost; the chunk that meets ITL on 8B blows it on 70B. Re-tune per model.
- **Chunking fixed ITL but tanked TTFT for long prompts** — spreading prefill over iterations delays first token; rebalance chunk size or disaggregate long prompts.
- **Ragged-batch kernel inefficiency** — naive kernels pad/serialize mixed batches, eating the throughput chunking was supposed to preserve.
- **Token budget too small** — prefill crawls over many tiny chunks at poor MFU; throughput drops even though ITL is smooth.

***

## Performance Numbers & Benchmarks
| Config | P99 ITL | Prefill throughput | TTFT |
|---|---|---|---|
| Full prefill in batch | spikes (∝ prompt len) | high | low for that req |
| Chunked (512) | bounded near SLO | ~high | slightly higher for long prompts |
| Sarathi-Serve vs vLLM-baseline (2024) | up to ~3–5× lower tail ITL | maintained | balanced |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How does chunked prefill prevent ITL spikes?"* — Expected: bound per-iteration prefill tokens; interleave with decode; iteration time ≤ SLO.
- *"What does chunk size trade off?"* — Expected: prefill MFU/TTFT vs ITL smoothness/overhead.
- *"Chunked prefill vs disaggregation — when each?"* — Expected: chunking simpler single-engine for moderate prompts; disaggregation for long prompts/skewed P/D/scale.
- *"How is chunked prefill implemented in the scheduler?"* — Expected: per-iteration token budget = decode + prefill chunk; ragged batch.

***

## Open Problems & Active Research (2025–2026)
- **Dynamic/adaptive chunk sizing** based on live load and SLO headroom.
- **Chunked prefill + context parallelism** for very long prompts ([§10](../10_long_context/02_ring_attention_and_context_parallelism.md)).
- **Unified policy** that fluidly chooses chunking vs disaggregation per request.

***

## References
- Agrawal, A., et al. (2024). "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." *OSDI 2024*. arXiv:2403.02310.
- Agrawal, A., et al. (2023). "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills." arXiv:2308.16369.
- Holmes, C., et al. (2024). "DeepSpeed-FastGen: High-throughput Text Generation via Dynamic SplitFuse." arXiv:2401.08671.
