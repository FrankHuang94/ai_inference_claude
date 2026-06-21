# Prefill vs Decode: The Two Phases of LLM Inference

> **Section:** 00_fundamentals
> **Last Updated:** June 2026
> **Related Files:** [03_memory_bandwidth_bound_compute.md](03_memory_bandwidth_bound_compute.md), [../03_batching_and_scheduling/02_chunked_prefill.md](../03_batching_and_scheduling/02_chunked_prefill.md), [../03_batching_and_scheduling/03_prefill_decode_disaggregation.md](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md)
> **Must-Read Papers:** Agrawal et al. (2024, MLSys/OSDI) "Sarathi-Serve"; Patel et al. (2024, ISCA) "Splitwise"; Zhong et al. (2024, OSDI) "DistServe"
> **Estimated Study Time:** 35 minutes

***

## TL;DR
- **Prefill** processes the entire prompt in one parallel forward pass → **compute-bound**, high arithmetic intensity, behaves like training. Determines **TTFT** (time-to-first-token).
- **Decode** generates tokens one at a time → **memory-bandwidth-bound**, low arithmetic intensity. Determines **ITL/TPOT** (inter-token / time-per-output-token).
- The two phases sit on **opposite sides of the roofline ridge** and want different optimizations and even different hardware.
- **Mixing them in one batch causes prefill-decode interference**: a long prefill stalls co-located decodes, spiking everyone's ITL. This is the central scheduling problem.
- Two fixes dominate 2024–2026: **chunked prefill** (interleave prefill chunks with decode in one engine) and **disaggregation** (separate prefill and decode instances).

***

## Overview
A request's lifetime has two phases with qualitatively different performance physics. **Prefill** ("context encoding") runs the prompt of length P through the model *in parallel* — all P positions at once — producing the KV cache for the prompt and the first output token's logits. Because P positions share each weight load, the arithmetic intensity is high (∝ P), the Tensor Cores are well-fed, and prefill is **compute-bound**, much like a training forward pass. Prefill latency is roughly linear in P (quadratic in the attention term for very long P) and sets the **TTFT**.

**Decode** then generates the remaining output tokens one at a time, each a forward pass with a single query position attending to the growing KV cache. Arithmetic intensity is ∝ batch size (≈1 at batch=1), so decode is **memory-bandwidth-bound** and dominated by streaming weights. Decode latency per token is the **ITL**, and the user-perceived speed of a streaming response is governed by ITL, while the wait before text appears is TTFT.

This phase split is *the* organizing principle of modern serving. Almost every architectural decision — batching policy, whether to disaggregate, how to schedule, what hardware to buy for which pool — flows from the fact that prefill and decode have opposite bottlenecks and therefore conflict when they share a GPU. If you understand this file deeply, much of [§03](../03_batching_and_scheduling/00_static_vs_continuous_batching.md) and [§09](../09_distributed_inference/02_disaggregated_prefill_decode_systems.md) becomes obvious.

***

## Core Concepts & Mechanics

### Compute profiles
For a model with parameters `N`, per-token forward FLOPs ≈ `2N` (plus attention term). 📐

- **Prefill** of prompt length P, batch B: FLOPs ≈ `2N · B · P` + attention `O(B · P² · d)`. Bytes moved ≈ weights (read once) + KV writes. AI ≈ `~P` (high) → compute-bound. Latency ≈ `2N·B·P / (MFU · peak_FLOPs)`.
- **Decode** step, batch B, current length S: FLOPs ≈ `2N · B` + attention `O(B · S · d)`. Bytes ≈ weights (read once) + KV read `O(B · S · d)`. AI ≈ `~B` (low) → memory-bound. Latency ≈ `weight_bytes / bandwidth` (B small) plus KV-read time (grows with S).

The asymmetry: prefill does ~P× more FLOPs per weight load than a single decode step, which is exactly why prefill saturates compute while decode saturates bandwidth.

### Roofline positions
On the roofline (see [04_roofline_model_for_llm.md](04_roofline_model_for_llm.md)): prefill sits to the **right of the ridge** (compute-bound, near peak TFLOPS); a single decode step sits far to the **left** (memory-bound, performance = AI × bandwidth). Raising decode batch slides it rightward toward the ridge — the geometric picture of "batching fixes decode."

### The latency metrics
- **TTFT** = time from request arrival to first token = queue wait + prefill time (+ any KV transfer if disaggregated). Dominated by prefill compute and prompt length.
- **ITL / TPOT** = time between successive output tokens ≈ decode step latency. Dominated by bandwidth and batch.
- **E2E latency** = TTFT + (output_len − 1) × ITL. For short prompts/long outputs, decode dominates; for long prompts/short outputs (e.g., classification, RAG), prefill dominates.

### Prefill-decode interference
When a continuous-batching engine mixes a long prefill request into a batch that also contains decodes, the *iteration time* of that batch balloons to the prefill's compute time. Every decode in that batch waits, so their ITL spikes — a long prompt from one user degrades latency for everyone co-batched. This is **HOL blocking at the iteration level** and the core motivation for chunked prefill and disaggregation.

***

## Key Challenges
1. **Opposite bottlenecks, shared hardware.** A GPU tuned/configured for compute-bound prefill is wasteful for bandwidth-bound decode and vice versa; co-locating forces a compromise.
2. **Interference / HOL blocking.** Long prefills inflate iteration time and wreck decode ITL for co-batched requests; this is the dominant cause of P99 ITL spikes.
3. **TTFT vs ITL trade.** Prioritizing prefill (good TTFT) starves decode (bad ITL) and vice versa; a single queue cannot satisfy both SLOs without a smarter policy.
4. **Long-context prefill cost.** Prefill is ~linear in P with a quadratic attention term; at 100k–1M tokens prefill alone can take seconds, making TTFT the binding SLO (see [§10](../10_long_context/00_long_context_challenges.md)).

***

## Solutions & Current Best Practices
- **Chunked prefill** (Sarathi-Serve, Agrawal et al. 2024): split a long prefill into fixed chunks (e.g., 512 tokens) and interleave with decodes so each iteration stays short, bounding ITL inflation while keeping prefill efficient. Now default-on in vLLM/SGLang. See [§03](../03_batching_and_scheduling/02_chunked_prefill.md).
- **Disaggregation** (Splitwise, Patel et al. 2024; DistServe, Zhong et al. 2024): dedicated prefill instances and decode instances, KV transferred over RDMA/NVLink. Each pool is independently optimized and scaled; eliminates interference. See [§03](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md) and [§09](../09_distributed_inference/02_disaggregated_prefill_decode_systems.md).
- **Prefix caching** to skip prefill entirely for shared prompts (system prompts, multi-turn) — see [§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md).
- **Prioritized scheduling** that bounds how much prefill work enters a decode batch per iteration ([§03](../03_batching_and_scheduling/04_request_scheduling_policies.md)).

Convergence: chunked prefill is the default for single-engine serving; disaggregation wins at scale and for workloads with skewed input/output length ratios.

***

## Implementation Notes
- The **chunk size** is a tuning knob: larger chunks → better prefill MFU but more ITL inflation; smaller chunks → smoother ITL but lower prefill efficiency and more overhead. Tune per hardware and SLO.
- In vLLM, `enable_chunked_prefill=True` and `max_num_batched_tokens` control the prefill/decode token budget per iteration — set the budget so a single iteration meets your ITL SLO.
- For disaggregation, the **KV transfer must overlap** with decode startup or it shows up directly in TTFT; budget interconnect bandwidth accordingly (see [§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **"P99 ITL spikes randomly."** Almost always a long prefill landing in a decode batch. The fix is chunked prefill or disaggregation, not a bigger GPU.
- **Disaggregation can *raise* TTFT** if KV transfer isn't overlapped or the interconnect is undersized — you pay prefill time *plus* transfer time. A single H100 can emit ~tens of GB/s of KV; cross-node without good IB/RoCE, transfer dominates.
- **Chunk size that's great for one model is wrong for another** — the optimal depends on hidden size, attention cost, and your ITL target. Re-tune per deployment.
- **Short-prompt workloads barely benefit from disaggregation** — if prefill is tiny, the interference and the transfer overhead aren't worth the operational complexity; colocation + chunked prefill wins.

***

## Performance Numbers & Benchmarks
| Hardware | Model | Scenario | Metric | Result |
|---|---|---|---|---|
| 1× H100 | 13B | prefill 2k tokens | TTFT | ~tens of ms (compute-bound, near peak) |
| 1× H100 | 13B | decode batch=1 | ITL | ~7–10 ms/token |
| 1× H100 | 70B (TP2) | prefill 100k | TTFT | seconds (quadratic attention term) |
| Sarathi-Serve vs vLLM-baseline | 13B/70B | mixed traffic | P99 ITL | up to ~3–5× lower ITL tail via chunked prefill |
| DistServe vs colocated | 13B–66B | tight TTFT+ITL SLOs | goodput | reported up to ~4–7× more SLO-meeting goodput |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Contrast the roofline position and bottleneck of prefill vs decode."* — Expected: compute-bound/high-AI vs bandwidth-bound/low-AI, with the FLOPs-per-weight-load argument.
- *"Your P99 inter-token latency is spiking under load. Diagnose."* — Expected: prefill-decode interference → chunked prefill or disaggregation.
- *"When does disaggregation beat colocation, and when does it not?"* — Expected: skewed P/D length ratios and tight dual SLOs favor disagg; short prompts and limited interconnect favor colocation.
- *"How do TTFT and ITL map to user experience, and how do you optimize each?"* — Expected: TTFT=prefill+queue, ITL=decode; prefix caching and prefill scheduling for TTFT, batching/quantization for ITL.

***

## Open Problems & Active Research (2025–2026)
- **Optimal P/D ratio and dynamic re-balancing** as traffic shifts between prompt-heavy and generation-heavy workloads (reasoning models break old assumptions).
- **Disaggregation economics** — when the interconnect and operational cost pay off vs chunked-prefill colocation is still empirically litigated.
- **Long-context prefill acceleration** — making 1M-token TTFT tolerable via context parallelism, sparse prefill, and KV reuse ([§10](../10_long_context/02_ring_attention_and_context_parallelism.md)).
- **Unified schedulers** that fluidly move between chunked-colocation and disaggregation based on live load.

***

## References
- Agrawal, A., et al. (2024). "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." *OSDI 2024* (chunked prefill). arXiv:2403.02310.
- Patel, P., et al. (2024). "Splitwise: Efficient Generative LLM Inference Using Phase Splitting." *ISCA 2024*. arXiv:2311.18677.
- Zhong, Y., et al. (2024). "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving." *OSDI 2024*. arXiv:2401.09670.
- Kwon, W., et al. (2023). "PagedAttention/vLLM." *SOSP 2023*. arXiv:2309.06180.
