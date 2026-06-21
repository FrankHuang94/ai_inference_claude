# Prefill-Decode Disaggregation

> **Section:** 03_batching_and_scheduling
> **Last Updated:** June 2026
> **Related Files:** [02_chunked_prefill.md](02_chunked_prefill.md), [../09_distributed_inference/02_disaggregated_prefill_decode_systems.md](../09_distributed_inference/02_disaggregated_prefill_decode_systems.md), [../00_fundamentals/02_prefill_vs_decode_phases.md](../00_fundamentals/02_prefill_vs_decode_phases.md)
> **Must-Read Papers:** Patel et al. (2024, ISCA) "Splitwise"; Zhong et al. (2024, OSDI) "DistServe"; Qin et al. (2024) "Mooncake"
> **Estimated Study Time:** 35 minutes

***

## TL;DR
- **Insight:** prefill is compute-bound, decode is memory-bandwidth-bound — opposite needs → run them on **separate, independently-scaled instances**, transferring the KV cache between them.
- **Splitwise** (Patel 2024), **DistServe** (Zhong 2024), **Mooncake** (Qin 2024) are the canonical systems; report up to ~**2–7× goodput** under tight dual SLOs.
- Eliminates **prefill-decode interference** at the source and lets each pool use ideal hardware (e.g., compute-heavy GPU for prefill, bandwidth/capacity GPU for decode).
- Cost: **KV transfer** over the interconnect (a busy prefill GPU emits tens of GB/s of KV) — must be RDMA and overlapped, or it inflates TTFT.
- Key tuning knob: the **P/D ratio** (number of prefill vs decode instances), set by the workload's input/output length distribution.

***

## Overview
Disaggregation is the most important architectural trend in 2024–2026 inference systems. It starts from the phase asymmetry established in [§00](../00_fundamentals/02_prefill_vs_decode_phases.md): prefill saturates compute (Tensor Cores) while decode saturates memory bandwidth. Co-locating them on one GPU forces a compromise and creates interference (a long prefill stalls decodes). Disaggregation instead splits serving into a **prefill pool** and a **decode pool**: a request is prefilled on a prefill instance, its KV cache is transferred to a decode instance, and generation proceeds there. Each pool is sized, scaled, and even hardware-matched independently — compute-dense GPUs for prefill, bandwidth/capacity-rich GPUs (H200/MI300X) for decode — and neither phase interferes with the other.

The payoff, demonstrated by **DistServe** and **Splitwise**, is large **goodput** gains under strict TTFT *and* ITL SLOs, because you can optimize each phase for its own SLO without the other dragging it down. DistServe reports up to several× more SLO-meeting goodput vs colocation; Splitwise additionally exploits *heterogeneous* hardware and power profiles per phase. **Mooncake** (Moonshot AI's production system) takes it further with a **KV-cache-centric** design — a global, disaggregated KV pool shared across prefill/decode nodes — emphasizing that in disaggregation the KV cache becomes the central, movable object of the architecture.

The cost and risk is the **KV transfer**. A request's KV (e.g., ~1.25 GB for 70B@4k, [§02](../02_kv_cache/00_kv_cache_fundamentals.md)) must move from prefill to decode GPU; a busy prefill instance generates tens of GB/s of KV, requiring RDMA (InfiniBand/NVLink) and careful overlap so transfer hides behind compute rather than adding to TTFT. If the interconnect is undersized or transfer isn't pipelined, disaggregation can *raise* TTFT and lose to chunked-prefill colocation. So disaggregation wins at scale, with good interconnect, and for workloads with skewed or heavy prefill — and chunked prefill ([§02](02_chunked_prefill.md)) remains the better choice for simpler/smaller deployments.

***

## Core Concepts & Mechanics

### Architecture
```
client → router → [Prefill pool] --KV transfer (RDMA)--> [Decode pool] → stream tokens
```
- **Prefill instances**: ingest prompt, produce KV + first token. Compute-bound; benefit from FLOPs (H100/B200), can run higher TP for long prompts.
- **Decode instances**: receive KV, run the generation loop. Bandwidth/capacity-bound; benefit from H200/MI300X and large batches.
- **KV transfer**: move the prompt's KV blocks over the interconnect; overlap with decode startup.

### KV transfer cost
📐 Transfer time per request ≈ `KV_bytes / interconnect_BW`. For 1.25 GB over NDR IB (~50 GB/s): ~25 ms — comparable to a long prefill, so it must overlap or it doubles TTFT impact. Aggregate KV rate from a prefill GPU = `requests/s × KV_bytes`; can reach tens of GB/s, demanding RDMA and possibly KV compression ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).

### The P/D ratio
📐 Let prefill instance throughput = `R_p` req/s and decode instance capacity = `R_d` req/s (depends on output length). Balance: `n_prefill · R_p = n_decode · R_d`. Workloads with long prompts/short outputs need more prefill instances; long outputs (reasoning) need more decode. The ratio must adapt as traffic shifts — a live control problem.

### Operating modes
- **Prefill-bound**: long prompts dominate (RAG, doc QA) → more prefill capacity.
- **Decode-bound**: long outputs dominate (chat, reasoning) → more decode capacity.
- **Heterogeneous disaggregation**: cheaper/compute-dense GPUs for prefill, bandwidth/capacity GPUs for decode — Splitwise's power/cost optimization.

***

## Key Challenges
1. **KV transfer overhead.** Moving GBs per request over the interconnect can dominate TTFT if not RDMA + overlapped; undersized fabric negates the benefit.
2. **P/D ratio tuning.** The optimal ratio depends on the live input/output length distribution; static ratios waste capacity when traffic shifts.
3. **Routing and KV placement.** Which decode instance receives a given KV (load + cache locality) is a non-trivial routing problem ([§09](../09_distributed_inference/01_load_balancing_strategies.md)).
4. **Operational complexity.** Two pools, a transfer fabric, and a global KV layer are far more complex to run than a single engine — only worth it at scale.

***

## Solutions & Current Best Practices
- **Disaggregate at scale** with RDMA KV transfer and overlap; **chunked-prefill colocation** below that scale ([§02](02_chunked_prefill.md)).
- **Heterogeneous pools**: FLOPs-rich for prefill, bandwidth/capacity-rich for decode ([§01](../01_hardware/05_hardware_selection_decision_framework.md)).
- **Dynamic P/D ratio control** driven by live length distributions and queue depths.
- **KV-centric design** (Mooncake): global KV pool + cache-aware routing for prefix reuse across pools.

***

## Implementation Notes
- vLLM and SGLang now ship PD-disaggregation support; KV transfer via libraries like **NIXL/Mooncake transfer engine/LMCache**.
- Budget the KV transfer into the TTFT SLO; verify overlap with Nsight Systems (transfer should hide behind compute).
- Monitor per-pool queues and the P/D balance; auto-rebalance instances as traffic shifts.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Disaggregation raised TTFT** — KV transfer not overlapped or IB undersized; transfer time added to first-token latency. Overlap + provision fabric.
- **Static P/D ratio wasted half the fleet** — traffic shifted from prompt-heavy to generation-heavy; prefill pool idle, decode pool overloaded. Make the ratio dynamic.
- **Short-prompt workload gained nothing** — tiny prefills mean little interference and the transfer overhead isn't worth it; colocation + chunked prefill wins.
- **Prefix-cache hits lost across pools** — without a global KV layer + cache-aware routing, disaggregation can break prefix reuse that colocation had ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)).

***

## Performance Numbers & Benchmarks
| System | Setting | Result |
|---|---|---|
| DistServe (2024) | tight TTFT+ITL SLOs | up to ~4–7× goodput vs colocation |
| Splitwise (2024) | heterogeneous pools | better throughput/$ and throughput/W |
| Mooncake (2024) | production, KV-centric | high cache reuse, scaled disaggregation |
| Chunked-prefill colocation | moderate prompts | competitive, far simpler |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why disaggregate prefill and decode? What does it cost?"* — Expected: opposite bottlenecks, independent scaling, no interference; cost = KV transfer.
- *"Estimate the interconnect bandwidth needed for KV transfer."* — Expected: KV_bytes × req/s; compare to IB; overlap requirement.
- *"What sets the P/D ratio and why must it be dynamic?"* — Expected: input/output length distribution; traffic shifts.
- *"When is chunked prefill better than disaggregation?"* — Expected: moderate prompts, limited interconnect, simplicity; small scale.

***

## Open Problems & Active Research (2025–2026)
- **Dynamic, traffic-driven P/D rebalancing** and unified schedulers across pools.
- **Cheap KV transfer** (compression, RoCE/Ethernet) to make disaggregation viable without premium fabric ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).
- **Global KV pools with cross-pool prefix reuse** (Mooncake-style) as a standard component.

***

## References
- Patel, P., et al. (2024). "Splitwise: Efficient Generative LLM Inference Using Phase Splitting." *ISCA 2024*. arXiv:2311.18677.
- Zhong, Y., et al. (2024). "DistServe." *OSDI 2024*. arXiv:2401.09670.
- Qin, R., et al. (2024). "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving." arXiv:2407.00079.
