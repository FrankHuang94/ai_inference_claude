# Disaggregated Prefill-Decode Systems (Splitwise, DistServe, Mooncake)

> **Section:** 09_distributed_inference
> **Last Updated:** June 2026
> **Related Files:** [../03_batching_and_scheduling/03_prefill_decode_disaggregation.md](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md), [03_kv_cache_migration_and_transfer.md](03_kv_cache_migration_and_transfer.md), [../00_fundamentals/02_prefill_vs_decode_phases.md](../00_fundamentals/02_prefill_vs_decode_phases.md)
> **Must-Read Papers:** Patel et al. (2024, ISCA) "Splitwise"; Zhong et al. (2024, OSDI) "DistServe"; Qin et al. (2024) "Mooncake"; Hu et al. (2024) "TetriInfer"
> **Estimated Study Time:** 35 minutes

***

## TL;DR
- The most important 2024–2026 architectural trend: run **prefill (compute-bound)** and **decode (bandwidth-bound)** on **separate, independently-scaled instances**, transferring KV between them.
- **Splitwise** (ISCA 2024): phase-splitting + heterogeneous hardware/power optimization per phase.
- **DistServe** (OSDI 2024): co-optimizes per-phase parallelism + placement for **goodput** under TTFT+ITL SLOs (up to ~4–7×).
- **Mooncake** (Moonshot AI, production): **KV-cache-centric** design with a global disaggregated KV pool; high cache reuse at scale.
- **TetriInfer**: prediction-based scheduling and resource partitioning for disaggregation.
- Cost/risk: **KV transfer** over the interconnect (see [§03](03_kv_cache_migration_and_transfer.md)); wins at scale with good fabric and skewed P/D ratios.

***

## Overview
This file deepens the disaggregation concept introduced in [§03 batching](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md) with the specific production/research systems. The shared premise: prefill and decode have **opposite resource profiles** ([§00](../00_fundamentals/02_prefill_vs_decode_phases.md)), so co-locating them forces compromise and causes interference; separating them into dedicated **prefill** and **decode** pools lets each be optimized, scaled, and hardware-matched independently, transferring the prompt's KV cache from prefill to decode over a fast interconnect. The result is large **goodput** gains under strict dual (TTFT + ITL) SLOs.

The landmark systems each contribute a distinct angle. **DistServe** (Zhong et al. 2024) frames the problem as **goodput optimization**: it independently chooses parallelism and placement for prefill and decode to meet TTFT and ITL SLOs, reporting up to ~4–7× more SLO-meeting goodput than colocation. **Splitwise** (Patel et al. 2024) emphasizes **phase splitting with heterogeneous hardware** — using compute-dense GPUs for prefill and different (cost/power-optimized) GPUs for decode — and analyzes power/cost benefits. **Mooncake** (Qin et al. 2024), Moonshot AI's production system behind Kimi, is **KV-cache-centric**: it builds a **global, disaggregated KV pool** (across CPU/GPU/SSD tiers and nodes) that all prefill/decode instances share, maximizing cross-request KV reuse and decoupling KV from any single GPU. **TetriInfer** adds prediction-based scheduling and resource partitioning to reduce interference.

The unifying engineering challenges are the **KV transfer** (moving GBs/request — detailed in [§03](03_kv_cache_migration_and_transfer.md)), the **P/D ratio** (how many prefill vs decode instances, set by the input/output length distribution and dynamically rebalanced), and **routing** (which decode instance receives a prefill's KV, ideally cache-aware). Disaggregation wins at **scale**, with **good interconnect**, and for **skewed P/D workloads** (e.g., long prompts/short outputs or vice versa, and reasoning models with huge decode); chunked-prefill colocation remains better for small/simple deployments. This file surveys the systems and their design choices.

***

## Core Concepts & Mechanics

### Common architecture
```
router → Prefill pool (compute-optimized) --KV transfer (RDMA)--> Decode pool (bandwidth/capacity-optimized) → stream
```
Each pool independently scaled/parallelized/hardware-matched; KV moves between them.

### System contributions
| System | Key idea |
|---|---|
| **Splitwise** | phase split + heterogeneous hardware per phase; power/cost analysis |
| **DistServe** | goodput-optimal per-phase parallelism & placement under TTFT+ITL SLOs |
| **Mooncake** | KV-cache-centric: global disaggregated KV pool, high reuse, tiered storage |
| **TetriInfer** | prediction-based scheduling, resource partitioning to cut interference |

### The P/D ratio
📐 Balance: `n_prefill·R_prefill = n_decode·R_decode`. Long prompts/short outputs → more prefill; long outputs (reasoning) → more decode. Must adapt dynamically as traffic shifts ([§03](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md)).

### KV-centric design (Mooncake)
Treat KV as the central, movable, storable object: a **global KV pool** across GPU/CPU/SSD and nodes, shared by all instances, with cache-aware routing → high cross-request prefix reuse and decoupling from individual GPUs. Influential blueprint for large-scale disaggregation.

***

## Key Challenges
1. **KV transfer overhead.** Moving GBs/request must be RDMA + overlapped or it inflates TTFT ([§03](03_kv_cache_migration_and_transfer.md)).
2. **Dynamic P/D balancing.** Optimal ratio shifts with traffic; static ratios waste capacity.
3. **Routing & KV placement.** Which decode instance, with cache locality, is a non-trivial routing problem ([§01](01_load_balancing_strategies.md)).
4. **Operational complexity.** Two pools + transfer fabric + global KV layer; only justified at scale.

***

## Solutions & Current Best Practices
- **Disaggregate at scale** with RDMA KV transfer + overlap; heterogeneous pools (Splitwise) ([§01](../01_hardware/05_hardware_selection_decision_framework.md)).
- **Goodput-driven per-phase optimization** (DistServe).
- **Global KV pool + cache-aware routing** (Mooncake) for reuse.
- **Dynamic P/D rebalancing** by live length distribution; **chunked-prefill colocation** below disaggregation scale ([§03](../03_batching_and_scheduling/02_chunked_prefill.md)).

***

## Implementation Notes
- vLLM/SGLang now support PD disaggregation; KV transfer via NIXL/Mooncake transfer engine/LMCache ([§03](03_kv_cache_migration_and_transfer.md)).
- Budget KV transfer into TTFT; verify overlap (Nsight Systems).
- Monitor per-pool queues and rebalance P/D; integrate KV-aware routing.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **TTFT rose after disaggregating** — KV transfer not overlapped/fabric undersized; fix overlap + provision IB ([§03](03_kv_cache_migration_and_transfer.md)).
- **Static P/D wasted the fleet** — traffic shifted phase-heaviness; make ratio dynamic.
- **Lost prefix-cache reuse across pools** — no global KV layer/cache-aware routing; add Mooncake-style pool.
- **Disaggregated a small/short-prompt service** — overhead not worth it; colocate + chunked prefill.

***

## Performance Numbers & Benchmarks
| System | Result |
|---|---|
| DistServe | up to ~4–7× goodput vs colocation (tight SLOs) |
| Splitwise | better throughput/$ and throughput/W (heterogeneous) |
| Mooncake | high KV reuse, production scale (Kimi) |
| Chunked-prefill colocation | competitive, far simpler |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Compare Splitwise, DistServe, and Mooncake."* — Expected: heterogeneous phase-split / goodput-optimal parallelism / KV-centric global pool.
- *"Why does disaggregation improve goodput under dual SLOs?"* — Expected: independent per-phase optimization, no interference.
- *"What's Mooncake's central abstraction?"* — Expected: global disaggregated KV pool (KV-centric).
- *"When is disaggregation not worth it?"* — Expected: small scale, short prompts, limited interconnect.

***

## Open Problems & Active Research (2025–2026)
- **Cheap KV transfer** (compression, commodity Ethernet) to broaden disaggregation viability ([§03](03_kv_cache_migration_and_transfer.md)).
- **Dynamic, unified schedulers** spanning prefill/decode pools.
- **Disaggregation for reasoning models** (huge decode) economics ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

***

## References
- Patel, P., et al. (2024). "Splitwise." *ISCA 2024*. arXiv:2311.18677.
- Zhong, Y., et al. (2024). "DistServe." *OSDI 2024*. arXiv:2401.09670.
- Qin, R., et al. (2024). "Mooncake." arXiv:2407.00079.
- Hu, C., et al. (2024). "TetriInfer / Inference without Interference." arXiv:2401.11181.
