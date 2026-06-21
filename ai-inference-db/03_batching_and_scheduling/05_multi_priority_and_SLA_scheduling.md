# Multi-Priority and SLA-Aware Scheduling

> **Section:** 03_batching_and_scheduling
> **Last Updated:** June 2026
> **Related Files:** [04_request_scheduling_policies.md](04_request_scheduling_policies.md), [../13_production_systems/06_SLO_definition_and_enforcement.md](../13_production_systems/06_SLO_definition_and_enforcement.md), [../13_production_systems/04_multi_tenant_serving.md](../13_production_systems/04_multi_tenant_serving.md)
> **Must-Read Papers:** Wu et al. (2023) "FastServe"; Zhong et al. (2024, OSDI) "DistServe" (goodput/SLO); classic: Demers et al. (1989) "Weighted Fair Queueing"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- Production serving has **multiple SLO tiers** (premium/free, interactive/batch) sharing one fleet; scheduling must enforce per-tier TTFT/ITL targets and fairness.
- **Priority scheduling** runs higher tiers first but risks **starving** lower tiers; **weighted fair queueing (WFQ)** and **token-bucket rate limiting** bound each tier's share.
- **Preemption** lets a high-priority request bump a low-priority one (swap/recompute its KV); essential for tight premium SLOs.
- **Goodput per tier** (SLO-meeting throughput) is the optimization target, not aggregate throughput.
- Batch/offline traffic should **backfill** idle capacity without harming interactive SLOs.

***

## Overview
Real inference services are multi-tenant and multi-tier: a premium API customer expects sub-second TTFT and fast ITL; a free tier tolerates more latency; internal batch jobs (evals, data generation) care only about throughput and can run whenever there's spare capacity. All share the same expensive GPU fleet. The scheduler must therefore enforce **differentiated SLOs** — giving each tier the latency it pays for — while keeping utilization high by filling troughs with lower-priority work. This is a classic QoS problem adapted to the LLM iteration-level setting.

The tools are borrowed from networking and OS scheduling. **Strict priority** runs higher tiers first — simple and effective for protecting premium latency, but it can **starve** lower tiers under sustained premium load. **Weighted fair queueing (WFQ)** and **token-bucket / leaky-bucket rate limiting** instead allocate each tier a guaranteed share of capacity, preventing starvation while still favoring higher tiers. **Preemption** (cheap via KV swap/recompute, [§02](../02_kv_cache/04_kv_cache_offloading.md)) lets an arriving premium request bump a running batch job to meet its SLO. And **backfill** schedules best-effort/batch work only into capacity that interactive traffic isn't using, so the fleet stays busy without violating interactive SLOs.

The right objective is **per-tier goodput** — the rate of requests served *within each tier's SLO* — rather than raw aggregate throughput, which could be maximized by starving premium users with cheap batch work. Disaggregation and routing interact here too: you might route tiers to different pools, or reserve decode capacity for interactive traffic. This file covers the mechanisms and their failure modes.

***

## Core Concepts & Mechanics

### Tier model
| Tier | TTFT/ITL SLO | Scheduling treatment |
|---|---|---|
| Premium/interactive | tight (e.g., TTFT<300ms, ITL<40ms) | highest priority, preempt others |
| Standard/free | relaxed | fair share, lower priority |
| Batch/offline | none (throughput) | backfill idle capacity only |

### Priority vs fairness
- **Strict priority**: always serve highest non-empty tier. Protects premium; starves lower tiers under load.
- **WFQ**: assign weights `w_i`; tier i gets `w_i / Σw` of capacity over time. 📐 Guarantees a minimum share, bounding worst-case latency per tier while still biasing toward heavy weights.
- **Token bucket**: each tier draws tokens at rate `r_i` with burst `b_i`; requests wait if the bucket is empty — enforces rate caps and smooths bursts.

### Preemption for SLO
A premium arrival when the GPU is full preempts a batch/low-tier request (swap or recompute its KV), freeing a slot immediately. 📐 Preemption cost must be < the SLO slack it buys; recompute short victims, swap long ones. Anti-starvation: cap how often a given low-tier request can be preempted (aging).

### Backfill
Best-effort/batch work is admitted only when interactive queues are empty and KV headroom exists, and is the first to be preempted. Keeps utilization high (good $/token, [§01](../01_hardware/06_tco_and_cost_modeling.md)) without harming interactive SLOs.

***

## Key Challenges
1. **Starvation vs prioritization.** Strict priority protects premium but starves others; pure fairness fails to protect premium. Hybrid (priority + minimum shares) is needed.
2. **Preemption overhead and thrash.** Over-preempting batch jobs wastes work (recompute) and can livelock; under-preempting misses premium SLOs.
3. **Per-tier goodput accounting.** Measuring and optimizing SLO-conditioned throughput per tier (not aggregate) requires careful instrumentation.
4. **Cross-tier interference.** Even with priority, a low-tier long prefill in the batch can spike a premium request's ITL — needs chunking/isolation ([§02](02_chunked_prefill.md)).

***

## Solutions & Current Best Practices
- **Hybrid priority + WFQ**: strict-ish priority for premium with guaranteed minimum shares for lower tiers to prevent starvation.
- **Token-bucket rate limiting** per tenant/tier for fairness and abuse control.
- **Preemption with aging** so premium SLOs are met without starving or thrashing batch jobs.
- **Backfill batch into idle capacity**; preempt it first.
- **Optimize per-tier goodput**, track SLO compliance per tier ([§13](../13_production_systems/06_SLO_definition_and_enforcement.md)).

***

## Implementation Notes
- Tag requests with tier/priority at the gateway; carry through to the iteration scheduler.
- Reserve a small **headroom of decode slots** for interactive traffic so a burst doesn't wait behind batch work.
- Consider **separate pools** (or disaggregated decode pools) per tier when interference can't be tamed within one engine ([§03](03_prefill_decode_disaggregation.md)).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Strict priority starved the free tier into timeouts** — sustained premium load left nothing for others; add minimum shares (WFQ).
- **Batch backfill spiked premium ITL** — a batch long-prefill landed in the premium batch; chunk it and/or isolate tiers.
- **Preemption thrash wasted GPU** — premium bursts repeatedly preempted/recomputed batch jobs; cap preemption rate, prefer swap for long victims.
- **Aggregate throughput looked great, premium SLO violated** — optimizing the wrong metric; track per-tier goodput.

***

## Performance Numbers & Benchmarks
| Mechanism | Protects premium? | Prevents starvation? | Utilization |
|---|---|---|---|
| Strict priority | yes | no | high |
| WFQ / shares | partially | yes | high |
| Token bucket | rate caps | yes | smooths bursts |
| Priority + WFQ + backfill | yes | yes | high (fills troughs) |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Design scheduling for a service with premium, free, and batch tiers on one fleet."* — Expected: priority + minimum shares (WFQ), preemption for premium, backfill batch, per-tier goodput.
- *"How do you prevent low tiers from starving under strict priority?"* — Expected: WFQ guaranteed shares, aging, rate limits.
- *"How does preemption help SLO enforcement, and what's the risk?"* — Expected: bump low-tier to meet premium; thrash/overhead risk, aging.
- *"What's the right objective for multi-tier serving?"* — Expected: per-tier goodput, not aggregate throughput.

***

## Open Problems & Active Research (2025–2026)
- **SLO-optimal multi-tier scheduling** under bursty, heterogeneous load with provable guarantees.
- **Cross-pool tier isolation** in disaggregated systems.
- **Economic scheduling** that prices preemption/priority dynamically to maximize goodput-per-dollar ([§01](../01_hardware/06_tco_and_cost_modeling.md)).

***

## References
- Wu, B., et al. (2023). "FastServe." arXiv:2305.05920.
- Zhong, Y., et al. (2024). "DistServe." *OSDI 2024*. arXiv:2401.09670.
- Demers, A., Keshav, S., Shenker, S. (1989). "Analysis and Simulation of a Fair Queueing Algorithm." *SIGCOMM 1989*.
