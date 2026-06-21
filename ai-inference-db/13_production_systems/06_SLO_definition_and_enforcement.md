# SLO Definition and Enforcement

> **Section:** 13_production_systems
> **Last Updated:** June 2026
> **Related Files:** [../00_fundamentals/06_key_metrics_latency_throughput_cost.md](../00_fundamentals/06_key_metrics_latency_throughput_cost.md), [01_observability_and_profiling.md](01_observability_and_profiling.md), [../03_batching_and_scheduling/05_multi_priority_and_SLA_scheduling.md](../03_batching_and_scheduling/05_multi_priority_and_SLA_scheduling.md)
> **Must-Read Papers:** Zhong et al. (2024, OSDI) "DistServe" (goodput); Dean & Barroso (2013) "The Tail at Scale"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- An LLM SLO is defined on **TTFT and ITL percentiles** (e.g., TTFT P99 < 500ms, ITL P99 < 50ms), per use case and tier.
- **Goodput** (throughput meeting the SLO) is the enforcement target — optimize it, not raw throughput ([§00](../00_fundamentals/06_key_metrics_latency_throughput_cost.md)).
- Enforcement levers: **admission control / load shedding**, **chunked prefill / disaggregation** (protect ITL), **priority/WFQ** (per tier), **autoscaling** (capacity), **degradation** (fallback).
- SLOs are **multi-dimensional and conflicting** (TTFT vs ITL vs throughput vs cost) — you choose an operating point.
- **Reasoning models** break TTFT-based SLOs → need new metrics ($/solved-task, time-to-final-answer) ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

***

## Overview
A Service-Level Objective makes "fast enough" precise and measurable, and for LLMs it's expressed on the two latency metrics that matter — **TTFT** (time-to-first-token) and **ITL** (inter-token latency) — at specified **percentiles** for specified use cases. Real-time chat might require TTFT P99 < 500ms and ITL P99 < 50ms (≈20+ tok/s, faster than reading); a coding assistant tighter; batch/offline jobs essentially none (throughput-optimized). Defining the SLO at a **percentile** (P99, not mean) is essential because users feel the tail (Dean & Barroso), and per-tier because different customers buy different guarantees ([§03 multi-priority](../03_batching_and_scheduling/05_multi_priority_and_SLA_scheduling.md)).

The right enforcement target is **goodput** — the rate of requests served *within* the SLO — not raw throughput, which can be high while violating the SLO for everyone ([§00](../00_fundamentals/06_key_metrics_latency_throughput_cost.md)). DistServe (Zhong et al. 2024) crystallized goodput as the optimization objective, and it should be the headline KPI ([§01](01_observability_and_profiling.md)). The system's job is to **maximize goodput-per-dollar** subject to the SLOs, which is exactly what the techniques throughout this database serve.

Enforcement combines several levers: **admission control / load shedding** (reject or queue-with-deadline rather than admit work that would breach everyone's SLO); **chunked prefill / disaggregation** to protect ITL from prefill interference ([§03](../03_batching_and_scheduling/02_chunked_prefill.md)); **priority/WFQ** to meet per-tier SLOs ([§03 multi-priority](../03_batching_and_scheduling/05_multi_priority_and_SLA_scheduling.md)); **autoscaling** to add capacity ([§02](02_autoscaling_and_capacity_planning.md)); and **graceful degradation** (fallback to a smaller model) under overload ([§00](00_production_serving_architecture.md)). Because the SLO dimensions conflict (tighter latency → smaller batch → lower throughput → higher cost), you choose an operating point on the frontier. Finally, **reasoning models** break the TTFT-centric framing (TTFT is irrelevant when output is the thinking) and need new SLO definitions like time-to-final-answer or $/solved-task ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)). This file covers defining, measuring, and enforcing SLOs.

***

## Core Concepts & Mechanics

### Defining the SLO
| Use case | TTFT | ITL | Posture |
|---|---|---|---|
| Real-time chat | P99 < 300–500ms | P99 < 50ms | latency-first |
| Coding assistant | P99 < 1s | < 30–50ms | latency-first |
| Batch/offline | none | none | throughput-first |
| Reasoning | (TTFT N/A) | cost/step; time-to-answer | cost/throughput-first |
Define at **percentiles** (P99/P999), **per tier/tenant**.

### Goodput as the target
📐 **Goodput** = requests/s (or tokens/s) meeting TTFT *and* ITL SLOs. Optimize goodput-per-dollar, not raw throughput. A high-throughput system at batch=256 violating ITL has near-zero goodput.

### Enforcement levers
- **Admission control / load shedding**: don't admit work that breaches the SLO; queue with deadlines.
- **Chunked prefill / disaggregation**: protect ITL from prefill interference ([§03](../03_batching_and_scheduling/02_chunked_prefill.md), [§03 disagg](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md)).
- **Priority / WFQ**: per-tier SLO guarantees ([§03 multi-priority](../03_batching_and_scheduling/05_multi_priority_and_SLA_scheduling.md)).
- **Autoscaling**: add capacity for load ([§02](02_autoscaling_and_capacity_planning.md)).
- **Degradation**: fallback to smaller model under overload ([§03](03_model_routing_and_cascading.md)).

### Conflicting dimensions
Tighter latency → smaller batch → lower throughput → higher $/token. Choose an operating point on the latency-throughput-cost frontier per SLO.

***

## Key Challenges
1. **Tail enforcement.** Meeting P99 (not mean) requires controlling interference, preemption, cold start — the tail causes.
2. **Conflicting objectives.** TTFT vs ITL vs throughput vs cost can't all be maxed; choosing the operating point is a business decision.
3. **Multi-tier SLOs.** Different guarantees on shared hardware need priority/WFQ + isolation ([§04](04_multi_tenant_serving.md)).
4. **Reasoning models.** TTFT-based SLOs don't apply; new metrics needed ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

***

## Solutions & Current Best Practices
- **Define SLOs as TTFT+ITL percentiles per tier**; track **goodput** as the KPI ([§01](01_observability_and_profiling.md)).
- **Protect ITL** with chunked prefill/disaggregation; **TTFT** with prefix caching + admission ([§02 prefix](../02_kv_cache/02_prefix_caching_and_radix_attention.md)).
- **Priority/WFQ + admission control + autoscaling + degradation** to enforce under load.
- **Define reasoning SLOs** on time-to-answer / $-per-solved-task ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

***

## Implementation Notes
- Measure and alert on **per-tier P99 TTFT/ITL and goodput**, not averages/aggregates.
- Set admission/load-shedding thresholds to protect the SLO under burst; queue with deadlines.
- Tune batch/token-budget so worst-case iteration meets ITL ([§03](../03_batching_and_scheduling/02_chunked_prefill.md)).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Met mean latency, breached P99 SLO** — defined/optimized the wrong statistic; SLOs are percentiles.
- **High throughput, ~zero goodput** — batch too large violated ITL; optimize goodput, cap batch.
- **No admission control → everyone breached under burst** — admit work that can't meet SLO; add shedding/queue-with-deadline.
- **Applied chat SLO to a reasoning model** — TTFT meaningless; redefine on time-to-answer/cost.

***

## Performance Numbers & Benchmarks
| Metric | Target example | Lever |
|---|---|---|
| TTFT P99 | < 500ms (chat) | prefix cache, admission, prefill priority |
| ITL P99 | < 50ms (chat) | chunked prefill, batch cap, disaggregation |
| Goodput | maximize @ SLO | all of the above |
| $/token @ SLO | minimize | batch/quantization within SLO |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How do you define an SLO for an LLM chat service?"* — Expected: TTFT+ITL percentiles per tier; goodput as target.
- *"Why optimize goodput not throughput?"* — Expected: SLO-conditioned; high throughput can violate ITL → unusable.
- *"What levers enforce ITL SLO under load?"* — Expected: chunked prefill/disaggregation, batch cap, admission, priority.
- *"How do SLOs change for reasoning models?"* — Expected: TTFT irrelevant; time-to-answer / $-per-solved-task.

***

## Open Problems & Active Research (2025–2026)
- **Goodput-optimal control** jointly meeting TTFT+ITL under bursty, heterogeneous load.
- **SLO frameworks for reasoning/agentic** workloads (new metrics).
- **Automated operating-point selection** on the latency-throughput-cost frontier.

***

## References
- Zhong, Y., et al. (2024). "DistServe" (goodput). *OSDI 2024*. arXiv:2401.09670.
- Dean, J., Barroso, L. (2013). "The Tail at Scale." *CACM* 56(2).
- Little, J.D.C. (1961). "L = λW." *Operations Research* 9(3).
