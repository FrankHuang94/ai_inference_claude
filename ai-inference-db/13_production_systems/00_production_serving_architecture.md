# Production Serving Architecture (Full Stack)

> **Section:** 13_production_systems
> **Last Updated:** June 2026
> **Related Files:** [02_autoscaling_and_capacity_planning.md](02_autoscaling_and_capacity_planning.md), [05_cold_start_and_model_loading.md](05_cold_start_and_model_loading.md), [../08_serving_frameworks/00_serving_system_architecture.md](../08_serving_frameworks/00_serving_system_architecture.md)
> **Must-Read Papers:** Crankshaw et al. (2017, NSDI) "Clipper"; Qin et al. (2024) "Mooncake"; Dean & Barroso (2013) "The Tail at Scale"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- A production LLM service is more than the engine: **API gateway → auth/rate-limit → router → inference cluster (engines) → KV/cache layer → response**, with observability and autoscaling around it.
- The request journey adds **gateway, routing, multi-tenancy, and fallback** concerns on top of the serving engine ([§08](../08_serving_frameworks/00_serving_system_architecture.md)).
- **Cold start** (loading a 70B model takes 30–60s) shapes autoscaling and availability ([§05](05_cold_start_and_model_loading.md)).
- **Graceful degradation** (fallback to smaller model, shed load) keeps SLOs under overload.
- Multi-region/multi-datacenter for latency and availability; the KV/cache layer (Mooncake-style) increasingly central.

***

## Overview
A production LLM serving system wraps the inference engine ([§08](../08_serving_frameworks/00_serving_system_architecture.md)) in a full request-handling stack. The journey: a client request hits an **API gateway** (TLS, OpenAI-compatible protocol), passes **authentication, rate limiting, and quota** checks, is **routed** to an appropriate inference cluster/replica (cache-aware, load-aware, tier-aware — [§09](../09_distributed_inference/01_load_balancing_strategies.md)), processed by an **inference engine** (continuous batching, paged KV), possibly using a shared **KV/cache layer**, and streamed back. Around this sit **observability** ([§01](01_observability_and_profiling.md)), **autoscaling** ([§02](02_autoscaling_and_capacity_planning.md)), and **fault tolerance** ([§09](../09_distributed_inference/05_fault_tolerance_in_serving.md)). The engine is the heart, but production reliability lives in these surrounding layers.

Several concerns are unique to (or amplified in) production. **Cold start**: loading a large model into GPU memory takes 30–60+ seconds ([§05](05_cold_start_and_model_loading.md)), so you can't scale instantly to a traffic spike — this forces warm pools, predictive autoscaling, and over-provisioning. **Graceful degradation**: under overload, the system must shed load, queue, or **fall back to a smaller/cheaper model** rather than violate SLOs for everyone or fall over — a deliberate quality-vs-availability tradeoff. **Multi-tenancy**: many customers share infrastructure with isolation, fairness, and per-tenant SLOs ([§04](04_multi_tenant_serving.md)). **Multi-region**: deploy across datacenters for latency (serve near users) and availability (survive region failure), with routing and data/cache considerations.

The architectural trend is toward a **KV-cache-centric** design ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)): as prefix caching, disaggregation, and reasoning workloads make the KV cache central, a shared KV/cache layer (Mooncake-style) becomes a first-class component spanning the cluster, decoupling KV from individual engines and enabling cross-instance reuse and disaggregation. This file gives the end-to-end architecture, the request journey, and the production-specific concerns (cold start, degradation, multi-region) that the rest of the section details.

***

## Core Concepts & Mechanics

### The request journey
```
Client → API Gateway (TLS, protocol)
       → Auth / Rate-limit / Quota
       → Router (cache-aware, load-aware, tier-aware) [§09]
       → Inference Engine (continuous batching, paged KV) [§08]
       → KV/Cache layer (prefix cache, global KV pool) [§02,§09]
       → stream response
   (around: observability [§01], autoscaling [§02], fault tolerance [§09])
```

### Production-specific components
- **API gateway**: protocol, TLS, request validation, streaming (SSE).
- **Auth/rate-limit/quota**: per-tenant keys, token buckets ([§05 multi-priority](../03_batching_and_scheduling/05_multi_priority_and_SLA_scheduling.md)).
- **Router**: cache/load/tier-aware request distribution ([§09](../09_distributed_inference/01_load_balancing_strategies.md)).
- **Inference cluster**: DP replicas (each TP/PP) of engines ([§04](../04_parallelism/03_data_parallelism_for_serving.md)).
- **KV/cache layer**: prefix cache, global KV pool ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).

### Cold start & degradation
- **Cold start**: 30–60s model load → warm pools, predictive scaling ([§05](05_cold_start_and_model_loading.md)).
- **Graceful degradation**: under overload, shed/queue/fallback-to-smaller-model to protect SLOs.

### Multi-region
Deploy across datacenters for latency (geo-routing) and availability (region failover); replicate models, route to nearest healthy region.

***

## Key Challenges
1. **Cold start vs elasticity.** Can't scale instantly (slow model load); spikes require warm capacity / prediction ([§05](05_cold_start_and_model_loading.md)).
2. **SLO protection under overload.** Must degrade gracefully (shed/fallback) rather than violate all SLOs or crash.
3. **End-to-end latency budget.** Gateway, auth, routing, retrieval all add to the budget beyond the engine; each must be lean.
4. **Multi-region complexity.** Geo-routing, replication, cache locality, and failover across datacenters.

***

## Solutions & Current Best Practices
- **Warm pools + predictive autoscaling** to mask cold start ([§02](02_autoscaling_and_capacity_planning.md), [§05](05_cold_start_and_model_loading.md)).
- **Graceful degradation**: load shedding, queueing with deadlines, fallback to smaller models ([§03](03_model_routing_and_cascading.md)).
- **Cache/load/tier-aware routing** and a **KV/cache layer** ([§09](../09_distributed_inference/01_load_balancing_strategies.md)).
- **Multi-region** deployment with geo-routing and failover; lean gateway/auth on the hot path.

***

## Implementation Notes
- Keep gateway/auth/routing overhead minimal (they're on every request's latency budget).
- Instrument the full journey (not just the engine) for end-to-end latency attribution ([§01](01_observability_and_profiling.md)).
- Define degradation policies explicitly (when to shed, queue, or fall back) and test them.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Spike caused outage because scaling couldn't keep up** — 60s cold start; needed warm pool / predictive scaling.
- **Gateway/auth added 100ms** — non-engine overhead ate the latency budget; optimize the hot path.
- **No degradation policy → cascading failure** — overload violated all SLOs and crashed; add shed/queue/fallback.
- **Region failure took down the service** — no multi-region failover; replicate and geo-route.

***

## Performance Numbers & Benchmarks
| Layer | Latency contribution | Concern |
|---|---|---|
| Gateway/auth | ~ms | keep lean |
| Routing | ~ms | cache/load-aware |
| Retrieval (RAG) | ~tens ms | adds to TTFT |
| Engine (prefill+decode) | dominant | the core |
| Cold start | 30–60s | scaling/availability |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Design the full production architecture for an LLM API."* — Expected: gateway→auth→router→engine cluster→KV layer; observability, autoscaling, degradation, multi-region.
- *"How does cold start shape your architecture?"* — Expected: warm pools, predictive scaling, can't scale instantly.
- *"How do you protect SLOs under overload?"* — Expected: load shedding, queueing, fallback to smaller model.
- *"What's becoming a first-class component in modern serving?"* — Expected: the KV/cache layer (Mooncake-style).

***

## Open Problems & Active Research (2025–2026)
- **Sub-second cold start** (fast loading, snapshotting) for true elasticity ([§05](05_cold_start_and_model_loading.md)).
- **Global KV/cache layers** as standard infrastructure ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).
- **Automated degradation/routing policies** optimizing goodput under overload.

***

## References
- Crankshaw, D., et al. (2017). "Clipper." *NSDI 2017*.
- Qin, R., et al. (2024). "Mooncake." arXiv:2407.00079.
- Dean, J., Barroso, L. (2013). "The Tail at Scale." *CACM* 56(2).
