# Multi-Tenant Serving

> **Section:** 13_production_systems
> **Last Updated:** June 2026
> **Related Files:** [../03_batching_and_scheduling/05_multi_priority_and_SLA_scheduling.md](../03_batching_and_scheduling/05_multi_priority_and_SLA_scheduling.md), [../02_kv_cache/05_cross_request_kv_sharing.md](../02_kv_cache/05_cross_request_kv_sharing.md), [06_SLO_definition_and_enforcement.md](06_SLO_definition_and_enforcement.md)
> **Must-Read Papers:** Sheng et al. (2023, OSDI) "S-LoRA"; Chen et al. (2023) "Punica"; Demers et al. (1989) "Weighted Fair Queueing"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- Multi-tenant serving shares one fleet across many customers, requiring **isolation** (data + timing), **fairness** (no noisy neighbors), and **per-tenant SLOs**.
- **S-LoRA** (Sheng et al. 2023): serve **one base model + hundreds/thousands of LoRA adapters** concurrently — swap small adapters, not whole models, for per-customer fine-tunes.
- Fairness via **token-bucket rate limiting + weighted fair queueing**; isolation via **per-tenant KV/quotas** ([§03 multi-priority](../03_batching_and_scheduling/05_multi_priority_and_SLA_scheduling.md)).
- **Privacy**: shared prefix caching across tenants risks **data leakage / timing side-channels** — scope to public content only ([§02](../02_kv_cache/05_cross_request_kv_sharing.md)).
- **Noisy neighbor**: one tenant's burst/long requests must not starve others.

***

## Overview
Inference clouds and internal platforms serve many customers on shared GPUs to achieve the high utilization that makes the economics work ([§01](../01_hardware/06_tco_and_cost_modeling.md)). Multi-tenancy introduces three requirements absent in single-tenant serving: **isolation** (one tenant must not see another's data, nor infer it via timing), **fairness** (no tenant monopolizes resources — the "noisy neighbor" problem), and **per-tenant SLOs/quotas** (different customers pay for different guarantees). These are classic cloud problems specialized to LLM serving's quirks (KV cache, prefix sharing, LoRA).

The standout LLM-specific capability is **multi-LoRA serving**. Customers often want fine-tuned models, but loading a separate full model per customer is infeasible. **S-LoRA** (Sheng et al. 2023) and **Punica** (Chen et al. 2023) serve **one shared base model plus many small LoRA adapters** concurrently: the base weights are shared across all tenants, and each request applies its tenant's lightweight adapter (low-rank deltas) via batched LoRA kernels. This lets a single deployment serve hundreds to thousands of per-customer fine-tunes with near-base-model efficiency — a huge multi-tenant cost win, and a frequent interview topic. The serving challenge is **batched heterogeneous-adapter execution** (different requests in a batch use different adapters) and adapter memory management.

The **fairness and isolation** mechanisms come from scheduling and resource management ([§03 multi-priority](../03_batching_and_scheduling/05_multi_priority_and_SLA_scheduling.md)): **token-bucket rate limiting** per tenant, **weighted fair queueing** to guarantee shares, **per-tenant KV/quota** caps, and **preemption** to protect premium SLOs. The subtle **privacy** hazard is **cross-tenant prefix caching**: sharing KV for a common system prompt is fine, but sharing user-specific content can leak data, and even a cache *hit* (faster response) is a **timing side-channel** revealing that another tenant sent the same prefix ([§02](../02_kv_cache/05_cross_request_kv_sharing.md)). Production systems scope cross-tenant sharing to genuinely public content and isolate user data. This file covers S-LoRA, fairness/isolation, and the privacy pitfalls.

***

## Core Concepts & Mechanics

### Multi-LoRA serving (S-LoRA / Punica)
- **Base model shared**; each tenant has a small **LoRA adapter** (low-rank `BA`, e.g., rank 8–64).
- Per request: `Wx + B(Ax)` — base GEMM (shared) + adapter delta (small). **Batched LoRA kernels** apply different adapters to different requests in one batch (S-LoRA's heterogeneous batching, custom CUDA).
- Adapters paged in/out of GPU memory (small); serve hundreds–thousands concurrently.
- 📐 Memory: base (once) + Σ adapters (tiny each) ≪ N full models.

### Fairness & isolation
- **Token-bucket rate limiting** per tenant (rate r, burst b).
- **Weighted fair queueing**: guaranteed capacity shares per tenant/tier ([§03 multi-priority](../03_batching_and_scheduling/05_multi_priority_and_SLA_scheduling.md)).
- **Per-tenant KV/quota** caps to bound resource use.
- **Preemption** to protect premium SLOs from noisy neighbors.

### Privacy pitfalls
- **Cross-tenant prefix caching**: safe for public/shared system prompts; **unsafe** for user content (data leak).
- **Timing side-channel**: a cache hit → faster TTFT → reveals someone else sent that prefix. Scope sharing; consider per-tenant cache partitions.

***

## Key Challenges
1. **Noisy neighbor.** One tenant's burst/long requests can starve others without rate limiting + fair scheduling.
2. **Heterogeneous-adapter batching.** Applying different LoRA adapters to different requests in one batch efficiently (S-LoRA kernels) is non-trivial.
3. **Privacy/side-channels.** Cross-tenant KV/prefix sharing risks data leakage and timing channels; must scope carefully.
4. **Per-tenant SLO enforcement.** Differentiated guarantees on shared hardware require WFQ/priority + isolation ([§06](06_SLO_definition_and_enforcement.md)).

***

## Solutions & Current Best Practices
- **S-LoRA/Punica** for per-customer fine-tunes (base + many adapters) — the multi-tenant cost win.
- **Token-bucket + WFQ + per-tenant quotas + preemption** for fairness/isolation ([§03 multi-priority](../03_batching_and_scheduling/05_multi_priority_and_SLA_scheduling.md)).
- **Scope cross-tenant sharing to public content only**; partition/disable user-content sharing; mitigate timing channels ([§02](../02_kv_cache/05_cross_request_kv_sharing.md)).
- **Per-tenant goodput monitoring** ([§01](01_observability_and_profiling.md)).

***

## Implementation Notes
- vLLM (LoRAX) / S-LoRA support multi-LoRA; manage adapter loading/eviction (adapters are small but numerous).
- Enforce per-tenant rate limits at the gateway; WFQ in the scheduler.
- For prefix caching, allowlist shared (public) prefixes; never share user-specific KV across tenants.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **One tenant's burst starved everyone** — noisy neighbor; add token-bucket + WFQ + per-tenant quotas.
- **Cross-tenant prefix cache leaked / side-channeled** — shared user content or timing revealed others' prompts; scope to public, partition.
- **Adapter-switch overhead hurt throughput** — frequent LoRA swaps; batch by adapter / keep hot adapters resident.
- **Premium tenant breached SLO under shared load** — no priority/preemption; add WFQ + preemption ([§03 multi-priority](../03_batching_and_scheduling/05_multi_priority_and_SLA_scheduling.md)).

***

## Performance Numbers & Benchmarks
| Capability | Mechanism | Benefit |
|---|---|---|
| Per-customer fine-tunes | S-LoRA (base + adapters) | 100s–1000s tenants, ~base efficiency |
| Fairness | token bucket + WFQ | no noisy neighbor |
| Isolation | per-tenant KV/quota | bounded blast radius |
| Privacy | scoped prefix sharing | no cross-tenant leak |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How do you serve thousands of per-customer fine-tuned models?"* — Expected: S-LoRA — shared base + many small LoRA adapters, batched heterogeneous-adapter kernels.
- *"How do you prevent noisy neighbors?"* — Expected: token-bucket rate limiting + WFQ + per-tenant quotas + preemption.
- *"What are the privacy risks of cross-tenant prefix caching?"* — Expected: data leakage + timing side-channel; scope to public content.
- *"How do you enforce per-tenant SLOs on shared hardware?"* — Expected: WFQ/priority + isolation + per-tenant goodput monitoring.

***

## Open Problems & Active Research (2025–2026)
- **Secure cross-tenant KV sharing** without side-channels.
- **Scaling multi-LoRA** to more adapters / higher ranks efficiently.
- **Provable per-tenant SLO guarantees** under shared, bursty load.

***

## References
- Sheng, Y., et al. (2023). "S-LoRA: Serving Thousands of Concurrent LoRA Adapters." *OSDI 2024*. arXiv:2311.03285.
- Chen, L., et al. (2023). "Punica: Multi-Tenant LoRA Serving." arXiv:2310.18547.
- Demers, A., et al. (1989). "Weighted Fair Queueing." *SIGCOMM 1989*.
