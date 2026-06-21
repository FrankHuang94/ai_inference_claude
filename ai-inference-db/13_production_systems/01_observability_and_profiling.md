# Observability and Profiling in Production

> **Section:** 13_production_systems
> **Last Updated:** June 2026
> **Related Files:** [00_production_serving_architecture.md](00_production_serving_architecture.md), [06_SLO_definition_and_enforcement.md](06_SLO_definition_and_enforcement.md), [../06_kernel_optimization/05_kernel_profiling_and_benchmarking.md](../06_kernel_optimization/05_kernel_profiling_and_benchmarking.md)
> **Must-Read Papers:** Dean & Barroso (2013) "The Tail at Scale"; industry SRE/observability practice; NVIDIA DCGM docs
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- Observe **request-level** (TTFT, ITL, E2E percentiles, goodput per tier), **engine-level** (batch size, KV utilization, queue depth, preemptions), and **hardware-level** (GPU util, MFU, power, HBM, NVLink) metrics.
- Report **percentiles (P50/P90/P99)**, not averages — tails are what SLOs and users feel ([§00](../00_fundamentals/06_key_metrics_latency_throughput_cost.md)).
- **Goodput** (SLO-meeting throughput) is the headline business metric, broken down per tenant/tier.
- **Distributed tracing** to attribute latency across gateway→router→engine→KV; **DCGM/Prometheus** for hardware.
- The hard part is **attribution**: knowing *which layer* caused a P99 spike (interference? preemption? cold start? comm?).

***

## Overview
You cannot operate or optimize a serving system you can't see. Production observability spans three levels. **Request-level**: per-request TTFT, ITL/TPOT, E2E latency (as **percentile histograms**, not means), throughput, and **goodput** (SLO-meeting rate) broken down per tenant/tier/model. **Engine-level**: running batch size, KV-cache utilization, waiting-queue depth, preemption/eviction counts, prefix-cache hit rate, speculative-decoding acceptance — the internal state that explains performance. **Hardware-level**: GPU utilization and **MFU**, power draw, HBM usage, NVLink/IB bandwidth, temperature/throttling — via DCGM/Prometheus. Together these let you detect SLO violations, diagnose causes, and plan capacity.

The recurring methodological points (from [§00](../00_fundamentals/06_key_metrics_latency_throughput_cost.md)): **percentiles over averages** (a fine mean P50 can hide a terrible P99 from interference or preemption — the tail is what matters, Dean & Barroso); **goodput as the headline** (raw throughput is misleading if it violates SLOs); and **per-tier/tenant breakdown** (aggregate metrics hide that the premium tier is suffering). A dashboard that only shows average latency and total throughput is dangerously blind to the failure modes that actually hurt.

The genuinely hard problem is **attribution**: when P99 ITL spikes, *why*? Was it prefill-decode interference ([§00](../00_fundamentals/02_prefill_vs_decode_phases.md)), preemption thrashing ([§02](../02_kv_cache/01_paged_attention_vllm.md)), a cold-start replica ([§05](05_cold_start_and_model_loading.md)), all-reduce contention ([§04](../04_parallelism/00_tensor_parallelism.md)), or a noisy neighbor ([§04](04_multi_tenant_serving.md))? **Distributed tracing** (correlating a request across gateway→router→engine→KV) plus correlated engine/hardware metrics is what turns a symptom into a root cause. And for deep kernel issues, you drop to Nsight ([§06](../06_kernel_optimization/05_kernel_profiling_and_benchmarking.md)). This file covers what to measure, how to report it, and how to attribute problems.

***

## Core Concepts & Mechanics

### The three levels
| Level | Metrics | Tool |
|---|---|---|
| Request | TTFT, ITL, TPOT, E2E (P50/P90/P99), throughput, **goodput**, per-tier/tenant | app metrics, tracing |
| Engine | batch size, KV utilization, queue depth, preemptions, prefix-hit rate, spec-accept | engine metrics (Prometheus) |
| Hardware | GPU util, **MFU**, power, HBM, NVLink/IB BW, temp/throttle | DCGM, Prometheus |

### Percentiles and goodput
📐 Report P50/P90/P99(/P999). **Goodput** = throughput conditioned on meeting TTFT+ITL SLOs ([§06](06_SLO_definition_and_enforcement.md)). Break down per tier/tenant — aggregates hide premium suffering.

### Distributed tracing & attribution
Correlate a request ID across gateway → router → engine → KV layer; align with engine/hardware metrics at that timestamp to attribute a spike to interference / preemption / cold start / comm / noisy neighbor.

### Hardware monitoring
DCGM exports GPU util, MFU-relevant counters, power, HBM, NVLink/IB; but remember **GPU util ≠ MFU ≠ efficiency** ([§01 GPU arch](../01_hardware/00_gpu_architecture_for_inference.md)) — track tokens/s/GPU and goodput, not just util%.

***

## Key Challenges
1. **Attribution.** Mapping a P99 spike to its root cause (interference/preemption/cold-start/comm/neighbor) requires correlated tracing + metrics; hard without it.
2. **Percentile/per-tier blindness.** Averages and aggregates hide tail and per-tenant problems; must track percentiles and breakdowns.
3. **Metric overload vs signal.** Too many metrics obscure; need the right SLO-aligned KPIs (goodput, P99 TTFT/ITL).
4. **Overhead.** Tracing/profiling must be low-overhead in production (sampling) vs full Nsight (offline).

***

## Solutions & Current Best Practices
- **Track goodput + P50/P90/P99 TTFT/ITL per tier/tenant** as headline KPIs ([§06](06_SLO_definition_and_enforcement.md)).
- **Distributed tracing** (request ID across layers) + correlated engine/hardware metrics for attribution.
- **DCGM/Prometheus + Grafana** for hardware/engine; **tokens/s/GPU**, not util%, for efficiency.
- **Sampled tracing** in prod; **Nsight** offline for kernel deep-dives ([§06](../06_kernel_optimization/05_kernel_profiling_and_benchmarking.md)).

***

## Implementation Notes
- Emit per-request TTFT/ITL histograms and tag with tenant/tier/model.
- Export engine internals (batch, KV util, queue, preemptions, hit/accept rates) — they explain performance.
- Build dashboards around SLOs/goodput; alert on P99 and goodput, not averages.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Average latency green, users complained** — P99 was terrible (interference); track percentiles.
- **Aggregate throughput great, premium tier breached SLO** — no per-tier breakdown; segment goodput.
- **Couldn't find the cause of a spike** — no tracing/attribution; add request-ID tracing + correlated metrics.
- **Chased GPU util% as efficiency** — high util but low MFU/throughput; use tokens/s/GPU and goodput.

***

## Performance Numbers & Benchmarks
| KPI | Why |
|---|---|
| Goodput (per tier) | SLO-meeting throughput; the business metric |
| P99 TTFT / ITL | tail UX; SLO compliance |
| KV utilization / preemptions | engine health; OOM/thrash signals |
| tokens/s/GPU | true efficiency (not util%) |
| MFU | prefill efficiency (not for decode) |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"What would you put on an LLM serving dashboard?"* — Expected: goodput + P50/P90/P99 TTFT/ITL per tier, engine internals (KV/batch/queue/preempt), hardware (tokens/s/GPU, HBM).
- *"P99 ITL spiked — how do you find the cause?"* — Expected: tracing + correlated metrics → interference/preemption/cold-start/comm.
- *"Why report goodput not throughput?"* — Expected: SLO-conditioned; throughput alone can be unusable.
- *"Why isn't GPU util% a good efficiency metric?"* — Expected: util ≠ MFU ≠ useful work; use tokens/s/GPU.

***

## Open Problems & Active Research (2025–2026)
- **Automated root-cause attribution** for latency spikes from traces+metrics.
- **Low-overhead always-on profiling** approaching Nsight detail.
- **Goodput-centric, SLO-aware** observability standards for LLM serving.

***

## References
- Dean, J., Barroso, L. (2013). "The Tail at Scale." *CACM* 56(2).
- NVIDIA. "DCGM (Data Center GPU Manager)" documentation.
- Industry SRE/observability practice (Prometheus, OpenTelemetry, Grafana).
