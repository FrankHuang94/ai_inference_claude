# Autoscaling and Capacity Planning

> **Section:** 13_production_systems
> **Last Updated:** June 2026
> **Related Files:** [05_cold_start_and_model_loading.md](05_cold_start_and_model_loading.md), [00_production_serving_architecture.md](00_production_serving_architecture.md), [../00_fundamentals/06_key_metrics_latency_throughput_cost.md](../00_fundamentals/06_key_metrics_latency_throughput_cost.md)
> **Must-Read Papers:** Little (1961) "L=λW"; Gandhi et al. (2012) "Autoscale"; industry capacity-planning practice
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- Capacity planning uses **Little's Law** (`L=λW`) + queueing theory to size replicas for a target SLO at a given load, with utilization margin (ρ≈0.5–0.7) to protect P99.
- **Autoscaling** adjusts replica count (DP) with load — but **cold start (30–60s)** means you can't react instantly, forcing **predictive scaling** and **warm pools** ([§05](05_cold_start_and_model_loading.md)).
- Scale on **goodput/queue/SLO signals**, not raw GPU util (which is misleading for decode).
- Over-provisioning protects SLOs but raises $/token (utilization↓); the core tension ([§01](../01_hardware/06_tco_and_cost_modeling.md)).
- Reasoning/variable-length workloads make demand unpredictable → harder scaling ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

***

## Overview
Capacity planning answers "how many GPUs/replicas do I need?" and autoscaling answers "how do I adjust that with load?" — both governed by queueing theory and complicated by cold start. The foundation is **Little's Law** ([§00](../00_fundamentals/06_key_metrics_latency_throughput_cost.md)): for arrival rate λ and time-in-system W (bounded by the SLO), the in-flight concurrency `L = λW` must be served by your aggregate capacity. Map L to replicas via each replica's safe concurrent capacity (max batch within ITL SLO), and add a **utilization margin** — because as utilization ρ→1, queueing delay explodes (M/M/1: `W=1/(μ−λ)`), so you provision for ρ≈0.5–0.7 to protect P99. Under-provisioning breaches SLOs under bursts; over-provisioning wastes capex (lower utilization → higher $/token, [§01](../01_hardware/06_tco_and_cost_modeling.md)). The whole game is finding the margin that meets P99 at minimum cost.

**Autoscaling** then tracks demand by adding/removing DP replicas. The defining constraint unique to LLM serving is **cold start**: loading a large model into GPU memory takes 30–60+ seconds ([§05](05_cold_start_and_model_loading.md)), so reactive autoscaling (scale up *after* load rises) arrives too late for a spike — by the time the new replica is ready, the spike may have already breached SLOs. This forces **predictive/proactive scaling** (forecast load from time-of-day/trends and pre-warm) and **warm pools** (keep spare loaded replicas ready), trading idle cost for responsiveness.

The right **scaling signal** matters: scaling on raw GPU utilization is misleading (decode runs at low MFU even when saturated, [§01](../01_hardware/00_gpu_architecture_for_inference.md)); better signals are **queue depth, goodput, in-flight tokens, or SLO headroom**. And **demand predictability** is increasingly a problem: reasoning/agentic workloads with variable per-query compute ([§12](../12_reasoning_model_inference/03_dynamic_compute_allocation.md)) make load harder to forecast, widening the margin needed. This file covers the capacity math, autoscaling under cold start, scaling signals, and the cost tension.

***

## Core Concepts & Mechanics

### Capacity math (Little's Law + queueing)
📐 `L = λW`. Replicas needed `≈ L / (per_replica_concurrency)`, where per-replica concurrency = max batch meeting ITL SLO. Add margin: provision so ρ = λ/(capacity·μ) ≈ 0.5–0.7. M/M/1 intuition: `W = 1/(μ−λ)` → delay blows up near ρ=1, so margin protects P99.

Example: λ=100 req/s, SLO W=2s → L=200 in-flight; replica handles 50 → 4 replicas for concurrency, ~6–8 with margin/redundancy.

### Autoscaling under cold start
- **Reactive** (scale on current load): too slow given 30–60s cold start; spikes breach SLO before capacity arrives.
- **Predictive/proactive**: forecast (time-of-day, trends, events) and pre-scale.
- **Warm pools**: keep spare loaded replicas (idle cost) for instant activation ([§05](05_cold_start_and_model_loading.md)).

### Scaling signals
Scale on **queue depth / goodput / in-flight tokens / SLO headroom**, not GPU util% (misleading for decode). Set thresholds with hysteresis to avoid flapping.

### Cost tension
📐 $/token ∝ 1/utilization ([§01](../01_hardware/06_tco_and_cost_modeling.md)). Margin/warm pools lower utilization → higher cost. Optimize **goodput-per-dollar**: smallest margin that meets P99.

***

## Key Challenges
1. **Cold start defeats reactive scaling.** Slow model load means you must predict/pre-warm; reactive is too late for spikes.
2. **Margin vs cost.** Protecting P99 needs headroom (low ρ) → low utilization → high $/token; the central tradeoff.
3. **Demand unpredictability.** Bursty and (for reasoning) variable-compute traffic is hard to forecast, widening required margin.
4. **Right signal/flapping.** GPU util misleads; thresholds without hysteresis cause oscillation (scale up/down churn, each paying cold start).

***

## Solutions & Current Best Practices
- **Size with Little's Law + queueing margin** (ρ≈0.5–0.7) for P99.
- **Predictive autoscaling + warm pools** to mask cold start ([§05](05_cold_start_and_model_loading.md)).
- **Scale on queue/goodput/in-flight-tokens**, with hysteresis.
- **Optimize goodput-per-dollar**; use spot/committed mixes and backfill batch work into troughs ([§01](../01_hardware/06_tco_and_cost_modeling.md)).

***

## Implementation Notes
- Measure per-replica safe concurrency (max batch within ITL SLO) empirically; feed into capacity math.
- Forecast load (time-series) and pre-warm; keep a small warm pool sized to spike slope.
- Use queue-depth/goodput-based autoscalers (e.g., KEDA-style custom metrics), not CPU/GPU-util triggers.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Reactive autoscaler missed the spike** — 60s cold start; SLO breached before capacity arrived. Predict + warm pool.
- **Scaled on GPU util, never triggered** — decode runs low MFU even when saturated; util didn't reflect load. Use queue/goodput.
- **Autoscaler flapped** — no hysteresis; churned replicas, each paying cold start. Add cooldown/hysteresis.
- **Margin too thin, P99 exploded under burst** — ρ near 1; provision more headroom.

***

## Performance Numbers & Benchmarks
| Quantity | Rule |
|---|---|
| In-flight concurrency | L = λW (Little's Law) |
| Replicas | L / per-replica concurrency × margin |
| Utilization target | ρ ≈ 0.5–0.7 (protect P99) |
| Cold start | 30–60s → predictive + warm pool |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Size a cluster for 100 req/s at 2s E2E SLO."* — Expected: L=λW=200 in-flight, map to replicas, add margin/redundancy.
- *"Why can't you just reactively autoscale LLM serving?"* — Expected: cold start (30–60s); need predictive + warm pools.
- *"What signal do you autoscale on and why not GPU util?"* — Expected: queue/goodput/in-flight tokens; util misleads for decode.
- *"What's the cost tension in capacity planning?"* — Expected: margin protects P99 but lowers utilization → higher $/token.

***

## Open Problems & Active Research (2025–2026)
- **Fast cold start** to enable true reactive elasticity ([§05](05_cold_start_and_model_loading.md)).
- **Forecasting variable-compute (reasoning) demand** for tighter margins.
- **Goodput-per-dollar-optimal autoscaling** with spot/committed mixes.

***

## References
- Little, J.D.C. (1961). "L = λW." *Operations Research* 9(3).
- Gandhi, A., et al. (2012). "AutoScale: Dynamic, Robust Capacity Management." *ACM TOCS*.
- Industry capacity-planning / SRE practice.
