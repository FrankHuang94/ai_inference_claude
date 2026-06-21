# Key Serving Metrics: Latency, Throughput, and Cost

> **Section:** 00_fundamentals
> **Last Updated:** June 2026
> **Related Files:** [02_prefill_vs_decode_phases.md](02_prefill_vs_decode_phases.md), [../13_production_systems/06_SLO_definition_and_enforcement.md](../13_production_systems/06_SLO_definition_and_enforcement.md), [../01_hardware/06_tco_and_cost_modeling.md](../01_hardware/06_tco_and_cost_modeling.md)
> **Must-Read Papers:** Zhong et al. (2024, OSDI) "DistServe" (goodput); Little (1961) "A Proof for the Queuing Formula L=λW"; Agrawal et al. (2024, OSDI) "Sarathi-Serve"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- **TTFT** (time-to-first-token) = queue + prefill; **ITL/TPOT** (inter-token / time-per-output-token) = decode step latency; **E2E = TTFT + (out_len−1)·ITL**.
- **Throughput** (tokens/s, req/s) and **goodput** (throughput that *meets SLOs*) are the cost-side metrics; goodput is what actually matters in production.
- **MFU** (model FLOP utilization) is a useful efficiency number for prefill/training but **misleading for decode**; use **tokens/s/GPU** and **$/M tokens** instead.
- **Latency and throughput trade off**: bigger batches raise throughput but raise TTFT/ITL. You pick a point per SLO; you cannot maximize both.
- **Little's Law** `L = λW` governs queue occupancy and is the basis of capacity planning.

***

## Overview
Serving is judged on three axes — **latency** (user experience), **throughput** (capacity), and **cost** (unit economics) — and they are in tension. A precise, shared vocabulary for these metrics is essential because optimizations that improve one often harm another, and because SLOs are written in these terms. This file defines each metric, the relationships among them, and the queuing math that ties offered load to latency.

The central operational truth is the **latency–throughput frontier**: for a given model and hardware, you can run small batches (low latency, low throughput, high $/token) or large batches (high latency, high throughput, low $/token), tracing a Pareto frontier. Production systems choose an operating point on this frontier to satisfy SLOs at minimum cost, and the entire scheduling/batching apparatus (see [§03](../03_batching_and_scheduling/00_static_vs_continuous_batching.md)) exists to push that frontier outward and to keep the system at the chosen point under varying load.

Finally, the right *cost* metric is **goodput-aware**: throughput that violates SLOs is worthless (those responses are too slow to use), so the metric the field has converged on for system comparison is **goodput** — requests/second served *within* TTFT and ITL targets — and its economic shadow, **$/M tokens at SLO**.

***

## Core Concepts & Mechanics

### The latency metrics
- **TTFT** = arrival → first token = `queue_wait + prefill_time (+ KV_transfer if disaggregated)`. Dominated by prompt length and load.
- **ITL** (inter-token latency) / **TPOT** (time per output token) = steady-state time between output tokens ≈ decode iteration time. (ITL is per-gap; TPOT often reported as the mean.)
- **E2E latency** = `TTFT + (output_len − 1) × ITL`. 📐 For long outputs decode dominates; for long prompts/short outputs prefill dominates.
- **Percentiles**: report **P50/P90/P99** (and sometimes P999). Tails are set by interference, preemption, and queueing; P99 ITL is the classic victim of prefill-decode interference ([§00](02_prefill_vs_decode_phases.md)).

### Throughput, goodput, utilization
- **Throughput**: output tokens/s (or req/s, or total tokens/s incl. prompt). The capacity number.
- **Goodput**: tokens/s (or req/s) served *within SLO*. DistServe (Zhong et al. 2024) popularized goodput as the optimization target — a system can have high throughput but low goodput if it meets neither TTFT nor ITL SLOs under load.
- **GPU utilization** (SM busy %) vs **MFU** (achieved FLOPs / peak FLOPs). 📐 `MFU = (6·N·tokens/s) / peak_FLOPs` (the 6N is fwd+bwd for training; use 2N for inference forward). For decode, MFU is tiny (<10%) by nature — *do not* treat low decode MFU as a bug.

### Cost metrics
- **$/M tokens** = `GPU_hourly_cost / (tokens_per_hour)` (split input vs output pricing because output tokens cost far more — they're decode, the expensive phase). See [§01](../01_hardware/06_tco_and_cost_modeling.md).
- Output tokens are typically priced 2–5× input tokens because decode is bandwidth-bound and serial while prefill is amortized over the prompt.

### Little's Law and queueing
📐 **Little's Law**: `L = λ · W`, where L = average number of requests in the system, λ = arrival rate, W = average time in system. Applied to inference:
- If you must keep `W` (E2E latency) under an SLO and λ is the offered load, then the in-flight concurrency `L = λW` tells you the **batch/replica capacity** you must provision.
- Example: λ = 100 req/s, target W = 2 s ⇒ L = 200 concurrent requests must be in flight; if one GPU sustains a max in-flight batch of 50 within ITL SLO, you need ≥4 GPUs just for concurrency (before redundancy).
- As utilization ρ→1, queueing delay blows up (M/M/1: `W = 1/(μ−λ)`), so you provision for ρ well below 1 (often 0.5–0.7) to protect P99.

### The latency–throughput tradeoff
Increasing batch size B raises tokens/s (throughput) but each iteration takes longer (higher ITL) and prefills wait longer in queue (higher TTFT). The optimal B is the largest that still meets ITL/TTFT SLOs — found empirically and enforced by the scheduler's token budget per iteration.

***

## Key Challenges
1. **SLOs are multi-dimensional and conflicting.** TTFT and ITL pull batching in opposite directions; satisfying both simultaneously under bursty load is the core scheduling problem.
2. **Tail latency is hard.** P99 is dominated by rare interference/preemption events, not the mean; optimizing average latency can leave P99 untouched or worse.
3. **MFU misleads for decode.** Teams that optimize MFU on decode-heavy traffic chase the wrong target and miss the real levers (batch, KV memory, quantization).
4. **Cost attribution across phases.** Input vs output token costs differ greatly; flat per-token accounting misprices reasoning/long-output workloads.

***

## Solutions & Current Best Practices
- **Define SLOs as TTFT and ITL percentiles** (e.g., real-time chat: TTFT P99 < 500 ms, ITL P99 < 50 ms; batch jobs: throughput-optimized, latency relaxed) and optimize **goodput** against them ([§13](../13_production_systems/06_SLO_definition_and_enforcement.md)).
- **Chunked prefill / disaggregation** to protect ITL while serving long prefills ([§03](../03_batching_and_scheduling/02_chunked_prefill.md)).
- **Capacity planning via Little's Law + queueing margins**, provisioning for ρ ≈ 0.5–0.7 ([§13](../13_production_systems/02_autoscaling_and_capacity_planning.md)).
- **Report $/M tokens split by input/output** for honest unit economics ([§01](../01_hardware/06_tco_and_cost_modeling.md)).

***

## Implementation Notes
- Instrument **per-request TTFT and per-token ITL histograms**, not just averages; alert on P99.
- Track **goodput** as a first-class dashboard metric: throughput conditioned on SLO compliance.
- Separate **prefill tokens/s** and **decode tokens/s** in monitoring; they have different physics and different regressions.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **High throughput, low goodput.** A system maxing tokens/s with batch=256 may violate ITL SLO for everyone — throughput looks great, product is unusable. Always condition on SLO.
- **Mean ITL fine, P99 terrible.** A long prefill every few seconds spikes the tail; the average hides it. Track percentiles and the cause (interference).
- **Little's Law applied with the wrong W.** Use E2E time-in-system including queue, not just service time, or you under-provision and tails explode as ρ→1.
- **Comparing $/M tokens without context length / batch.** Cost is meaningless without the workload's prompt/output distribution and SLO; vendors quote best-case batch-saturated numbers.

***

## Performance Numbers & Benchmarks
| Use case | TTFT target | ITL target | Optimization posture |
|---|---|---|---|
| Real-time chat | P99 < 300–500 ms | P99 < 50 ms (≈20+ tok/s) | latency-first, modest batch |
| Coding assistant | P99 < 1 s | < 30–50 ms | latency-first |
| Batch/offline | seconds–minutes OK | relaxed | throughput-first, max batch |
| Reasoning (long CoT) | TTFT irrelevant | cost/step matters | throughput + cost-first |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Define TTFT, ITL, and goodput, and explain why goodput is the right system metric."* — Expected: SLO-conditioned throughput; throughput alone can be unusable.
- *"Apply Little's Law to size a cluster for 100 req/s at 2 s E2E."* — Expected: L=λW=200 in-flight; map to batch capacity and replicas with utilization margin.
- *"Why is MFU a bad target for decode?"* — Expected: decode is bandwidth-bound, MFU intrinsically low; use tok/s/GPU and $/token.
- *"How do you optimize P99 ITL specifically?"* — Expected: kill prefill-decode interference (chunked prefill/disagg), bound per-iteration token budget, preemption policy.

***

## Open Problems & Active Research (2025–2026)
- **SLO definitions for reasoning models** where TTFT is meaningless and "time-to-final-answer" / cost-per-solved-task are the real metrics ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).
- **Goodput-optimal schedulers** that jointly satisfy TTFT and ITL under bursty, heterogeneous load.
- **Predictive autoscaling** using output-length prediction to keep ρ in the safe zone without over-provisioning.

***

## References
- Little, J.D.C. (1961). "A Proof for the Queuing Formula L = λW." *Operations Research* 9(3).
- Zhong, Y., et al. (2024). "DistServe." *OSDI 2024*. arXiv:2401.09670.
- Agrawal, A., et al. (2024). "Sarathi-Serve." *OSDI 2024*. arXiv:2403.02310.
- Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference." *MLSys 2023*. arXiv:2211.05102.
