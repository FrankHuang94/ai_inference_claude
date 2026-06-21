# Request Scheduling Policies

> **Section:** 03_batching_and_scheduling
> **Last Updated:** June 2026
> **Related Files:** [01_iteration_level_scheduling.md](01_iteration_level_scheduling.md), [05_multi_priority_and_SLA_scheduling.md](05_multi_priority_and_SLA_scheduling.md), [../00_fundamentals/06_key_metrics_latency_throughput_cost.md](../00_fundamentals/06_key_metrics_latency_throughput_cost.md)
> **Must-Read Papers:** Yu et al. (2022, OSDI) "Orca"; Wu et al. (2023) "Fast Distributed Inference Serving" (FastServe, SJF/MLFQ); Zheng et al. (2023) "Response Length Perception (S3)"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- Scheduling policy chooses **order and priority** of requests; it shapes P50 vs P99 latency and fairness, on top of the per-iteration mechanics ([§01](01_iteration_level_scheduling.md)).
- **FCFS**: fair, simple, but suffers **head-of-line (HOL) blocking** — a long request delays everyone behind it.
- **SJF/SRPT** (shortest-job/remaining-time-first) minimizes mean latency but needs **output-length prediction** (hard — the model doesn't know its own length) and can **starve** long requests.
- **MLFQ** (FastServe): preemptive multi-level feedback queues approximate SJF without known lengths, reducing HOL blocking.
- Output-length prediction methods (regression on features, the model's own "length perception," conservative bounds) are imperfect; robust schedulers degrade gracefully when predictions are wrong.

***

## Overview
Given the per-iteration batching machinery, the *policy* question is: when capacity is scarce, whose requests run first, who waits, and who gets preempted? This choice doesn't change the total work but dramatically reshapes the latency distribution and fairness. The default, **first-come-first-served (FCFS)**, is fair and predictable but vulnerable to **head-of-line blocking**: a request with a long prompt (expensive prefill) or long output occupies capacity and delays shorter requests queued behind it, inflating their TTFT and the system's P99. In interactive serving this is the dominant tail-latency pathology after prefill-decode interference.

Classical scheduling theory says **shortest-job-first (SJF)** / **shortest-remaining-processing-time (SRPT)** minimizes mean response time, and indeed prioritizing short requests slashes average latency and HOL blocking. The catch unique to LLMs is that **job length is unknown** — an autoregressive request's output length isn't known until it emits EOS, and the model itself can't reliably predict it. So LLM schedulers either *predict* output length (with regression models, the model's own length-perception signal, or conservative bounds) or *approximate* SJF without lengths via **multi-level feedback queues (MLFQ)**, as in **FastServe** (Wu et al. 2023): new requests start high-priority and are demoted as they consume more compute, so short requests finish quickly while long ones drop to lower priority — approximating SRPT with preemption (enabled by cheap KV swap/recompute).

The practical art is balancing **mean latency (favor short jobs)** against **fairness/starvation (don't perpetually defer long jobs)** and **prediction robustness (don't catastrophically mis-schedule when length estimates are wrong)**. Production systems usually run FCFS or MLFQ variants with anti-starvation aging and SLO-tier priorities ([§05](05_multi_priority_and_SLA_scheduling.md)), and increasingly fold in length prediction where it's reliable.

***

## Core Concepts & Mechanics

### Policies
- **FCFS**: process in arrival order. Fair; HOL blocking; good P50 for uniform loads, bad P99 with length skew.
- **SJF/SRPT**: prioritize shortest (remaining) job. Minimizes mean latency; needs length prediction; starves long jobs without aging.
- **MLFQ** (FastServe): multiple priority queues; new requests enter top queue; demote as they accrue service; preempt lower queues. Approximates SRPT without known lengths.
- **Priority/SLO-based**: explicit tiers (premium vs free); see [§05](05_multi_priority_and_SLA_scheduling.md).

### Head-of-line blocking
📐 Under FCFS with a long job of service time `S_long` arriving first, all jobs behind wait ≥ `S_long`. With heavy-tailed length distributions, a few long jobs dominate P99. SJF/MLFQ move long jobs out of the critical path.

### Output-length prediction
The crux for SJF. Approaches:
- **Feature regression**: predict length from prompt features/task type (e.g., S3, Zheng et al. 2023 "response length perception"); moderate accuracy.
- **Model self-signal**: ask/the model's hidden state correlates with remaining length; noisy.
- **Conservative bounds / quantile estimates**: schedule by predicted quantile, hedge for error.
Predictions are imperfect; misprediction (short predicted, long actual) reintroduces HOL blocking — so schedulers must **re-evaluate** (MLFQ demotion handles this naturally).

### Preemption as enabler
SJF/MLFQ require **preempting** running long jobs to let short ones through. KV swap/recompute ([§02](../02_kv_cache/04_kv_cache_offloading.md)) makes preemption affordable; without cheap preemption, only non-preemptive ordering is possible.

***

## Key Challenges
1. **Length unpredictability.** Output length is fundamentally unknown a priori; SJF's optimality assumes known lengths, so real systems pay a prediction-error tax.
2. **Starvation.** Pure SJF/MLFQ can defer long jobs indefinitely under continuous short-job arrivals; needs aging/guarantees.
3. **Preemption cost.** Frequent preemption (swap/recompute) adds overhead and can thrash if overused.
4. **P50 vs P99 vs fairness trilemma.** Optimizing mean (SJF) can hurt fairness and specific tenants; policy must reflect business SLOs, not just averages.

***

## Solutions & Current Best Practices
- **MLFQ-style preemptive scheduling** (FastServe) to approximate SRPT without length oracles, cutting HOL blocking.
- **Length prediction where reliable** (task-type priors, regression) to inform admission/ordering, with graceful fallback on error.
- **Aging / fairness guarantees** to prevent starvation of long jobs.
- **SLO-tier priorities** layered on top for multi-tenant ([§05](05_multi_priority_and_SLA_scheduling.md)).

***

## Implementation Notes
- Combine policy with the token budget and chunked prefill: a long prefill should be chunked *and* schedulable at lower priority to protect interactive decodes.
- Track per-request service consumed (for MLFQ demotion) and waiting time (for aging).
- Validate on the **P99/P999**, not mean — policies that win on average can lose on the tail.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **SJF starved long requests in production** — continuous short-job arrivals deferred a long generation forever; add aging or per-request deadlines.
- **Length mispredictions reintroduced HOL blocking** — a predicted-short job ran long, blocking the queue; MLFQ demotion or re-estimation needed.
- **Preemption thrash from over-aggressive SJF** — constant swap/recompute overhead exceeded the latency savings; bound preemption frequency.
- **Optimized mean, P99 got worse** — favoring short jobs lengthened the worst-case for long jobs; SLOs are about tails and fairness, not means.

***

## Performance Numbers & Benchmarks
| Policy | Mean latency | P99 / HOL | Fairness | Needs length? |
|---|---|---|---|---|
| FCFS | moderate | poor (HOL) | high | no |
| SJF/SRPT | best mean | good (short), bad (long) | low (starvation) | yes |
| MLFQ (FastServe) | near-SJF | good | medium (with aging) | no |
| Priority/SLO | per-tier | per-tier | by policy | optional |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"What is head-of-line blocking in LLM serving and how do you mitigate it?"* — Expected: long job delays queue; SJF/MLFQ + chunked prefill + preemption.
- *"Why is SJF hard for LLMs, and how does MLFQ help?"* — Expected: unknown output length; MLFQ approximates SRPT via demotion without prediction.
- *"How would you predict output length, and what if you're wrong?"* — Expected: feature regression/task priors; graceful fallback, re-estimation, MLFQ.
- *"How do you prevent starvation under SJF?"* — Expected: aging, deadlines, fairness guarantees.

***

## Open Problems & Active Research (2025–2026)
- **Reliable output-length prediction**, especially for reasoning models with highly variable CoT length ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).
- **Goodput- and SLO-aware scheduling** that jointly optimizes tails, fairness, and throughput.
- **Learned, online schedulers** adapting policy to live traffic.

***

## References
- Yu, G.-I., et al. (2022). "Orca." *OSDI 2022*.
- Wu, B., et al. (2023). "Fast Distributed Inference Serving for LLMs" (FastServe, MLFQ). arXiv:2305.05920.
- Zheng, Z., et al. (2023). "Response Length Perception and Sequence Scheduling (S3)." *NeurIPS 2023*. arXiv:2305.13144.
