# Iteration-Level Scheduling Internals

> **Section:** 03_batching_and_scheduling
> **Last Updated:** June 2026
> **Related Files:** [00_static_vs_continuous_batching.md](00_static_vs_continuous_batching.md), [04_request_scheduling_policies.md](04_request_scheduling_policies.md), [../08_serving_frameworks/01_vllm_deep_dive.md](../08_serving_frameworks/01_vllm_deep_dive.md)
> **Must-Read Papers:** Yu et al. (2022, OSDI) "Orca"; Agrawal et al. (2024, OSDI) "Sarathi-Serve"; Zheng et al. (2024) "SGLang"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- The scheduler runs once **per forward iteration**: decide the batch, allocate KV, run, sample, evict finished, admit waiting — all within the time of one decode step.
- Core knobs: **max running sequences**, **per-iteration token budget** (prefill+decode), **admission watermark**, **preemption policy**.
- The scheduler itself must be **cheap** — at >100 tok/s it runs every few ms; a Python-heavy scheduler becomes the bottleneck (SGLang's "zero-overhead scheduler" and overlap scheduler address this).
- **Admission control** balances filling the GPU vs avoiding interference/OOM; **overlap** hides CPU scheduling behind GPU compute.
- Getting this loop right is what separates a fast framework from a slow one at high QPS.

***

## Overview
Continuous batching ([§00](00_static_vs_continuous_batching.md)) is implemented as a tight control loop: the **scheduler** decides, before each model forward pass, exactly which sequences run, how much prefill to admit, and which to preempt. Because this decision happens every iteration — potentially every few milliseconds at high decode rates — the scheduler is both a *policy* engine (what to run for good latency/throughput) and a *performance-critical* component (its own CPU time can stall the GPU). Modern frameworks invest heavily here: vLLM's scheduler, SGLang's zero-overhead and overlap schedulers, and TensorRT-LLM's in-flight batching manager are all sophisticated implementations of this loop.

The policy decisions per iteration are: (1) **eviction** — remove sequences that hit EOS/length and free KV; (2) **admission** — pull requests from the waiting queue into the running batch, bounded by KV availability and a token budget; (3) **prefill handling** — for admitted requests, schedule their prompt prefill, possibly chunked and interleaved with decode; (4) **preemption** — under memory pressure, evict running requests (swap or recompute) to keep others moving. Each decision affects TTFT (admission/prefill speed), ITL (how much prefill lands in a decode batch), and throughput (how full the batch stays).

The performance dimension is subtle but critical. At batch decode of, say, 5 ms/iteration, a scheduler that takes 2 ms of Python per iteration adds 40% overhead and caps QPS. SGLang's contribution was a near-zero-overhead scheduler plus an **overlap scheduler** that runs the next iteration's CPU scheduling concurrently with the current iteration's GPU compute, hiding scheduling latency entirely. This is why scheduler engineering — not just policy — is a real differentiator and a frequent interview topic.

***

## Core Concepts & Mechanics

### The per-iteration loop
```
loop每 forward pass:
  1. evict finished sequences; free KV blocks
  2. compute available KV budget and token budget
  3. admit from waiting queue (respect budgets, SLO priority)
  4. for admitted: schedule prefill (chunked if enabled)
  5. build ragged batch (mixed prefill chunks + decodes)
  6. run forward pass; sample; append tokens; write KV
  7. update queues, metrics
```

### Token budget
📐 `max_num_batched_tokens` caps total tokens processed per iteration (sum of prefill-chunk tokens + 1 per decode sequence). With pure decode, tokens ≈ batch size. When admitting a prefill of P tokens, the iteration processes ~P + batch tokens, so a large P inflates iteration time — the budget bounds this, protecting ITL. Choose budget so worst-case iteration ≤ ITL SLO.

### Admission watermark and KV headroom
Admit only if enough free KV blocks exist for the new request to run at least one step (and ideally a safety margin), or you'll immediately preempt it. A **watermark** reserves headroom to avoid thrashing.

### Preemption policy
Under pressure: choose victims (often most-recently-admitted or lowest-priority), and **swap** (KV→CPU) or **recompute** (drop KV, re-prefill later). 📐 recompute cost ≈ prefill compute; swap cost ≈ KV/PCIe bandwidth round-trip — pick per sequence length ([§02](../02_kv_cache/04_kv_cache_offloading.md)).

### Scheduler overhead and overlap
The CPU scheduling work (queue ops, block allocation, building tensors) must be small relative to the GPU step. **Overlap scheduling** (SGLang) computes iteration *t+1*'s schedule while iteration *t* runs on the GPU, so scheduling is off the critical path. Avoiding Python GIL contention (C++/async) is key at high QPS.

***

## Key Challenges
1. **Scheduler-as-bottleneck.** At high token rates the per-iteration CPU cost can exceed/rival the GPU step; naive Python schedulers cap throughput.
2. **Interference control.** Deciding how much prefill to admit per iteration trades TTFT (admit fast) vs ITL (don't inflate the step) — hard under bursty load.
3. **Preemption tuning.** Wrong victim selection or swap/recompute choice spikes tail latency and can thrash.
4. **Fairness/priority.** Mixing SLO tiers and avoiding starvation (e.g., long requests perpetually preempted) requires careful policy ([§05](05_multi_priority_and_SLA_scheduling.md)).

***

## Solutions & Current Best Practices
- **Low-overhead / overlap schedulers** (SGLang) to hide CPU scheduling behind GPU compute ([§08](../08_serving_frameworks/02_sglang_deep_dive.md)).
- **Token-budget + chunked prefill** to bound interference ([§02](02_chunked_prefill.md)).
- **Watermark-based admission** with KV headroom to prevent thrashing.
- **Length-aware / priority-aware policies** for SLO compliance ([§04](04_request_scheduling_policies.md), [§05](05_multi_priority_and_SLA_scheduling.md)).

***

## Implementation Notes
- Profile scheduler CPU time per iteration separately from GPU time; if it's a meaningful fraction, move to async/overlap or C++.
- Pre-build/reuse batch tensors and block tables to avoid per-iteration allocation churn.
- Expose budgets (`max_num_seqs`, `max_num_batched_tokens`) and tune to the ITL/TTFT SLO and model/hardware.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **GPU shows idle gaps between iterations** — the scheduler (CPU) isn't overlapped; the GPU waits for the next batch. Enable overlap/async scheduling.
- **High QPS collapses despite spare GPU** — Python GIL-bound scheduler saturates a CPU core; the GPU starves. Classic vLLM-vs-SGLang difference.
- **Admission watermark too low → preemption storm** — admit, OOM, preempt, recompute, loop; throughput craters under load.
- **Long requests starve** — greedy admission of short requests perpetually preempts a long one; add aging/priority to guarantee progress.

***

## Performance Numbers & Benchmarks
| Aspect | Naive Python scheduler | Overlap/low-overhead (SGLang) |
|---|---|---|
| Per-iteration CPU | ms-scale, on critical path | hidden behind GPU |
| High-QPS ceiling | GIL/CPU-bound | GPU-bound |
| GPU idle gaps | present | minimized |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Walk through what the scheduler does each iteration."* — Expected: evict, compute budgets, admit, prefill (chunk), build ragged batch, run, update.
- *"Why can the scheduler itself become the bottleneck, and how do you fix it?"* — Expected: per-iteration CPU vs GPU step; overlap/async/C++ scheduling.
- *"How does the token budget protect ITL?"* — Expected: caps per-iteration tokens incl. prefill chunks → bounds iteration time.
- *"Design a preemption policy."* — Expected: victim selection + swap vs recompute by length; headroom watermark; anti-starvation.

***

## Open Problems & Active Research (2025–2026)
- **Learned schedulers** that predict output length/interference to optimize admission online.
- **Goodput-optimal iteration scheduling** jointly meeting TTFT and ITL SLOs.
- **Scheduling across disaggregated prefill/decode pools** as one logical scheduler ([§03](03_prefill_decode_disaggregation.md)).

***

## References
- Yu, G.-I., et al. (2022). "Orca." *OSDI 2022*.
- Agrawal, A., et al. (2024). "Sarathi-Serve." *OSDI 2024*. arXiv:2403.02310.
- Zheng, L., et al. (2024). "SGLang." arXiv:2312.07104.
- Kwon, W., et al. (2023). "vLLM." *SOSP 2023*. arXiv:2309.06180.
