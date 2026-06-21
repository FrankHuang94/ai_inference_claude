# When Speculative Decoding Fails

> **Section:** 07_speculative_decoding
> **Last Updated:** June 2026
> **Related Files:** [00_speculative_decoding_fundamentals.md](00_speculative_decoding_fundamentals.md), [01_draft_model_selection.md](01_draft_model_selection.md), [../00_fundamentals/02_prefill_vs_decode_phases.md](../00_fundamentals/02_prefill_vs_decode_phases.md)
> **Must-Read Papers:** Leviathan et al. (2023, ICML); Chen et al. (2024) "Sequoia"; Liu et al. (2024) "Online Speculative Decoding"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- Speculative decoding is a **low-batch / latency** optimization; at **high batch** the target is already compute-bound, so verifying extra tokens costs real compute → **net regression**.
- 📐 There's a **batch-size threshold** (often ~4–16, config-dependent) above which speculation stops helping.
- **Low acceptance rate** (domain shift, instruction-following, strict formats, code) kills the speedup and can make it net-negative.
- Other failure modes: **draft overhead/memory**, **draft drift** after target fine-tuning, **tree-verification compute** at scale, **scheduling complexity**.
- The senior skill: **measure net tokens/s and acceptance in situ**, and **gate speculation** on batch size and measured α.

***

## Overview
Speculative decoding is often presented as free throughput, but it's a conditional optimization that **fails or backfires** in common production regimes. The core reason traces to the roofline ([§00](../00_fundamentals/04_roofline_model_for_llm.md)): speculation works by spending the *idle compute* of bandwidth-bound, low-batch decode to verify multiple tokens per weight load. But as batch size grows, decode moves toward the **compute roof** — the GPU is no longer idle — so the extra verification tokens (especially tree-based) now compete for the *binding* resource (compute), and the "free" verification becomes a real cost. Past a **batch-size threshold**, speculative decoding stops helping and then *hurts*, because you're doing more compute (drafting + verifying rejected tokens) for no latency benefit.

The second major failure is **low acceptance rate**. The speedup formula `(1−α^{k+1})/(1−α)` collapses toward 1 as α drops, and below a break-even α the draft+verify overhead exceeds the savings → net slowdown. Low α happens when the draft and target disagree often: **domain shift** (draft trained on general text, serving code or a specialized domain), **instruction-following / strict formatting** (where exact tokens matter and the draft guesses wrong), and **high-temperature sampling** (more entropy → lower agreement). So a draft that gives 3× on chat can give <1× on code or JSON-mode output.

The remaining failure modes are operational: the **draft model's own latency and memory** (a too-large or slow draft eats the gains, [§01](01_draft_model_selection.md)); **draft drift** when the target is fine-tuned and the draft no longer matches (acceptance silently degrades); **tree-verification compute** that scales poorly with batch (SpecInfer/Medusa trees); and the **scheduling complexity** of mixing speculative and non-speculative requests in continuous batching. The senior-engineer behavior is to treat speculation as **conditional**: instrument net tokens/s and acceptance per workload, **gate it on batch size** (enable at low batch, disable at high), and re-validate after model updates. This file enumerates the failure modes and the mitigations.

***

## Core Concepts & Mechanics

### The batch-size threshold
📐 At low batch, verify cost ≈ one decode step (bandwidth-bound) regardless of k tokens → speculation ~free. As batch B grows, decode becomes compute-bound; verifying `k` extra candidate tokens × B now adds compute proportional to the work. Net benefit > 0 only while `saved_steps × step_cost > extra_verify_compute`. Empirically the crossover is often **B ≈ 4–16** (depends on model, draft, tree size). Above it: **regression**.

### Low acceptance regimes
- **Domain shift**: draft mismatched to serving domain (code, legal, medical) → low α.
- **Instruction/format-strict**: exact tokens matter (JSON, function calls); draft guesses wrong → low α.
- **High temperature**: more entropy → target and draft agree less.
- Below break-even α, draft+verify overhead > savings → **slower than plain decode**.

### Other failure modes
- **Draft overhead/memory**: large/slow draft; autoregressive drafting latency; extra HBM.
- **Draft drift**: target fine-tuned, draft stale → α silently drops ([§01](01_draft_model_selection.md)).
- **Tree compute at scale**: tree methods verify many tokens; worse at high batch.
- **Scheduling**: mixing speculative (variable accepted-token counts) with regular requests complicates continuous batching and KV management.

***

## Key Challenges
1. **Regime-dependence.** The same config is a 3× win or a regression depending on batch and workload; static "always on" is wrong.
2. **Silent acceptance drift.** Model updates or traffic shifts degrade α without obvious symptoms; needs monitoring.
3. **Throughput vs latency conflict.** Speculation favors latency at low batch but can cut aggregate throughput at high batch; you may want it per-request, not globally.
4. **Verification correctness under sampling.** High-temperature/exact-match settings stress the rejection rule and acceptance.

***

## Solutions & Current Best Practices
- **Gate speculation on batch size**: enable at low batch / latency-critical, disable above the threshold. Some systems (Sequoia, dynamic spec) adapt automatically.
- **Monitor acceptance and net tokens/s per workload**; disable when net-negative.
- **Match/distill the draft to the serving domain**; re-distill after target updates ([§01](01_draft_model_selection.md)).
- **Adaptive draft length / tree size** (EAGLE-2) to scale proposals with confidence and batch.

***

## Implementation Notes
- Instrument α (acceptance rate) and compare speculative vs non-speculative tokens/s online.
- Configure batch-size thresholds in the serving framework to auto-disable speculation under load.
- For mixed traffic, consider per-request speculation decisions (latency-tier on, batch-tier off).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Speculation helped in dev (batch=1), hurt in prod (batch=32)** — crossed the batch threshold into compute-bound; gate on batch size.
- **3× on chat, <1× on code** — domain/format mismatch dropped α below break-even; domain-match the draft.
- **Acceptance silently dropped after a model update** — draft drift; re-distill and monitor α.
- **Aggregate throughput fell with speculation on** — optimized latency at the cost of throughput at high batch; make it conditional.

***

## Performance Numbers & Benchmarks
| Regime | Outcome |
|---|---|
| Low batch, well-matched draft | ~2–4× speedup |
| High batch (compute-bound) | ~1× or regression |
| Low α (domain/format mismatch) | net-negative |
| Adaptive (gated on batch/α) | win where applicable, neutral elsewhere |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why does speculative decoding stop helping at high batch?"* — Expected: decode becomes compute-bound; extra verification no longer free.
- *"What is the batch-size threshold and how do you handle it?"* — Expected: ~4–16 config-dependent; gate speculation on batch size.
- *"Give workloads where speculation fails."* — Expected: code/format-strict, domain shift, high temperature → low α.
- *"How do you decide if speculation is net-positive in deployment?"* — Expected: measure α and net tokens/s in situ; gate accordingly.

***

## Open Problems & Active Research (2025–2026)
- **Batch-aware / high-batch speculation** that stays net-positive (Sequoia, batched tree verification).
- **Online/adaptive speculation** that auto-tunes α and disables when unhelpful (Online Speculative Decoding).
- **Robust drafts for code/reasoning/format-strict** workloads.

***

## References
- Leviathan, Y., et al. (2023). "Speculative Decoding." *ICML 2023*. arXiv:2211.17192.
- Chen, Z., et al. (2024). "Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding." arXiv:2402.12374.
- Liu, X., et al. (2024). "Online Speculative Decoding." *ICML 2024*. arXiv:2310.07177.
