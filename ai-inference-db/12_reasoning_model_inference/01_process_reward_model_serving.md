# Process Reward Model Serving

> **Section:** 12_reasoning_model_inference
> **Last Updated:** June 2026
> **Related Files:** [02_tree_search_and_mcts_serving.md](02_tree_search_and_mcts_serving.md), [04_thinking_budget_and_inference_time_scaling.md](04_thinking_budget_and_inference_time_scaling.md), [00_long_chain_of_thought_serving.md](00_long_chain_of_thought_serving.md)
> **Must-Read Papers:** Lightman et al. (2024, ICLR) "Let's Verify Step by Step (PRM)"; Wang et al. (2024) "Math-Shepherd"; Snell et al. (2024) "Test-Time Compute"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- A **Process Reward Model (PRM)** scores each *step* of a reasoning chain (vs an Outcome RM scoring only the final answer), guiding search/selection toward better reasoning paths.
- Serving a PRM means running a **second model** (the verifier) alongside the generator, often **many times** (per step × per candidate) — a major compute multiplier.
- Common pattern: generator produces N candidate steps/paths; PRM scores them; keep the best (best-of-N, beam, or tree search) — **inference-time scaling** ([§04](04_thinking_budget_and_inference_time_scaling.md)).
- Serving challenges: **two-model orchestration**, PRM call volume, batching generator + verifier, latency of the score-then-continue loop.
- The PRM can be smaller than the generator, but its **call frequency** makes total cost significant.

***

## Overview
Process Reward Models are central to inference-time-scaling pipelines for reasoning. Where an **Outcome Reward Model (ORM)** judges only the final answer, a **PRM** (Lightman et al. 2024, "Let's Verify Step by Step") scores **each intermediate reasoning step**, providing dense feedback that lets a search procedure prune bad paths early and steer toward correct reasoning. PRMs substantially improve reasoning accuracy when used to guide best-of-N selection, beam search, or tree search over reasoning steps ([§02](02_tree_search_and_mcts_serving.md), [§04](04_thinking_budget_and_inference_time_scaling.md)). From a serving standpoint, the key fact is that using a PRM means running **two models in concert** — the generator and the verifier — with the verifier invoked **frequently** (potentially per step, per candidate), turning a single generation into a multi-model, many-call workload.

The serving cost comes from **call volume**, not per-call size. A PRM can be smaller than the generator, but in a guided search you might generate N candidate continuations at each of K steps and score every one, so the PRM runs O(N×K) times per query — a large multiplier on top of the already decode-heavy reasoning generation ([§00](00_long_chain_of_thought_serving.md)). This makes PRM serving a **two-model orchestration and batching** problem: you must interleave generator decode (producing candidate steps) with PRM scoring (evaluating them), batch each model's calls efficiently, and manage the latency of the generate→score→select→continue loop, which serializes generation and verification per step.

The practical serving strategies: **batch PRM scoring** across candidates (the PRM processes many candidate steps at once — a prefill-like, parallelizable workload); **co-locate or disaggregate** generator and verifier depending on their relative load; **reuse KV** across candidate continuations sharing a prefix (the reasoning trunk — RadixAttention/prefix caching shines, [§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)); and **bound the search** (N, K) to control cost. This file covers PRM serving mechanics, the orchestration/batching challenges, and the optimizations.

***

## Core Concepts & Mechanics

### PRM vs ORM
- **ORM**: one score for the final answer. Cheap (one call/solution) but sparse signal.
- **PRM**: a score per reasoning step. Dense signal → better search guidance, but **many calls** (per step × per candidate).

### The guided-search loop
```
for each step:
  generator produces N candidate next-steps (decode)
  PRM scores each candidate (verify, batchable)
  select best (best-of-N / beam / tree) → continue
```
📐 PRM calls ≈ O(N × K) per query (N candidates, K steps). Even a small PRM × high frequency = significant cost.

### Serving the two models
- **Batch PRM scoring**: score all N candidates together (prefill-like, parallel, compute-bound).
- **Generator decode**: standard reasoning decode (memory-bound, long).
- **Orchestration**: interleave generate→score→select per step; the loop serializes the two stages.

### KV reuse across candidates
Candidate continuations share the reasoning **trunk**; **prefix caching / RadixAttention** reuses the shared prefix's KV across candidates (and the PRM can reuse context too) — a big saving ([§02](../02_kv_cache/05_cross_request_kv_sharing.md)).

***

## Key Challenges
1. **PRM call volume.** O(N×K) verifier calls multiply cost; even a small PRM is expensive at high frequency.
2. **Two-model orchestration.** Interleaving generator and verifier, batching each, and managing the per-step loop latency is complex.
3. **Serialized generate→score loop.** Per-step verification serializes with generation, adding latency; overlap is hard.
4. **Resource balancing.** Generator (decode-bound) and PRM (score-bound) have different profiles; co-location vs disaggregation tradeoff.

***

## Solutions & Current Best Practices
- **Batch PRM scoring** across candidates (parallel, compute-bound) for efficiency.
- **Prefix-cache the reasoning trunk** across candidates (RadixAttention) ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)).
- **Bound search width/depth (N, K)** to control cost; use dynamic compute (more search on hard queries) ([§03](03_dynamic_compute_allocation.md)).
- **Co-locate or disaggregate** generator/PRM by load; smaller PRM where quality permits.

***

## Implementation Notes
- Run the PRM as a batched scoring service; score all candidates of a step in one call.
- Use RadixAttention/prefix caching for the shared trunk across candidates.
- Track PRM-call count and total verifier cost; it often dominates the search overhead.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **PRM cost dwarfed generation** — O(N×K) calls; high frequency, not size, was the cost. Bound N/K, batch scoring.
- **No trunk KV reuse across candidates** — re-prefilled the shared reasoning prefix per candidate; use prefix caching.
- **Generate→score loop serialized badly** — per-step verification added latency; batch candidates, overlap where possible.
- **Generator/PRM resource imbalance** — one starved the other; co-locate/disaggregate by load.

***

## Performance Numbers & Benchmarks
| Pattern | PRM calls/query | Cost driver |
|---|---|---|
| Best-of-N (outcome) | ~N (ORM) | generation |
| PRM-guided beam | O(N×K) | PRM frequency |
| Tree search + PRM | high | PRM + branching ([§02](02_tree_search_and_mcts_serving.md)) |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"PRM vs ORM — serving implications?"* — Expected: per-step (many calls) vs per-answer; PRM call-volume cost.
- *"How do you serve PRM-guided search efficiently?"* — Expected: batch scoring, prefix-cache trunk, bound N/K, balance two models.
- *"Where does the cost go in PRM-guided reasoning?"* — Expected: PRM call frequency O(N×K), not per-call size.
- *"How do candidates share compute?"* — Expected: shared reasoning trunk via prefix caching/RadixAttention.

***

## Open Problems & Active Research (2025–2026)
- **Cheaper verification** (smaller/faster PRMs, partial scoring) without quality loss.
- **Overlapping generation and verification** to cut loop latency.
- **Optimal search width/depth allocation** per query difficulty ([§03](03_dynamic_compute_allocation.md)).

***

## References
- Lightman, H., et al. (2024). "Let's Verify Step by Step (PRM)." *ICLR 2024*. arXiv:2305.20050.
- Wang, P., et al. (2024). "Math-Shepherd: Verify and Reinforce LLMs Step-by-step." arXiv:2312.08935.
- Snell, C., et al. (2024). "Scaling Test-Time Compute." arXiv:2408.03314.
