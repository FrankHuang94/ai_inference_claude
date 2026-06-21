# Tree Search and MCTS Serving

> **Section:** 12_reasoning_model_inference
> **Last Updated:** June 2026
> **Related Files:** [01_process_reward_model_serving.md](01_process_reward_model_serving.md), [04_thinking_budget_and_inference_time_scaling.md](04_thinking_budget_and_inference_time_scaling.md), [../02_kv_cache/05_cross_request_kv_sharing.md](../02_kv_cache/05_cross_request_kv_sharing.md)
> **Must-Read Papers:** Yao et al. (2023, NeurIPS) "Tree of Thoughts"; Silver et al. (2016) "AlphaGo/MCTS"; AlphaProof/AlphaCode reports
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- Tree-search reasoning (Tree-of-Thoughts, MCTS-style) explores **branching** reasoning paths, expanding/evaluating/selecting nodes — each node is an LLM generation, each evaluation a (PRM/value) call.
- Serving = **massive branching workload**: many parallel generations sharing prefixes (the tree trunk) + frequent scoring → throughput- and KV-intensive.
- **Prefix sharing is the key optimization**: sibling branches share their parent's KV → RadixAttention/copy-on-write make this efficient ([§02](../02_kv_cache/05_cross_request_kv_sharing.md)).
- MCTS adds **sequential dependency** (selection→expansion→evaluation→backprop) that limits parallelism vs simple best-of-N.
- Cost scales with branching factor × depth × evaluations — bounded by a compute/thinking budget ([§04](04_thinking_budget_and_inference_time_scaling.md)).

***

## Overview
The most compute-intensive inference-time-scaling methods explore a **tree** of reasoning paths rather than a single chain. **Tree-of-Thoughts** (Yao et al. 2023) generates multiple candidate thoughts at each step, evaluates them, and explores the promising branches; **MCTS-style** search (à la AlphaGo, applied to reasoning in AlphaProof/AlphaCode-style systems) adds the classic selection → expansion → evaluation → backpropagation loop with a value/reward model guiding exploration. Each tree **node** corresponds to an LLM generation (a reasoning step or partial solution), and each **evaluation** is a scoring call (a PRM or value model, [§01](01_process_reward_model_serving.md)). Serving this means orchestrating **many LLM generations and evaluations per query**, with the total cost scaling as branching factor × depth × evaluations.

From a serving systems view, two properties dominate. First, **prefix sharing**: all branches descending from a node share that node's reasoning prefix, so the KV cache of the shared trunk should be reused across siblings rather than recomputed — this is exactly what **RadixAttention** (and copy-on-write paged KV) was built for, and it's the single most important optimization for tree-search serving ([§02](../02_kv_cache/05_cross_request_kv_sharing.md)). Without prefix sharing, you'd re-prefill the common reasoning path for every branch, an enormous waste. Second, **parallelism structure**: simple methods like best-of-N or breadth-first ToT expansion are **embarrassingly parallel** (generate all candidates at a level at once — great for batching), whereas **MCTS** has inherent **sequential dependency** (the next node to expand depends on backpropagated values from prior simulations), limiting batch parallelism and adding loop latency.

The practical serving strategies: exploit **prefix sharing aggressively** (RadixAttention is ideal — SGLang was partly designed for this), **batch the parallel parts** (sibling generations, candidate evaluations), **bound the search** (branching/depth/simulations) via a compute budget ([§04](04_thinking_budget_and_inference_time_scaling.md)), and **balance generator vs evaluator** load ([§01](01_process_reward_model_serving.md)). Tree search is the high end of the inference-time-compute spectrum — most accurate, most expensive — and its serving is a throughput+KV+orchestration problem. This file covers the search structures, the prefix-sharing optimization, and the parallelism limits.

***

## Core Concepts & Mechanics

### Search structures
- **Best-of-N**: N independent full solutions, pick best (by ORM/vote). Embarrassingly parallel.
- **Tree-of-Thoughts**: branch at each step (breadth b, depth d), evaluate, explore promising. Parallel per level.
- **MCTS**: selection → expansion → evaluation → backpropagation, guided by value/UCB. **Sequential** dependency.

### Cost scaling
📐 Generations ≈ O(b^d) (full tree) or O(b×d) (beam-limited); evaluations similar. Total cost = generations × gen_cost + evaluations × eval_cost. Bounded by a compute/thinking budget ([§04](04_thinking_budget_and_inference_time_scaling.md)).

### Prefix sharing (the key optimization)
All children of a node share its reasoning prefix → reuse the trunk's KV across siblings via **RadixAttention** / copy-on-write paged KV ([§02](../02_kv_cache/05_cross_request_kv_sharing.md), [§02 PagedAttention](../02_kv_cache/01_paged_attention_vllm.md)). 📐 Without sharing: re-prefill trunk per branch (O(branches × trunk)); with sharing: prefill trunk once.

### Parallelism structure
- **Best-of-N / level-wise ToT**: parallel → batch all candidates of a level → high throughput.
- **MCTS**: selection depends on backpropagated values → sequential simulations → limited parallelism, more loop latency. (Parallel MCTS variants exist but add complexity.)

***

## Key Challenges
1. **Compute explosion.** Branching × depth × evaluations multiplies cost; must bound via budget.
2. **MCTS sequential dependency.** The select→expand→eval→backprop loop limits batch parallelism and adds latency vs parallel methods.
3. **KV management for the tree.** Many live branches' KV (sharing trunks) must be managed with copy-on-write/radix structures without leaks ([§02](../02_kv_cache/05_cross_request_kv_sharing.md)).
4. **Generator/evaluator balance.** Heavy generation + heavy evaluation (PRM) need co-scheduling ([§01](01_process_reward_model_serving.md)).

***

## Solutions & Current Best Practices
- **Aggressive prefix sharing** (RadixAttention/CoW) for tree trunks — the defining optimization.
- **Batch parallel search** (best-of-N, level-wise ToT candidates, evaluations).
- **Bound search** (branching/depth/simulations) via compute/thinking budget ([§04](04_thinking_budget_and_inference_time_scaling.md)).
- **Prefer parallel methods** (best-of-N/ToT) when adequate; reserve MCTS for when its guidance is worth the serialization.

***

## Implementation Notes
- Use SGLang/RadixAttention (or vLLM CoW) to share trunk KV across branches automatically.
- Batch sibling generations and candidate evaluations; manage tree KV lifecycle (free pruned branches).
- For MCTS, consider parallel/virtual-loss variants to recover some batch parallelism.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Re-prefilled the trunk per branch** — no prefix sharing; massive waste. Use RadixAttention/CoW.
- **MCTS underutilized the GPU** — sequential select→backprop limited batching; use parallel variants or simpler search.
- **Tree KV leaked** — pruned branches not freed; KV exhausted. Manage lifecycle.
- **Search cost unbounded** — branching×depth blew the budget; cap search width/depth.

***

## Performance Numbers & Benchmarks
| Method | Parallelism | KV sharing | Cost |
|---|---|---|---|
| Best-of-N | full | independent | N× generation |
| Tree-of-Thoughts | per level | trunk shared | b^d-ish (bounded) |
| MCTS | limited (sequential) | tree shared | simulations × (gen+eval) |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How would you serve Tree-of-Thoughts / MCTS reasoning efficiently?"* — Expected: prefix-share trunks (RadixAttention/CoW), batch parallel parts, bound search.
- *"Why is MCTS harder to serve than best-of-N?"* — Expected: sequential select→backprop dependency limits batching.
- *"What's the key KV optimization for tree search?"* — Expected: share trunk KV across branches.
- *"How does tree-search cost scale and how do you control it?"* — Expected: branching×depth×eval; compute/thinking budget.

***

## Open Problems & Active Research (2025–2026)
- **Efficient parallel MCTS** serving recovering batch throughput.
- **Optimal search-shape allocation** per query difficulty ([§03](03_dynamic_compute_allocation.md)).
- **KV-sharing-aware schedulers** for large branching workloads at scale.

***

## References
- Yao, S., et al. (2023). "Tree of Thoughts." *NeurIPS 2023*. arXiv:2305.10601.
- Silver, D., et al. (2016). "Mastering the Game of Go (AlphaGo/MCTS)." *Nature*.
- Google DeepMind (2024). AlphaProof / AlphaGeometry reports.
