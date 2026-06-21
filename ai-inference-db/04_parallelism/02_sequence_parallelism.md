# Sequence Parallelism

> **Section:** 04_parallelism
> **Last Updated:** June 2026
> **Related Files:** [00_tensor_parallelism.md](00_tensor_parallelism.md), [../10_long_context/02_ring_attention_and_context_parallelism.md](../10_long_context/02_ring_attention_and_context_parallelism.md), [05_parallelism_strategy_selection.md](05_parallelism_strategy_selection.md)
> **Must-Read Papers:** Korthikanti et al. (2022, MLSys) "Reducing Activation Recomputation" (sequence parallelism); Jacobs et al. (2023) "DeepSpeed-Ulysses"; Liu et al. (2023) "Ring Attention"
> **Estimated Study Time:** 20 minutes

***

## TL;DR
- Sequence parallelism (SP) shards tensors along the **sequence dimension**, distributing activations (and the normalization/elementwise regions TP leaves replicated) across GPUs.
- Originally an activation-memory optimization (Korthikanti et al. 2022) that complements TP: SP handles LayerNorm/dropout regions, TP handles attention/FFN, with conversions between them.
- For **long-context inference**, SP/context parallelism distributes the sequence so each GPU holds part of the KV — the key to 1M-token serving ([§10](../10_long_context/02_ring_attention_and_context_parallelism.md)).
- Two communication patterns: **all-to-all** (Ulysses) vs **ring** (Ring Attention) — different bandwidth/latency tradeoffs.
- Mainly relevant to **prefill and long context**; less impactful for short-context decode.

***

## Overview
Tensor parallelism shards the attention and FFN matmuls but leaves the LayerNorm, dropout, and residual regions **replicated** across TP ranks, so their activations (and the associated memory) aren't distributed. **Sequence parallelism** (Korthikanti et al. 2022) closes this gap by sharding those regions along the **sequence dimension**: each GPU owns a slice of the tokens for the norm/elementwise parts, then the system converts (via all-gather/reduce-scatter) into TP's layout for the attention/FFN parts. The original motivation was reducing **activation memory** during training, but the same idea — distributing work and state along the sequence axis — is central to **long-context inference**, where the sequence (and its KV cache) is too large for one GPU.

In the inference/long-context framing, SP generalizes to **context parallelism**: split a very long sequence across GPUs so each holds part of the tokens and part of the KV cache, then exchange the information needed for attention (which is inherently all-to-all across positions). Two communication strategies dominate: **DeepSpeed-Ulysses** uses an **all-to-all** to redistribute the head/sequence dimensions so each GPU computes full attention for a subset of heads over the full sequence; **Ring Attention** passes KV blocks around a **ring** so each GPU eventually attends to the whole sequence while overlapping communication with compute ([§10](../10_long_context/02_ring_attention_and_context_parallelism.md)). They trade off differently: all-to-all is bandwidth-heavy but latency-light; ring overlaps comm with compute but needs careful scheduling.

For typical short-context serving, SP adds little (decode is one token; norms are cheap). Its payoff is in **prefill of long prompts** and **long-context decode**, where distributing the sequence is the only way to fit and to parallelize the quadratic attention work. So SP/context parallelism is best understood as the long-context member of the parallelism family, complementary to TP (within layer) and PP (across layers).

***

## Core Concepts & Mechanics

### SP + TP composition (Korthikanti)
- **TP regions**: attention, FFN (sharded by heads/hidden dim).
- **SP regions**: LayerNorm, dropout, residual (sharded by sequence).
- Transitions: **all-gather** (SP→TP) before attention/FFN, **reduce-scatter** (TP→SP) after. Net communication volume is comparable to TP's all-reduce, but activation memory drops by the TP degree.

### Context parallelism for long sequences
Split sequence of length S across `c` GPUs: each holds `S/c` tokens and their KV. Attention requires each query to see all keys → cross-GPU exchange:
- **Ulysses (all-to-all)**: redistribute so each GPU has all positions for a subset of heads; compute local attention; all-to-all back. 📐 Comm ≈ `O(S·d)` per all-to-all, latency-light, bandwidth-heavy.
- **Ring Attention**: arrange GPUs in a ring; each step, compute attention against the local KV block, then pass KV to the neighbor; after `c` steps each query has attended to all keys. Overlaps comm with compute; memory `O(S/c)` per GPU. Enables effectively unbounded context across GPUs.

### When it matters
- **Long prefill** (10k–1M tokens): SP/CP distributes the quadratic attention and the activation memory.
- **Long-context decode**: KV spread across GPUs (each step still cheap, but KV fits).
- **Short context**: minimal benefit.

***

## Key Challenges
1. **Communication overhead.** All-to-all (Ulysses) is bandwidth-intensive; ring requires careful overlap or comm stalls compute — both add cost.
2. **Load balance across the sequence.** Causal masking makes later positions attend to more keys; naive equal splits imbalance work (some CP schemes interleave tokens to balance).
3. **Composition complexity.** SP + TP + PP (+ EP) together require careful layout conversions and correctness — easy to get wrong.
4. **Limited decode benefit.** SP's wins are in prefill/long-context; for short-context decode it adds complexity for little gain.

***

## Solutions & Current Best Practices
- **Use SP with TP** to cut activation memory in long-prefill serving (Korthikanti scheme).
- **Ring Attention / Ulysses (context parallelism)** for 1M-token contexts ([§10](../10_long_context/02_ring_attention_and_context_parallelism.md)).
- **Balance causal load** via token interleaving in CP.
- **Reserve SP/CP for long-context workloads**; don't add it to short-context serving.

***

## Implementation Notes
- vLLM/SGLang and Megatron support context/sequence parallelism for long context; configure CP degree alongside TP/PP.
- Choose Ulysses vs Ring by fabric: all-to-all favors high-bandwidth NVLink domains; ring favors overlap on longer/looser fabrics.
- Validate attention correctness across the CP boundary (causal masking across GPU-local blocks is a common bug).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Causal load imbalance** — equal sequence splits give later GPUs more attention work; interleave tokens to balance.
- **Ring comm not overlapped** — KV passing stalls compute instead of hiding behind it; the whole benefit is overlap.
- **Added SP to short-context serving** — complexity with no benefit; SP is for long context/prefill.
- **Wrong masking across CP boundaries** — silent correctness bug where queries attend to wrong keys.

***

## Performance Numbers & Benchmarks
| Technique | Comm pattern | Best fabric | Use |
|---|---|---|---|
| SP+TP (Korthikanti) | all-gather/reduce-scatter | NVLink | activation memory, long prefill |
| Ulysses | all-to-all | NVLink (high BW) | long context |
| Ring Attention | ring (overlapped) | NVLink/IB | 1M+ context, memory-bound |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"What does sequence parallelism shard that tensor parallelism doesn't?"* — Expected: norm/dropout/residual along sequence; activation memory.
- *"Contrast Ulysses all-to-all vs Ring Attention for long context."* — Expected: bandwidth-heavy/latency-light vs comm-compute overlap/unbounded.
- *"Why is causal load imbalance a problem in context parallelism?"* — Expected: later tokens attend to more keys; interleave to balance.
- *"When is SP not worth it?"* — Expected: short-context decode.

***

## Open Problems & Active Research (2025–2026)
- **Optimal context-parallel scheduling** balancing causal load and overlapping communication.
- **Combining CP with KV compression/MLA** for extreme context ([§10](../10_long_context/05_kv_cache_management_at_1M_tokens.md)).
- **Auto-selecting Ulysses vs Ring** by sequence length and fabric.

***

## References
- Korthikanti, V., et al. (2022). "Reducing Activation Recomputation in Large Transformer Models." *MLSys 2023*. arXiv:2205.05198.
- Jacobs, S., et al. (2023). "DeepSpeed-Ulysses." arXiv:2309.14509.
- Liu, H., Zaharia, M., Abbeel, P. (2023). "Ring Attention with Blockwise Transformers for Near-Infinite Context." arXiv:2310.01889.
