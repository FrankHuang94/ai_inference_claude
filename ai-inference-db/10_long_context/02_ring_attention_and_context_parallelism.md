# Ring Attention and Context Parallelism

> **Section:** 10_long_context
> **Last Updated:** June 2026
> **Related Files:** [../04_parallelism/02_sequence_parallelism.md](../04_parallelism/02_sequence_parallelism.md), [00_long_context_challenges.md](00_long_context_challenges.md), [05_kv_cache_management_at_1M_tokens.md](05_kv_cache_management_at_1M_tokens.md)
> **Must-Read Papers:** Liu et al. (2023) "Ring Attention with Blockwise Transformers"; Jacobs et al. (2023) "DeepSpeed-Ulysses"; Korthikanti et al. (2022) "Sequence Parallelism"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- Context parallelism (CP) distributes a long sequence across GPUs so each holds part of the tokens + KV, enabling contexts that exceed one GPU's memory and parallelizing the O(N²) attention.
- **Ring Attention** (Liu et al. 2023): GPUs in a ring; each computes attention on its local KV block, then passes KV around the ring — overlapping communication with compute — until every query has attended to the full sequence. Memory O(N/P) per GPU, near-unbounded context.
- **DeepSpeed-Ulysses**: uses **all-to-all** to redistribute heads/sequence so each GPU computes full attention for a subset of heads — different comm pattern (bandwidth-heavy, latency-light).
- **Causal load imbalance**: later positions attend to more keys; naive splits imbalance work → interleave tokens.
- The key enabler of 1M-token serving on multi-GPU.

***

## Overview
When a sequence is too long for one GPU's memory (the linear-KV problem, [§00](00_long_context_challenges.md)) or its O(N²) attention is too slow on one device, you distribute the **sequence dimension** across GPUs — **context parallelism**. Each GPU holds a contiguous chunk of the tokens and their KV, so per-GPU memory is O(N/P), letting aggregate context scale with the number of GPUs. The challenge is that attention is inherently **all-to-all across positions** — every query must attend to every key — so the GPUs must exchange KV information. Two communication strategies dominate, trading off differently.

**Ring Attention** (Liu et al. 2023) arranges the P GPUs in a logical **ring**. Each GPU starts with its local Q, K, V; it computes the partial attention of its local queries against its local KV (using online softmax, like FlashAttention), then **passes its KV block to the next GPU** in the ring while receiving the previous neighbor's block, and accumulates. After P steps, every query has attended to all keys — exact full attention — with per-GPU memory O(N/P) and, crucially, the KV passing **overlapped with the attention compute** so communication is largely hidden. This enables effectively unbounded context (add more GPUs) and is the basis of many 1M-token systems. **DeepSpeed-Ulysses** takes a different route: an **all-to-all** redistributes the data so that after the exchange each GPU holds the *full sequence* for a *subset of attention heads*, computes standard attention locally, then all-to-alls back. Ulysses is bandwidth-heavy (large all-to-all) but latency-light and simple; Ring is comms-overlapped and memory-optimal but needs careful scheduling.

A subtle but important issue is **causal load imbalance**: with causal masking, token *i* attends to *i* keys, so later positions do more work. A naive contiguous split gives the GPU holding the last chunk far more attention work, creating a straggler. CP implementations **interleave tokens** (e.g., zig-zag assignment) across GPUs to balance the causal load. CP composes with tensor parallelism (TP within a GPU group, CP across the sequence) and is supported in Megatron, vLLM, and SGLang for long-context serving. This file covers Ring vs Ulysses, the overlap mechanics, load balancing, and composition.

***

## Core Concepts & Mechanics

### Ring Attention
- P GPUs in a ring; each holds Q,K,V for N/P tokens.
- Step loop (P iterations): compute local-query × current-KV-block partial attention (online softmax accumulate), **send KV block to next GPU, receive from previous**.
- After P steps: every query has attended to all keys → **exact** full attention.
- 📐 Memory O(N/P) per GPU; communication (KV passing) **overlapped** with compute → near-free if per-step transfer < per-step compute.

### DeepSpeed-Ulysses
- **All-to-all** redistributes: each GPU ends up with the full sequence for a subset of heads; compute standard attention locally; all-to-all back.
- 📐 Comm ≈ O(N·d) per all-to-all; bandwidth-heavy, latency-light; simpler kernels (reuses standard attention).

### Ring vs Ulysses
| | Ring Attention | Ulysses |
|---|---|---|
| Comm pattern | ring (P2P, overlapped) | all-to-all |
| Bandwidth | lower (overlapped) | higher |
| Latency sensitivity | needs overlap | latency-light |
| Memory | O(N/P) | O(N/P) (after redistribute) |
| Best fabric | NVLink/IB w/ overlap | high-BW NVLink |

### Causal load balancing
📐 Token i attends to i keys → contiguous split overloads the last GPU. **Interleave/zig-zag** token assignment so each GPU gets a mix of early and late positions, equalizing work.

### Composition
CP across the sequence + TP within GPU groups; choose CP degree to fit KV (O(N/P)) and parallelize attention. NVL72 makes large CP groups feasible on NVLink ([§01](../01_hardware/02_nvidia_h100_b200_architecture.md)).

***

## Key Challenges
1. **Comm-compute overlap (Ring).** The benefit hinges on hiding KV passing behind attention compute; poor overlap → comm stalls dominate.
2. **All-to-all bandwidth (Ulysses).** Large all-to-all needs high-bandwidth fabric; scales poorly on slow interconnects.
3. **Causal imbalance.** Naive splits create stragglers; requires token interleaving.
4. **Kernel/scheduling complexity.** Ring needs online-softmax accumulation across ring steps + correct causal masking across GPU boundaries — easy to get wrong.

***

## Solutions & Current Best Practices
- **Ring Attention** for memory-optimal, overlapped long-context (1M+) on multi-GPU.
- **Ulysses** when high-bandwidth NVLink is available and simplicity is valued.
- **Token interleaving** for causal load balance.
- **CP + TP composition**; use NVL72 for large CP groups ([§00](00_long_context_challenges.md)).

***

## Implementation Notes
- vLLM/SGLang/Megatron support context/sequence parallelism; set CP degree to fit KV and parallelize attention.
- Verify causal masking correctness across CP boundaries (a common bug) and that comm is overlapped (Nsight Systems).
- Combine with KV quantization/compression for extreme context ([§05](05_kv_cache_management_at_1M_tokens.md)).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Ring comm not overlapped** — KV passing stalled compute instead of hiding behind it; the whole benefit is overlap.
- **Last GPU was the straggler** — contiguous causal split overloaded late positions; interleave tokens.
- **Wrong masking across CP boundaries** — queries attended to wrong keys; silent correctness bug.
- **Ulysses all-to-all saturated IB** — needs high-BW fabric; Ring may be better cross-node.

***

## Performance Numbers & Benchmarks
| Method | Per-GPU memory | Comm | Context enabled |
|---|---|---|---|
| Ring Attention | O(N/P) | overlapped ring | 1M+ (add GPUs) |
| Ulysses | O(N/P) | all-to-all | long, high-BW fabric |
| CP + TP | O(N/P)/group | combined | very long + large model |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Explain Ring Attention and why it enables near-unbounded context."* — Expected: ring KV passing + online softmax, O(N/P) memory, overlapped comm.
- *"Ring vs Ulysses — tradeoffs?"* — Expected: overlapped P2P vs all-to-all; bandwidth/latency/fabric.
- *"What is causal load imbalance and how is it fixed?"* — Expected: late tokens attend more; interleave/zig-zag.
- *"How does CP compose with TP?"* — Expected: CP across sequence, TP within group.

***

## Open Problems & Active Research (2025–2026)
- **Optimal CP scheduling** balancing overlap and causal load at very large P.
- **CP + KV compression/MLA** for extreme context ([§05](05_kv_cache_management_at_1M_tokens.md)).
- **Auto-selecting Ring vs Ulysses** by length and fabric.

***

## References
- Liu, H., Zaharia, M., Abbeel, P. (2023). "Ring Attention with Blockwise Transformers for Near-Infinite Context." arXiv:2310.01889.
- Jacobs, S., et al. (2023). "DeepSpeed-Ulysses." arXiv:2309.14509.
- Korthikanti, V., et al. (2022). "Sequence Parallelism / Reducing Activation Recomputation." *MLSys 2023*. arXiv:2205.05198.
