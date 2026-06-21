# KV Cache Management at 1M Tokens

> **Section:** 10_long_context
> **Last Updated:** June 2026
> **Related Files:** [02_ring_attention_and_context_parallelism.md](02_ring_attention_and_context_parallelism.md), [../02_kv_cache/03_kv_cache_compression.md](../02_kv_cache/03_kv_cache_compression.md), [../05_quantization/05_kv_cache_quantization.md](../05_quantization/05_kv_cache_quantization.md)
> **Must-Read Papers:** Hooper et al. (2024) "KVQuant"; Liu et al. (2024) "KIVI"; DeepSeek-AI (2024) "MLA"; Qin et al. (2024) "Mooncake"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- At 1M tokens, a single request's KV can be **tens to hundreds of GB** — the binding constraint on long-context serving.
- No single technique suffices; production stacks **combine**: **MLA / GQA** (architectural reduction) + **KV quantization** (INT4/FP8) + **compression/eviction** + **context parallelism** + **offloading/tiered KV**.
- 📐 These are roughly **multiplicative**: MLA (≫ reduction) × INT4 (4×) × CP (÷P across GPUs) × offload (capacity) makes 1M tractable.
- **Prefix caching / KV reuse** is essential when long documents are queried repeatedly.
- The trend (Mooncake) is treating KV as a **tiered, global, movable object** across GPU/CPU/SSD/nodes.

***

## Overview
Serving 1M-token contexts is fundamentally a **KV-memory engineering** problem: the linear-in-length KV cache ([§02](../02_kv_cache/00_kv_cache_fundamentals.md)) reaches tens to hundreds of GB for a single request (e.g., ~320 GB for a 70B GQA model at 1M tokens in FP16), which no single GPU can hold and which dwarfs the model weights. There is no silver bullet; instead, production long-context systems **stack** every KV-reduction and KV-distribution technique in this database, exploiting that their savings are roughly **multiplicative**. The art is composing them so they don't conflict.

The stack, from architectural to systems: (1) **Architectural KV reduction** — GQA shrinks KV ~8×, and **MLA** (DeepSeek's low-rank latent KV) shrinks it far more by caching a small latent per token instead of full per-head K,V ([§02](../02_kv_cache/00_kv_cache_fundamentals.md)); this is the single biggest lever and why MLA-based models are attractive for long context. (2) **KV quantization** — INT4/FP8 KV (KIVI, KVQuant) gives another ~4× on top, with per-channel-keys/per-token-values granularity ([§05](../05_quantization/05_kv_cache_quantization.md)). (3) **Compression/eviction** — StreamingLLM/H2O/SnapKV drop tokens unlikely to be attended ([§02](../02_kv_cache/03_kv_cache_compression.md)), though carefully given lost-in-the-middle. (4) **Context parallelism** — distribute the sequence (and its KV) across P GPUs, O(N/P) each ([§02](02_ring_attention_and_context_parallelism.md)). (5) **Offloading/tiered KV** — spill cold KV to CPU/NVMe with prefetch/overlap ([§02](../02_kv_cache/04_kv_cache_offloading.md)). Together: MLA × INT4 × CP(÷P) × offload makes a 1M-token request feasible.

Two cross-cutting concerns complete the picture. **KV reuse**: long documents are often queried repeatedly (RAG, doc QA, multi-turn over a document), so **prefix caching** the document's KV avoids re-prefilling 1M tokens per query — a massive win when applicable ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)). And the **systems framing**: Mooncake's global, tiered, movable KV pool ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)) treats 1M-token KV as a managed object across GPU/CPU/SSD/nodes, enabling reuse and capacity beyond any single GPU. This file covers composing the stack and its gotchas.

***

## Core Concepts & Mechanics

### The multiplicative stack
📐 Effective KV per token ≈ `base / (GQA_factor × MLA_factor × quant_factor)`, distributed by CP (÷P), with offload for capacity.
- GQA: ~8× (vs MHA). MLA: large further reduction (small latent/token). INT4 KV: 4×. CP: ÷P across GPUs. Offload: tiered capacity.
- Example: 1M-token 70B — GQA FP16 ~320 GB → INT4 ~80 GB → CP over 8 GPUs ~10 GB/GPU → feasible; MLA models even less.

### Architectural reduction (biggest lever)
- **GQA** (standard) and especially **MLA** (DeepSeek low-rank latent KV) cut KV at the source ([§02](../02_kv_cache/00_kv_cache_fundamentals.md)). MLA is why DeepSeek serves long context economically.

### Quantization + compression
- **INT4/FP8 KV** (KIVI/KVQuant): per-channel keys, per-token values; validate long-context recall ([§05](../05_quantization/05_kv_cache_quantization.md)).
- **Eviction** (StreamingLLM/H2O/SnapKV): drop low-value tokens, but beware lost-in-the-middle for recall tasks ([§02](../02_kv_cache/03_kv_cache_compression.md)).

### Distribution + offload
- **Context parallelism** (Ring/Ulysses): O(N/P) KV per GPU ([§02](02_ring_attention_and_context_parallelism.md)).
- **Offloading/tiered KV**: cold KV to CPU/NVMe with prefetch/overlap; global tiered pool (Mooncake) ([§02](../02_kv_cache/04_kv_cache_offloading.md), [§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).

### KV reuse
**Prefix-cache** repeated long documents to skip 1M-token re-prefill across queries — essential for doc-QA/RAG workloads ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)).

***

## Key Challenges
1. **Composing techniques.** Quantization + compression + paging + prefix caching + CP must interoperate (e.g., quantized blocks vs exact prefix match) without conflict.
2. **Recall preservation.** Aggressive quant/eviction degrades long-range recall (lost-in-the-middle); must validate on needle/RULER.
3. **Offload latency.** Tiered KV must overlap transfers or it dominates long-context decode ([§02](../02_kv_cache/04_kv_cache_offloading.md)).
4. **Prefill cost remains.** Even with KV managed, the O(N²) prefill compute for 1M tokens persists (needs CP / prefix caching) ([§00](00_long_context_challenges.md)).

***

## Solutions & Current Best Practices
- **Stack MLA/GQA + INT4/FP8 KV + CP + offload**; compose carefully.
- **Prefix-cache repeated long documents** to amortize prefill ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)).
- **Global tiered KV pool** (Mooncake) for cross-request/cross-node reuse and capacity ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).
- **Validate recall** at the target length, not just that it runs.

***

## Implementation Notes
- Choose MLA-based models for long-context-heavy workloads where possible.
- Combine INT4 KV (per-channel keys) + CP; ensure prefix caching handles quantized blocks or accept reduced hits.
- For repeated documents, build a KV cache layer (LMCache/Mooncake) keyed on document content.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Quantized KV broke prefix-cache hits for the shared document** — quantized blocks didn't match; re-prefilled 1M tokens. Verify framework support.
- **Aggressive eviction failed long-doc recall** — dropped the needed mid-document token; validate recall.
- **Offload latency exposed in decode** — cold KV reads weren't overlapped; ITL spiked. Prefetch/overlap.
- **Managed KV but prefill still took seconds** — O(N²) prefill remains; need CP / prefix caching.

***

## Performance Numbers & Benchmarks
| Technique | KV reduction (cumulative) |
|---|---|
| MHA → GQA | ~8× |
| + MLA (DeepSeek) | large further reduction |
| + INT4 KV | ~4× |
| + CP over P GPUs | ÷P per GPU |
| + offload | tiered capacity |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How would you serve a 1M-token context?"* — Expected: stack MLA/GQA + KV quant + compression + CP + offload + prefix caching; multiplicative.
- *"Why is MLA important for long context?"* — Expected: low-rank latent KV → large cache reduction at the source.
- *"What breaks when combining KV quantization with prefix caching?"* — Expected: quantized blocks may not match; reduced hits.
- *"Does managing KV solve long context?"* — Expected: no — O(N²) prefill remains; need CP/prefix caching.

***

## Open Problems & Active Research (2025–2026)
- **Sub-linear KV with preserved recall** for 10M+ contexts.
- **Composable quant + compression + prefix caching** that interoperate cleanly.
- **Standard global tiered KV pools** (Mooncake/LMCache convergence) for long-doc reuse ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).

***

## References
- Hooper, C., et al. (2024). "KVQuant" (10M-context aim). *NeurIPS 2024*. arXiv:2401.18079.
- Liu, Z., et al. (2024). "KIVI." *ICML 2024*. arXiv:2402.02750.
- DeepSeek-AI (2024). "DeepSeek-V2" (MLA). arXiv:2405.04434.
- Qin, R., et al. (2024). "Mooncake." arXiv:2407.00079.
