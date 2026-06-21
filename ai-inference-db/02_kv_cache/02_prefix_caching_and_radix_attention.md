# Prefix Caching and RadixAttention

> **Section:** 02_kv_cache
> **Last Updated:** June 2026
> **Related Files:** [01_paged_attention_vllm.md](01_paged_attention_vllm.md), [05_cross_request_kv_sharing.md](05_cross_request_kv_sharing.md), [../08_serving_frameworks/02_sglang_deep_dive.md](../08_serving_frameworks/02_sglang_deep_dive.md)
> **Must-Read Papers:** Zheng et al. (2024) "SGLang: Efficient Execution of Structured LM Programs" (RadixAttention); Kwon et al. (2023, SOSP) "vLLM"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- **Prefix caching** reuses the KV cache of a shared prompt prefix across requests, **skipping prefill** for the shared part — huge TTFT and cost wins for shared system prompts, multi-turn chat, RAG.
- **vLLM exact prefix caching**: hash-based, reuses KV only for an *exact* matching prefix (block-aligned).
- **RadixAttention (SGLang, Zheng et al. 2024)**: a **radix tree (trie)** of token sequences indexing KV blocks, enabling reuse of *any* shared prefix — including nested and branching prefixes — automatically.
- Eviction is **LRU over tree nodes**; cache hit rate directly drives TTFT and throughput.
- Biggest benefit when prefixes are long and widely shared (system prompts, few-shot, conversation history, common RAG context).

***

## Overview
Much production traffic shares large prompt prefixes: a fixed system prompt prepended to every request, few-shot exemplars reused across calls, the growing history of a multi-turn conversation, or a common document prefixed to many RAG queries. Recomputing the prefill (and re-storing the KV) for these shared tokens on every request is pure waste — the KV is identical. **Prefix caching** keeps the shared prefix's KV blocks in HBM and reuses them, so a new request with that prefix skips prefill for the shared portion and only prefills its unique suffix. This slashes TTFT (often the dominant latency for long shared prompts) and increases effective throughput by removing redundant compute.

vLLM implements **automatic prefix caching** via hashing block-aligned prefixes; if an incoming request's leading blocks hash-match cached blocks, they're reused (ref-counted, copy-on-write on divergence — built on PagedAttention's sharing primitives, [§01](01_paged_attention_vllm.md)). The limitation is that matching is essentially exact and prefix-anchored. **RadixAttention** (SGLang, Zheng et al. 2024) generalizes this with a **radix tree** keyed by token sequences: every cached sequence is a path from the root, and any new request walks the tree matching the longest shared prefix, reusing all KV blocks along that path. Because the tree naturally represents branching and nested prefixes, RadixAttention reuses KV across complex sharing patterns (e.g., a conversation that branches into multiple continuations) with no manual cache management — the cache is the tree, and eviction is LRU over its nodes.

The payoff is workload-dependent but can be dramatic: when system prompts or few-shot contexts dominate the prompt, cache hit rates of 50–90% are common, turning most prefill into a cheap tree lookup. This file covers the mechanics, the matching/eviction algorithms, and when prefix caching helps most — and its gotchas.

***

## Core Concepts & Mechanics

### Exact prefix caching (vLLM)
- Prompts are tokenized and split into blocks (block_size, e.g., 16). Each prefix-block sequence is **hashed**; the hash → physical block mapping is stored.
- New request: hash its leading blocks; reuse matching physical blocks (ref-count++), prefill only the unmatched suffix.
- **Copy-on-write** when generation diverges from a shared block. Matching is block-aligned and prefix-anchored (must match from token 0).

### RadixAttention (SGLang)
- A **radix tree** where edges are token substrings and each node owns the KV blocks for its path. Root = empty; a full cached sequence = a leaf path.
- **Insertion:** after processing a request, insert its token sequence, creating/extending nodes; KV blocks attach to nodes.
- **Matching:** for a new request, traverse from root matching the longest common prefix; reuse KV along the matched path; prefill only the remainder.
- **Branching:** two requests sharing a prefix then diverging share the common path and branch into separate child nodes — automatic nested/branched reuse.
- **Eviction:** LRU over leaf/least-recently-used nodes; evicting frees the node's KV blocks. Reference-counted so in-use paths aren't evicted.

📐 If shared prefix length is `S_shared` of total prompt `P`, prefill work drops from `O(P)` to `O(P − S_shared)`; TTFT improves proportionally for the (compute-bound) prefill. Cache hit rate `h = E[S_shared]/E[P]` predicts savings.

### When it helps most
- **Shared system prompts** (every request shares a long preamble) → near-total prefill skip.
- **Multi-turn chat**: turn *t+1* shares the entire history of turn *t* → reuse all prior KV; only prefill the new user turn.
- **Few-shot / RAG with common context**: shared exemplars/documents.
- **Tree-of-thought / parallel sampling**: branches share the trunk (RadixAttention shines).

***

## Key Challenges
1. **Memory vs hit-rate tradeoff.** Cached prefixes occupy HBM that could hold active KV; aggressive caching can reduce batch capacity. Eviction policy must balance reuse value vs memory.
2. **Exact-match brittleness (vLLM).** A one-token difference at the start (e.g., a timestamp in the system prompt) defeats prefix caching entirely; prompts must be structured for cache-friendliness.
3. **Quantized/compressed KV complicates matching.** Quantized blocks may not match bit-identically; mixing prefix caching with KV quantization needs care ([§05](../05_quantization/05_kv_cache_quantization.md)).
4. **Privacy across tenants.** Sharing prefixes across users (e.g., common system prompt) is fine, but accidentally sharing user-specific content is a data-leak risk ([§13](../13_production_systems/04_multi_tenant_serving.md)).

***

## Solutions & Current Best Practices
- **Enable prefix caching by default** for shared-prompt workloads (vLLM `enable_prefix_caching`, SGLang RadixAttention on by default).
- **Structure prompts for cache-friendliness**: put stable content (system prompt, exemplars) first, volatile content (timestamps, user input) last.
- **Tune cache size / eviction** to balance reuse against active-KV capacity.
- **Scope cross-tenant sharing** to non-sensitive shared prefixes only ([§05](05_cross_request_kv_sharing.md)).

***

## Implementation Notes
- Measure **cache hit rate** as a first-class metric; it directly predicts TTFT/throughput gains.
- For multi-turn, keep the conversation's KV resident (or quickly re-attachable) so each turn reuses history; otherwise you re-prefill the whole conversation.
- RadixAttention's tree is per-instance; in multi-instance/disaggregated serving, route requests with shared prefixes to the same instance (cache-aware routing, [§09](../09_distributed_inference/01_load_balancing_strategies.md)).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **A timestamp/UUID at the top of the system prompt destroys all prefix reuse** — exact match fails from token 0. Move volatile fields to the end.
- **Cache-aware routing is required at scale** — without routing shared-prefix requests to the same instance, each instance rebuilds the cache and hit rate craters.
- **Prefix caching + KV quantization mismatch** — quantized blocks may not match across requests with different scales; verify the framework handles it.
- **Over-caching reduces active batch** — pinned prefix KV eats HBM; under memory pressure it can hurt throughput. Bound cache size.

***

## Performance Numbers & Benchmarks
| Workload | Shared prefix | Effect |
|---|---|---|
| Long system prompt (1k tokens) on short queries | ~90% | TTFT/prefill cost down ~10× for shared part |
| Multi-turn chat | full history | each turn skips prior-turn prefill |
| Tree-of-thought / parallel samples | trunk | RadixAttention reuses trunk across branches |
| SGLang vs vLLM on shared-prefix bench (Zheng 2024) | — | up to several× higher throughput |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How does prefix caching reduce TTFT, and what's the math?"* — Expected: skip prefill for shared prefix; savings ∝ hit rate.
- *"Contrast vLLM exact prefix caching with SGLang RadixAttention."* — Expected: hash exact-match vs radix-tree longest-prefix incl. branching/nesting.
- *"Why is cache-aware routing necessary in multi-instance serving?"* — Expected: keep shared prefixes on one instance to preserve hit rate.
- *"What can silently defeat prefix caching?"* — Expected: volatile tokens at prompt start; quantized-block mismatch.

***

## Open Problems & Active Research (2025–2026)
- **Global, cross-instance prefix caches** (shared KV store) for disaggregated/multi-node serving ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).
- **Approximate/semantic prefix matching** beyond exact token match.
- **Prefix caching with compressed/quantized KV** and with MLA's latent KV.

***

## References
- Zheng, L., et al. (2024). "SGLang: Efficient Execution of Structured Language Model Programs" (RadixAttention). arXiv:2312.07104.
- Kwon, W., et al. (2023). "vLLM/PagedAttention." *SOSP 2023*. arXiv:2309.06180.
- Gim, I., et al. (2024). "Prompt Cache: Modular Attention Reuse for Low-Latency Inference." *MLSys 2024*. arXiv:2311.04934.
