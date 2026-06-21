# Cross-Request KV Sharing

> **Section:** 02_kv_cache
> **Last Updated:** June 2026
> **Related Files:** [02_prefix_caching_and_radix_attention.md](02_prefix_caching_and_radix_attention.md), [04_kv_cache_offloading.md](04_kv_cache_offloading.md), [../13_production_systems/04_multi_tenant_serving.md](../13_production_systems/04_multi_tenant_serving.md)
> **Must-Read Papers:** Zheng et al. (2024) "RadixAttention/SGLang"; Gim et al. (2024, MLSys) "Prompt Cache"; Kwon et al. (2023, SOSP) "vLLM"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- Cross-request KV sharing = reusing one copy of KV blocks across **multiple concurrent requests** that share content (prefix caching is the common case; this file generalizes it).
- **Copy-on-write + ref-counting** (from PagedAttention) is the enabling primitive; **RadixAttention** is the automatic, branching-aware mechanism.
- Beyond prefixes: **modular/non-contiguous reuse** (Prompt Cache) lets reusable text *segments* (a document, a tool spec) share KV even when not a strict prefix.
- **Routing is essential**: shared-content requests must land on the instance holding the KV (cache-aware load balancing).
- **Privacy/isolation**: sharing across tenants is safe only for non-sensitive shared content; user data must not leak via shared KV.

***

## Overview
Prefix caching ([§02](02_prefix_caching_and_radix_attention.md)) is the most common form of a more general idea: many requests in flight share *some* token content, and that content's KV need only exist once. Cross-request sharing keeps a single physical copy of shared KV blocks and points multiple requests' block tables at it (ref-counted), cloning on divergence (copy-on-write). This saves both the **compute** of re-prefilling shared content and the **memory** of storing duplicate KV — which in turn raises achievable batch size and throughput. At scale, with long shared system prompts or documents, the savings are large enough to change the cost structure of a service.

The generalization beyond strict prefixes is the frontier. **Prompt Cache** (Gim et al. 2024) introduces "prompt modules": reusable text segments (a system prompt, a few-shot block, a document) whose KV is precomputed and reused even when they appear at different positions or in different combinations — using positional handling to make non-contiguous reuse valid. **RadixAttention** captures branching/nested sharing automatically via its trie. Together these turn KV into a library of reusable building blocks, especially powerful for agentic and RAG systems that recombine the same tools/documents across many calls.

Two hard constraints govern production use. First, **routing**: a shared KV block lives on a specific instance, so to get a cache hit you must route requests with shared content to that instance — naive round-robin destroys hit rates ([§09](../09_distributed_inference/01_load_balancing_strategies.md)). Second, **isolation**: sharing KV across users is only acceptable for content that is genuinely common (a public system prompt); sharing user-specific content risks leaking one tenant's data into another's context or via timing side-channels ([§13](../13_production_systems/04_multi_tenant_serving.md)).

***

## Core Concepts & Mechanics

### The enabling primitive: CoW + ref-counting
- Shared physical blocks carry a **reference count**; multiple requests' block tables map to them.
- On a **write that diverges** (a request generates/contains a different next token in a shared block), the block is **copied** and the writer gets a private copy (copy-on-write); ref-count of the original decrements.
- Eviction only frees blocks with ref-count 0. This is exactly PagedAttention's sharing mechanism ([§01](01_paged_attention_vllm.md)).

### Forms of sharing
1. **Prefix sharing** (most common): shared leading tokens; exact (vLLM hash) or trie-based (RadixAttention).
2. **Parallel sampling / beam / tree search**: N candidates of one request share the trunk; branches diverge with CoW. RadixAttention is ideal.
3. **Modular segment reuse** (Prompt Cache): non-prefix reusable segments (document, tool spec) share KV across requests/positions.
4. **Conversation reuse**: a session's KV reused across turns (a degenerate single-tenant case).

### Routing for hits
📐 Effective hit rate `h_eff = h_potential × P(routed to holder)`. Round-robin across K instances gives `P ≈ 1/K` for a given shared block — so cache-aware routing (hash on prefix, sticky sessions, or a shared KV store) is required to realize the potential `h`. SGLang and production routers implement prefix-aware routing.

### Cross-instance sharing
A single instance's cache is bounded; **global KV stores** (LMCache, Mooncake's KV pool) let instances share KV over the network, decoupling the cache from any one GPU ([§04](04_kv_cache_offloading.md), [§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).

***

## Key Challenges
1. **Routing complexity at scale.** Maximizing hits requires content-aware routing that fights load balancing (you may overload the instance holding a hot prefix) — a real tension ([§09](../09_distributed_inference/01_load_balancing_strategies.md)).
2. **Privacy and isolation.** Shared KV across tenants can leak data or create timing side-channels (a fast response reveals a cache hit, hence what others sent); must scope sharing carefully.
3. **Positional correctness for non-prefix reuse.** A segment reused at a different position has different positional encodings; Prompt Cache must handle RoPE/position offsets to keep reuse valid.
4. **Eviction of hot shared blocks.** Evicting a widely-shared block forces many requests to re-prefill; eviction must weight ref-count/sharing value, not just recency.

***

## Solutions & Current Best Practices
- **RadixAttention / automatic prefix caching** for transparent prefix and branch sharing.
- **Prompt Cache-style modules** for RAG/agentic systems recombining documents and tool specs.
- **Cache-aware (prefix-hash / sticky) routing** to realize hit potential ([§09](../09_distributed_inference/01_load_balancing_strategies.md)).
- **Tenant-scoped sharing**: share only public/common content; isolate user data; consider per-tenant cache partitions ([§13](../13_production_systems/04_multi_tenant_serving.md)).

***

## Implementation Notes
- Weight eviction by **sharing value** (ref-count × prefix length) so hot shared prefixes survive.
- For multi-tenant, gate cross-tenant sharing behind an allowlist of shared system prompts; never share user-specific segments.
- Use a **global KV store** for cross-instance/disaggregated reuse; budget network bandwidth for reloads.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Cache-aware routing creates hot spots** — routing all shared-prefix traffic to one instance overloads it; balance hit rate vs load (e.g., replicate hot prefixes).
- **Timing side-channel from shared KV** — a suspiciously fast TTFT reveals a cache hit, leaking that someone else sent that prefix; a real multi-tenant privacy concern.
- **Non-prefix reuse with wrong positions corrupts output** — reusing a segment's KV at a different position without position handling breaks attention.
- **Evicting a hot shared prefix causes a thundering herd** — many requests re-prefill simultaneously; protect high-share blocks.

***

## Performance Numbers & Benchmarks
| Sharing form | Mechanism | Typical benefit |
|---|---|---|
| Long shared system prompt | prefix cache | skip ~all shared prefill |
| Parallel sampling N=8 | CoW trunk | ~N× prompt-KV savings |
| RAG shared document | Prompt Cache module | reuse doc KV across queries |
| Multi-turn session | conversation reuse | each turn skips history prefill |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How do multiple requests safely share KV blocks?"* — Expected: ref-counted shared blocks + copy-on-write on divergence.
- *"Why does cross-request sharing require special routing?"* — Expected: KV lives on one instance; cache-aware routing to realize hits.
- *"What are the privacy risks of sharing KV across tenants?"* — Expected: data leakage and timing side-channels; scope to public content.
- *"How would you reuse a document's KV across many RAG queries?"* — Expected: Prompt Cache modules / prefix cache with position handling.

***

## Open Problems & Active Research (2025–2026)
- **Balancing hit-rate-maximizing routing with load balancing** (replication of hot prefixes, hybrid policies).
- **Secure cross-tenant KV sharing** that prevents side-channels while keeping savings.
- **Global, compressed KV libraries** for agentic systems recombining many reusable segments ([§12](../12_reasoning_model_inference/02_tree_search_and_mcts_serving.md)).

***

## References
- Zheng, L., et al. (2024). "SGLang/RadixAttention." arXiv:2312.07104.
- Gim, I., et al. (2024). "Prompt Cache: Modular Attention Reuse for Low-Latency Inference." *MLSys 2024*. arXiv:2311.04934.
- Kwon, W., et al. (2023). "vLLM/PagedAttention." *SOSP 2023*. arXiv:2309.06180.
- Qin, R., et al. (2024). "Mooncake." arXiv:2407.00079.
