# Load Balancing Strategies

> **Section:** 09_distributed_inference
> **Last Updated:** June 2026
> **Related Files:** [../04_parallelism/03_data_parallelism_for_serving.md](../04_parallelism/03_data_parallelism_for_serving.md), [../02_kv_cache/02_prefix_caching_and_radix_attention.md](../02_kv_cache/02_prefix_caching_and_radix_attention.md), [00_multi_node_serving.md](00_multi_node_serving.md)
> **Must-Read Papers:** Zheng et al. (2024) "SGLang" (cache-aware routing); Crankshaw et al. (2017, NSDI) "Clipper"; Qin et al. (2024) "Mooncake"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- Routing requests across replicas/instances is where DP-scaled serving's performance is won or lost.
- **Round-robin** ignores load and cache; **least-outstanding-requests** balances heterogeneous costs; **cache-aware (prefix-hash) routing** preserves prefix-cache hits; **session affinity** keeps multi-turn KV on one instance.
- The central tension: **cache-locality routing creates hot spots** (overload the instance holding a hot prefix) vs **load balancing** spreads work but loses cache hits.
- Solution: **hybrid** policies — prefix-aware with load-aware tie-breaking, plus **replication of hot prefixes**.
- LLM-specific: route by KV locality, not just CPU/connection count (unlike classic web LB).

***

## Overview
Once you scale serving with data-parallel replicas ([§04](../04_parallelism/03_data_parallelism_for_serving.md)), the router in front of them determines the system's actual performance. Classic web load balancing (round-robin, least-connections) is insufficient for LLMs because of **KV-cache locality**: a request that shares a prefix with content already cached on a specific instance (a system prompt, a conversation history, a RAG document) should be routed *there* to get a prefix-cache hit and skip prefill ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)). Routing it elsewhere forces a redundant prefill and wastes the cache. So LLM routing must be **cache-aware**, not just load-aware.

But cache-aware routing creates a tension with load balancing. If all requests sharing a hot prefix are routed to the one instance caching it, that instance becomes a **hot spot** while others idle — great cache hit rate, terrible load distribution. Conversely, pure least-loaded routing spreads work evenly but scatters shared prefixes across instances, each rebuilding the cache, collapsing hit rate. The practical answer is **hybrid policies**: route by prefix locality when it yields a hit and the target isn't overloaded, otherwise fall back to least-loaded; and **replicate hot prefixes** across several instances so popular shared content has multiple homes, combining cache hits with load spreading. SGLang and production routers implement prefix-aware routing with load-aware tie-breaking.

Beyond cache locality, LLM routing must account for **heterogeneous request cost** (output length is unknown and varies hugely, so "least connections" misestimates load — better to track in-flight tokens or use least-outstanding-requests), **session affinity** (multi-turn conversations should stick to the instance holding their KV), and, in disaggregated systems, **KV-aware routing** (which decode instance receives a prefill's KV) ([§03](03_kv_cache_migration_and_transfer.md)). This file covers the policies, the locality-vs-balance tension, and the hybrid solutions.

***

## Core Concepts & Mechanics

### Policies
- **Round-robin**: even distribution, ignores load and cache. Baseline only.
- **Least-outstanding-requests / least in-flight tokens**: route to the least-busy instance; handles heterogeneous LLM request costs better than least-connections.
- **Cache-aware (prefix-hash) routing**: hash the prompt prefix; route to the instance owning that prefix's KV → prefix-cache hit.
- **Session affinity (sticky)**: route a conversation's turns to the same instance to reuse its KV.
- **KV-aware (disaggregation)**: route a prefill's KV to a chosen decode instance ([§03](03_kv_cache_migration_and_transfer.md)).

### The locality–balance tension
📐 Effective hit rate `h_eff = h_potential × P(routed to holder)`. Pure prefix routing maximizes `P` but concentrates load (hot spot). Pure load routing maximizes balance but `P→1/K` (K instances) → low hits. Hybrid + hot-prefix replication optimizes both.

### Hybrid policy
- Prefer the prefix-owning instance **if** its load < threshold; else least-loaded.
- **Replicate hot prefixes** across R instances; route among them by load → cache hits + spreading.
- Track per-instance load (in-flight tokens) and prefix ownership.

### Health & lifecycle
Router also does health checks, retries on failure, and **draining** (stop sending to an instance being replaced) ([§05](05_fault_tolerance_in_serving.md)).

***

## Key Challenges
1. **Locality vs balance.** The core tension; naive cache-aware routing hot-spots, naive load routing loses cache.
2. **Unknown request cost.** Output length is unknown; load estimation (connections) misleads — use in-flight tokens.
3. **Hot-prefix dynamics.** Popular prefixes shift over time; the router must adapt replication/ownership.
4. **Disaggregation routing.** Choosing the decode instance for a prefill's KV adds a dimension ([§03](03_kv_cache_migration_and_transfer.md)).

***

## Solutions & Current Best Practices
- **Hybrid prefix-aware + load-aware** routing; **replicate hot prefixes** ([§02](../02_kv_cache/05_cross_request_kv_sharing.md)).
- **Least-outstanding / in-flight-token** load metric for heterogeneous requests.
- **Session affinity** for multi-turn KV reuse.
- **Adaptive hot-prefix detection/replication** as traffic shifts.

***

## Implementation Notes
- Use SGLang's router or a custom gateway supporting prefix-aware + load-aware policies.
- Measure **cache hit rate** and **per-instance load**; alert on hot spots.
- For disaggregation, integrate KV-aware routing with the global KV pool ([§03](03_kv_cache_migration_and_transfer.md)).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Cache-aware routing created a hot instance** — all hot-prefix traffic concentrated; replicate hot prefixes / load-aware tie-break.
- **Round-robin collapsed hit rate** — shared prefixes scattered; each instance rebuilt cache.
- **Least-connections misjudged load** — long-output requests counted as one connection but cost far more; use in-flight tokens.
- **Multi-turn re-prefilled every turn** — no session affinity; conversation bounced across instances.

***

## Performance Numbers & Benchmarks
| Policy | Cache hits | Load balance | LLM fit |
|---|---|---|---|
| Round-robin | poor | good | baseline |
| Least-outstanding | poor | good | better load |
| Cache-aware (prefix) | high | hot-spot risk | high (with care) |
| Hybrid + replication | high | good | best |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why is LLM load balancing different from web load balancing?"* — Expected: KV-cache locality; route by prefix, not just connections.
- *"What's the tension in cache-aware routing and how do you resolve it?"* — Expected: locality vs balance; hybrid + hot-prefix replication.
- *"What load metric should an LLM router use?"* — Expected: in-flight tokens / least-outstanding, since output length varies.
- *"How do you handle multi-turn conversations?"* — Expected: session affinity to reuse KV.

***

## Open Problems & Active Research (2025–2026)
- **Optimal hybrid routing** with provable locality/balance tradeoffs at scale.
- **Predictive routing** using output-length/cost estimates.
- **Global-KV-aware routing** for disaggregated multi-node serving ([§03](03_kv_cache_migration_and_transfer.md)).

***

## References
- Zheng, L., et al. (2024). "SGLang" (cache-aware routing). arXiv:2312.07104.
- Crankshaw, D., et al. (2017). "Clipper." *NSDI 2017*.
- Qin, R., et al. (2024). "Mooncake." arXiv:2407.00079.
