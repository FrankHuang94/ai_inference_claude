# SGLang Deep Dive

> **Section:** 08_serving_frameworks
> **Last Updated:** June 2026
> **Related Files:** [01_vllm_deep_dive.md](01_vllm_deep_dive.md), [../02_kv_cache/02_prefix_caching_and_radix_attention.md](../02_kv_cache/02_prefix_caching_and_radix_attention.md), [06_framework_selection_matrix.md](06_framework_selection_matrix.md)
> **Must-Read Papers:** Zheng et al. (2024) "SGLang: Efficient Execution of Structured Language Model Programs"; Kwon et al. (2023) "vLLM"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- SGLang is a high-performance serving framework whose defining innovations are **RadixAttention** (trie-based prefix KV reuse) and a **zero-overhead / overlap CPU scheduler**.
- **RadixAttention** automatically reuses KV for *any* shared prefix (nested, branching) — excellent for shared system prompts, multi-turn, and structured/agentic workloads ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)).
- The **overlap scheduler** hides CPU scheduling behind GPU compute, avoiding the Python-GIL bottleneck that limited vLLM at high QPS.
- Also: chunked prefill, **PD disaggregation**, FP8 attention, Triton kernels, and a **DSL** for structured generation (constrained decoding, branching).
- Often the **throughput leader** in 2025–2026 benchmarks for shared-prefix / high-QPS / structured workloads.

***

## Overview
SGLang (Zheng et al. 2024) is a serving framework that started from the observation that real LLM workloads — multi-turn chat, few-shot prompting, agentic/tree-of-thought programs, RAG with shared context — have enormous **prefix sharing** that exact, prefix-anchored caching underexploits. Its flagship contribution, **RadixAttention**, indexes cached KV blocks in a **radix tree (trie)** keyed by token sequences, so any new request automatically reuses the KV of the longest matching prefix, including **nested and branching** prefixes, with LRU eviction over tree nodes ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)). For workloads with heavy shared context this yields large throughput and TTFT gains over exact prefix caching, and it composes naturally with SGLang's **DSL** for structured LM programs (parallel branches, constrained/JSON generation) where branches share a trunk.

SGLang's second major contribution is **scheduler engineering**. It identified that the per-iteration CPU scheduling cost — Python, GIL-bound — was a real ceiling at high QPS in earlier engines ([§03](../03_batching_and_scheduling/01_iteration_level_scheduling.md)). SGLang's **zero-overhead scheduler** minimizes that cost, and its **overlap scheduler** runs the next iteration's CPU scheduling *concurrently* with the current iteration's GPU compute, taking scheduling off the critical path entirely. The effect is that SGLang stays GPU-bound (not CPU-bound) at high request rates, a key reason it frequently tops throughput benchmarks. On top of this it implements the modern feature set: chunked prefill, prefill-decode disaggregation, FP8 (including FP8 attention), Triton-based custom kernels, speculative decoding, and tensor/pipeline/expert parallelism (with strong MLA/DeepSeek support).

The practical positioning: SGLang tends to **win on high-request-rate workloads with shared prefixes, structured/constrained generation, and multi-turn conversations**, and is a strong general performer. vLLM remains the broadest/most-adopted and TensorRT-LLM the fixed-shape latency leader ([§06](06_framework_selection_matrix.md)); the three leapfrog each other release-to-release (a watchlist item). This file covers RadixAttention, the schedulers, the DSL, and where SGLang excels.

***

## Core Concepts & Mechanics

### RadixAttention
- KV blocks indexed in a **radix tree**; nodes own KV for their token-path. New requests traverse from root matching the **longest common prefix**, reusing KV along it; only the remainder is prefilled.
- Handles **nested/branching** sharing automatically (e.g., one system prompt branching into many continuations). LRU eviction over nodes; ref-counted for in-use paths.
- 📐 Prefill work drops from O(P) to O(P − S_shared); excels when shared prefixes are long/common.

### Zero-overhead & overlap schedulers
- **Zero-overhead**: minimize per-iteration CPU scheduling cost (avoid GIL contention, efficient data structures).
- **Overlap**: compute iteration *t+1*'s schedule on CPU while iteration *t* runs on GPU → scheduling off the critical path; stays GPU-bound at high QPS.

### Feature set
- **Chunked prefill**, **PD disaggregation**, **FP8** (weights + attention), **Triton kernels**, **speculative decoding** (EAGLE), **MLA/DeepSeek** support, TP/PP/EP.
- **SGLang DSL**: a frontend for structured LM programs — parallelism, branching, **constrained decoding** (regex/JSON/grammar), tool use — that maps onto RadixAttention's sharing.

### Benchmarks positioning
SGLang frequently leads throughput on **shared-prefix, high-QPS, structured, multi-turn** workloads; competitive elsewhere. Rankings shift across releases vs vLLM/TRT-LLM.

***

## Key Challenges
1. **RadixAttention memory pressure.** Caching many prefixes consumes HBM that could hold active KV; eviction policy must balance reuse vs capacity ([§02](../02_kv_cache/05_cross_request_kv_sharing.md)).
2. **Cache-aware routing at scale.** Multi-instance deployments need to route shared-prefix requests to the holding instance, or RadixAttention's benefit is lost ([§09](../09_distributed_inference/01_load_balancing_strategies.md)).
3. **Fast-moving codebase.** Like vLLM, rapid development; pin versions and re-validate.
4. **Feature parity churn.** The vLLM/SGLang/TRT-LLM leapfrog means "fastest" depends on the month and workload.

***

## Solutions & Current Best Practices
- **Use SGLang for shared-prefix, structured, multi-turn, high-QPS** workloads; enable RadixAttention (default).
- **Pair with cache-aware routing** in multi-instance setups to realize prefix reuse ([§09](../09_distributed_inference/01_load_balancing_strategies.md)).
- **Enable overlap scheduler + chunked prefill + FP8** for balanced high-throughput serving.
- **Use the DSL** for constrained/structured/agentic generation.

***

## Implementation Notes
- Structure prompts to maximize prefix sharing (stable content first) to exploit RadixAttention ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)).
- Tune RadixAttention cache size vs active-KV capacity.
- For DeepSeek/MoE, use SGLang's MLA/EP support.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **RadixAttention gains lost across instances** — without cache-aware routing each instance rebuilds the tree; route shared prefixes together.
- **Over-caching shrank active batch** — pinned prefix KV ate HBM; bound the cache.
- **Throughput crown flipped after a release** — vLLM/SGLang/TRT-LLM leapfrog; benchmark your workload, don't assume.
- **Structured-output overhead** — constrained decoding (grammar) adds CPU work; ensure it's not the bottleneck.

***

## Performance Numbers & Benchmarks
| Workload | SGLang strength |
|---|---|
| Shared system prompts / multi-turn | RadixAttention → high reuse |
| High QPS, small requests | overlap scheduler → GPU-bound |
| Structured/JSON/agentic | DSL + RadixAttention |
| DeepSeek/MoE | MLA + EP support |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"What is RadixAttention and how does it beat exact prefix caching?"* — Expected: radix-tree longest-prefix incl. nested/branching reuse.
- *"How does SGLang avoid the scheduler bottleneck?"* — Expected: zero-overhead + overlap scheduler hides CPU behind GPU.
- *"When would you pick SGLang over vLLM?"* — Expected: shared-prefix/structured/high-QPS/multi-turn.
- *"What's needed to realize RadixAttention across many instances?"* — Expected: cache-aware routing.

***

## Open Problems & Active Research (2025–2026)
- **Global/cross-instance RadixAttention** (shared KV store) for disaggregated serving.
- **Maintaining the throughput lead** as vLLM V1 / TRT-LLM advance (watchlist).
- **Constrained-decoding efficiency** for heavy structured/agentic workloads.

***

## References
- Zheng, L., et al. (2024). "SGLang: Efficient Execution of Structured Language Model Programs." arXiv:2312.07104.
- Kwon, W., et al. (2023). "vLLM/PagedAttention." *SOSP 2023*. arXiv:2309.06180.
- DeepSeek-AI (2024). "DeepSeek-V2" (MLA). arXiv:2405.04434.
