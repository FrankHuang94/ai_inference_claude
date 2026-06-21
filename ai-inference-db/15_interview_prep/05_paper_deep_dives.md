# Paper Deep Dives (10 Must-Know Papers)

> **Section:** 15_interview_prep
> **Last Updated:** June 2026
> **Related Files:** [../02_kv_cache/01_paged_attention_vllm.md](../02_kv_cache/01_paged_attention_vllm.md), [../06_kernel_optimization/01_flash_attention_deep_dive.md](../06_kernel_optimization/01_flash_attention_deep_dive.md), [00_inference_system_design_questions.md](00_inference_system_design_questions.md)
> **Must-Read Papers:** the 10 below
> **Estimated Study Time:** 45 minutes

***

## TL;DR
- Ten papers that define modern inference: **vLLM, Orca, FlashAttention (1/2), Speculative Decoding, SGLang, Splitwise/DistServe, S-LoRA, EAGLE, Sarathi-Serve**.
- For each: 1-paragraph summary, key contributions, the core equations/algorithms, what it gets right, limitations, likely interview questions.
- Be able to **derive the core idea** and **place it on the bandwidth-bound/roofline map**.

***

## 1. vLLM / PagedAttention — Kwon et al., SOSP 2023 (arXiv:2309.06180)
**Summary**: KV memory wasted 60–80% via fragmentation; PagedAttention applies OS paging (block tables → non-contiguous physical blocks) to eliminate it, enabling larger batches and 2–4× throughput. **Contributions**: paged KV, copy-on-write sharing, the vLLM scheduler. **Core**: waste ≤ 1 partial block/seq (<4%); attention kernel gathers scattered blocks. **Right**: foundational, ubiquitous. **Limits**: Python scheduler overhead at high QPS (→V1). **Q**: "Quantify fragmentation savings"; "explain CoW for parallel sampling." ([§02](../02_kv_cache/01_paged_attention_vllm.md))

## 2. Orca — Yu et al., OSDI 2022
**Summary**: Introduced **iteration-level (continuous) batching** — admit/evict requests at each iteration boundary instead of static batches — dramatically improving GPU utilization. **Contributions**: continuous batching, selective batching. **Core**: per-iteration scheduling; throughput ∝ mean not max length. **Right**: the basis of all modern serving. **Limits**: doesn't solve prefill-decode interference (→ Sarathi). **Q**: "Static vs continuous batching, quantitatively." ([§03](../03_batching_and_scheduling/00_static_vs_continuous_batching.md))

## 3. FlashAttention — Dao et al., NeurIPS 2022 (arXiv:2205.14135)
**Summary**: Exact attention without materializing the N×N matrix in HBM — tile into SRAM + online softmax → O(N) memory, far less HBM traffic. **Contributions**: IO-aware attention, online softmax. **Core**: HBM accesses Θ(N²d/M); running (max, sum) recurrence. **Right**: massive speedup, enables long context. **Limits**: superseded by FA-2/3 on parallelism/hardware. **Q**: "IO complexity + online softmax derivation." ([§06](../06_kernel_optimization/01_flash_attention_deep_dive.md))

## 4. FlashAttention-2 — Dao, 2023 (arXiv:2307.08691)
**Summary**: Better work partitioning (parallelize over sequence/query, fewer non-matmul FLOPs) → ~2× FA-1. **Contributions**: improved parallelism/occupancy, reduced rescaling. **Core**: separate Q/K/V work partitioning. **Right**: the production standard for prefill attention. **Limits**: pre-Hopper-async (→ FA-3). **Q**: "What did FA-2 improve over FA-1?" ([§06](../06_kernel_optimization/01_flash_attention_deep_dive.md))

## 5. Speculative Decoding — Leviathan et al., ICML 2023 (arXiv:2211.17192)
**Summary**: Draft model proposes k tokens; target verifies all in one pass; **rejection sampling** guarantees the exact target distribution → 2–3× decode at low batch. **Contributions**: draft-verify, exactness proof, speedup formula. **Core**: accept w.p. min(1,p/q), resample (p−q)_+; E[tokens]=(1−α^{k+1})/(1−α). **Right**: exact, widely used. **Limits**: fails at high batch / low acceptance. **Q**: "State the acceptance rule + speedup formula; when does it fail?" ([§07](../07_speculative_decoding/00_speculative_decoding_fundamentals.md))

## 6. SGLang / RadixAttention — Zheng et al., 2024 (arXiv:2312.07104)
**Summary**: **RadixAttention** (radix-tree KV reuse for any shared prefix incl. nested/branching) + zero-overhead/overlap scheduler → throughput leadership on shared-prefix/structured workloads. **Contributions**: trie-based prefix caching, overlap scheduler, structured-generation DSL. **Core**: longest-prefix match in a radix tree; LRU eviction over nodes. **Right**: best for shared-prefix/high-QPS. **Limits**: cache-aware routing needed across instances. **Q**: "RadixAttention vs exact prefix caching." ([§08](../08_serving_frameworks/02_sglang_deep_dive.md))

## 7. Splitwise / DistServe — Patel et al. ISCA 2024 / Zhong et al. OSDI 2024
**Summary**: **Disaggregate** prefill (compute-bound) and decode (bandwidth-bound) onto separate, independently-scaled instances with KV transfer → up to ~4–7× goodput under dual SLOs. **Contributions**: phase splitting (Splitwise: heterogeneous HW; DistServe: goodput-optimal placement). **Core**: P/D ratio balance; KV transfer must overlap. **Right**: the key 2024–26 architecture at scale. **Limits**: KV-transfer overhead; not for short prompts. **Q**: "When does disaggregation beat colocation? Estimate KV-transfer BW." ([§09](../09_distributed_inference/02_disaggregated_prefill_decode_systems.md))

## 8. S-LoRA — Sheng et al., 2023 (arXiv:2311.03285)
**Summary**: Serve **thousands of LoRA adapters** on one base model concurrently via batched heterogeneous-adapter kernels + unified paging → multi-tenant fine-tune serving at near-base efficiency. **Contributions**: batched multi-LoRA, adapter memory management. **Core**: `Wx + B(Ax)` shared base + per-request adapter; paged adapter memory. **Right**: enables per-customer fine-tunes economically. **Limits**: adapter-switch overhead, rank limits. **Q**: "How serve thousands of fine-tunes?" ([§13](../13_production_systems/04_multi_tenant_serving.md))

## 9. EAGLE — Li et al., 2024 (arXiv:2401.15077)
**Summary**: **Feature-level** speculative drafting — predict the target's next hidden state (reusing its features) → much higher acceptance than token-level drafts; EAGLE-2 adds dynamic draft trees. **Contributions**: feature-level autoregressive head, dynamic trees. **Core**: draft in feature space; tree verification. **Right**: SOTA acceptance/low cost. **Limits**: head training per target; high-batch erosion. **Q**: "Why does feature-level drafting beat token-level?" ([§07](../07_speculative_decoding/02_speculative_decoding_variants.md))

## 10. Sarathi-Serve / Chunked Prefill — Agrawal et al., OSDI 2024 (arXiv:2403.02310)
**Summary**: Split long prefills into chunks interleaved with decodes (bounded per-iteration token budget) → tames prefill-decode interference, lowering P99 ITL while keeping prefill efficient. **Contributions**: chunked prefill, stall-free batching. **Core**: iteration token budget = decode + prefill chunk ≤ ITL SLO. **Right**: default-on; simpler than disaggregation. **Limits**: chunk-size tuning; long-prompt TTFT. **Q**: "How does chunked prefill protect ITL? Chunk-size tradeoff." ([§03](../03_batching_and_scheduling/02_chunked_prefill.md))

***

## Interview Angles
> 💡 **Per paper, be ready to:**
- Give the **1-line problem + 1-line solution**.
- State the **core equation/algorithm** and derive it.
- Name **what it gets right and its limitation**.
- Place it on the **bandwidth-bound / roofline / prefill-decode** map.

***

## Complexity Gotchas
> ⚠️ **Common mistakes:**
- Confusing FlashAttention's memory reduction with compute reduction.
- Stating speculative decoding changes output quality (it's exact).
- Forgetting disaggregation's KV-transfer cost / when colocation wins.
- Treating RadixAttention as exact-prefix-only (it handles nested/branching).

***

## References
(All ten papers above, with arXiv IDs inline; see linked section files for full treatments.)
