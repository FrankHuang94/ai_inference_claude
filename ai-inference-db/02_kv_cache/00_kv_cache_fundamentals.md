# KV Cache Fundamentals

> **Section:** 02_kv_cache
> **Last Updated:** June 2026
> **Related Files:** [01_paged_attention_vllm.md](01_paged_attention_vllm.md), [03_kv_cache_compression.md](03_kv_cache_compression.md), [../00_fundamentals/01_autoregressive_decoding.md](../00_fundamentals/01_autoregressive_decoding.md)
> **Must-Read Papers:** Shazeer (2019) "Fast Transformer Decoding" (MQA); Ainslie et al. (2023, EMNLP) "GQA"; DeepSeek-AI (2024) "DeepSeek-V2" (MLA)
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- The KV cache stores per-token **keys and values** so each decode step attends to history without recomputing it — turning O(L²) recompute into O(L) memory.
- 📐 Footprint = `2 × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes`. For Llama-3-70B (GQA, 8 KV heads, 80 layers, d_h=128) at 4096 tokens FP16 ≈ **~1.25 GB/request** (and ~5 GB at 16k).
- **KV cache, not compute, caps batch size** for large models and long contexts — this is the central serving constraint.
- **MQA/GQA** shrink KV by sharing K/V across query heads (8–64×); **MLA** (DeepSeek) compresses KV into a low-rank latent — the biggest architectural KV reductions.
- KV cache often **exceeds model weights** in memory at large batch / long context.

***

## Overview
The KV cache is the single most important data structure in LLM serving. During autoregressive decoding, computing attention for the new token requires the keys and values of *all* previous tokens. Recomputing them every step would cost O(L²) work over a generation of length L; instead, each token's K and V are computed once (at prefill or when first generated) and **cached** in HBM, so each decode step computes K,V only for the one new token and reads the cached rest. This trades compute for memory: O(L) storage that grows linearly with sequence length.

That linear growth is the crux of serving economics. Because batching is the primary way to make decode efficient ([§00](../00_fundamentals/00_inference_vs_training.md)), and because the KV cache for every in-flight request must reside in HBM simultaneously, **the KV cache — not the GPU's compute — determines how many requests you can batch**. For large models and long contexts the KV cache can rival or exceed the weights in size, so essentially all of the field's serving innovations (PagedAttention, prefix caching, KV quantization, offloading, MLA, GQA) are attacks on KV-cache memory pressure. If you internalize the footprint formula and its implications, the rest of this section follows.

The architectural responses come in two flavors: **reduce KV per token** at the model level (MQA, GQA, MLA — fewer/smaller K,V tensors) and **manage KV memory better** at the system level (paging, sharing, compression, offload). This file covers the fundamentals and the model-level reductions; the system-level techniques are the rest of the section.

***

## Core Concepts & Mechanics

### The footprint formula
📐
```
KV_bytes = 2 (K and V)
         × n_layers
         × n_kv_heads
         × head_dim
         × seq_len
         × batch_size
         × bytes_per_element
```
**Worked: Llama-3-70B**, n_layers=80, n_kv_heads=8 (GQA), head_dim=128, FP16 (2 B):
- Per token per request: `2 × 80 × 8 × 128 × 2 = 327,680 B ≈ 320 KB/token`.
- At 4096 tokens: `320 KB × 4096 ≈ 1.25 GB/request`. At 16k: ~5 GB. At batch=32, 4k: ~40 GB — most of an 80 GB GPU after weights.

Compare **without GQA** (n_kv_heads=64): 8× larger — ~10 GB/request at 4k. GQA is doing enormous work.

### Why KV caps batch
Available HBM after weights = `HBM − weights/TP`. Max batch ≈ `(HBM − weights/TP) / KV_per_request`. For 70B FP16 on 2×H100 (160 GB − 140 GB = 20 GB free), at 1.25 GB/req you fit only ~16 concurrent 4k requests — far below the compute breakeven batch (~hundreds). Hence large models stay memory-bound ([§00](../00_fundamentals/03_memory_bandwidth_bound_compute.md)). Quantizing weights to FP8 frees ~70 GB → many more requests.

### MQA, GQA, MLA
- **MHA** (baseline): n_kv_heads = n_heads. Largest KV.
- **MQA** (Shazeer 2019): one shared K,V head for all query heads → KV ÷ n_heads. Big savings, some quality loss.
- **GQA** (Ainslie 2023): g groups, n_kv_heads = g (e.g., 8). Interpolates MHA↔MQA; the production standard (Llama-2/3, Mixtral).
- **MLA** (DeepSeek-V2/V3): project K,V into a shared **low-rank latent** cached instead of full K,V; reconstruct per head on the fly. Dramatic KV reduction (cache a small latent vector/token) while preserving quality — see [§06](../06_kernel_optimization/01_flash_attention_deep_dive.md). Changes the attention kernel.

### KV vs weights crossover
📐 KV exceeds weights when `batch × seq × KV_per_token > weights_bytes`. For 70B (140 GB FP16, 320 KB/token KV): crossover at `batch×seq ≈ 140e9/320e3 ≈ 437,500` token-slots — e.g., batch=32 at ~13.7k context. Beyond that, KV dominates HBM and KV optimizations matter more than weight quantization.

***

## Key Challenges
1. **Linear-in-length growth.** Long contexts (and long reasoning outputs) consume KV proportionally, forcing batch down mid-generation and degrading throughput exactly when you need it.
2. **Fragmentation.** Naive contiguous pre-allocation per request wastes 60–80% of KV memory due to variable lengths — solved by PagedAttention ([§01](01_paged_attention_vllm.md)).
3. **KV vs weights vs batch trilemma.** Fixed HBM must be split among weights, KV, and activations; raising any one squeezes the others. There is no free lunch.
4. **GQA/MLA complicate kernels and parallelism.** Fewer KV heads change TP sharding and require specialized attention kernels (esp. MLA), adding implementation complexity.

***

## Solutions & Current Best Practices
- **GQA by default** in modern models; **MLA** where the architecture supports it for extreme KV reduction.
- **PagedAttention** to eliminate fragmentation and enable large batches ([§01](01_paged_attention_vllm.md)).
- **Prefix caching / RadixAttention** to reuse KV across requests ([§02](02_prefix_caching_and_radix_attention.md)).
- **KV quantization (INT8/FP8/INT4)** to halve/quarter KV bytes ([§05](../05_quantization/05_kv_cache_quantization.md)).
- **Offloading** to CPU/NVMe for capacity at the cost of bandwidth ([§04](04_kv_cache_offloading.md)).

***

## Implementation Notes
- Always size HBM for **peak** KV (max concurrency × max context), with headroom for fragmentation/activations.
- Cache **post-RoPE keys** consistently; mixing conventions corrupts long-context attention ([§00](../00_fundamentals/05_transformer_inference_walkthrough.md)).
- With GQA and TP, ensure `n_kv_heads ≥ TP` or replicate KV heads across ranks; TP > n_kv_heads requires replication.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Throughput collapses as conversations lengthen.** KV grows, batch shrinks; the "regression" is the memory budget, not the model or kernel.
- **KV silently exceeds weights at long context** — teams quantize weights expecting big wins but KV is the dominant consumer; quantize KV instead.
- **TP=8 with 8 KV heads is fine; TP=16 needs replication** — exceeding n_kv_heads forces KV duplication, raising memory and breaking naive sharding assumptions.
- **Forgetting KV for speculative/parallel sampling** — multiple candidates multiply KV; beam/best-of-N need copy-on-write blocks or they OOM.

***

## Performance Numbers & Benchmarks
| Model | KV/token (FP16) | KV @ 4k/req | KV @ 32k/req |
|---|---|---|---|
| Llama-3-8B (GQA, 8 KV heads, 32L, d_h=128) | ~128 KB | ~0.5 GB | ~4 GB |
| Llama-3-70B (GQA, 8 KV heads, 80L) | ~320 KB | ~1.25 GB | ~10 GB |
| 70B MHA (64 KV heads) | ~2.5 MB | ~10 GB | ~80 GB |
| DeepSeek MLA | small latent/token | fraction of GQA | enables long context |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Derive the KV cache size for a 70B model at 4k context and explain what limits batch."* — Expected: the formula → ~1.25 GB/req; KV caps batch.
- *"Compare MHA, MQA, GQA, MLA in KV footprint and quality."* — Expected: ÷n_heads (MQA), groups (GQA), low-rank latent (MLA).
- *"When does KV cache exceed model weights?"* — Expected: batch×seq×KV_per_token > weights; long context / high batch.
- *"How does GQA interact with tensor parallelism?"* — Expected: n_kv_heads vs TP sharding/replication.

***

## Open Problems & Active Research (2025–2026)
- **Sub-linear KV** via compression/eviction without long-context accuracy loss ([§03](03_kv_cache_compression.md)).
- **MLA adoption and kernels** beyond DeepSeek; generalizing low-rank KV.
- **KV management for reasoning models** where single-request KV reaches hundreds of GB ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

***

## References
- Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150.
- Ainslie, J., et al. (2023). "GQA." *EMNLP 2023*. arXiv:2305.13245.
- DeepSeek-AI (2024). "DeepSeek-V2" (MLA). arXiv:2405.04434.
- Kwon, W., et al. (2023). "PagedAttention/vLLM." *SOSP 2023*. arXiv:2309.06180.
