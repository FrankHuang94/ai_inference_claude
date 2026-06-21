# KV Cache Quantization

> **Section:** 05_quantization
> **Last Updated:** June 2026
> **Related Files:** [../02_kv_cache/00_kv_cache_fundamentals.md](../02_kv_cache/00_kv_cache_fundamentals.md), [../02_kv_cache/03_kv_cache_compression.md](../02_kv_cache/03_kv_cache_compression.md), [04_fp8_inference_h100.md](04_fp8_inference_h100.md)
> **Must-Read Papers:** Liu et al. (2024) "KIVI"; Hooper et al. (2024) "KVQuant"; Sheng et al. (2023) "FlexGen" (group-wise KV)
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- KV cache often **exceeds weight memory** at large batch/long context, so quantizing it (INT8/FP8/INT4) is a high-leverage memory win that enables bigger batches/longer contexts.
- **Key and Value have different distributions**: Keys have per-channel outliers (quantize **per-channel**); Values are flatter (quantize **per-token**). KIVI exploits exactly this.
- **INT8 KV** is near-lossless and common; **INT4 KV** is more aggressive (KIVI/KVQuant) with small loss; **FP8 KV** is increasingly standard on Hopper.
- Long-context accuracy is the **most sensitive** to KV quantization; validate on long-context recall.
- Interacts awkwardly with **prefix caching** (quantized blocks may not match exactly) and adds (de)quant cost in the attention kernel.

***

## Overview
KV cache quantization reduces the bits per cached key/value entry, directly shrinking the data structure that — at large batch or long context — dominates HBM ([§02](../02_kv_cache/00_kv_cache_fundamentals.md)). Because batch size (and thus decode throughput) is capped by KV memory for large models, halving or quartering KV bytes can double or quadruple achievable batch/context, often a bigger practical win than weight quantization in long-context regimes. It also reduces the **KV-read bandwidth** in attention, which grows with sequence length and becomes a real decode cost at long context.

The defining technical insight, formalized by **KIVI** (Liu et al. 2024) and **KVQuant** (Hooper et al. 2024), is that **keys and values have different distributions and need different quantization granularity**. Keys exhibit strong **per-channel outliers** (certain channels consistently large, like activation outliers), so keys should be quantized **per-channel** (along the channel dimension). Values are flatter and quantize well **per-token**. Mixing this up — e.g., per-token quantization of keys — incurs large error because a channel outlier in one token forces a bad scale. KIVI quantizes keys per-channel and values per-token to INT2/INT4 with surprisingly small quality loss; KVQuant pushes to ~INT4 (and below) with per-channel keys plus outlier isolation.

The catch is sensitivity and integration. **Long-context tasks** (retrieval, long-doc QA) are the most degraded by aggressive KV quantization — fluency/perplexity can look fine while needle-in-a-haystack recall drops. And quantized KV interacts poorly with **exact prefix caching** (two requests' shared prefix may quantize to different scales/blocks, breaking the bit-identical match) and adds (de)quantization work in the attention kernel that must be efficient. FP8 KV on Hopper sidesteps some issues (float range, hardware support) and is becoming a clean default. This file covers the K/V asymmetry, the methods, and the production gotchas.

***

## Core Concepts & Mechanics

### Why KV quantization pays off
📐 KV/token (70B GQA) ≈ 320 KB FP16 → 160 KB INT8 → 80 KB INT4. At batch=32, 16k context: 80 GB FP16 → 20 GB INT4, freeing ~60 GB for more batch/context. The decode KV-read bandwidth shrinks proportionally too.

### Key vs Value distributions (KIVI)
- **Keys**: per-channel outliers (consistent large channels) → quantize **per-channel** (scale per channel). Per-token would let one outlier channel ruin a token's scale.
- **Values**: flatter, no strong channel structure → quantize **per-token** (scale per token), which is also kernel-friendly for the `·V` step.
- KIVI: keys per-channel, values per-token, group-wise INT2/INT4; small accuracy loss.

### KVQuant
Per-channel key quantization + **outlier isolation** (keep a few outliers in higher precision) + non-uniform/​sensitivity-aware quantization → ~INT4 (even ~3-bit) with low loss, targeting long context.

### FP8 KV
Store K,V in FP8 (E4M3); the exponent handles ranges without elaborate per-channel schemes; hardware-supported on Hopper; near-lossless and simple. Increasingly the default where available.

### Granularity & kernel
Group-wise scales (e.g., per 128 along the quantized axis) balance accuracy and metadata. The attention kernel must **dequantize K,V on the fly** during QKᵀ and ·V; this must be fused/efficient or it negates the bandwidth savings.

***

## Key Challenges
1. **Long-context recall sensitivity.** Aggressive KV quant preserves fluency but degrades specific long-range retrieval — the most important failure mode, hidden by perplexity.
2. **K/V asymmetry.** Wrong granularity (per-token keys) causes large error; you must respect the per-channel-keys / per-token-values structure.
3. **Prefix-cache interaction.** Quantized blocks may not match bit-identically across requests, breaking exact prefix caching ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)).
4. **Kernel (de)quant cost.** On-the-fly dequant in attention must be fused/cheap or the bandwidth win evaporates.

***

## Solutions & Current Best Practices
- **INT8 or FP8 KV** as a near-lossless default; **INT4 (KIVI/KVQuant)** when memory is binding and some loss is acceptable.
- **Per-channel keys, per-token values** (KIVI structure); outlier isolation (KVQuant) for very low bits.
- **Validate on long-context recall** (RULER, needle-in-haystack), not just PPL.
- **Combine with KV compression** (eviction) — multiplicative memory savings ([§02](../02_kv_cache/03_kv_cache_compression.md)).

***

## Implementation Notes
- vLLM/SGLang support FP8/INT8 KV; choose granularity per the K/V asymmetry.
- Ensure the attention kernel fuses dequant; profile that KV-read bandwidth actually drops.
- If using prefix caching, verify the framework handles quantized-block matching (or accept reduced hit rate).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **INT4 KV passed perplexity, failed long-doc retrieval** — recall is the sensitive metric; test it explicitly.
- **Per-token key quantization tanked accuracy** — keys have per-channel outliers; must quantize keys per-channel (KIVI).
- **KV quant broke prefix-cache hits** — quantized blocks didn't match across requests; hit rate dropped. Check framework support.
- **No speedup despite smaller KV** — unfused dequant in attention ate the savings; fuse it.

***

## Performance Numbers & Benchmarks
| Scheme | KV bytes vs FP16 | Quality | Note |
|---|---|---|---|
| FP8 KV | 0.5× | near-lossless | simple, Hopper |
| INT8 KV | 0.5× | near-lossless | common |
| INT4 KIVI | 0.25× | small loss | per-chan keys/per-tok values |
| ~INT3 KVQuant | ~0.19× | small loss (long ctx) | outlier isolation |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why quantize the KV cache, and when does it matter more than weight quantization?"* — Expected: KV dominates at long context/high batch; enables more batch/context.
- *"Why do keys and values need different quantization?"* — Expected: keys have per-channel outliers (per-channel), values flat (per-token).
- *"Which tasks are most sensitive to KV quantization?"* — Expected: long-context recall; PPL hides it.
- *"How does KV quantization interact with prefix caching?"* — Expected: quantized blocks may not match; reduced hits.

***

## Open Problems & Active Research (2025–2026)
- **Sub-4-bit KV** with preserved long-context recall.
- **Quantization-friendly prefix caching** (matching across scales) and KV-transfer compression ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).
- **FP8/FP4 attention kernels** end-to-end on Blackwell.

***

## References
- Liu, Z., et al. (2024). "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache." *ICML 2024*. arXiv:2402.02750.
- Hooper, C., et al. (2024). "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization." *NeurIPS 2024*. arXiv:2401.18079.
- Sheng, Y., et al. (2023). "FlexGen." *ICML 2023*. arXiv:2303.06865.
