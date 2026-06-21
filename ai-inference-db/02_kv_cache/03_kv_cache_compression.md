# KV Cache Compression: Eviction, Sparsity, and Token Selection

> **Section:** 02_kv_cache
> **Last Updated:** June 2026
> **Related Files:** [00_kv_cache_fundamentals.md](00_kv_cache_fundamentals.md), [../05_quantization/05_kv_cache_quantization.md](../05_quantization/05_kv_cache_quantization.md), [../10_long_context/05_kv_cache_management_at_1M_tokens.md](../10_long_context/05_kv_cache_management_at_1M_tokens.md)
> **Must-Read Papers:** Xiao et al. (2024, ICLR) "StreamingLLM"; Zhang et al. (2023, NeurIPS) "H2O"; Li et al. (2024) "SnapKV"; Cai et al. (2024) "PyramidKV"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- KV compression reduces the **number of tokens** kept (eviction/sparsity), complementing **quantization** (fewer bits/token, [§05](../05_quantization/05_kv_cache_quantization.md)).
- **StreamingLLM** (Xiao et al. 2024): keep **attention sinks** (first few tokens) + a recent window → unbounded streaming with bounded KV, but loses true long-range recall.
- **H2O** (Zhang et al. 2023): evict non-"heavy-hitter" tokens identified by accumulated attention scores → ~keep 20% of KV with small quality loss.
- **SnapKV / PyramidKV**: prompt-time KV selection / layer-wise budget allocation (more KV in lower layers) for long-context efficiency.
- Core question: **which cached tokens actually matter?** Attention is sparse and recency-biased, but the "lost in the middle" effect means naive eviction can drop crucial middle context.

***

## Overview
Quantization shrinks each KV entry's bit-width; **compression** instead reduces *how many* KV entries you keep, exploiting the empirical sparsity of attention — at any decode step most attention mass concentrates on a small subset of past tokens (recent tokens, plus a few persistent "sink"/"heavy-hitter" tokens). If you can identify and keep only the tokens that future queries will actually attend to, you bound KV memory independent of (or sub-linear in) sequence length, enabling long-context and high-batch serving that linear KV growth would forbid.

The techniques form a spectrum of how they pick survivors. **StreamingLLM** is the simplest and most robust for *streaming*: keep the first few tokens (which empirically act as "attention sinks" the model dumps excess attention onto) plus a sliding recent window; evict the middle. This gives constant KV for infinite streams and stable perplexity, but by construction it *forgets* the middle — fine for chat-style streaming, wrong for tasks needing full-context recall. **H2O** is adaptive: it tracks accumulated attention scores and evicts low-score ("non-heavy-hitter") tokens, keeping the dynamically important ones. **SnapKV** compresses at prompt time by selecting important prompt tokens (using attention from an observation window). **PyramidKV** allocates a *layer-wise* budget — more KV in lower layers (broader attention) and less in higher layers — reflecting that attention patterns differ by depth.

The unifying tension is accuracy vs memory and the **"lost in the middle"** phenomenon (Liu et al. 2023): models already attend less reliably to mid-context, and aggressive eviction can amplify this, silently dropping information needed for long-context tasks (retrieval, summarization of long docs). So compression is workload-sensitive: excellent for streaming/chat, risky for tasks requiring faithful long-range recall.

***

## Core Concepts & Mechanics

### Attention sinks (StreamingLLM)
Xiao et al. observed that models allocate large attention to the **first tokens** regardless of content — they act as a "sink" stabilizing the softmax. Dropping them (naive sliding window) collapses quality; keeping ~4 sink tokens + a recent window of size W restores stable streaming. 📐 KV memory = `(n_sink + W) × KV_per_token` — constant, independent of stream length. Limitation: no recall beyond the window + sinks.

### Heavy hitters (H2O)
Zhang et al. maintain per-token accumulated attention scores; at each step, among non-recent tokens, evict those with the lowest accumulated score, keeping a budget of "heavy hitters" + recent tokens. Greedy, online, O(1) extra per step. 📐 Keep ~20% of KV with minor degradation on many tasks. Risk: a token unimportant early but needed later was already evicted (irreversibility).

### Prompt-time selection (SnapKV)
Uses attention from a small **observation window** at the end of the prompt to score prompt tokens, then keeps only the top-scoring KV per head before decoding — compressing long prompts up-front. Good for long-input/short-output (RAG, doc QA).

### Layer-wise budgets (PyramidKV)
Allocates more KV budget to lower layers (which attend broadly) and less to upper layers (sparser, more local), giving a "pyramid" of KV across depth — better accuracy per memory than uniform budgets.

### The selection question
All methods approximate "which tokens will future queries attend to?" Attention is sparse and recency-biased, but **important tokens can be anywhere**, and eviction is usually irreversible. This is why these methods trade controllable accuracy loss for memory and why none is universally safe.

***

## Key Challenges
1. **Irreversible eviction.** A token dropped now cannot be recovered if a later query needs it; online methods (H2O, StreamingLLM) gamble on future attention patterns.
2. **"Lost in the middle" amplification.** Naive compression worsens the model's existing mid-context weakness, hurting long-context retrieval/summarization.
3. **Per-head/per-layer heterogeneity.** Different heads/layers have different sparsity; uniform budgets are suboptimal, but per-head budgets complicate kernels and memory layout.
4. **Interaction with paging/prefix caching.** Evicting tokens fragments paged blocks and breaks prefix-cache exact matches; compressed KV is harder to share across requests.

***

## Solutions & Current Best Practices
- **StreamingLLM** for unbounded chat/streaming where long recall isn't required.
- **H2O / SnapKV / PyramidKV** for long-context where memory is binding and some accuracy loss is acceptable — validate per task.
- **Combine with KV quantization** (compression × quantization are multiplicative) ([§05](../05_quantization/05_kv_cache_quantization.md)).
- **Task-aware policy**: disable aggressive eviction for retrieval/long-doc tasks; enable for streaming chat.

***

## Implementation Notes
- Keep ~4 sink tokens for StreamingLLM; tune window W to memory budget.
- For H2O, the heavy-hitter budget + recent window sizes are the key knobs; measure on your eval set.
- Validate on **long-context recall benchmarks** (e.g., needle-in-a-haystack, RULER), not just perplexity — perplexity hides recall failures.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Perplexity looks fine, retrieval breaks.** Compression preserves fluency but drops the specific mid-context token a query needed; always test recall, not just PPL.
- **Sliding window without sinks collapses quality** — removing the first tokens destabilizes softmax; StreamingLLM's whole point is keeping sinks.
- **Eviction breaks prefix caching** — compressed/evicted KV no longer matches other requests' prefixes; the two optimizations can conflict.
- **Heavy-hitter eviction is irreversible** — a token unimportant early but critical later is gone; risky for agentic/long-horizon tasks.

***

## Performance Numbers & Benchmarks
| Method | KV kept | Quality impact | Best for |
|---|---|---|---|
| StreamingLLM | sinks + window (constant) | stable PPL, no long recall | infinite streaming chat |
| H2O | ~20% | small on many tasks | long context, memory-bound |
| SnapKV | top-k prompt tokens | small on long-input tasks | RAG / doc QA |
| PyramidKV | layer-wise budget | better acc/memory than uniform | long context |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"What are attention sinks and why does StreamingLLM keep the first tokens?"* — Expected: softmax stabilization; window+sinks for constant KV.
- *"How does H2O decide what to evict, and what's the risk?"* — Expected: accumulated attention scores; irreversible eviction of later-needed tokens.
- *"Why can KV compression pass perplexity but fail retrieval?"* — Expected: fluency preserved, specific recall dropped; 'lost in the middle'.
- *"How does KV compression interact with prefix caching?"* — Expected: eviction breaks exact prefix matches; tension.

***

## Open Problems & Active Research (2025–2026)
- **Lossless or recall-preserving compression** for long-context retrieval tasks.
- **Learned/predictive token selection** rather than greedy attention-score heuristics.
- **Unified compression + quantization + paging + MLA** that compose cleanly ([§10](../10_long_context/05_kv_cache_management_at_1M_tokens.md)).

***

## References
- Xiao, G., et al. (2024). "Efficient Streaming Language Models with Attention Sinks" (StreamingLLM). *ICLR 2024*. arXiv:2309.17453.
- Zhang, Z., et al. (2023). "H2O: Heavy-Hitter Oracle for Efficient Generative Inference." *NeurIPS 2023*. arXiv:2306.14048.
- Li, Y., et al. (2024). "SnapKV: LLM Knows What You are Looking for Before Generation." arXiv:2404.14469.
- Cai, Z., et al. (2024). "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling." arXiv:2406.02069.
- Liu, N., et al. (2023). "Lost in the Middle." *TACL 2024*. arXiv:2307.03172.
