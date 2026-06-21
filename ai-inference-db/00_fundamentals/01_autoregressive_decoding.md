# Autoregressive Decoding: The Token-by-Token Loop

> **Section:** 00_fundamentals
> **Last Updated:** June 2026
> **Related Files:** [02_prefill_vs_decode_phases.md](02_prefill_vs_decode_phases.md), [03_memory_bandwidth_bound_compute.md](03_memory_bandwidth_bound_compute.md), [../02_kv_cache/00_kv_cache_fundamentals.md](../02_kv_cache/00_kv_cache_fundamentals.md)
> **Must-Read Papers:** Holtzman et al. (2020, ICLR) "The Curious Case of Neural Text Degeneration" (nucleus sampling); Fan et al. (2018, ACL) "Top-k"; Vaswani et al. (2017, NeurIPS) "Attention Is All You Need"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- Decode generates **one token per forward pass**, feeding each output back as the next input — an inherently sequential loop with a data dependency across time.
- Each decode step **loads all model weights from HBM to produce a single token** (at batch=1); this is the core inefficiency that makes decode bandwidth-bound.
- The **KV cache** turns per-step attention from O(t²) recompute into O(t) — without it, decode would re-attend to all prior tokens every step.
- Decode throughput at batch=1 ≈ `HBM_bandwidth / model_size_in_bytes`. Everything else (sampling, layernorm) is rounding error by comparison.
- **Sampling** (greedy / temperature / top-k / top-p) is cheap but governs output quality; **beam search is rarely used in production** serving because it multiplies KV cost and hurts latency for marginal gains on open-ended generation.

***

## Overview
After the prompt is processed (prefill), the model enters the **decode loop**: compute logits for the next position, sample a token, append it to the sequence, and repeat until an EOS token or length limit. Each iteration is a *full forward pass* through every layer, but with a sequence length of exactly **one new query token** — the previous tokens' keys and values are already cached. This is the regime where LLM serving spends most of its wall-clock time for chat workloads and essentially all of its time for reasoning models.

The defining property is the **temporal data dependency**: token *t+1*'s logits depend on token *t* having been sampled and written into the KV cache. You cannot run step *t+1* before step *t* finishes within a single sequence. This serializes the most expensive part of generation and is why a single GPU running one stream achieves single-digit MFU. The two escape hatches are *batching across many independent sequences* (raising arithmetic intensity, see [§03](../03_batching_and_scheduling/00_static_vs_continuous_batching.md)) and *speculative decoding* (predicting several tokens then verifying in parallel, see [§07](../07_speculative_decoding/00_speculative_decoding_fundamentals.md)).

Because each step is bandwidth-bound, the cost is dominated by **streaming weights**, not by the tiny per-token matmuls or the sampling step. This is the single most important fact about decode and the lens through which to read the rest of this section.

***

## Core Concepts & Mechanics

### The decode throughput equation
📐 At batch=1, ignoring KV traffic (small relative to weights early in a sequence), each token requires reading all weights once:

```
tokens_per_second ≈ HBM_bandwidth / (model_params × bytes_per_param)
```

Examples (single GPU, weights only):
- 7B FP16 on H100: `3.35e12 / (7e9 × 2) ≈ 239 tok/s` (real-world ~140–170 after KV + overhead).
- 70B FP16 on H100 (if it fit): `3.35e12 / 140e9 ≈ 24 tok/s`.
- 70B INT4 on H100: `3.35e12 / 35e9 ≈ 96 tok/s` — ~4× from cutting bytes/param, the bandwidth-bound payoff.

This formula is a favorite interview opener; be able to derive and apply it instantly.

### KV cache growth
At decode step *t*, attention needs keys/values for all *t* prior positions. Without caching you would recompute K,V for all positions each step: O(t) work per step, O(L²) over a generation of length L. The **KV cache** stores K,V once when each token is first processed, so each decode step only computes K,V for the *one* new token and reads the cached rest. Cache size grows linearly: `+2 × n_layers × n_kv_heads × head_dim × bytes` per token (full formula in [../02_kv_cache/00_kv_cache_fundamentals.md](../02_kv_cache/00_kv_cache_fundamentals.md)). This linear growth is what eventually forces batch size down on long generations.

### Sampling mechanics
Given logits `z ∈ R^V` (V = vocab size), the next-token distribution and selection:

- **Greedy:** `argmax(z)`. Deterministic; can be repetitive/degenerate on open-ended text.
- **Temperature T:** `softmax(z / T)`. T<1 sharpens, T>1 flattens. T→0 ≈ greedy. 📐 Numerical stability: always subtract `max(z)` before exp (`softmax(z) = exp(z - max z)/Σ exp(z - max z)`) to avoid overflow.
- **Top-k** (Fan et al. 2018): keep the k highest-logit tokens, renormalize, sample. Fixed support size.
- **Top-p / nucleus** (Holtzman et al. 2020): sort by probability, keep the smallest set whose cumulative prob ≥ p, renormalize, sample. *Adaptive* support — wider when the model is uncertain, narrower when confident. This is the production default for quality.
- **Combined / min-p / typical:** real servers apply temperature → top-k → top-p in sequence; repetition penalties and logit bias are applied to `z` before softmax.

The sampling kernel touches only the V-dimensional logits for the batch and is negligible versus the weight streaming — but a *naive* full sort for top-p across a large vocab (V≈128k for Llama-3) can become a measurable fraction of a fast decode step, which is why frameworks use partial/radix selection.

### Beam search and why it's rare in serving
Beam search maintains B partial hypotheses, expanding and pruning each step to maximize sequence likelihood. It improves closed-ended tasks (translation, constrained generation) but for open-ended chat it (a) tends to produce bland, high-likelihood text, (b) multiplies KV cache by B, (c) complicates batching and continuous scheduling, and (d) adds latency. Production LLM APIs almost universally serve with sampling, not beam search. (vLLM supports it via copy-on-write KV blocks, but it's a niche path.)

***

## Key Challenges
1. **The serial dependency.** Token *t+1* cannot start before *t* completes within a sequence, so single-stream decode cannot fill the GPU. Hidden parallelism only exists *across* independent requests.
2. **Memory-bandwidth bottleneck at small batch.** Each step pays the full weight-read cost regardless of how little compute it does, so per-token efficiency is dreadful until batched.
3. **KV cache growth caps batch and context.** Linear-in-length KV consumption means long outputs progressively shrink the feasible batch, degrading throughput mid-generation.
4. **Sampling correctness under quantization/fused kernels.** Subtle bugs (missing max-subtraction, wrong order of temperature vs top-k, fp16 softmax overflow) silently change the output distribution and are hard to detect without distribution-level tests.

***

## Solutions & Current Best Practices
- **Continuous batching** to share each weight load across many sequences ([§03](../03_batching_and_scheduling/00_static_vs_continuous_batching.md)).
- **Paged KV cache** to fit larger batches and longer contexts ([§02](../02_kv_cache/01_paged_attention_vllm.md)).
- **Weight quantization (INT4/FP8)** for near-linear decode speedups ([§05](../05_quantization/02_weight_only_quantization.md)).
- **Speculative decoding** to emit multiple tokens per target forward pass ([§07](../07_speculative_decoding/00_speculative_decoding_fundamentals.md)).
- **Fused sampling kernels** (partial top-k/top-p selection, fused temperature+penalty) to keep the sampling step off the critical path.

***

## Implementation Notes
- Apply logit processors in a **defined, documented order**; mismatches between training-time and serving-time sampling are a common source of "the model got dumber in prod."
- Use **CUDA graphs** to capture the decode step and eliminate per-iteration kernel-launch overhead — at batch=1, launch latency can be a real fraction of a sub-10ms step ([§06](../06_kernel_optimization/02_fused_kernels_and_operator_fusion.md)).
- Seed and RNG handling matters for reproducibility; per-request RNG state must be threaded through continuous batching correctly or "deterministic" requests become non-deterministic when co-batched.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Top-p with a huge vocab can dominate a fast step.** For 7B at >150 tok/s, a step is ~6ms; a naive O(V log V) sort over 128k vocab × batch can eat a meaningful slice. Use partial selection.
- **fp16 softmax overflow in sampling.** Large logits without max-subtraction → `inf`/`nan` → garbage tokens, often only on rare prompts. Always stabilize, and prefer fp32 accumulation for the final softmax.
- **Repetition penalty interacts with KV/prefix caching.** Penalties depend on the generated history; if you cache/share prefixes you must be careful that per-request penalty state isn't corrupted by sharing.
- **EOS handling under continuous batching.** A sequence that hits EOS must be evicted at the *iteration boundary* and its KV blocks freed; getting this wrong leaks memory or truncates neighbors.

***

## Performance Numbers & Benchmarks
| Hardware | Model | Config | Metric | Result |
|---|---|---|---|---|
| 1× H100 | 7B FP16 | batch=1 decode | tokens/s | ~140–170 |
| 1× H100 | 7B INT4 (W4A16) | batch=1 decode | tokens/s | ~300–400 |
| 2× H100 TP2 | 70B FP16 | batch=1 decode | tokens/s | ~20–30 |
| 1× H100 | 7B FP16 | batch=64 | aggregate tok/s | thousands (bandwidth amortized) |
| 1× H100 | 70B INT4 | batch=1 | tokens/s | ~80–100 |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Derive single-stream decode throughput for a 70B model on an H100."* — Expected: `bandwidth / weight_bytes`, the ~24 tok/s FP16 / ~96 tok/s INT4 numbers.
- *"Why does the KV cache exist and what does it cost?"* — Expected: avoids O(L²) recompute at O(L) memory; linear growth formula and its batch-limiting effect.
- *"Walk me through top-p sampling and its numerical pitfalls."* — Expected: adaptive nucleus, max-subtraction, order of operations, partial-sort optimization.
- *"Why isn't beam search used in chat serving?"* — Expected: KV ×B cost, latency, bland outputs, batching complexity.

***

## Open Problems & Active Research (2025–2026)
- **Parallel/non-autoregressive decoding** (diffusion LMs, consistency-style decoders) that break the serial dependency without quality loss remain unproven at frontier scale.
- **Sampling for reasoning models**: how temperature/top-p interact with long-CoT self-consistency and search is under-studied ([§12](../12_reasoning_model_inference/04_thinking_budget_and_inference_time_scaling.md)).
- **Hardware-aware sampling** for very large vocabularies as models push V toward 256k+.

***

## References
- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS 2017*. arXiv:1706.03762.
- Holtzman, A., et al. (2020). "The Curious Case of Neural Text Degeneration." *ICLR 2020*. arXiv:1904.09751.
- Fan, A., Lewis, M., Dauphin, Y. (2018). "Hierarchical Neural Story Generation." *ACL 2018*. arXiv:1805.04833.
- Kwon, W., et al. (2023). "PagedAttention/vLLM." *SOSP 2023*. arXiv:2309.06180.
