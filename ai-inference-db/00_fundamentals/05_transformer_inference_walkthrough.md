# Transformer Inference Walkthrough: A Tensor-by-Tensor Trace

> **Section:** 00_fundamentals
> **Last Updated:** June 2026
> **Related Files:** [04_roofline_model_for_llm.md](04_roofline_model_for_llm.md), [01_autoregressive_decoding.md](01_autoregressive_decoding.md), [../06_kernel_optimization/01_flash_attention_deep_dive.md](../06_kernel_optimization/01_flash_attention_deep_dive.md)
> **Must-Read Papers:** Vaswani et al. (2017, NeurIPS) "Attention Is All You Need"; Shazeer (2019, arXiv) "Fast Transformer Decoding" (MQA); Ainslie et al. (2023, EMNLP) "GQA"
> **Estimated Study Time:** 35 minutes

***

## TL;DR
- A forward pass is: **tokenize → embed → [N × decoder layer] → final norm → LM head → sample**, where each layer is **attention (with KV cache) + FFN**, both wrapped in residual + normalization.
- Per **decode** step the query length is 1; the expensive part is **reading weights** (and KV) from HBM, not the tiny matmuls.
- Per-layer weight bytes (FP16) ≈ `2 × (4·d² + 3·d·d_ff)` for a SwiGLU MLP — memorize the shape accounting.
- **GQA/MQA** shrink the K/V projections and KV cache by sharing KV heads across query heads — the single biggest practical KV reduction.
- Worked example: **7B Llama-style, batch=1, decode** on H100 → ~14 GB weight read/token → ~140–170 tok/s.

***

## Overview
This file traces exactly what tensors flow through a decoder-only transformer at inference, with shapes, FLOP/byte accounting, and which tensors come from HBM vs are computed on the fly. The goal is that you can stand at a whiteboard and reproduce the forward pass, annotate each step's bottleneck, and immediately reason about how an optimization (FlashAttention, GQA, FP8, fusion) changes the picture. We use a concrete reference model: **Llama-3-8B-class** — hidden size `d = 4096`, `n_layers = 32`, `n_heads = 32`, `head_dim = 128`, GQA with `n_kv_heads = 8`, FFN intermediate `d_ff ≈ 14336` (SwiGLU), vocab `V ≈ 128256`.

We trace **one decode step** (the common, bandwidth-bound case): batch B=1, a single new query token, current context length S. Prefill is the same computation but with the query dimension equal to the whole prompt P (so every matmul's effective batch is B·P and the operation becomes compute-bound — see [02_prefill_vs_decode_phases.md](02_prefill_vs_decode_phases.md)). Where it matters, we note the prefill difference.

Throughout, remember the central lesson from [00_inference_vs_training.md](00_inference_vs_training.md): at batch=1 the FLOPs are trivial; what costs time is streaming the ~14 GB of weights (and the KV cache) from HBM. Every shape below is in service of computing exactly how many bytes move.

***

## Core Concepts & Mechanics — The Trace

### 0. Tokenization & embedding
- Input token id → **embedding lookup** in `E ∈ R^{V×d}` (here 128256×4096). One row read: `d` values. Output `x ∈ R^{B×1×d}`. Negligible compute; a gather from HBM. (Embedding table is often tied with the LM head.)

### 1. Per decoder layer (×32)
Each layer: `x → RMSNorm → attention → +residual → RMSNorm → FFN → +residual`.

**(a) RMSNorm:** elementwise, reads `x` and a `d`-vector of weights. Memory-bound, tiny. 📐 `y = x / rms(x) · g`, `rms(x)=sqrt(mean(x²)+ε)`.

**(b) QKV projection:**
- `W_q ∈ R^{d×d}` → Q `∈ R^{B×1×d}` (32 heads × 128).
- With **GQA**, `W_k, W_v ∈ R^{d×(n_kv_heads·head_dim)} = R^{4096×1024}` → K,V each `∈ R^{B×1×1024}` (8 KV heads).
- Bytes (FP16): `2·(d² + 2·d·1024) = 2·(16.8M + 8.4M) ≈ 50 MB` per layer. FLOPs: `2·B·(d² + 2·d·1024)` — for B=1, ~50 MFLOP. **AI ≈ 1 → memory-bound.**

**(c) RoPE:** apply rotary positional encoding to Q and the new K. Elementwise, cheap.

**(d) KV cache append + attention:**
- Append new K,V (1024 floats each) to the cache at position S.
- Attention reads **all** cached K,V: `K,V ∈ R^{B×S×1024}`. Bytes read = `2 · B · S · 1024 · 2` (K and V, FP16). At S=4096: ≈ 16 MB/layer just for KV.
- Compute `scores = Q·Kᵀ / √head_dim` → softmax → `·V`. With GQA each KV head serves 4 query heads. FLOPs ≈ `4·h·S·head_dim` ≈ small; **AI ≈ 2 → deeply memory-bound** (FlashAttention/​paged-attention kernels minimize HBM traffic here, see [§06](../06_kernel_optimization/01_flash_attention_deep_dive.md)).
- **Output projection** `W_o ∈ R^{d×d}` → `∈ R^{B×1×d}`. Bytes ≈ `2·d² ≈ 33 MB`.

**(e) FFN (SwiGLU):**
- `gate = x·W_g`, `up = x·W_u`, `W_g,W_u ∈ R^{d×d_ff}` (4096×14336); `h = SiLU(gate) ⊙ up`; `down = h·W_d`, `W_d ∈ R^{d_ff×d}`.
- Bytes (FP16): `2·(2·d·d_ff + d_ff·d) = 2·3·d·d_ff ≈ 2·3·4096·14336 ≈ 352 MB` per layer. **FFN dominates the layer's weight bytes (~85%).** FLOPs `2·B·3·d·d_ff`; AI ≈ B → memory-bound at decode.

**Per-layer total weight bytes (FP16) ≈ 50 MB (QKV) + 33 MB (O) + 352 MB (FFN) ≈ 435 MB.** × 32 layers ≈ **~14 GB** — matches the 8B-weights ≈ 16 GB FP16 figure (difference is embedding/LM head and rounding).

### 2. Final RMSNorm + LM head
- `W_lm ∈ R^{d×V}` (4096×128256). Bytes ≈ `2·d·V ≈ 1.05 GB` — a *single* big read, non-trivial! Produces logits `∈ R^{B×1×V}`. This is why large vocabularies add measurable per-token cost.

### 3. Sampling
- Apply temperature/top-k/top-p over V logits, sample one token (see [01_autoregressive_decoding.md](01_autoregressive_decoding.md)). Cheap relative to weight streaming, but watch large-vocab top-p cost.

### Putting it together (decode, B=1, S=4096, H100)
- Weight bytes/token ≈ 14 GB + 1.05 GB LM head ≈ 15 GB. KV read ≈ 32 × 16 MB ≈ 0.5 GB.
- Time ≈ 15.5e9 / 3.35e12 ≈ **4.6 ms/token → ~215 tok/s** ceiling; real ~140–170 after overhead and <100% bandwidth. **Prefill** of P=512 does the same matmuls with effective batch 512 → compute-bound, near peak FLOPs.

***

## Key Challenges
1. **FFN dominates bytes, attention dominates the long-context tax.** Optimizations must target FFN weight loading (quantization) and KV traffic (GQA, FlashAttention, KV quant) separately.
2. **The LM head is a hidden ~1 GB read per token** for large vocabularies — easy to forget in back-of-envelope models and a real cost at high token rates.
3. **Shapes must be exact for parallelism.** Splitting heads (TP) and FFN dimensions requires divisibility; GQA changes how KV heads shard across TP ranks.
4. **Elementwise/norm ops are individually tiny but numerous** — without fusion their launch + HBM round-trips add up at small batch.

***

## Solutions & Current Best Practices
- **GQA/MQA** (Shazeer 2019; Ainslie et al. 2023) to cut K/V projection and KV cache ~4–8× — universal in modern models ([§02](../02_kv_cache/00_kv_cache_fundamentals.md)).
- **FlashAttention** for the attention step to avoid N×N HBM traffic ([§06](../06_kernel_optimization/01_flash_attention_deep_dive.md)).
- **Operator fusion + CUDA graphs** for norms/RoPE/elementwise to remove launch overhead ([§06](../06_kernel_optimization/02_fused_kernels_and_operator_fusion.md)).
- **Weight quantization** to shrink the dominant FFN/QKV byte cost ([§05](../05_quantization/02_weight_only_quantization.md)).

***

## Implementation Notes
- Pre-allocate KV in paged blocks (see [§02](../02_kv_cache/01_paged_attention_vllm.md)); the attention kernel must gather non-contiguous KV blocks.
- Tie embedding/LM-head weights when the architecture does, to save ~1 GB of HBM and one big read.
- For TP, column-parallel QKV/up/gate and row-parallel O/down minimize all-reduces (see [§04](../04_parallelism/00_tensor_parallelism.md)).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Forgetting the LM-head read** undercounts per-token cost by ~7% for 8B and more for small models with large vocab — it can dominate for a 1–2B model with 128k+ vocab.
- **GQA changes TP sharding of KV heads** — with 8 KV heads and TP=8 each rank holds one KV head; TP>n_kv_heads requires replication, an easy correctness/perf trap.
- **RoPE applied to cached K vs new K** — you cache *post-RoPE* K (or apply RoPE consistently); mixing conventions silently corrupts attention at long context.
- **Prefill reuses the same kernels but is compute-bound** — tuning a kernel only at batch=1 can leave prefill performance on the table.

***

## Performance Numbers & Benchmarks
| Step (8B, B=1, S=4096, H100) | Weight bytes | KV bytes | Bound |
|---|---|---|---|
| QKV proj (GQA) ×32 | ~1.6 GB | — | memory |
| Attention ×32 | — | ~0.5 GB | memory |
| Output proj ×32 | ~1.1 GB | — | memory |
| FFN (SwiGLU) ×32 | ~11.3 GB | — | memory |
| LM head | ~1.05 GB | — | memory |
| **Total/token** | **~15 GB** | ~0.5 GB | ~140–170 tok/s |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Walk through a single decode forward pass for an 8B model and tell me where the bytes go."* — Expected: FFN ~75–85% of weight bytes, KV grows with S, LM head ~1 GB.
- *"How does GQA change the shapes and the KV cache?"* — Expected: K/V projection to n_kv_heads·head_dim; KV cache shrinks by n_heads/n_kv_heads.
- *"Why is the LM head non-negligible per token?"* — Expected: d×V ≈ 1 GB read; worse for small models / big vocab.
- *"Which steps are memory-bound vs compute-bound at decode vs prefill?"* — Expected: all memory-bound at decode B=1; matmuls compute-bound at prefill.

***

## Open Problems & Active Research (2025–2026)
- **Cheaper LM heads / vocabulary methods** (factorized or hierarchical softmax) as vocabularies grow past 256k.
- **Attention variants (MLA, GQA successors)** that further cut KV bytes without quality loss ([§06](../06_kernel_optimization/01_flash_attention_deep_dive.md)).
- **Fusion across the whole layer** (mega-kernels) to minimize HBM round-trips for small-batch decode.

***

## References
- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS 2017*. arXiv:1706.03762.
- Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need" (MQA). arXiv:1911.02150.
- Ainslie, J., et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models." *EMNLP 2023*. arXiv:2305.13245.
- Touvron, H., et al. / Llama team (2023–2024). Llama model reports.
