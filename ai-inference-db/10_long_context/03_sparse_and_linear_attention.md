# Sparse and Linear Attention (and SSMs)

> **Section:** 10_long_context
> **Last Updated:** June 2026
> **Related Files:** [01_attention_complexity_solutions.md](01_attention_complexity_solutions.md), [00_long_context_challenges.md](00_long_context_challenges.md), [../06_kernel_optimization/01_flash_attention_deep_dive.md](../06_kernel_optimization/01_flash_attention_deep_dive.md)
> **Must-Read Papers:** Beltagy et al. (2020) "Longformer"; Katharopoulos et al. (2020) "Linear Transformers"; Gu & Dao (2023) "Mamba"; Dao & Gu (2024) "Mamba-2"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- **Sparse attention** restricts each token to attend to a subset (local window + global/strided tokens) → ~O(N·w) compute; quality depends on the pattern (Longformer, BigBird).
- **Linear attention** rewrites softmax attention with kernel feature maps so it becomes a **recurrence** with **constant state** → O(N) compute, no KV growth (Katharopoulos et al. 2020).
- **SSMs (Mamba/Mamba-2)**: selective state-space recurrence; O(N) compute, **constant per-step state**, hardware-efficient selective scan → huge serving win for long context.
- Inference upshot: constant-state models eliminate the linear KV cache — cheap, fast long-context decode — but trail full attention on **exact long-range recall**.
- **Hybrids** (attention + Mamba layers, e.g., Jamba) are the pragmatic 2025–2026 answer.

***

## Overview
This file digs into the sub-quadratic families that change *decode economics*: sparse attention, linear attention, and state-space models. **Sparse attention** keeps the attention mechanism but limits its scope — each token attends only to a local window plus a few global/strided tokens — reducing compute from O(N²) toward O(N·w) for window w. Longformer and BigBird showed this works for many long-document tasks, but a fixed pattern can miss dependencies it doesn't cover, and content-based (data-dependent) sparsity is harder to make hardware-efficient. Sparse attention still maintains a KV cache (reduced), so its decode win is partial.

**Linear attention** is more radical: it replaces the softmax similarity with a **kernel feature map** φ, so attention `softmax(QKᵀ)V` becomes `φ(Q)(φ(K)ᵀV)`, which by associativity can be computed as a **recurrence** maintaining a fixed-size state matrix `S = Σ φ(k)vᵀ`. This makes both training (O(N)) and, critically, **decode** O(1) per token with **constant memory** — no growing KV cache at all (Katharopoulos et al. 2020). **State-space models** (Mamba, Gu & Dao 2023; Mamba-2, 2024) generalize this idea with a **selective** linear recurrence — an input-dependent state-space system carrying a fixed hidden state — plus a hardware-efficient **selective scan** kernel. Mamba matches transformer quality at moderate scale with O(N) compute and **constant per-step state**, which for inference means *cheap, fast, memory-flat long-context decode* — arguably the most important serving property a sequence model can have.

The persistent catch is **exact long-range recall**: full attention can directly look up any past token's content, while constant-state models must compress all history into a fixed state and tend to underperform on "needle-in-a-haystack" retrieval and tasks needing precise recall of arbitrary earlier content. This quality gap — not efficiency — is why pure SSM/linear models haven't replaced transformers at the frontier, and why the field has converged (for now) on **hybrid** architectures that interleave a minority of full-attention layers (for recall) with a majority of Mamba/linear layers (for efficiency), e.g., Jamba, Zamba. This file covers the mechanics, the recall tradeoff, and hybrids.

***

## Core Concepts & Mechanics

### Sparse attention
- Patterns: **local window** (attend to ±w neighbors), **global tokens** (a few tokens attend to/are attended by all), **strided/dilated**. Longformer/BigBird combine these.
- 📐 Compute ~O(N·w + N·g). KV reduced but present. Quality depends on whether the pattern covers needed dependencies.

### Linear attention
- 📐 `Attn = φ(Q)(φ(K)ᵀ V)` — compute `S = φ(K)ᵀV` (d×d state) incrementally; per token: `out = φ(q)·S`, update `S += φ(k)vᵀ`. **Constant state** (d×d), O(1)/token decode, O(N) total.
- No softmax → approximation; quality historically below full attention, improving with better feature maps/gating (e.g., gated linear attention).

### SSMs (Mamba/Mamba-2)
- Linear recurrence `h_t = A h_{t-1} + B x_t`, `y_t = C h_t`, with **selective** (input-dependent) A,B,C. Fixed-size hidden state h.
- **Selective scan** kernel: hardware-efficient parallel scan (training) + recurrent (decode). O(N) compute, **constant per-step state/memory**.
- Mamba-2 connects SSMs and attention (state-space duality), improving efficiency and scale.

### Inference upshot
| Property | Full attn | Sparse | Linear | SSM |
|---|---|---|---|---|
| Decode/token | O(N) KV read | reduced | **O(1)** | **O(1)** |
| Per-req memory | linear KV | reduced | **constant** | **constant** |
| Recall quality | best | pattern-dep | lower | competitive (moderate) |

### Hybrids
Interleave attention layers (recall) with Mamba/linear layers (efficiency) — Jamba (Mamba+attention+MoE), Zamba. Most efficiency, near-full quality; the pragmatic direction.

***

## Key Challenges
1. **Exact recall gap.** Constant-state models compress history and miss precise long-range lookups; the main barrier to replacing attention.
2. **Kernel maturity.** Selective-scan / linear-attention kernels and serving-framework support lag FlashAttention's ecosystem.
3. **Sparse pattern coverage.** Fixed patterns miss uncovered dependencies; data-dependent sparsity is hard to make efficient.
4. **Hybrid design search.** Optimal attention/SSM ratio and placement is unsettled.

***

## Solutions & Current Best Practices
- **Hybrids (attention + Mamba)** for long-context serving needing both efficiency and recall.
- **SSM/linear** where decode cost/long context dominates and exact recall is less critical.
- **Sparse attention** for long-document tasks where the pattern fits.
- **Validate recall** (RULER/needle) before deploying any sub-quadratic model.

***

## Implementation Notes
- Ensure efficient selective-scan/linear-attention kernels exist in your serving stack; otherwise the theoretical win doesn't materialize.
- For hybrids, KV cache only the attention layers (the SSM layers have constant state) — a memory win to exploit.
- Benchmark long-context recall, not just perplexity/throughput.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Mamba/linear model failed needle tasks** — constant state can't do exact recall; use hybrid or full attention for recall-critical work.
- **No serving-framework kernel** — selective scan unsupported/slow; O(N) advantage unrealized.
- **Sparse pattern missed a dependency** — fixed window/global didn't cover the link; quality dropped silently.
- **Treated hybrid KV like full transformer** — only attention layers cache KV; mis-sizing memory.

***

## Performance Numbers & Benchmarks
| Model class | Long-context decode cost | Recall |
|---|---|---|
| Transformer (FlashAttention) | linear KV, O(N²) prefill | best |
| Sparse (Longformer) | reduced | pattern-dependent |
| Linear (gated) | constant state | improving |
| Mamba-2 | constant state | competitive (moderate scale) |
| Hybrid (Jamba) | mostly constant | near-full |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why is linear/SSM attention attractive for inference?"* — Expected: constant state, O(1)/token decode, no KV growth.
- *"What's the catch with constant-state models?"* — Expected: exact long-range recall; needle tasks.
- *"Derive how linear attention becomes a recurrence."* — Expected: associativity of φ(Q)(φ(K)ᵀV), incremental state S.
- *"Why hybrids?"* — Expected: attention layers for recall + SSM layers for efficiency.

***

## Open Problems & Active Research (2025–2026)
- **Closing the recall gap** for constant-state models at scale.
- **Optimal hybrid architectures** and KV-only-on-attention serving optimizations.
- **Mature, fast kernels** for selective scan / gated linear attention across frameworks.

***

## References
- Beltagy, I., et al. (2020). "Longformer." arXiv:2004.05150.
- Katharopoulos, A., et al. (2020). "Transformers are RNNs: Linear Attention." *ICML 2020*. arXiv:2006.16236.
- Gu, A., Dao, T. (2023). "Mamba." arXiv:2312.00752; Dao, T., Gu, A. (2024). "Mamba-2." arXiv:2405.21060.
- Lieber, O., et al. (2024). "Jamba." arXiv:2403.19887.
