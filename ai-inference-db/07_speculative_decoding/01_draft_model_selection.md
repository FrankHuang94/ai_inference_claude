# Draft Model Selection

> **Section:** 07_speculative_decoding
> **Last Updated:** June 2026
> **Related Files:** [00_speculative_decoding_fundamentals.md](00_speculative_decoding_fundamentals.md), [02_speculative_decoding_variants.md](02_speculative_decoding_variants.md), [03_medusa_and_hydra_heads.md](03_medusa_and_hydra_heads.md)
> **Must-Read Papers:** Leviathan et al. (2023, ICML); Li et al. (2024) "EAGLE"; Zhou et al. (2024) "DistillSpec"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- The draft model governs **acceptance rate α** (→ speedup) and **draft cost** (overhead); the goal is **high α at low cost**.
- Options: (1) a **smaller model of the same family** (shared tokenizer), (2) a **distilled draft** trained to mimic the target, (3) **self-drafting heads** (Medusa/EAGLE — no separate model), (4) **n-gram/retrieval** drafts (no model at all).
- 📐 Sweet spot: draft cost ≪ target cost (e.g., 10–50× smaller) *and* α high (~0.7+). Both matter; a cheap draft with low α is useless.
- **Tokenizer/vocab must match** the target (or be carefully mapped) for verification to work.
- **Distillation (DistillSpec)** and feature-level drafts (EAGLE) are the highest-α approaches.

***

## Overview
Given the speculative-decoding speedup formula `(1−α^{k+1})/(1−α)` discounted by draft cost ([§00](00_speculative_decoding_fundamentals.md)), the draft model is the single biggest design choice: it sets both the **acceptance rate α** (how often the target agrees, driving the numerator) and the **draft overhead** (its own forward-pass cost, in the denominator). The ideal draft is one that the target almost always agrees with (high α) yet costs almost nothing to run (low overhead). These pull against each other — a bigger, more accurate draft has higher α but more cost — so draft selection is an optimization of α-per-unit-cost.

The options span a spectrum. The classic choice is a **smaller model from the same family** sharing the target's tokenizer (e.g., a 1B draft for a 70B target) — simple, decent α. Better α comes from **distillation** (DistillSpec, Zhou et al. 2024): train the draft specifically to match the target's output distribution, raising agreement. The most effective modern approaches eliminate the separate model: **self-drafting** methods (Medusa adds extra prediction heads; EAGLE drafts at the *feature* level reusing the target's own hidden states) achieve high α with minimal extra cost because the "draft" is tightly coupled to the target ([§02](02_speculative_decoding_variants.md), [§03](03_medusa_and_hydra_heads.md)). At the cheap-and-cheerful end, **n-gram / retrieval** drafts (REST, prompt-lookup) propose tokens by matching against the prompt/datastore with *no model at all* — near-zero cost, lower α, but excellent for repetitive/templated text.

The practical constraints: the draft must produce candidates the target can verify, so **vocabulary/tokenizer compatibility** is mandatory (same tokenizer, or a careful mapping). And α is **workload-dependent** — a draft great on general text may collapse on code or strict-format instruction-following ([§05](05_when_speculative_decoding_fails.md)). The robust recipe in 2025–2026 is EAGLE-style self-drafting (or a distilled draft) for high α at low cost, with n-gram drafting as a zero-cost option for repetitive workloads. This file covers the options and the selection tradeoffs.

***

## Core Concepts & Mechanics

### What determines value
📐 Net speedup ≈ `E[tokens]/(1 + c·k)` where `E[tokens]=(1−α^{k+1})/(1−α)`, c = draft/target cost ratio. Maximize α, minimize c. A draft with α=0.8, c=0.05 is great; α=0.5, c=0.3 is near-useless.

### Options
| Draft type | α | Cost | Notes |
|---|---|---|---|
| Smaller same-family model | medium-high | moderate | needs shared tokenizer |
| Distilled draft (DistillSpec) | high | moderate | trained to mimic target |
| Self-drafting heads (Medusa) | medium-high | low | no separate model |
| Feature-level (EAGLE/EAGLE-2) | high | low | reuses target hidden states |
| n-gram / retrieval (REST) | low-medium | ~zero | great for repetitive text |

### Tokenizer/vocab compatibility
Verification compares draft tokens against target probabilities, so the draft must use the **same tokenization** (or a mapping). Mismatched vocab breaks the rejection rule. Same-family drafts and self-drafting avoid this by construction.

### Distillation for acceptance
DistillSpec trains the draft on the target's outputs (distribution matching, sometimes on-policy) to maximize agreement — directly raising α beyond a generic small model.

### Adaptive use
α varies by domain; some systems pick draft length k adaptively (EAGLE-2) or even disable speculation when measured α is low ([§05](05_when_speculative_decoding_fails.md)).

***

## Key Challenges
1. **α vs cost tradeoff.** Bigger/better drafts raise α but cost more; the optimum depends on target size and workload.
2. **Domain-dependent α.** A draft tuned on general text underperforms on code/format-strict tasks; α isn't stable across workloads.
3. **Tokenizer compatibility.** Cross-family drafts need vocab mapping; mismatches break verification.
4. **Draft maintenance.** Fine-tuning/updating the target requires re-distilling/re-aligning the draft (drift) — an operational burden.

***

## Solutions & Current Best Practices
- **EAGLE/EAGLE-2 self-drafting** for high α at low cost (current SOTA for many cases) ([§02](02_speculative_decoding_variants.md)).
- **Distilled same-family draft** when a separate model is preferred.
- **n-gram/REST** for repetitive/templated workloads at ~zero cost.
- **Measure α per workload**; adapt k or disable when α is low.

***

## Implementation Notes
- Match tokenizer; for cross-family, verify the framework supports vocab mapping.
- Track per-workload acceptance rate; route format-strict traffic differently if α collapses.
- Re-distill/re-train drafts when the target is updated to avoid acceptance drift.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Cheap draft, no speedup** — α too low; cost savings don't matter without acceptance. Use distilled/self-drafting.
- **Great on chat, useless on code** — α is domain-dependent; measure per workload and adapt.
- **Draft drifted after target fine-tune** — acceptance dropped; re-distill the draft on the new target.
- **Cross-tokenizer draft broke verification** — vocab mismatch invalidates the rejection rule; use compatible tokenizers.

***

## Performance Numbers & Benchmarks
| Draft | Typical α | Typical speedup |
|---|---|---|
| Generic small model | ~0.6–0.7 | ~1.5–2× |
| Distilled draft | ~0.75 | ~2–2.5× |
| EAGLE/EAGLE-2 | high | ~2.5–4× |
| n-gram/REST (repetitive) | varies | up to several× on templated text |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"What makes a good draft model?"* — Expected: high α, low cost; α-per-cost; tokenizer match.
- *"Why might a cheaper draft give less speedup than a bigger one?"* — Expected: low α dominates; speedup ∝ acceptance.
- *"How does distillation help speculative decoding?"* — Expected: train draft to match target distribution → higher α.
- *"What breaks if the draft uses a different tokenizer?"* — Expected: verification/rejection rule invalid.

***

## Open Problems & Active Research (2025–2026)
- **Maximizing α with near-zero-cost drafts** (better self-drafting, EAGLE-3+).
- **Workload-adaptive draft selection** and online α estimation.
- **Robust drafts for code/format-strict and reasoning** tasks ([§05](05_when_speculative_decoding_fails.md)).

***

## References
- Leviathan, Y., et al. (2023). "Speculative Decoding." *ICML 2023*. arXiv:2211.17192.
- Li, Y., et al. (2024). "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." arXiv:2401.15077.
- Zhou, Y., et al. (2024). "DistillSpec: Improving Speculative Decoding via Knowledge Distillation." arXiv:2310.08461.
