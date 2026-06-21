# Speculative Decoding Variants: SpecInfer, EAGLE, REST

> **Section:** 07_speculative_decoding
> **Last Updated:** June 2026
> **Related Files:** [00_speculative_decoding_fundamentals.md](00_speculative_decoding_fundamentals.md), [03_medusa_and_hydra_heads.md](03_medusa_and_hydra_heads.md), [04_lookahead_and_jacobi_decoding.md](04_lookahead_and_jacobi_decoding.md)
> **Must-Read Papers:** Miao et al. (2023) "SpecInfer"; Li et al. (2024) "EAGLE" & "EAGLE-2"; He et al. (2024) "REST"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- **SpecInfer** (Miao et al. 2023): **tree-based** speculation — draft *multiple* candidate sequences as a token tree, verify the whole tree in one target pass → higher expected accepted tokens than a single chain.
- **EAGLE** (Li et al. 2024): draft at the **feature (hidden-state) level**, reusing the target's own representations → much higher acceptance than token-level drafts; **EAGLE-2** adds **dynamic, context-aware draft trees**.
- **REST** (He et al. 2024): **retrieval-based** drafting from a datastore (no draft model) — propose continuations matched from a corpus; great for repetitive/known text.
- Trend: from single-chain → **tree verification**; from separate model → **feature-level self-drafting**. EAGLE-family is current SOTA for general use.
- All preserve exactness via the same rejection/verification principle.

***

## Overview
The base speculative-decoding algorithm drafts a single linear chain of k tokens, but acceptance is probabilistic — one early rejection truncates the rest. The variants improve on this along two axes: **what to draft** (a single chain vs a *tree* of candidates) and **how to draft** (a separate token-level model vs feature-level self-drafting vs retrieval). Each pushes the expected accepted tokens per target pass higher, which directly raises throughput.

**SpecInfer** introduced **tree-based speculation**: instead of one candidate sequence, the draft produces a *tree* of plausible continuations (multiple branches at each step), and the target verifies the entire tree in a single forward pass using a tree-structured attention mask. Because the tree covers more of the probability mass, the chance that *some* path matches the target is higher, so more tokens are accepted per pass — at the cost of verifying more candidate tokens. **EAGLE** changed *how* drafting works: rather than a separate model predicting tokens, it predicts the target's next **hidden-state feature** (then maps to a token), reusing the target's own representations. Feature-level drafting is far more accurate (the draft "thinks like" the target), yielding high acceptance at low cost; **EAGLE-2** adds **dynamic draft trees** whose shape adapts to the model's confidence in context, further boosting accepted tokens. EAGLE/EAGLE-2 are widely regarded as SOTA for general-purpose speculative decoding.

**REST** removes the model entirely for drafting: it **retrieves** candidate continuations from a datastore (e.g., a corpus or prior outputs) keyed on the recent context, proposing them for verification. With zero draft compute, it shines on repetitive, templated, or domain-specific text where good continuations exist in the store; acceptance is lower on novel text. Together these variants illustrate the design space — tree vs chain, model vs features vs retrieval — and all retain the **exactness guarantee** because the target still verifies every token with the rejection rule ([§00](00_speculative_decoding_fundamentals.md)). This file compares them and their tradeoffs.

***

## Core Concepts & Mechanics

### SpecInfer (tree speculation)
- Draft a **token tree** (multiple candidate branches), e.g., top-b tokens at each of several positions.
- Verify with a **tree attention mask** so all branches are checked in one target pass.
- Accept the longest verified path. 📐 Expected accepted tokens rises vs single chain because more candidates are covered; cost = verifying more tokens (still ~one weight load).

### EAGLE (feature-level drafting)
- Insight: predicting the next **feature** (hidden state) is easier/more accurate than predicting the next token directly, and reusing the target's features makes the draft highly aligned.
- A lightweight autoregressive head on top of the target's features drafts; tokens derived and verified. High α, low cost (no full separate model).
- **EAGLE-2**: **context-aware dynamic draft trees** — tree shape/depth adapt to confidence, increasing accepted tokens without wasted verification. **EAGLE-3** (successor) pushes acceptance further.

### REST (retrieval drafting)
- Maintain a **datastore**; key on recent tokens to retrieve likely continuations; propose them as drafts.
- **No draft model** → zero draft compute. α depends on store coverage; excellent for repetitive/known text, weak on novel content.

### Comparison
| Variant | Draft source | Structure | α | Cost |
|---|---|---|---|---|
| Vanilla | small model | chain | medium | moderate |
| SpecInfer | small model | tree | higher | more verify |
| EAGLE/EAGLE-2 | feature head | (dynamic) tree | high | low |
| REST | retrieval | candidates | varies | ~zero |

***

## Key Challenges
1. **Tree verification cost.** Trees verify more candidate tokens per pass; at high batch this extra compute can erode the benefit ([§05](05_when_speculative_decoding_fails.md)).
2. **Tree-mask kernel complexity.** Verifying a token tree needs custom attention masks/kernels; correctness and efficiency are non-trivial.
3. **EAGLE training/integration.** Feature-level heads must be trained per target and integrated into the framework; updating the target needs retraining.
4. **Retrieval coverage (REST).** α collapses on novel text outside the datastore; store construction/maintenance matters.

***

## Solutions & Current Best Practices
- **EAGLE-2 (or EAGLE-3)** for general-purpose high-α, low-cost speculation — current default SOTA.
- **SpecInfer-style trees** when more acceptance is worth extra verify (low-batch latency-critical).
- **REST** for repetitive/templated/domain workloads at zero draft cost.
- **Adapt tree size/draft length** to batch and confidence; disable at high batch ([§05](05_when_speculative_decoding_fails.md)).

***

## Implementation Notes
- vLLM/SGLang/TRT-LLM support EAGLE/Medusa/n-gram/draft-model speculation; pick per workload.
- Tree methods need the framework's tree-attention kernel; verify it's efficient.
- Train EAGLE heads on data matching production; re-train when the target changes.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Tree speculation slower at high batch** — extra verified tokens add compute when the target is already compute-bound; trees help most at low batch.
- **EAGLE head stale after target update** — feature head must be retrained; acceptance drops otherwise.
- **REST useless on novel prompts** — datastore miss → low α; great only where continuations are retrievable.
- **Custom tree-mask kernel bug** — wrong masking accepts incorrect paths or breaks exactness; validate distribution.

***

## Performance Numbers & Benchmarks
| Variant | Reported speedup (low batch) |
|---|---|
| Vanilla spec decoding | ~1.5–2× |
| SpecInfer (tree) | ~2–3× |
| EAGLE | ~2.5–3× |
| EAGLE-2 | ~3–4× |
| REST (repetitive text) | up to several× |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why does tree-based speculation accept more tokens than a chain?"* — Expected: covers more probability mass; verify tree in one pass.
- *"What's the key idea in EAGLE?"* — Expected: feature-level drafting reusing target hidden states → higher α at low cost.
- *"When would you use REST?"* — Expected: repetitive/templated/domain text; zero draft cost; retrieval coverage.
- *"What's the cost of tree verification and when does it hurt?"* — Expected: more verified tokens; high batch (compute-bound).

***

## Open Problems & Active Research (2025–2026)
- **EAGLE-3+ and beyond** — pushing the acceptance frontier (watchlist).
- **Batch-aware tree speculation** that stays net-positive at high batch.
- **Hybrid retrieval + feature drafting** for both novel and repetitive text.

***

## References
- Miao, X., et al. (2023). "SpecInfer: Accelerating LLM Serving with Tree-based Speculative Inference." arXiv:2305.09781.
- Li, Y., et al. (2024). "EAGLE" arXiv:2401.15077; "EAGLE-2" arXiv:2406.16858.
- He, Z., et al. (2024). "REST: Retrieval-Based Speculative Decoding." *NAACL 2024*. arXiv:2311.08252.
