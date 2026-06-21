# Lookahead and Jacobi Decoding

> **Section:** 07_speculative_decoding
> **Last Updated:** June 2026
> **Related Files:** [00_speculative_decoding_fundamentals.md](00_speculative_decoding_fundamentals.md), [02_speculative_decoding_variants.md](02_speculative_decoding_variants.md), [05_when_speculative_decoding_fails.md](05_when_speculative_decoding_fails.md)
> **Must-Read Papers:** Fu et al. (2024, ICML) "Lookahead Decoding"; Santilli et al. (2023) "Parallel Jacobi/GS Decoding"; Song et al. (2021) "Jacobi decoding for MT"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- These methods accelerate decoding **without a draft model** by reformulating generation as a **parallel fixed-point iteration**.
- **Jacobi decoding**: treat the next n tokens as unknowns and iteratively refine them in parallel until they converge to the autoregressive result (a fixed point).
- **Lookahead decoding** (Fu et al. 2024): combines Jacobi iteration with an **n-gram cache** of verified token sequences to propose and verify multiple tokens per step — guaranteed same output as greedy.
- Pros: no draft model, no training; Cons: gains are modest vs EAGLE, and benefits shrink at high batch (like all speculation).
- Trades extra parallel compute per step for fewer total steps.

***

## Overview
Lookahead and Jacobi decoding are **draft-model-free** acceleration methods that exploit a different idea than draft-verify speculation: reformulate autoregressive generation as solving a system of equations in parallel. Normally token *t+1* depends on *t*, forcing strict sequential decoding. But you can instead *guess* the next n tokens, then **iteratively refine** all of them in parallel — each iteration updates every position using the current guesses — until the sequence stops changing (a **fixed point**), which provably equals the autoregressive output. This is **Jacobi iteration** (and the related Gauss-Seidel variant) applied to decoding: it converts some sequential steps into parallel refinement passes, and since each pass is a single (bandwidth-bound) forward over n positions for ~the cost of one token, you can finish a chunk of tokens in fewer passes than naive decoding.

Pure Jacobi decoding converges slowly in practice, so **Lookahead decoding** (Fu et al. 2024) makes it practical by adding two ingredients: a **lookahead branch** that generates n-grams via Jacobi iteration, and an **n-gram cache** ("verification branch") that stores previously-seen token n-grams and proposes them as candidates to verify in the same forward pass. The model both *refines* future tokens (Jacobi) and *matches/verifies* cached n-grams, accepting any that the model confirms — emitting multiple tokens per step. Lookahead is **exact for greedy decoding** (guaranteed identical output) and requires **no draft model and no training**, which is its main appeal: you can enable it on any model immediately.

The tradeoffs are real. Lookahead/Jacobi trade **extra compute per step** (refining/verifying n positions) for **fewer steps**, so they help when decode is bandwidth-bound (low batch) and underutilizing compute — and, like all speculation, the benefit shrinks at high batch where the extra parallel work isn't free ([§05](05_when_speculative_decoding_fails.md)). Their speedups are generally **more modest than EAGLE/Medusa** because the proposals (Jacobi guesses, n-gram matches) are less accurate than a trained feature-level drafter. So the niche is: zero-setup acceleration where you can't or don't want to train heads/drafts. This file covers the fixed-point reformulation, Lookahead's n-gram mechanism, and the tradeoffs.

***

## Core Concepts & Mechanics

### Decoding as a fixed point
📐 Autoregressive generation of `y_{1..n}` satisfies `y_i = f(y_{<i}, x)` for all i. Jacobi iteration: start with a guess `y^{(0)}`, repeatedly apply `y_i^{(k+1)} = f(y_{<i}^{(k)}, x)` for all i **in parallel**; iterate until `y^{(k+1)} = y^{(k)}` (fixed point) = the autoregressive output. Converges in ≤ n iterations (worst case = sequential), often fewer.

### Lookahead decoding
- **Lookahead branch**: maintains a 2D window of future-token guesses, refined by Jacobi steps each forward pass, generating candidate **n-grams**.
- **Verification branch**: an **n-gram pool** of previously-generated sequences; matching n-grams are proposed and verified in the same pass.
- Each forward pass: refine future tokens (Jacobi) + verify n-gram candidates; accept confirmed tokens. **Exact for greedy.**
- Knobs: window size W (lookahead breadth) and n-gram size N — bigger → more candidates, more compute/step.

### vs draft-verify speculation
| | Lookahead/Jacobi | Draft-verify (EAGLE/Medusa) |
|---|---|---|
| Draft model | none | yes (or self-heads) |
| Training | none | usually yes |
| Speedup | modest | higher |
| Setup | instant | train/integrate |

***

## Key Challenges
1. **Slow Jacobi convergence.** Naive Jacobi often needs many iterations; without the n-gram cache, speedups are small.
2. **Compute-per-step overhead.** Refining/verifying a window of positions adds FLOPs per pass; net win only when decode is bandwidth-bound (low batch).
3. **Modest gains vs trained drafters.** Proposals are less accurate than EAGLE/Medusa → lower acceptance → smaller speedup.
4. **Sampling exactness.** Strong guarantees are for **greedy**; matching arbitrary-temperature sampling distributions is more involved.

***

## Solutions & Current Best Practices
- **Lookahead decoding** when you want **zero-setup** acceleration (no draft, no training) — e.g., quick wins on any model.
- **Tune window/n-gram sizes** to the batch and latency target.
- **Prefer EAGLE/Medusa** when you can train heads and want larger speedups ([§02](02_speculative_decoding_variants.md), [§03](03_medusa_and_hydra_heads.md)).
- **Disable at high batch** where compute overhead dominates ([§05](05_when_speculative_decoding_fails.md)).

***

## Implementation Notes
- Lookahead is available in some frameworks/libraries; configure window (W) and n-gram (N) sizes.
- Best for greedy or low-temperature decoding; verify behavior under your sampling settings.
- Measure net tokens/s vs plain decode at your batch size — the gain is workload/batch-dependent.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Tiny speedup vs EAGLE** — expected; Jacobi/n-gram proposals are weaker than trained drafters.
- **Regression at high batch** — extra per-step compute isn't free when compute-bound; disable.
- **Big window slowed it down** — too-large lookahead/n-gram added compute beyond the step savings; tune down.
- **Sampling output differed** — exactness guarantees are strongest for greedy; check sampling-mode behavior.

***

## Performance Numbers & Benchmarks
| Method | Setup | Speedup (low batch, greedy) |
|---|---|---|
| Pure Jacobi | none | small |
| Lookahead | none | ~1.5–2× |
| EAGLE-2 (contrast) | train heads | ~3–4× |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How does Jacobi decoding parallelize autoregressive generation?"* — Expected: fixed-point iteration over n tokens; converges to AR output.
- *"What does Lookahead add to make Jacobi practical?"* — Expected: n-gram cache + lookahead branch; verify candidates per pass; exact for greedy.
- *"Lookahead vs draft-based speculation — when use which?"* — Expected: zero-setup vs higher-speedup-with-training.
- *"Why do gains shrink at high batch?"* — Expected: extra per-step compute not free when compute-bound.

***

## Open Problems & Active Research (2025–2026)
- **Faster-converging parallel decoding** (better initialization, hybrid with retrieval).
- **Exact arbitrary-temperature** lookahead with strong guarantees.
- **Combining lookahead with feature-level drafting** for best-of-both.

***

## References
- Fu, Y., et al. (2024). "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding." *ICML 2024*. arXiv:2402.02057.
- Santilli, A., et al. (2023). "Accelerating Transformer Inference for Translation via Parallel Decoding." *ACL 2023*. arXiv:2305.10427.
- Song, Y., et al. (2021). "Jacobi/Gauss-Seidel parallel decoding."
