# Speculative Decoding Fundamentals

> **Section:** 07_speculative_decoding
> **Last Updated:** June 2026
> **Related Files:** [01_draft_model_selection.md](01_draft_model_selection.md), [05_when_speculative_decoding_fails.md](05_when_speculative_decoding_fails.md), [../00_fundamentals/01_autoregressive_decoding.md](../00_fundamentals/01_autoregressive_decoding.md)
> **Must-Read Papers:** Leviathan et al. (2023, ICML) "Fast Inference from Transformers via Speculative Decoding"; Chen et al. (2023) "Accelerating LLM Decoding with Speculative Sampling"; Stern et al. (2018) "Blockwise Parallel Decoding"
> **Estimated Study Time:** 35 minutes

***

## TL;DR
- Decode at small batch is **memory-bandwidth-bound**: a forward pass spends most time loading weights, so verifying **k tokens at once costs nearly the same as 1**.
- A small **draft model** proposes k tokens autoregressively; the large **target model verifies all k+1 in one parallel pass**; a **rejection-sampling** rule guarantees the output distribution **exactly matches** the target.
- 📐 Expected tokens/step = `(1 − α^{k+1})/(1 − α)` where α = acceptance rate; speedup ≈ this / (cost ratio).
- It converts bandwidth-bound decode into more **compute-bound verification** — free lunch *only while the GPU is underutilized* (low batch).
- Fails/regresses at **high batch** (verification becomes compute-bound) and **low acceptance** (domain mismatch) — see [§05](05_when_speculative_decoding_fails.md).

***

## Overview
Speculative decoding exploits the central inefficiency of autoregressive decode: at small batch the target model is memory-bandwidth-bound, so a single forward pass (which streams all weights) produces just one token while the compute units idle ([§00](../00_fundamentals/00_inference_vs_training.md)). The key realization (Leviathan et al. 2023; Chen et al. 2023) is that the target can **verify multiple candidate tokens in one forward pass for almost the same cost as one token**, because the bottleneck is the weight load, not the per-token compute. So if a cheap **draft model** guesses the next k tokens, the target can check all of them in a single pass and accept the longest correct prefix — emitting multiple tokens per expensive target forward pass.

Crucially, this is done with a **rejection-sampling** acceptance rule that makes the output distribution **provably identical** to sampling from the target model alone — it is exact, not an approximation. For each drafted token, you accept it with probability `min(1, p_target/p_draft)`; on rejection, you resample from an adjusted distribution `(p_target − p_draft)_+` normalized. This guarantees that, in expectation, you produce exactly the target model's distribution while the draft only affects *speed* (acceptance rate), never *correctness*. This exactness is what makes speculative decoding safe to deploy and a favorite interview topic — be able to state and justify the acceptance rule.

The speedup depends on the **acceptance rate α** (how often the target agrees with the draft) and the **cost ratio** (draft+verify vs plain decode). With acceptance α and draft length k, the expected number of accepted tokens per target pass is `(1−α^{k+1})/(1−α)`; high α and the right k give 2–3× speedups in practice. The technique converts bandwidth-bound decode into somewhat more **compute-bound** verification — which is exactly why it stops helping (or hurts) at **high batch**, where the target is already compute-bound and the extra verification work isn't free ([§05](05_when_speculative_decoding_fails.md)). This file covers the protocol, the rejection-sampling proof sketch, and the speedup math.

***

## Core Concepts & Mechanics

### The draft-verify protocol
1. **Draft**: the small model generates k candidate tokens `x₁..x_k` autoregressively (cheap).
2. **Verify**: the target model runs **one parallel forward pass** over the prompt + k drafts, producing target probabilities `p_target` at each of the k+1 positions.
3. **Accept/reject**: walk the k drafts; accept each with the rejection rule; on first rejection, resample that position from the adjusted distribution and stop; if all k accepted, sample one extra "bonus" token from the target at position k+1.
4. Emit accepted tokens (1 to k+1), repeat.

### The rejection-sampling rule (exactness)
📐 For draft token `x` with draft prob `q(x)` and target prob `p(x)`:
- Accept with prob `min(1, p(x)/q(x))`.
- If rejected, sample from `norm((p − q)_+)` (the positive part).
This yields samples **exactly distributed as p** (the target). Proof idea: the accept + resample probabilities compose to `p(x)` for every x. So the draft never changes the output distribution — only throughput.

### Speedup math
📐 Let α = expected per-token acceptance probability. Expected accepted tokens per target pass:
```
E[tokens] = (1 − α^{k+1}) / (1 − α)
```
Speedup ≈ `E[tokens] / (1 + c·k)` where c = draft cost / target cost per token (draft is cheap, c small). High α and moderate k (k≈4–8) maximize it. At α→1, E[tokens]→k+1.

### Why it works (hardware view)
At low batch the target verify pass is bandwidth-bound (same weight load as 1 token) but now does useful work for k+1 positions → higher arithmetic intensity, more tokens per weight load. It "fills" the idle compute with verification.

***

## Key Challenges
1. **Acceptance rate dependence.** Speedup hinges on α; a poorly-matched draft (domain/instruction mismatch) gives low α and little/no gain ([§01](01_draft_model_selection.md)).
2. **Draft overhead and memory.** The draft model adds latency (autoregressive drafting) and HBM; must be cheap relative to the target.
3. **Batch-size sensitivity.** At high batch the target is already compute-bound; verifying extra tokens costs real compute, eroding or reversing the benefit ([§05](05_when_speculative_decoding_fails.md)).
4. **Choosing k.** Too small → little speedup; too large → wasted draft+verify on tokens likely rejected. Often adaptive.

***

## Solutions & Current Best Practices
- **Match draft to target** (same family/tokenizer, distilled draft) to maximize α ([§01](01_draft_model_selection.md)).
- **Self-drafting / feature-level drafts** (EAGLE, Medusa) to raise α and cut draft cost ([§02](02_speculative_decoding_variants.md), [§03](03_medusa_and_hydra_heads.md)).
- **Adaptive draft length** (EAGLE-2) based on confidence.
- **Disable/scale down at high batch** where it stops helping ([§05](05_when_speculative_decoding_fails.md)).

***

## Implementation Notes
- vLLM/SGLang/TRT-LLM support speculative decoding (draft model, EAGLE, Medusa, n-gram); configure draft and k.
- Measure **acceptance rate** and **net tokens/s** in your deployment — it's workload-dependent; verify net-positive.
- Ensure the verify pass uses the same sampling params; the rejection rule must use the actual target/draft probabilities.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **No speedup at production batch sizes** — at high batch the target is compute-bound; verification isn't free. Speculative decoding is a *low-batch/latency* tool.
- **Low acceptance on instruction-following/format tasks** — draft diverges on exact formatting; α drops, gain vanishes. Distill/match the draft.
- **Draft cost ate the gains** — draft too large or slow; use a much cheaper draft or self-drafting heads.
- **Assumed quality loss** — there is none if the rejection rule is implemented correctly; output matches the target exactly.

***

## Performance Numbers & Benchmarks
| Setting | Acceptance α | Speedup (low batch) |
|---|---|---|
| Well-matched draft, k≈4–5 | ~0.7–0.8 | ~2–3× |
| EAGLE/EAGLE-2 | higher (feature-level) | ~2.5–4× |
| Poor draft / domain mismatch | <0.5 | ~1× or worse |
| High batch (compute-bound) | — | ~1× or regression |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why does speculative decoding speed up decode without changing the output distribution?"* — Expected: bandwidth-bound verify of k tokens ~free; rejection-sampling exactness.
- *"State the acceptance rule and the expected-tokens formula."* — Expected: `min(1,p/q)` + resample `(p−q)_+`; `(1−α^{k+1})/(1−α)`.
- *"When does speculative decoding stop helping?"* — Expected: high batch (compute-bound), low acceptance.
- *"How do you choose draft length k?"* — Expected: balance speedup vs wasted draft/verify; adaptive.

***

## Open Problems & Active Research (2025–2026)
- **Speculative decoding at high batch** (batch-aware tree verification) that stays net-positive.
- **Maximizing acceptance** via better self-drafting (EAGLE-3+) and distillation ([§01](01_draft_model_selection.md)).
- **Speculative decoding for reasoning models** with long, structured CoT ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

***

## References
- Leviathan, Y., Kalman, M., Matias, Y. (2023). "Fast Inference from Transformers via Speculative Decoding." *ICML 2023*. arXiv:2211.17192.
- Chen, C., et al. (2023). "Accelerating Large Language Model Decoding with Speculative Sampling." arXiv:2302.01318.
- Stern, M., Shazeer, N., Uszkoreit, J. (2018). "Blockwise Parallel Decoding." *NeurIPS 2018*. arXiv:1811.03115.
