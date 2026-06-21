# Medusa and Hydra: Self-Drafting Heads

> **Section:** 07_speculative_decoding
> **Last Updated:** June 2026
> **Related Files:** [00_speculative_decoding_fundamentals.md](00_speculative_decoding_fundamentals.md), [02_speculative_decoding_variants.md](02_speculative_decoding_variants.md), [01_draft_model_selection.md](01_draft_model_selection.md)
> **Must-Read Papers:** Cai et al. (2024) "Medusa"; Ankner et al. (2024) "Hydra"; Stern et al. (2018) "Blockwise Parallel Decoding"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- **Medusa** (Cai et al. 2024): add multiple lightweight **prediction heads** on top of the target model that each predict a *future* token (head i → token t+i), enabling self-speculation with **no separate draft model**.
- Heads are trained (frozen backbone) cheaply; combined with **tree attention** to verify many candidate combinations per pass.
- **Hydra** (Ankner et al. 2024): make the heads **sequentially dependent** (each head sees previous heads' predictions) → higher acceptance than Medusa's independent heads.
- Simpler to deploy than a separate draft (no second model, shared KV/features) but α is lower than feature-level EAGLE.
- Self-drafting trades a small amount of extra compute/params for eliminating the draft model and its overhead.

***

## Overview
Medusa attacks the draft-model overhead and integration burden ([§01](01_draft_model_selection.md)) by making the target model draft *itself*. It adds a handful of small **prediction heads** to the final hidden state, where head *i* is trained to predict the token at position *t+i* (head 1 → next token as usual, head 2 → token after that, etc.). In one forward pass the model thus proposes several future tokens directly from its own representation — no separate draft model, no second tokenizer, no extra weight set to stream. These proposals are then assembled into a **tree of candidate continuations** (since each head outputs top-k options) and verified by the target in a single pass with **tree attention**, accepting the longest valid path. The heads are cheap to train (the backbone is frozen; only the heads learn), making Medusa easy to bolt onto an existing model.

The limitation Medusa has is that its heads are **independent** — head 3 predicts token t+3 without knowing what heads 1 and 2 actually chose — so the joint proposal is less coherent and acceptance is bounded. **Hydra** fixes this by making the heads **sequentially dependent**: each head conditions on the previous heads' predicted tokens, producing a coherent draft sequence and substantially higher acceptance. This moves Hydra closer to (but still distinct from) **EAGLE's** feature-level autoregressive drafting ([§02](02_speculative_decoding_variants.md)); the spectrum runs Medusa (independent heads) → Hydra (dependent heads) → EAGLE (feature-level autoregressive head), with acceptance rate generally increasing along it.

The practical appeal of self-drafting (Medusa/Hydra) is operational simplicity: one model, shared KV cache and features, no draft-model maintenance or vocab matching, and a cheap training step. The cost is a modest amount of extra parameters/compute for the heads and tree verification, and acceptance that — for Medusa especially — trails the best feature-level methods. In 2025–2026, EAGLE-2/3 generally lead on raw acceptance, but Medusa/Hydra remain attractive when you want the simplest possible self-speculation on top of an existing deployment. This file covers the head designs, tree attention, training, and tradeoffs.

***

## Core Concepts & Mechanics

### Medusa heads
- Add `H` heads on the last hidden state `h_t`; head *i* predicts a distribution over token *t+i*. 📐 One forward pass yields proposals for the next `H` positions.
- Each head outputs top-k candidates → combine into a **candidate tree** of size up to `k^H` (pruned in practice).
- **Tree attention** verifies all tree paths in one target pass; accept longest valid prefix (exactness via verification).

### Training
- Freeze the backbone; train only the heads (cross-entropy to predict future tokens) on a modest dataset — cheap and fast. Optionally fine-tune backbone (Medusa-2) for higher α at more cost.

### Hydra: sequential dependence
- Medusa heads are **independent** → incoherent joint draft. Hydra makes head *i* condition on heads `1..i−1`'s predictions (an autoregressive head chain), yielding coherent drafts and higher acceptance. Still no separate model.

### Spectrum
| Method | Head structure | α | Complexity |
|---|---|---|---|
| Medusa | independent heads | medium | low |
| Hydra | sequentially dependent heads | higher | low-medium |
| EAGLE | feature-level autoregressive | high | medium |

### Tree pruning
The candidate tree (k^H) is pruned to a fixed budget of the most probable paths to bound verification cost — a key tuning knob (more paths → higher α, more verify compute).

***

## Key Challenges
1. **Independent-head incoherence (Medusa).** Heads ignore each other's choices, capping acceptance; Hydra/EAGLE address this with dependence.
2. **Tree size vs verify cost.** Larger candidate trees raise α but cost more verification; at high batch this erodes benefit ([§05](05_when_speculative_decoding_fails.md)).
3. **Head training/maintenance.** Heads must be trained per target and retrained when the backbone changes (drift).
4. **Lower α than EAGLE.** Self-heads (esp. Medusa) trail feature-level drafting on acceptance; a speed ceiling.

***

## Solutions & Current Best Practices
- **Hydra over Medusa** when you want self-drafting with higher α; **EAGLE-2/3** if you want the best acceptance ([§02](02_speculative_decoding_variants.md)).
- **Tune tree size/pruning** to the batch and latency target.
- **Retrain heads** on production-like data and when the backbone updates.
- **Disable/shrink at high batch** where verification isn't free ([§05](05_when_speculative_decoding_fails.md)).

***

## Implementation Notes
- vLLM/TRT-LLM support Medusa; framework provides tree-attention verification.
- Budget the candidate tree size; profile net tokens/s, not just α.
- Keep heads in sync with the deployed backbone version.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Medusa α plateaued** — independent heads produce incoherent joint drafts; switch to Hydra/EAGLE for higher acceptance.
- **Big tree, no net speedup** — verification of many paths cost more than it saved, especially at higher batch; prune the tree.
- **Heads stale after backbone fine-tune** — acceptance dropped; retrain heads.
- **Assumed quality change** — none; the target verifies every token (exact), heads only affect speed.

***

## Performance Numbers & Benchmarks
| Method | Reported speedup (low batch) |
|---|---|
| Medusa | ~2–2.8× |
| Hydra | ~2.5–3.6× |
| EAGLE-2 (for contrast) | ~3–4× |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How does Medusa do speculation without a draft model?"* — Expected: extra heads predict future tokens; tree-verify in one pass.
- *"Why does Hydra outperform Medusa?"* — Expected: sequentially dependent heads → coherent drafts → higher α.
- *"What's the cost of larger candidate trees?"* — Expected: more verified tokens; high-batch compute erosion.
- *"Medusa vs EAGLE — tradeoffs?"* — Expected: simplicity (self-heads) vs higher α (feature-level).

***

## Open Problems & Active Research (2025–2026)
- **Closing the gap to EAGLE** with cheaper self-drafting heads.
- **Adaptive tree shaping** per context/confidence (as EAGLE-2 does).
- **Training-free or instantly-adaptable heads** to avoid retraining on backbone updates.

***

## References
- Cai, T., et al. (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." arXiv:2401.10774.
- Ankner, Z., et al. (2024). "Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding." arXiv:2402.05109.
- Stern, M., et al. (2018). "Blockwise Parallel Decoding." *NeurIPS 2018*. arXiv:1811.03115.
