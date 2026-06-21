# Quantization-Aware Training (QAT)

> **Section:** 05_quantization
> **Last Updated:** June 2026
> **Related Files:** [01_post_training_quantization.md](01_post_training_quantization.md), [02_weight_only_quantization.md](02_weight_only_quantization.md), [00_quantization_fundamentals.md](00_quantization_fundamentals.md)
> **Must-Read Papers:** Jacob et al. (2018, CVPR) "Quantization and Training (STE)"; Bengio et al. (2013) "Estimating Gradients (STE)"; Liu et al. (2023) "LLM-QAT"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- QAT trains/fine-tunes the model with **simulated quantization in the forward pass**, so weights adapt to quantization error — recovering accuracy PTQ can't.
- The **straight-through estimator (STE)** passes gradients through the non-differentiable rounding op, enabling backprop.
- QAT is needed when **PTQ degrades too much** — typically at very low bits (W4A4, INT3/INT2) or for sensitive models/tasks.
- Cost: requires training compute, data, and pipeline — far more than PTQ; hence used selectively.
- For LLMs, **QAT via fine-tuning / distillation** (LLM-QAT, QLoRA-style) is the practical form; full QAT from scratch is rare.

***

## Overview
Post-training quantization is cheap and usually sufficient down to INT4 weights or FP8/INT8 weights+activations, but at aggressive settings (4-bit activations, sub-4-bit weights) or on sensitive models, PTQ's accuracy loss becomes unacceptable because the model's weights were never optimized to tolerate quantization error. **Quantization-aware training** fixes this by **simulating quantization during training**: in the forward pass, weights (and optionally activations) are quantized and dequantized ("fake quant") so the loss reflects quantization error, and the optimizer learns weights that are robust to it. The model effectively co-adapts to its quantized deployment form, recovering much of the lost accuracy.

The technical enabler is the **straight-through estimator (STE)**. Quantization's rounding operation has zero gradient almost everywhere (it's a step function), so naive backprop can't train through it. STE (Bengio et al. 2013; Jacob et al. 2018) approximates the rounding's gradient as the identity (pass the gradient straight through), so the network can learn despite the non-differentiable op. With STE, the forward pass uses quantized values while the backward pass updates the full-precision "shadow" weights, which are re-quantized each step. This is the standard machinery behind QAT in vision and, adapted, in LLMs.

For LLMs, full from-scratch QAT is expensive and rare; the practical forms are **QAT fine-tuning** (take a pretrained model, fine-tune with fake quant for a relatively short schedule — LLM-QAT, Liu et al. 2023, even uses data-free/​self-generated data) and **quantized-LoRA-style** approaches that combine low-rank adaptation with quantized bases. The decision is economic: QAT costs training compute and data and a pipeline, so you use it only when PTQ leaves too much on the table — typically to unlock W4A4 or sub-4-bit deployment where the inference savings justify the training cost. This file covers STE, QAT mechanics, and when it's worth it.

***

## Core Concepts & Mechanics

### Fake quantization in the forward pass
Each forward pass applies `x̂ = dequant(quant(x))` ("fake quant") to weights/activations, so the network computes with quantized values and the loss includes quantization error. Scales may be fixed (from calibration) or **learned** (learnable step size, LSQ).

### Straight-through estimator
📐 For `q = round(x)`, `dq/dx = 0` a.e. STE approximates `dq/dx ≈ 1` within the representable range (and 0 outside, to handle clipping). So gradients flow to the full-precision shadow weights, which are updated and re-quantized next step. Variants clip the STE gradient at the quantization bounds.

### QAT vs PTQ
| | PTQ | QAT |
|---|---|---|
| Cost | minutes–hours, small calib set | training compute + data |
| Accuracy at low bits | degrades | recovers more |
| When | INT4 weights, FP8 | W4A4, sub-4-bit, sensitive |
| LLM practicality | default | selective (fine-tune/distill) |

### LLM-specific QAT
- **LLM-QAT** (Liu et al. 2023): data-free QAT using the model's own generations as training data, quantizing weights, activations, and KV cache; recovers accuracy at low bits.
- **Distillation during QAT**: the full-precision model teaches the quantized student, improving recovery.
- **QLoRA** (related): quantize the frozen base to 4-bit and train LoRA adapters in higher precision — primarily a *fine-tuning memory* trick but conceptually adjacent.

***

## Key Challenges
1. **Training cost.** QAT needs compute, data, and a pipeline far beyond PTQ; for large LLMs this is significant.
2. **STE approximation error.** The identity-gradient hack is biased; training can be unstable or under-recover, needing careful clipping/learnable scales.
3. **Activation/KV QAT complexity.** Simulating activation and KV quantization during training (LLM-QAT) is more involved and costly than weight-only QAT.
4. **Data requirements.** Representative (or self-generated) data is needed; mismatched data limits recovery.

***

## Solutions & Current Best Practices
- **Default to PTQ** (AWQ/GPTQ/SmoothQuant/FP8); reach for QAT only when PTQ degrades too much ([§01](01_post_training_quantization.md)).
- **QAT fine-tuning / LLM-QAT** for W4A4 or sub-4-bit deployment, with **distillation** from the FP model.
- **Learnable scales (LSQ)** and **STE clipping** for stability.
- **Self-generated/data-free data** (LLM-QAT) when training data is unavailable.

***

## Implementation Notes
- Insert fake-quant modules; keep full-precision shadow weights; use STE with bound-aware gradient clipping.
- Short QAT fine-tune schedules often recover most accuracy at a fraction of pretraining cost.
- Validate the QAT model in the **exact deployment precision/kernels** to avoid train/serve mismatch.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **QAT model degraded at serving** — train-time fake quant didn't match the deployment kernel's quantization scheme; align them exactly.
- **STE training unstable** — unclipped identity gradients past the quant range; clip STE at bounds, use learnable scales.
- **Spent QAT compute for INT4 weights** — PTQ (AWQ) would have sufficed; QAT is for the *hard* (W4A4/sub-4-bit) cases.
- **Poor recovery from bad data** — unrepresentative QAT data limits accuracy; use distillation/self-generated data.

***

## Performance Numbers & Benchmarks
| Setting | PTQ accuracy | QAT accuracy |
|---|---|---|
| INT4 weights | ~near-lossless (AWQ) | marginal gain (not worth it) |
| W4A4 | significant loss | much improved (QAT worthwhile) |
| INT3/INT2 weights | large loss | recovers substantially |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"What is the straight-through estimator and why is it needed for QAT?"* — Expected: rounding has no gradient; STE passes identity to train shadow weights.
- *"When do you use QAT instead of PTQ?"* — Expected: PTQ degrades too much — W4A4, sub-4-bit, sensitive models.
- *"How is QAT done for LLMs practically?"* — Expected: fine-tune with fake quant + distillation (LLM-QAT), often self-generated data.
- *"What's the risk of train/serve mismatch in QAT?"* — Expected: deployment quant scheme must match training fake-quant.

***

## Open Problems & Active Research (2025–2026)
- **Cheap QAT for frontier-scale LLMs** (efficient fine-tune schedules, data-free methods).
- **Stable low-bit (W4A4/W2) QAT** via better estimators (beyond STE) and rotations.
- **Unified QAT for weights+activations+KV** at minimal cost.

***

## References
- Jacob, B., et al. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR 2018*. arXiv:1712.05877.
- Bengio, Y., Léonard, N., Courville, A. (2013). "Estimating or Propagating Gradients Through Stochastic Neurons (STE)." arXiv:1308.3432.
- Liu, Z., et al. (2023). "LLM-QAT: Data-Free Quantization Aware Training for LLMs." arXiv:2305.17888.
- Esser, S., et al. (2020). "Learned Step Size Quantization (LSQ)." *ICLR 2020*. arXiv:1902.08153.
