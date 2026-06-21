# Together AI

> **Section:** 14_company_deep_dives
> **Last Updated:** June 2026
> **Related Files:** [00_fireworks_ai.md](00_fireworks_ai.md), [07_company_comparison_matrix.md](07_company_comparison_matrix.md), [../06_kernel_optimization/01_flash_attention_deep_dive.md](../06_kernel_optimization/01_flash_attention_deep_dive.md)
> **Must-Read Papers:** Dao et al. (2022/2023) "FlashAttention 1/2" (Tri Dao is Together's Chief Scientist); Together engineering blog; Zheng et al. (2024) "SGLang"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- Together AI is an inference-and-training cloud combining a **fast inference engine** with **strong open research** — Tri Dao (FlashAttention author) is Chief Scientist.
- Differentiator: deep **kernel/algorithm research → production** pipeline (FlashAttention lineage, custom kernels, speculative decoding, quantization) plus broad open-model hosting and **GPU clusters** for training.
- Products: OpenAI-compatible inference API, fine-tuning, dedicated endpoints, GPU cluster rental.
- Competes with Fireworks/Replicate on inference; differentiates on **research depth** and **end-to-end (train + serve)**.
- **Interview focus**: kernel optimization, attention algorithms, quantization, serving systems — research-leaning systems roles.

***

## Overview
Together AI is a full-stack AI cloud spanning **training and inference**, distinguished by an unusually strong research bench feeding production. Most notably, **Tri Dao** — author of **FlashAttention** (the most important attention kernel, [§06](../06_kernel_optimization/01_flash_attention_deep_dive.md)) — is Together's Chief Scientist, and the company has a track record of translating algorithmic/kernel research (FlashAttention-2/3, FlashDecoding, Medusa-style speculation, efficient quantization, Mamba/SSM work) into its serving stack. This research-to-production pipeline is its core differentiator versus pure serving shops: Together tends to be early on cutting-edge inference techniques.

On the product side, Together offers an **OpenAI-compatible inference API** over a broad catalog of open models, **fine-tuning**, **dedicated endpoints**, and — distinctively — **GPU cluster rental for training**, making it an end-to-end "build and serve open models" platform. Its **Together Inference Engine** incorporates the custom kernels and serving optimizations (FlashAttention-class attention, continuous batching, paged KV, quantization, speculative decoding) that this database covers, with the research team continually upstreaming improvements. This positions Together against Fireworks and Replicate on inference cost/speed, while differentiating on research depth and the train+serve combination.

For candidates, Together's interview emphasis leans toward the **research-systems** intersection: **attention algorithms and kernels** (given the FlashAttention heritage), **quantization**, **speculative decoding**, and **serving systems**, with more openness to algorithmic novelty than a pure ops shop. Roles often value both the systems depth of §06 and the algorithmic understanding of §07 (speculative decoding) and §10 (long context / SSMs). This file covers the company, its research-to-production model, products, and interview emphasis.

***

## Core Concepts & Mechanics

### Business model
End-to-end AI cloud: **inference API** (hosted open models) + **fine-tuning** + **dedicated endpoints** + **GPU clusters for training**. Win on inference efficiency + research-driven features + train/serve breadth.

### Research-to-production differentiator
- **FlashAttention lineage** (Tri Dao, Chief Scientist): FA-1/2/3, FlashDecoding → in the serving engine ([§06](../06_kernel_optimization/01_flash_attention_deep_dive.md)).
- **Speculative decoding, quantization, SSM/Mamba** research → production.
- **Together Inference Engine**: custom kernels + continuous batching + paged KV + quantization + speculation.

### Products
- OpenAI-compatible **inference API** (broad open-model catalog).
- **Fine-tuning** and **dedicated endpoints**.
- **GPU cluster rental** (training infrastructure) — the train+serve combination.

### Interview emphasis
Research-systems: attention kernels/algorithms, quantization, speculative decoding, long-context/SSMs, serving systems. Values algorithmic depth + systems.

***

## Key Challenges (for the company)
1. **Research lead → product lead.** Converting research advantage into durable serving cost/speed leadership as OSS catches up.
2. **Breadth (train + serve).** Running both training clusters and inference at competitive economics.
3. **Unit economics.** Same utilization/efficiency pressures as all inference clouds ([§01](../01_hardware/06_tco_and_cost_modeling.md)).
4. **Model catalog freshness.** Rapidly supporting new open models (DeepSeek, Llama, Qwen, etc.).

***

## Solutions & Current Best Practices (what they exemplify)
- **Research-driven kernels/algorithms** (FlashAttention family) as differentiation ([§06](../06_kernel_optimization/01_flash_attention_deep_dive.md)).
- **Full stack** (train + serve) for open-model builders.
- **Standard efficiency moat**: kernels + quantization + serving + utilization.

***

## Implementation Notes (for interview prep)
- Know FlashAttention 1/2/3 deeply (the heritage) — IO complexity, online softmax, Hopper specifics ([§06](../06_kernel_optimization/01_flash_attention_deep_dive.md)).
- Be conversant in speculative decoding, quantization, and SSM/long-context ([§07](../07_speculative_decoding/00_speculative_decoding_fundamentals.md), [§10](../10_long_context/03_sparse_and_linear_attention.md)).
- Be ready to discuss translating a paper/algorithm into an efficient kernel.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Research ≠ automatic product win** — algorithmic lead must be productionized and maintained vs OSS ([§08](../08_serving_frameworks/06_framework_selection_matrix.md)).
- **Benchmark claims are workload-specific** — verify on your workload.
- **Train+serve breadth** stretches focus vs pure-inference specialists.

***

## Performance Numbers & Benchmarks
| Aspect | Together |
|---|---|
| Differentiator | research-to-production (FlashAttention lineage) |
| Scope | inference + fine-tune + training clusters |
| Engine | Together Inference Engine (custom kernels) |
| Heritage | Tri Dao / FlashAttention |

***

## Interview Angles
> 💡 **What Together (and similar) actually ask:**
- *"Explain FlashAttention and its versions."* — Expected: [§06](../06_kernel_optimization/01_flash_attention_deep_dive.md) in depth.
- *"How would you productionize a new attention/speculation algorithm?"* — Expected: kernel implementation (Triton/CUDA), integration into continuous batching, profiling.
- *"Compare SSM/Mamba vs transformer for serving."* — Expected: [§10](../10_long_context/03_sparse_and_linear_attention.md).
- *"Design a fast open-model inference engine."* — Expected: kernels + batching + paged KV + quantization + speculation ([§15](../15_interview_prep/00_inference_system_design_questions.md)).

***

## Open Problems & Active Research (2025–2026)
- **Next-gen attention/SSM kernels** (FA-4, Blackwell, Mamba serving).
- **Sustaining research-to-production speed** advantage.
- **Reasoning-model and long-context** serving economics ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md), [§10](../10_long_context/05_kv_cache_management_at_1M_tokens.md)).

***

## References
- Dao, T., et al. (2022/2023/2024). "FlashAttention 1/2/3." arXiv:2205.14135, 2307.08691, 2407.08608.
- Together AI engineering blog (Inference Engine, kernels).
- Gu, A., Dao, T. (2023). "Mamba." arXiv:2312.00752.
