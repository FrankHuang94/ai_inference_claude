# Framework Selection Matrix

> **Section:** 08_serving_frameworks
> **Last Updated:** June 2026
> **Related Files:** [01_vllm_deep_dive.md](01_vllm_deep_dive.md), [02_sglang_deep_dive.md](02_sglang_deep_dive.md), [03_tensorrt_llm_deep_dive.md](03_tensorrt_llm_deep_dive.md)
> **Must-Read Papers:** Kwon et al. (2023) "vLLM"; Zheng et al. (2024) "SGLang"; Agrawal et al. (2024) "Sarathi-Serve"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- Core techniques (continuous batching, paged KV, FlashAttention, quantization) have **converged**; choose on **differentiators + ecosystem + workload**.
- **vLLM**: broad default; diverse lengths, multi-LoRA, big community.
- **SGLang**: shared-prefix / structured / high-QPS / multi-turn / DeepSeek-MoE → RadixAttention + overlap scheduler.
- **TensorRT-LLM**: fixed-config, NVIDIA, latency-critical → compiled engines (lowest overhead).
- **TGI**: HuggingFace-centric ops; **LMDeploy**: C++ TurboMind efficiency / CN ecosystem.
- Decide by: request-rate, length distribution, prefix sharing, latency-vs-throughput, hardware, operational tolerance.

***

## Overview
By 2025–2026 the major serving frameworks all implement the same core techniques, so selection is a multi-dimensional fit problem rather than a hunt for the one fastest engine — and "fastest" flips between vLLM, SGLang, and TensorRT-LLM release-to-release anyway. The right approach is to characterize *your workload* (request rate, prompt/output length distribution, prefix-sharing pattern, latency vs throughput priority, hardware, and operational constraints) and match it to each framework's **differentiator**. This file provides a comparison table and a decision tree for that purpose, synthesizing the deep-dives.

The differentiators that actually drive the choice: **vLLM** offers the broadest model support, multi-LoRA, and the largest community — the safe default for diverse workloads and fast iteration. **SGLang** wins where **prefix sharing and structured generation** dominate (RadixAttention) and at **high QPS** (overlap scheduler avoids the CPU bottleneck), plus strong DeepSeek/MoE/MLA support. **TensorRT-LLM** delivers the **lowest overhead and best latency** for **fixed configurations on NVIDIA hardware**, at the cost of compile-time inflexibility. **TGI** is the **HuggingFace-ecosystem** default with good ops; **LMDeploy/TurboMind** brings **C++ efficiency** and is strong in the InternLM/Qwen ecosystem.

The meta-advice for interviews and practice: never answer "which framework is best" unconditionally. Instead, state the workload dimensions, map them to differentiators, and **benchmark the top one or two candidates on your actual model and traffic** — because the only reliable comparison is $/token-at-SLO on your workload, and headline numbers are version- and workload-specific. This file gives the structured decision aid; the watchlist notes that the ranking is a moving target.

***

## Core Concepts & Mechanics

### Comparison matrix
| Dimension | vLLM | SGLang | TensorRT-LLM | TGI | LMDeploy |
|---|---|---|---|---|---|
| Ease of use / iteration | high | high | low (compile) | high | medium |
| Throughput (general) | high | very high | high (fixed) | high | high |
| High-QPS scheduler | good (V1) | best (overlap) | good (C++) | good | good (C++) |
| TTFT / fixed-shape latency | good | good | best | good | good |
| Prefix caching | exact hash | **RadixAttention** | basic | basic | yes |
| Quantization | AWQ/GPTQ/FP8/INT8KV | FP8/AWQ/INT8KV | FP8/INT4/INT8 | many | W4A16/INT8KV |
| Speculative decoding | draft/EAGLE/Medusa | EAGLE | Medusa/draft | some | some |
| Multi-LoRA | **strong (S-LoRA)** | yes | plugin | yes | some |
| Disaggregation | maturing | yes | maturing | — | — |
| MoE/MLA (DeepSeek) | yes | **strong** | yes | partial | yes |
| Structured output | yes | **DSL/strong** | basic | guided | basic |
| Hardware | NVIDIA(+AMD) | NVIDIA(+AMD) | NVIDIA only | NVIDIA(+) | NVIDIA |
| Community/adoption | **largest** | growing fast | NVIDIA-backed | HF | CN ecosystem |

### Decision tree
```
- Fixed config, NVIDIA, latency-critical, compile OK? → TensorRT-LLM (+ Triton Server)
- Heavy shared prefixes / structured / high QPS / multi-turn / DeepSeek? → SGLang
- HuggingFace-centric ops, easy deploy? → TGI
- InternLM/Qwen / want C++ TurboMind efficiency? → LMDeploy
- Diverse workloads, multi-LoRA, broad models, fast iteration? → vLLM (default)
- Unsure → benchmark vLLM vs SGLang on your workload; pick by $/token @ SLO.
```

***

## Key Challenges
1. **Moving target.** Rankings change per release; a choice optimal today may not be in six months (watchlist).
2. **Workload-specificity.** No framework wins all workloads; the answer depends on length distribution, prefix sharing, QPS, hardware.
3. **Feature gaps.** A needed feature (RadixAttention, disaggregation, multi-LoRA) may only be strong in one framework.
4. **Migration cost.** Switching frameworks later has cost; pick with some headroom for future needs.

***

## Solutions & Current Best Practices
- **Match workload dimensions to differentiators** (table above), then **benchmark** the top candidates on your model/traffic.
- **Default to vLLM** unless a specific differentiator (SGLang RadixAttention, TRT-LLM fixed-shape latency) clearly fits.
- **Decide on $/token at SLO**, not headline throughput.
- **Re-evaluate periodically** as frameworks evolve.

***

## Implementation Notes
- Pin versions; re-benchmark on upgrades.
- Test with realistic prompt/output distributions and prefix-sharing patterns, not synthetic uniform loads.
- Validate the specific features you need are mature in the chosen framework.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Picked the "fastest" from a blog, lost on our workload** — benchmarks are workload/version-specific; test your own.
- **Chose a framework lacking a needed feature** — e.g., needed RadixAttention/disaggregation; verify feature maturity first.
- **TRT-LLM's compile cost surprised the team** — great latency but slow iteration; ensure the workflow tolerates it.
- **Assumed ranking permanence** — leapfrogging means re-evaluate; don't lock in on a stale comparison.

***

## Performance Numbers & Benchmarks
| Workload | Best fit |
|---|---|
| Diverse chat, multi-LoRA | vLLM |
| Shared system prompts, agentic/structured | SGLang |
| Fixed config, ultra-low latency, NVIDIA | TensorRT-LLM |
| HF-centric production | TGI |
| InternLM/Qwen, C++ efficiency | LMDeploy |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Which serving framework would you choose for [workload]?"* — Expected: map dimensions → differentiator; mention benchmarking $/token@SLO.
- *"vLLM vs SGLang vs TRT-LLM — key differentiators?"* — Expected: breadth/multi-LoRA vs RadixAttention/scheduler vs compiled latency.
- *"Have frameworks converged?"* — Expected: yes on core; differ on caching/scheduler/compile/ecosystem/features.
- *"How do you actually pick?"* — Expected: workload characterization + benchmark on real traffic.

***

## Open Problems & Active Research (2025–2026)
- **Standardized serving benchmarks** to cut through version churn.
- **Convergence vs specialization** — will one framework dominate or will niches persist?
- **Disaggregation/feature parity** across frameworks.

***

## References
- Kwon, W., et al. (2023). "vLLM." *SOSP 2023*. arXiv:2309.06180.
- Zheng, L., et al. (2024). "SGLang." arXiv:2312.07104.
- NVIDIA. "TensorRT-LLM"; HuggingFace "TGI"; InternLM "LMDeploy" docs.
