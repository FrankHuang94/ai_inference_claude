# TGI and LMDeploy

> **Section:** 08_serving_frameworks
> **Last Updated:** June 2026
> **Related Files:** [01_vllm_deep_dive.md](01_vllm_deep_dive.md), [02_sglang_deep_dive.md](02_sglang_deep_dive.md), [06_framework_selection_matrix.md](06_framework_selection_matrix.md)
> **Must-Read Papers:** HuggingFace TGI docs; InternLM LMDeploy/TurboMind docs; Kwon et al. (2023) "vLLM"
> **Estimated Study Time:** 20 minutes

***

## TL;DR
- **TGI (Text Generation Inference)**: HuggingFace's production serving framework — continuous batching, paged/FlashAttention, quantization, tight HF ecosystem integration; the easy default for HF-centric stacks.
- **LMDeploy (TurboMind)**: InternLM's toolkit with a high-performance **C++ TurboMind** engine — strong throughput/latency, persistent KV, good quantization (W4A16, KV INT8), popular in the Chinese LLM ecosystem.
- Both implement the modern feature set (continuous batching, paged KV, quantization); they compete with vLLM/SGLang on performance and ecosystem fit.
- TGI's edge is **HuggingFace integration and ops maturity**; LMDeploy's is **TurboMind's C++ efficiency**.
- Framework choice is increasingly about **ecosystem fit and specific features**, since core techniques have converged ([§06](06_framework_selection_matrix.md)).

***

## Overview
Beyond vLLM, SGLang, and TensorRT-LLM, two other serving frameworks are worth knowing. **TGI (Text Generation Inference)** is HuggingFace's production serving solution, designed to deploy HF models with minimal friction. It implements continuous batching, paged/FlashAttention, tensor parallelism, a range of quantization methods (bitsandbytes, GPTQ, AWQ, FP8, EETQ), and an OpenAI-compatible API, with deep integration into the HuggingFace Hub and ecosystem. Its appeal is **ops maturity and ecosystem fit**: if your stack is HuggingFace-centric, TGI is the path of least resistance, with solid production features (telemetry, safetensors loading, guided generation). Performance is competitive, though vLLM/SGLang often lead on raw throughput for specific workloads.

**LMDeploy** (from the InternLM team) centers on **TurboMind**, a high-performance C++/CUDA inference engine, alongside a PyTorch engine. TurboMind delivers strong throughput and latency via efficient C++ runtime (low overhead, like TRT-LLM's philosophy but more flexible), persistent KV cache management, blocked KV, and good quantization support (W4A16 weight-only, INT8 KV cache). It's widely used in the Chinese LLM ecosystem and for InternLM/Qwen-family models, and is a strong general performer. Its differentiator is the C++ engine's efficiency combined with easier deployment than TRT-LLM's compile-everything model.

The broader point these two illustrate: the **core serving techniques have converged** — every serious framework now does continuous batching, paged KV, FlashAttention, and quantization. So framework selection in 2025–2026 is less about "who has continuous batching" and more about **ecosystem fit** (HF → TGI, NVIDIA-fixed → TRT-LLM, shared-prefix/structured → SGLang, general/broad → vLLM, InternLM/Chinese ecosystem → LMDeploy), **specific features** (RadixAttention, multi-LoRA, disaggregation), and **operational preferences** (compile vs flexible, C++ vs Python). This file briefly covers TGI and LMDeploy and folds into the selection matrix ([§06](06_framework_selection_matrix.md)).

***

## Core Concepts & Mechanics

### TGI
- **Continuous batching**, paged/FlashAttention, tensor parallelism, OpenAI-compatible API.
- **Quantization**: bitsandbytes, GPTQ, AWQ, EETQ, FP8.
- **HF integration**: Hub model loading, safetensors, guided/structured generation, telemetry.
- **Ops**: production-oriented (Rust router + Python/Server shards historically), good observability.

### LMDeploy / TurboMind
- **TurboMind**: C++/CUDA engine — low overhead, persistent + blocked KV, efficient kernels.
- **Quantization**: W4A16 (AWQ-style), **INT8 KV cache**, good compression support.
- **PyTorch engine** option for flexibility/broader model support.
- Strong on InternLM/Qwen and the Chinese LLM ecosystem; competitive throughput/latency.

### Positioning vs vLLM/SGLang/TRT-LLM
- TGI: HF-ecosystem default; competitive but not usually the throughput leader.
- LMDeploy: C++ efficiency, strong throughput; ecosystem-specific popularity.
- All share converged core techniques; choose by fit/features.

***

## Key Challenges
1. **Performance leapfrogging.** vLLM/SGLang/TRT-LLM frequently lead specific benchmarks; TGI/LMDeploy must be evaluated per workload.
2. **Feature coverage variance.** Specific features (RadixAttention, advanced disaggregation, certain speculation variants) may lag the leaders.
3. **Ecosystem lock-in considerations.** TGI ties to HF; LMDeploy strongest in its ecosystem — fit matters.
4. **Benchmark honesty.** As always, benchmark your model/workload; headline numbers vary by version.

***

## Solutions & Current Best Practices
- **TGI** for HuggingFace-centric stacks wanting ops maturity and easy deployment.
- **LMDeploy/TurboMind** for C++-efficiency, INT8 KV, and InternLM/Qwen ecosystems.
- **Benchmark against vLLM/SGLang** on your workload before committing ([§06](06_framework_selection_matrix.md)).
- **Choose by ecosystem fit + needed features**, since core techniques have converged.

***

## Implementation Notes
- TGI: deploy from HF Hub; configure quantization and TP; use guided generation if needed.
- LMDeploy: use TurboMind for performance, PyTorch engine for flexibility; enable W4A16 + INT8 KV for memory/throughput.
- Validate feature support (speculation, disaggregation, prefix caching) for your needs.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Assumed TGI = vLLM performance** — varies by workload/version; benchmark, don't assume.
- **Missing a needed feature** — e.g., RadixAttention-level prefix reuse; check coverage vs SGLang.
- **Ecosystem mismatch** — using a framework outside its strong ecosystem can mean rough edges; pick for fit.
- **Version-sensitive numbers** — fast-moving; pin and re-benchmark on upgrade.

***

## Performance Numbers & Benchmarks
| Framework | Strength | Ecosystem |
|---|---|---|
| TGI | HF integration, ops maturity | HuggingFace |
| LMDeploy/TurboMind | C++ efficiency, INT8 KV | InternLM/Qwen/CN |
| vLLM | breadth, multi-LoRA | general/OSS |
| SGLang | RadixAttention, scheduler | shared-prefix/structured |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"What's TGI and when would you use it?"* — Expected: HF serving framework; HF-centric stacks, ops maturity.
- *"What is TurboMind/LMDeploy's differentiator?"* — Expected: C++ engine efficiency, INT8 KV, ecosystem.
- *"Have serving frameworks converged? How do you choose now?"* — Expected: yes on core techniques; choose by ecosystem/features/ops.
- *"How do you fairly compare frameworks?"* — Expected: benchmark your model/workload; pin versions.

***

## Open Problems & Active Research (2025–2026)
- **Feature parity** across frameworks (disaggregation, advanced caching/speculation).
- **Standardized, reproducible serving benchmarks** to cut through version-to-version churn.
- **Ecosystem consolidation** vs continued specialization.

***

## References
- HuggingFace. "Text Generation Inference (TGI)" documentation.
- InternLM. "LMDeploy / TurboMind" documentation.
- Kwon, W., et al. (2023). "vLLM." *SOSP 2023*. arXiv:2309.06180.
