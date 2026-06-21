# Serving System Architecture (Engine Anatomy)

> **Section:** 08_serving_frameworks
> **Last Updated:** June 2026
> **Related Files:** [01_vllm_deep_dive.md](01_vllm_deep_dive.md), [06_framework_selection_matrix.md](06_framework_selection_matrix.md), [../03_batching_and_scheduling/01_iteration_level_scheduling.md](../03_batching_and_scheduling/01_iteration_level_scheduling.md)
> **Must-Read Papers:** Kwon et al. (2023, SOSP) "vLLM"; Yu et al. (2022, OSDI) "Orca"; Zheng et al. (2024) "SGLang"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- A serving engine has four layers: **API server** (HTTP/OpenAI-compatible) → **scheduler** (continuous batching, admission, KV management) → **worker(s)** (TP/PP model execution) → **model runner** (the actual forward pass + kernels).
- The **scheduler + KV block manager** are the heart; everything else feeds them ([§03](../03_batching_and_scheduling/01_iteration_level_scheduling.md)).
- Cross-cutting concerns: **paged KV**, **CUDA graphs**, **prefix caching**, **quantization**, **speculative decoding**, **metrics**.
- Frameworks differ mainly in scheduler efficiency, KV management (paged vs radix), and kernel quality — not the overall shape.
- Knowing this anatomy lets you reason about any framework and answer "where would you add feature X?"

***

## Overview
Despite differing names, all modern LLM serving engines (vLLM, SGLang, TensorRT-LLM, TGI, LMDeploy) share the same architecture, because they all solve the same problem: turn a stream of heterogeneous requests into efficient, continuously-batched GPU forward passes while managing KV memory. Understanding this common anatomy is more valuable than memorizing any one framework, and lets you place every technique in this database into its slot. The layers, from the request inward: an **API server** (usually OpenAI-compatible HTTP, handling auth/streaming), a **scheduler** that implements continuous batching, admission control, and KV block management, one or more **workers** that hold the (possibly TP/PP-sharded) model, and the **model runner** that executes the forward pass with optimized kernels (FlashAttention, fused/quantized GEMM, CUDA graphs).

The **scheduler and KV block manager** are where the intelligence lives. The scheduler runs the per-iteration loop ([§03](../03_batching_and_scheduling/01_iteration_level_scheduling.md)): evict finished sequences, compute KV/token budgets, admit waiting requests (with chunked prefill), build the ragged batch, and hand it to the workers. The block manager implements paged KV ([§02](../02_kv_cache/01_paged_attention_vllm.md)), tracking free blocks, ref-counts for sharing/prefix-caching, and preemption (swap/recompute). These two components determine throughput, latency, and memory efficiency; the API server and worker plumbing are comparatively mechanical.

The cross-cutting features — paged KV, prefix caching, quantization, CUDA graphs, speculative decoding, and observability — are integrated across these layers, and a framework's quality is largely how well it implements the scheduler (overhead, policy), the KV manager (paged vs radix-tree), and the kernels. This is why the framework deep-dives that follow focus on exactly those differentiators. This file gives the reference architecture and the request lifecycle so the rest of the section reads as variations on a theme.

***

## Core Concepts & Mechanics

### The layers
```
Client → API Server (OpenAI-compatible, streaming, auth/rate-limit)
       → Scheduler (continuous batching, admission, KV budget, chunked prefill, preemption)
       → KV Block Manager (paged blocks, ref-counts, prefix cache, swap/recompute)
       → Worker(s) (TP/PP execution, NCCL collectives)
       → Model Runner (forward pass: FlashAttention, fused/quantized GEMM, CUDA graphs, sampling)
```

### Request lifecycle
1. Request arrives at API server → tokenized → enqueued with params (sampling, SLO tier).
2. Scheduler admits it (KV available, token budget), prefills (chunked) the prompt.
3. Enters the running batch; each iteration produces one token/sequence; tokens streamed back.
4. On EOS/length: evict, free KV blocks, finalize response.

### Key components
- **Scheduler**: the per-iteration policy + mechanics ([§03](../03_batching_and_scheduling/01_iteration_level_scheduling.md)).
- **KV block manager**: paging, sharing, prefix cache, eviction/preemption ([§02](../02_kv_cache/01_paged_attention_vllm.md)).
- **Worker / distributed executor**: TP/PP, NCCL ([§04](../04_parallelism/00_tensor_parallelism.md)).
- **Model runner / kernels**: attention, GEMM, sampling, CUDA graphs ([§06](../06_kernel_optimization/00_cuda_kernel_basics_for_inference.md)).

### Where features plug in
- Quantization → model runner (kernels) + checkpoint loading.
- Prefix caching → KV block manager.
- Speculative decoding → scheduler + model runner (draft + verify).
- Disaggregation → split scheduler/workers into prefill/decode pools + KV transfer ([§03](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md)).

***

## Key Challenges
1. **Scheduler efficiency.** The per-iteration CPU work must not bottleneck the GPU at high QPS (overlap/async/C++) ([§03](../03_batching_and_scheduling/01_iteration_level_scheduling.md)).
2. **KV manager correctness.** Paging, ref-counting, and preemption are subtle; bugs cause leaks or cross-request corruption.
3. **Kernel integration.** Matching the right kernel (FlashAttention/FlashDecoding, quantized GEMM) to each path and keeping CUDA graphs valid.
4. **Feature composition.** Prefix caching + quantized KV + speculation + disaggregation interact; clean composition is hard.

***

## Solutions & Current Best Practices
- **Adopt a mature framework** (vLLM/SGLang/TRT-LLM) rather than building; they encode years of these lessons.
- **Understand the four layers** to debug and extend; most issues localize to scheduler or KV manager.
- **Profile end-to-end** (Nsight Systems) to find which layer bottlenecks ([§06](../06_kernel_optimization/05_kernel_profiling_and_benchmarking.md)).
- **Choose by differentiators** (scheduler overhead, KV management, kernels) — see [§06](06_framework_selection_matrix.md).

***

## Implementation Notes
- The OpenAI-compatible API layer is largely standardized; focus engineering on scheduler/KV/kernels.
- Expose metrics from each layer (queue depth, batch size, KV utilization, kernel time) for observability ([§13](../13_production_systems/01_observability_and_profiling.md)).
- For custom features, identify the layer: it almost always lands in the scheduler or KV manager.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Bottleneck is the API/tokenizer layer, not the GPU** — Python tokenization or HTTP handling can cap QPS; profile end-to-end.
- **KV leak from the block manager** — sequences not freed at eviction shrink capacity over time.
- **Feature interaction bug** — e.g., prefix caching + quantized KV breaking matches; compose features carefully.
- **Scheduler CPU-bound at high QPS** — GPU starves; needs overlap/async scheduler ([§03](../03_batching_and_scheduling/01_iteration_level_scheduling.md)).

***

## Performance Numbers & Benchmarks
| Layer | Typical bottleneck symptom |
|---|---|
| API/tokenizer | QPS capped, GPU idle |
| Scheduler | GPU idle gaps at high QPS |
| KV manager | preemption thrash, OOM, leaks |
| Kernels | low MFU / high kernel time |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Draw the architecture of an LLM serving engine."* — Expected: API → scheduler → KV manager → workers → model runner.
- *"Where do prefix caching / speculation / quantization plug in?"* — Expected: KV manager / scheduler+runner / runner+loader.
- *"What are the two most important components and why?"* — Expected: scheduler + KV block manager (throughput/latency/memory).
- *"How would you add disaggregation to this architecture?"* — Expected: split into prefill/decode pools + KV transfer layer.

***

## Open Problems & Active Research (2025–2026)
- **Unified engines** that fluidly do colocated/disaggregated serving with one scheduler.
- **Composable feature stacks** (prefix cache + quant KV + speculation + CP) without conflicts.
- **Lower-overhead schedulers** approaching zero CPU cost at very high QPS.

***

## References
- Kwon, W., et al. (2023). "vLLM/PagedAttention." *SOSP 2023*. arXiv:2309.06180.
- Yu, G.-I., et al. (2022). "Orca." *OSDI 2022*.
- Zheng, L., et al. (2024). "SGLang." arXiv:2312.07104.
