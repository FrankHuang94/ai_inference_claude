# vLLM Deep Dive

> **Section:** 08_serving_frameworks
> **Last Updated:** June 2026
> **Related Files:** [00_serving_system_architecture.md](00_serving_system_architecture.md), [02_sglang_deep_dive.md](02_sglang_deep_dive.md), [../02_kv_cache/01_paged_attention_vllm.md](../02_kv_cache/01_paged_attention_vllm.md)
> **Must-Read Papers:** Kwon et al. (2023, SOSP) "vLLM/PagedAttention"; Sheng et al. (2023) "S-LoRA"; Agrawal et al. (2024) "Sarathi-Serve" (chunked prefill)
> **Estimated Study Time:** 35 minutes

***

## TL;DR
- vLLM is the **most widely-adopted open-source serving engine**, built around **PagedAttention** + continuous batching, with an OpenAI-compatible server.
- Core components: **AsyncLLMEngine**, **Scheduler** (FCFS + chunked prefill), **BlockManager** (paged KV, prefix caching, swap/recompute), **Worker** (TP/PP), **ModelRunner** (kernels + CUDA graphs).
- Strengths: broad model support, prefix caching, **multi-LoRA** (S-LoRA/LoRAX), quantization (AWQ/GPTQ/FP8), large community, fast release cadence.
- Known weakness historically: **Python scheduler overhead** at very high QPS (vLLM **V1** re-architecture and improvements target this).
- The default choice for diverse workloads; SGLang/TRT-LLM may edge it on specific axes ([§06](06_framework_selection_matrix.md)).

***

## Overview
vLLM is the reference open-source LLM serving engine and the most common production and research choice. It originated the **PagedAttention** KV management scheme ([§02](../02_kv_cache/01_paged_attention_vllm.md)) and pairs it with Orca-style continuous batching to deliver 2–4× throughput over naive serving. Around that core it has accreted the full feature set expected of a modern engine — an OpenAI-compatible API server, prefix caching, chunked prefill, a wide range of quantization formats, multi-LoRA serving, speculative decoding, tensor/pipeline parallelism, and CUDA-graph-accelerated decode — backed by a large, fast-moving community that adds model support quickly. For most teams, "use vLLM" is the sensible default, and knowing its internals is expected in interviews.

Architecturally, vLLM instantiates the generic engine anatomy ([§00](00_serving_system_architecture.md)). The **AsyncLLMEngine** wraps the request lifecycle and streaming. The **Scheduler** runs continuous batching with FCFS ordering and optional chunked prefill, deciding admissions against KV and token budgets. The **BlockManager** implements paged KV: a pool of fixed-size blocks, per-sequence block tables, ref-counting for prefix sharing and copy-on-write (parallel sampling), and preemption via swap-to-CPU or recompute. **Workers** hold the TP/PP-sharded model and run NCCL collectives; the **ModelRunner** executes the forward pass with FlashAttention/paged-attention kernels, fused/quantized GEMM, and CUDA graphs for decode. This clean separation is why vLLM is also a popular base for research extensions.

vLLM's historical weak spot is **scheduler/CPU overhead**: its Python-centric scheduling could become a bottleneck at very high QPS or with many small requests, where the per-iteration CPU work competes with feeding the GPU — the gap SGLang exploited with its zero-overhead/overlap scheduler ([§02](02_sglang_deep_dive.md)). The **vLLM V1** re-architecture (a cleaner, faster core with reduced overhead and better async) directly targets this, narrowing or closing the gap. vLLM also tends to carry more overhead than TensorRT-LLM for *fixed-shape* workloads where TRT-LLM's compiled engines win. This file covers vLLM's components, features, tuning knobs, and where it shines vs struggles.

***

## Core Concepts & Mechanics

### Components
- **AsyncLLMEngine / LLMEngine**: request intake, streaming, lifecycle.
- **Scheduler**: continuous batching, FCFS, chunked prefill (`enable_chunked_prefill`), admission vs KV/token budget, preemption policy.
- **BlockManager**: paged KV pool, block tables, ref-counts (prefix cache `enable_prefix_caching`, CoW for parallel sampling), swap (`swap_space`) / recompute.
- **Worker / distributed executor**: TP (`tensor_parallel_size`), PP (`pipeline_parallel_size`), NCCL.
- **ModelRunner**: attention (paged/FlashAttention), quantized GEMM (AWQ/GPTQ/FP8/Marlin), sampling, **CUDA graphs** for decode.

### Key features
- **Prefix caching** (hash-based exact) ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)).
- **Multi-LoRA** serving (S-LoRA/LoRAX-style): one base + many adapters multiplexed ([§13](../13_production_systems/04_multi_tenant_serving.md)).
- **Quantization**: AWQ, GPTQ, FP8, INT8 KV; Marlin kernels for W4A16.
- **Speculative decoding**: draft model, EAGLE, Medusa, n-gram.
- **Chunked prefill** (Sarathi-Serve) to protect ITL.

### Key knobs
- `gpu_memory_utilization` (~0.9): fraction of HBM for KV blocks.
- `max_num_seqs`, `max_num_batched_tokens`: batch & per-iteration token budget (governs interference/ITL).
- `block_size` (16): paged block granularity.
- `enable_chunked_prefill`, `enable_prefix_caching`, quantization, TP/PP sizes.

### vLLM V1
A re-architected core reducing Python overhead, improving the scheduler and async paths, and cleaning up the KV/attention abstractions — addressing the high-QPS overhead criticism.

***

## Key Challenges
1. **Scheduler/CPU overhead at high QPS.** Python scheduling can starve the GPU with many small requests (mitigated by V1).
2. **Overhead vs compiled engines.** For fixed-shape, latency-critical workloads, TensorRT-LLM's compiled C++ runtime can be faster ([§03](03_tensorrt_llm_deep_dive.md)).
3. **Feature-interaction tuning.** Chunked prefill + prefix caching + quantized KV + speculation require careful configuration.
4. **Fast-moving codebase.** Rapid releases mean APIs/behaviors shift; pinning versions matters for production.

***

## Solutions & Current Best Practices
- **Default to vLLM** for diverse request lengths, prefix-sharing, and multi-LoRA workloads.
- **Enable chunked prefill + prefix caching + FP8** for balanced latency/throughput.
- **Use vLLM V1** for high-QPS deployments to reduce scheduler overhead.
- **Tune `max_num_batched_tokens`** to your ITL SLO; capture CUDA graphs for your batch sizes ([§06](../06_kernel_optimization/02_fused_kernels_and_operator_fusion.md)).

***

## Implementation Notes
- Set `gpu_memory_utilization` leaving headroom for fragmentation/activations.
- For multi-LoRA, serve a base model + adapters; watch adapter-switch overhead.
- Pin the vLLM version; validate behavior after upgrades (fast-moving project).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **High QPS, GPU underutilized** — Python scheduler overhead; use vLLM V1 / increase batch / reduce per-request overhead.
- **Uncaptured batch size → eager decode** — slow path; ensure CUDA graphs cover your batch sizes.
- **Prefix caching defeated by volatile prompt prefix** — move timestamps/UUIDs to the end ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)).
- **OOM under load** — `gpu_memory_utilization` too high or peak KV under-budgeted; leave headroom.

***

## Performance Numbers & Benchmarks
| Aspect | vLLM |
|---|---|
| Throughput vs naive HF | 2–4× (PagedAttention) |
| Model support | very broad |
| Prefix caching / multi-LoRA | yes |
| High-QPS scheduler | improved in V1 |
| Fixed-shape latency | TRT-LLM may edge it |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Walk through vLLM's architecture and the role of the BlockManager."* — Expected: engine/scheduler/block-manager/worker/runner; paged KV, ref-counts, preemption.
- *"What was vLLM's main weakness and how is it addressed?"* — Expected: Python scheduler overhead at high QPS; vLLM V1.
- *"How does vLLM serve hundreds of LoRA adapters?"* — Expected: base + multiplexed adapters (S-LoRA).
- *"Which knobs control the latency/throughput tradeoff?"* — Expected: max_num_batched_tokens, max_num_seqs, chunked prefill, gpu_memory_utilization.

***

## Open Problems & Active Research (2025–2026)
- **vLLM V1 maturation** and closing the scheduler-overhead gap with SGLang.
- **Disaggregation support** maturing in vLLM ([§03](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md)).
- **Better feature composition** (prefix cache + quant KV + speculation) defaults.

***

## References
- Kwon, W., et al. (2023). "vLLM/PagedAttention." *SOSP 2023*. arXiv:2309.06180.
- Sheng, Y., et al. (2023). "S-LoRA: Serving Thousands of Concurrent LoRA Adapters." arXiv:2311.03285.
- Agrawal, A., et al. (2024). "Sarathi-Serve." *OSDI 2024*. arXiv:2403.02310.
- vLLM project documentation and V1 design notes.
