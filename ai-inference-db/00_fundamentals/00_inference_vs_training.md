# Inference vs Training: Two Different Performance Disciplines

> **Section:** 00_fundamentals
> **Last Updated:** June 2026
> **Related Files:** [02_prefill_vs_decode_phases.md](02_prefill_vs_decode_phases.md), [03_memory_bandwidth_bound_compute.md](03_memory_bandwidth_bound_compute.md), [04_roofline_model_for_llm.md](04_roofline_model_for_llm.md)
> **Must-Read Papers:** Williams, Waterman & Patterson (2009, CACM) "Roofline"; Pope et al. (2022, MLSys) "Efficiently Scaling Transformer Inference"; Kwon et al. (2023, SOSP) "PagedAttention/vLLM"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- **Training is compute-bound**; **autoregressive inference (decode) is memory-bandwidth-bound** at the batch sizes typical in production. They live on opposite sides of the roofline ridge point.
- The dominant cost in decode is **reading model weights from HBM once per token**, not the matrix-multiply FLOPs. At batch=1, the GPU's Tensor Cores sit mostly idle.
- **Arithmetic intensity** (FLOPs per byte moved) is the single number that determines which regime you are in. Training and prefill have high AI; decode has AI ≈ batch size.
- Inference has **no backward pass, no optimizer state, no activations to stash** for gradients — so its memory pressure comes from *weights + KV cache*, not activations + gradients.
- Optimizing inference therefore means **raising arithmetic intensity** (batching, speculative decoding) and **shrinking bytes-per-token** (quantization), not just adding FLOPs.

***

## Overview
Training and inference are often discussed as if they were the same workload run "forward" vs "forward+backward," but from a systems standpoint they are different disciplines with different bottlenecks, different hardware sweet spots, and different optimization playbooks. Training a transformer is dominated by large dense GEMMs over big batches and long sequences processed in parallel; it is **compute-bound**, achieving high Model FLOP Utilization (MFU, often 40–55% on well-tuned clusters) and stressing Tensor Cores and inter-GPU collective bandwidth. The engineering problem is keeping thousands of accelerators fed and synchronized.

Inference splits into two sub-phases (covered in depth in [02_prefill_vs_decode_phases.md](02_prefill_vs_decode_phases.md)): **prefill**, which processes the whole prompt in parallel and looks like training (compute-bound), and **decode**, which emits one token at a time and is **memory-bandwidth-bound**. Decode is where most production wall-clock time and cost go for chat-style workloads, and it is pathological from a hardware-utilization view: to produce a *single* token at batch=1 you must stream *every* model weight from HBM through the compute units exactly once, performing only a couple of FLOPs per weight. The ratio of useful FLOPs to bytes moved — the arithmetic intensity — is roughly equal to the batch size, far below the ~hundreds of FLOPs/byte where modern GPUs become compute-bound.

This reframes the whole optimization problem. In training you fight for FLOP efficiency; in decode you fight to **amortize weight loads across more useful work**. That is precisely what batching, speculative decoding, and MoE sparsity do, and why quantization (fewer bytes per weight) gives near-linear decode speedups while it gives training much smaller gains. Internalizing this asymmetry is the prerequisite for everything else in this database.

***

## Core Concepts & Mechanics

### Arithmetic intensity
📐 Arithmetic intensity (AI), also "operational intensity," is:

```
AI = (useful FLOPs performed) / (bytes moved from/to HBM)   [FLOPs/byte]
```

A GPU is **compute-bound** when `AI > AI_ridge` and **memory-bound** when `AI < AI_ridge`, where the ridge point is:

```
AI_ridge = Peak_compute (FLOP/s) / Peak_bandwidth (bytes/s)
```

For an H100 SXM5 with FP16 ≈ 989 TFLOP/s and HBM3 ≈ 3.35 TB/s:

```
AI_ridge(H100, FP16) ≈ 989e12 / 3.35e12 ≈ 295 FLOPs/byte
```

So you need to do ~295 FLOPs per byte read to saturate the Tensor Cores. Decode at batch=1 does ~2 FLOPs per weight byte (one multiply-add per weight, ≈2 FLOPs, per 2 bytes in FP16 ⇒ AI≈1). You are ~300× below the ridge — the GPU is doing nothing but waiting on memory.

### Why the forward-only pass changes everything
- **No gradients / optimizer state.** Training must store activations for backprop and keep optimizer moments (Adam = 2× params in fp32). Inference stores *none* of this. Memory goes to **weights + KV cache** instead.
- **No backward pass.** Backprop is ~2× the FLOPs of the forward pass; removing it removes the part that was most compute-bound and keeps the autoregressive part that is bandwidth-bound.
- **Sequential dependency.** Token *t+1* needs token *t*'s output. You cannot parallelize across the time dimension within one sequence during decode (this is exactly what speculative decoding tries to break; see [§07](../07_speculative_decoding/00_speculative_decoding_fundamentals.md)).

### Worked example: AI of a 70B decode step
A dense 70B model in FP16 = ~140 GB of weights. Per decoded token you read ≈140 GB.

- **Batch = 1:** useful FLOPs ≈ 2 × 70e9 = 1.4e11 (one MAC per param). Bytes ≈ 1.4e11. AI ≈ 1 FLOP/byte → deeply memory-bound. Decode rate ≈ `3.35e12 / 1.4e11 ≈ 24 tokens/s` (bandwidth ÷ weight bytes) on a single H100 — and you need ~2 H100s just to fit weights+KV, so realistically TP=2/4.
- **Batch = 64:** weights are read once but reused across 64 sequences → useful FLOPs ≈ 64 × 1.4e11 = 9e12, bytes ≈ 1.4e11 (weights) + KV traffic. AI ≈ 64 FLOPs/byte → still below ridge but ~64× more efficient per weight load. This is *why* throughput-oriented serving batches aggressively.

The takeaway: **decode throughput per GPU scales nearly linearly with batch size until you approach the compute ridge or run out of KV-cache memory** — and KV cache is what limits batch size, which is why [§02](../02_kv_cache/00_kv_cache_fundamentals.md) is the most important section.

***

## Key Challenges
1. **Batch size is capped by KV-cache memory, not compute.** The lever that fixes decode efficiency (bigger batches) is bounded by HBM consumed by the KV cache, which grows with batch × sequence length. You are perpetually trading batch size against context length.
2. **Latency vs throughput are in direct tension.** Bigger batches raise throughput (good for cost) but raise per-token latency and TTFT (bad for UX). You cannot maximize both; you choose a point on the curve per SLO.
3. **The sequential decode dependency wastes hardware.** Single-stream generation cannot fill the compute units, so a $30k GPU runs at single-digit-percent MFU unless you batch or speculate.
4. **Variable shapes break the static-graph assumptions** that make training fast. Inference sees arbitrary prompt/output lengths, so kernels and memory allocators must handle dynamism without fragmenting memory or re-compiling.

***

## Solutions & Current Best Practices
- **Continuous batching** (Orca, Yu et al. 2022 OSDI) to raise effective batch size without head-of-line blocking — see [§03](../03_batching_and_scheduling/00_static_vs_continuous_batching.md). This is the highest-leverage single technique.
- **PagedAttention / vLLM** (Kwon et al. 2023, SOSP) to eliminate KV fragmentation so larger batches fit — see [§02](../02_kv_cache/01_paged_attention_vllm.md).
- **Quantization** (weight-only INT4, FP8) to cut bytes-per-token; near-linear decode speedup because decode is bandwidth-bound — see [§05](../05_quantization/00_quantization_fundamentals.md). Converged production default in 2025–26 is FP8 on Hopper/Blackwell.
- **Speculative decoding** to raise per-step useful work, converting bandwidth-bound decode into compute-bound verification — see [§07](../07_speculative_decoding/00_speculative_decoding_fundamentals.md).
- **Prefill/decode disaggregation** to run each phase on hardware suited to its regime — see [§03](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md).

The field has converged on: continuous batching + paged KV + FP8 weights as the baseline; speculative decoding and disaggregation as workload-dependent add-ons.

***

## Implementation Notes
- When you profile inference and see **low SM occupancy but high "memory throughput"** in Nsight Compute, that is the bandwidth bound, not a bug. Don't chase occupancy; chase batch size and bytes-per-token.
- **MFU is the wrong north-star metric for decode.** A decode-heavy server can be cost-optimal at <10% MFU because the binding constraint is bandwidth and KV memory. Use **tokens/s/GPU and $/M tokens** instead — see [06_key_metrics_latency_throughput_cost.md](06_key_metrics_latency_throughput_cost.md).
- Training-era intuitions ("use bigger tiles, fuse more matmuls") help prefill but barely move decode; for decode you want **fused dequant kernels, CUDA graphs to kill launch overhead, and KV-cache-friendly attention**.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **"We added more FLOPs (a bigger GPU) and decode barely sped up."** If you went A100→H100 the FP16 FLOPs ~doubled but HBM bandwidth went 2.0→3.35 TB/s (~1.7×); decode (bandwidth-bound) tracks the *bandwidth* ratio, not the FLOP ratio. H200/HBM3e helps decode more than H100 despite identical compute, precisely because bandwidth is the bottleneck.
- **Quantizing weights to INT4 gave ~2× decode but <10% prefill speedup.** Prefill is compute-bound and (often) still runs the matmul in FP16/BF16 after dequant; only the bandwidth-bound decode benefits from the smaller weight footprint. People misattribute this to "the kernel being bad."
- **Throughput cratered when context got long even though batch was constant.** KV cache grew, batch had to shrink to fit HBM, and you slid back toward the memory-bound, low-batch regime. The "regression" is the memory budget, not the model.

***

## Performance Numbers & Benchmarks
| Hardware | Model | Config | Metric | Result |
|---|---|---|---|---|
| 1× H100 SXM5 | 7B FP16 | batch=1 decode | tokens/s | ~140–170 (≈ 3.35TB/s ÷ 14GB) |
| 1× H100 | 7B FP16 | batch=64 decode | tokens/s (aggregate) | ~5–8k (near-linear until KV/compute limit) |
| 2× H100 (TP2) | 70B FP16 | batch=1 decode | tokens/s | ~20–30 (bandwidth ÷ 140GB / TP) |
| 8× H100 | 70B FP8 | high batch | aggregate tokens/s | 10k+ (FP8 ~2× over FP16) |
| 1× H200 vs H100 | 70B | decode | speedup | ~1.4× from bandwidth alone (4.8 vs 3.35 TB/s) |

(Order-of-magnitude figures for reasoning; exact numbers depend on framework, kernels, and sequence length.)

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why is LLM decode memory-bandwidth-bound but training compute-bound? Derive the arithmetic intensity at batch=1."* — Expected: the AI≈batch derivation and the ridge-point comparison above.
- *"You upgraded A100→H100 and decode only sped up ~1.6×, not 2×. Why?"* — Expected: bandwidth ratio vs FLOP ratio; decode tracks bandwidth.
- *"Where does memory go in inference vs training?"* — Expected: weights+KV (inference) vs weights+gradients+optimizer+activations (training); no backward pass.
- *"Given a fixed GPU, how do you make decode faster?"* — Expected: raise AI (batch, speculation, MoE), cut bytes/token (quantization), reduce KV pressure to enable bigger batches.

***

## Open Problems & Active Research (2025–2026)
- **Breaking the sequential decode dependency** without quality loss at scale — speculative decoding helps but degrades at high batch (see [§07](../07_speculative_decoding/05_when_speculative_decoding_fails.md)); diffusion-LM and parallel decoding remain open.
- **Architectures that shift decode's arithmetic intensity** — MoE raises effective AI by activating few params; SSM/linear-attention hybrids change the bandwidth profile entirely (see [§10](../10_long_context/03_sparse_and_linear_attention.md)).
- **Hardware co-design for the bandwidth bound** — whether HBM bandwidth scaling (HBM3e→HBM4) or on-package SRAM (Groq-style) is the right answer for decode is unsettled (see [§01](../01_hardware/04_alternative_accelerators.md)).
- **Cost models for reasoning models** where decode dominates massively (10k–100k output tokens) — a new economic regime (see [§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

***

## References
- Williams, S., Waterman, A., Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *CACM* 52(4).
- Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference." *MLSys 2023*. arXiv:2211.05102.
- Kwon, W., et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023*. arXiv:2309.06180.
- Yu, G.-I., et al. (2022). "Orca: A Distributed Serving System for Transformer-Based Generative Models." *OSDI 2022*.
- Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*. arXiv:2205.14135.
