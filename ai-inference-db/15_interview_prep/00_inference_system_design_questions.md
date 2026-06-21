# Inference System Design Questions (Worked Answers)

> **Section:** 15_interview_prep
> **Last Updated:** June 2026
> **Related Files:** [../03_batching_and_scheduling/03_prefill_decode_disaggregation.md](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md), [../13_production_systems/00_production_serving_architecture.md](../13_production_systems/00_production_serving_architecture.md), [05_paper_deep_dives.md](05_paper_deep_dives.md)
> **Must-Read Papers:** Kwon et al. (2023) "vLLM"; Zhong et al. (2024) "DistServe"; Sheng et al. (2023) "S-LoRA"
> **Estimated Study Time:** 45 minutes

***

## TL;DR
- System-design answers should follow: **clarify SLOs → capacity math (Little's Law) → hardware/parallelism → KV/batching → optimizations → bottleneck/scaling → monitoring**.
- Always **start from the bandwidth bound** and the **prefill/decode split** — they justify every choice.
- Quantify: GPUs needed, KV memory, batch size, $/token at SLO.
- Name the techniques (continuous batching, paged KV, FP8, chunked prefill, disaggregation, speculative decoding) and **when each applies**.
- This file gives full worked answers to four canonical questions.

***

## Overview
Inference system-design interviews test whether you can translate the principles in this database into a coherent, quantified architecture under SLO constraints. The strongest answers follow a consistent structure: **clarify the SLOs and workload** (TTFT/ITL targets, request rate, prompt/output length distribution, prefix sharing), do the **capacity math** (Little's Law → in-flight concurrency → GPUs), choose **hardware and parallelism** (fit the model, meet latency), specify **KV and batching** strategy, layer **optimizations** (quantization, prefix caching, speculative decoding, disaggregation) justified by the bandwidth bound, identify the **bottleneck and how to scale**, and close with **monitoring** (goodput, P99). Below are four canonical questions with worked answers.

***

## Q1: Design a serving system for a 70B model — 10,000 req/s, P99 TTFT < 1s

**Clarify**: Assume avg prompt ~1k tokens, output ~500 tokens, ITL SLO ~50ms, mixed interactive traffic, some shared system prompts.

**Capacity (Little's Law)**: If avg E2E ≈ TTFT + 500×50ms ≈ 0.5s + 25s? — that's too long; revisit. For interactive, output must stream fast: at ITL 50ms, 500 tokens = 25s E2E, so this is a long-output workload. In-flight `L = λW = 10,000 × 25 ≈ 250,000` concurrent sequences. That's enormous → this is a **throughput-dominated** design.

**Hardware/parallelism**: 70B FP8 ≈ 70 GB → fits on 1×H100 (80GB) but leaves little for KV; use **TP=2** (H100) or **TP=4** for KV headroom and latency. Prefer **H200** (bandwidth+capacity) for decode. Per replica = TP group.

**KV/batching**: Paged KV ([§02](../02_kv_cache/01_paged_attention_vllm.md)); KV/token ~320KB → at 1.5k avg context, ~0.5GB/req. Continuous batching ([§03](../03_batching_and_scheduling/00_static_vs_continuous_batching.md)); per-replica batch bounded by KV headroom and ITL. **Chunked prefill** to protect ITL ([§03](../03_batching_and_scheduling/02_chunked_prefill.md)). **Prefix caching** for shared system prompts ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)).

**Optimizations**: **FP8** (half bytes, ~2× prefill) ([§05](../05_quantization/04_fp8_inference_h100.md)); **disaggregation** given the scale and the prompt/output skew — separate prefill and decode pools, decode on H200 ([§03](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md)); **speculative decoding** at low-batch decode pools if applicable.

**Scaling**: DP replicas behind a cache/load-aware router ([§09](../09_distributed_inference/01_load_balancing_strategies.md)); autoscale with warm pools (cold start) ([§13](../13_production_systems/02_autoscaling_and_capacity_planning.md)). For TTFT P99 < 1s: prefix caching + prefill pool sizing + admission control to bound queueing.

**Estimate**: with ~tens of decode tok/s/seq and large batches, each replica serves a few thousand concurrent decodes; reaching 250k concurrency needs ~tens–hundreds of replicas (hundreds–thousands of GPUs). Quantify per measured per-replica throughput; optimize $/token at SLO.

**Monitor**: goodput, P99 TTFT/ITL per tier, KV utilization, preemptions ([§13](../13_production_systems/01_observability_and_profiling.md)).

***

## Q2: Design a multi-tenant platform with per-customer SLAs

**Isolation**: per-tenant KV quotas, data isolation; scope cross-tenant prefix caching to public system prompts only (timing side-channel!) ([§13](../13_production_systems/04_multi_tenant_serving.md), [§02](../02_kv_cache/05_cross_request_kv_sharing.md)).

**Fairness/SLA**: token-bucket rate limiting per tenant + **weighted fair queueing** + priority/preemption for premium tiers ([§03 multi-priority](../03_batching_and_scheduling/05_multi_priority_and_SLA_scheduling.md)). Backfill batch traffic into idle capacity.

**Per-customer fine-tunes**: **S-LoRA** — one base + many LoRA adapters, batched heterogeneous-adapter kernels ([§13](../13_production_systems/04_multi_tenant_serving.md)).

**Enforcement**: per-tier goodput SLOs ([§13](../13_production_systems/06_SLO_definition_and_enforcement.md)); admission control; separate pools for tiers if interference can't be tamed.

***

## Q3: Reduce the cost of serving reasoning models (long CoT) by 50%

**Diagnose**: decode-dominated, huge per-request KV, length-unpredictable ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

**Levers** (stack them): **KV quantization (INT4/FP8) + compression** → fit bigger batches (more throughput/GPU) ([§05](../05_quantization/05_kv_cache_quantization.md), [§02](../02_kv_cache/03_kv_cache_compression.md)); **thinking-budget control + dynamic compute** → spend less on easy queries ([§12](../12_reasoning_model_inference/04_thinking_budget_and_inference_time_scaling.md), [§12](../12_reasoning_model_inference/03_dynamic_compute_allocation.md)); **cost-aware routing** → invoke reasoning model only when needed (cascade) ([§13](../13_production_systems/03_model_routing_and_cascading.md)); **speculative decoding** at low batch; **decode-optimized hardware** (H200/MI300X) + disaggregation. Combined, these readily exceed 50%.

***

## Q4: Design a RAG serving system

**Pipeline**: query → retrieval (vector search ~tens ms) → rerank → LLM (retrieved chunks + query) → stream ([§10](../10_long_context/04_retrieval_augmented_generation_vs_long_context.md)).

**Latency**: retrieval adds to TTFT — optimize/cache it; **chunked prefill** for the retrieved-context prompt; **prefix-cache shared documents** reused across queries ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)).

**Why RAG not long-context**: cost — prefill a few k tokens, not the whole corpus ([§10](../10_long_context/04_retrieval_augmented_generation_vs_long_context.md)).

**Orchestration**: compose stages (Ray Serve or custom) with independent autoscaling ([§14](../14_company_deep_dives/03_anyscale_and_ray_serve.md)).

***

## Interview Angles
> 💡 **How to deliver these:**
- **Always clarify SLOs and workload first**; never design in a vacuum.
- **Quantify** (Little's Law, KV math, GPUs, $/token); show the arithmetic.
- **Justify each choice from the bandwidth bound / prefill-decode split.**
- **Name techniques and when they apply** (and when they don't — e.g., disaggregation for skewed P/D, not short prompts).
- **Close with bottleneck + scaling + monitoring (goodput/P99).**

***

## Complexity Gotchas
> ⚠️ **Common interview mistakes:**
- Jumping to a solution without clarifying SLOs/workload.
- Optimizing MFU/throughput instead of **goodput at SLO**.
- Forgetting **peak KV** in capacity (OOM under load).
- Proposing disaggregation/speculation **everywhere** without the batch/workload conditions.
- Ignoring cold start in autoscaling.

***

## References
- Kwon et al. (2023) "vLLM" arXiv:2309.06180; Zhong et al. (2024) "DistServe" arXiv:2401.09670; Sheng et al. (2023) "S-LoRA" arXiv:2311.03285.
- This database, §00–§14.
