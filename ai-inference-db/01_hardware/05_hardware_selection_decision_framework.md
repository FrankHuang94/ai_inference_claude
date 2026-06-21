# Hardware Selection Decision Framework

> **Section:** 01_hardware
> **Last Updated:** June 2026
> **Related Files:** [02_nvidia_h100_b200_architecture.md](02_nvidia_h100_b200_architecture.md), [06_tco_and_cost_modeling.md](06_tco_and_cost_modeling.md), [../00_fundamentals/06_key_metrics_latency_throughput_cost.md](../00_fundamentals/06_key_metrics_latency_throughput_cost.md)
> **Must-Read Papers:** Pope et al. (2022, MLSys) "Efficiently Scaling Transformer Inference"; Williams et al. (2009) "Roofline"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- Start from the **binding constraint**: does the model *fit*? then is the workload **decode/latency-bound** (bandwidth+capacity, favor H200/MI300X) or **prefill/throughput-bound** (FLOPs, favor H100/B200)?
- **Memory capacity gate first**: weights + peak KV must fit (with TP) before anything else matters.
- **Bandwidth for decode, FLOPs for prefill, NVLink domain for large/MoE models.**
- **Latency-critical, small model** → consider Groq; **large MoE / 1M context** → NVL72/GB200; **AWS-committed** → Trainium/Inferentia.
- Always decide on **$/token at SLO and tok/s/W**, not peak specs or $/hour.

***

## Overview
Hardware selection is a constrained optimization: minimize $/token subject to meeting TTFT and ITL SLOs for your workload's prompt/output distribution. The decision is dominated by a short sequence of gates. First, **capacity**: the model's weights (at your chosen precision) plus the *peak* KV cache for your target concurrency must fit across the GPUs you'll use, given the tensor-parallel degree. If it doesn't fit, no other consideration applies. Second, the **bottleneck phase**: decode-heavy/latency-sensitive workloads are bandwidth- and capacity-bound and reward H200/MI300X; prefill-heavy/throughput workloads are compute-bound and reward H100/B200 FLOPs. Third, **scale-out coupling**: large dense or MoE models that must span many GPUs reward big NVLink domains (NVL72) to keep collectives fast.

The common mistake is to shop by headline FLOPs or $/hour. As established throughout [§00](../00_fundamentals/03_memory_bandwidth_bound_compute.md), decode tracks bandwidth, so an H200 (same FLOPs as H100) can be the better inference buy; and a cheaper-per-hour accelerator can be more expensive per token if kernels are immature or utilization is low. The correct currency is **$/token at your SLO**, computed from realistic, measured throughput on your model and traffic — not vendor best-case batch-saturated numbers.

This file gives a decision tree and a checklist you can run in an interview or a procurement meeting. It deliberately stays workload-first: the same model can warrant different hardware depending on whether you're serving real-time chat (latency), batch summarization (throughput), or long-CoT reasoning (decode-dominated cost).

***

## Core Concepts & Mechanics

### The decision tree
```
1. Does (weights@precision + peak_KV) fit on N GPUs with feasible TP?
   - No  → larger-memory GPU (H200/MI300X/B200) or more GPUs / smaller precision.
   - Yes → continue.
2. What is the binding phase?
   - Decode/latency-bound (chat, long output, reasoning)
        → maximize bandwidth + capacity: H200 > H100; MI300X if ROCm-ready.
        → ultra-low-latency small model? consider Groq.
   - Prefill/throughput-bound (RAG, classification, batch, long prompts)
        → maximize FLOPs: H100/B200; FP8/FP4.
3. Must the model span many GPUs (>8) or is it large MoE / 1M-context?
   - Yes → big NVLink domain (GB200 NVL72) to keep TP/EP/CP collectives on NVLink.
4. Cloud/commitment constraints?
   - AWS-committed → Trainium2/Inferentia2 for $/token.
   - Multi-cloud/portability → NVIDIA default.
5. Decide on measured $/token @ SLO and tok/s/W, not peak specs.
```

### Capacity math (the first gate)
📐 Required HBM ≈ `weights_bytes/TP + peak_concurrency × KV_per_request + activation_overhead`. With KV_per_request = `2·n_layers·n_kv_heads·head_dim·max_seq·bytes` (see [§02](../02_kv_cache/00_kv_cache_fundamentals.md)). If this exceeds aggregate HBM, raise TP, quantize weights/KV, or pick larger-memory parts.

### Matching part to phase
- **Decode-bound** payoff ∝ bandwidth and KV capacity → H200/MI300X. FP8 weights help (half bytes).
- **Prefill-bound** payoff ∝ FLOPs and FP8/FP4 → H100/B200.
- **Mixed** → H100/H200 with chunked prefill or disaggregation across part types ([§09](../09_distributed_inference/04_heterogeneous_cluster_inference.md)).

***

## Key Challenges
1. **KV capacity vs batch.** Long-context targets blow up KV, forcing larger-memory parts or smaller batch (hurting throughput) — the capacity gate and throughput goal conflict.
2. **Software readiness on non-NVIDIA parts.** A spec-superior part (MI300X) may underperform due to kernel maturity; selection must include a real benchmark, not specs.
3. **Workload heterogeneity.** A single fleet serving chat + batch + reasoning has conflicting optimal hardware; you may need heterogeneous pools.
4. **Power/cooling and rack constraints.** 700 W–1.2 kW parts and NVL72 racks require liquid cooling and high power density that some datacenters can't provide ([§01](06_tco_and_cost_modeling.md)).

***

## Solutions & Current Best Practices
- **Benchmark your exact model + traffic** on candidate hardware; compute $/token at SLO.
- **Default to H100/H200** for general NVIDIA serving; **H200 for decode/long-context**, **B200/NVL72 for very large/MoE**.
- **Use FP8 everywhere it's lossless** to relax the capacity gate and double prefill throughput ([§05](../05_quantization/04_fp8_inference_h100.md)).
- **Consider heterogeneous/disaggregated** fleets to match part to phase ([§03](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md)).

***

## Implementation Notes
- Provision for **peak** KV (max concurrency × max context), not average, or you OOM under load.
- Leave HBM headroom (~10–20%) for fragmentation, CUDA context, and activation spikes.
- Re-evaluate per generation: the right choice shifts as bandwidth/FLOP ratios and software support change.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Chose H100 on FLOPs, workload was decode-heavy** — H200's bandwidth/capacity would have given more tok/s/$; FLOPs were never the bottleneck.
- **Model "fit" computed without peak KV** — fit weights but OOM'd at target concurrency/context; always include peak KV in the capacity gate.
- **Picked the cheapest $/hour accelerator** — immature kernels/low utilization made $/token higher than NVIDIA.
- **Datacenter couldn't power/cool the chosen rack** — NVL72/1 kW parts need liquid cooling; a non-obvious hard constraint.

***

## Performance Numbers & Benchmarks
| Workload | Recommended | Why |
|---|---|---|
| Real-time chat (decode-bound) | H200 (or H100) | bandwidth+capacity for ITL |
| Ultra-low-latency, small model | Groq | deterministic SRAM latency |
| RAG / long-prompt (prefill-bound) | H100/B200 | FLOPs, FP8 |
| Very large / MoE / 1M context | GB200 NVL72 | NVLink domain for TP/EP/CP |
| Large memory-bound model, ROCm-ready | MI300X | 192 GB / 5.3 TB/s, price |
| AWS-committed batch | Inferentia2/Trn2 | $/token |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"You must serve a 70B model for real-time chat. Pick hardware and justify."* — Expected: capacity gate (TP for weights+KV), decode-bound → H200/H100, FP8.
- *"When is H200 worth it over H100?"* — Expected: decode/long-context bandwidth+capacity; same FLOPs.
- *"How do you decide $/token rather than $/hour?"* — Expected: measured throughput at SLO; utilization; kernel maturity.
- *"When would you pick non-NVIDIA hardware?"* — Expected: Groq latency niche, MI300X memory if ROCm-ready, AWS commitment.

***

## Open Problems & Active Research (2025–2026)
- **Automated hardware/parallelism co-selection** from a workload trace and SLO.
- **Heterogeneous fleet optimization** (which phase/model on which part) at cloud scale.
- **Power/cooling-constrained selection** as parts cross 1 kW.

***

## References
- Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference." *MLSys 2023*. arXiv:2211.05102.
- Williams, S., et al. (2009). "Roofline." *CACM* 52(4).
- NVIDIA / AMD / Google hardware whitepapers (see related files).
