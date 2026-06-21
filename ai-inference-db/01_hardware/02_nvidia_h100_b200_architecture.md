# NVIDIA H100, H200, and Blackwell (B200/GB200) for Inference

> **Section:** 01_hardware
> **Last Updated:** June 2026
> **Related Files:** [00_gpu_architecture_for_inference.md](00_gpu_architecture_for_inference.md), [03_interconnects_nvlink_infiniband.md](03_interconnects_nvlink_infiniband.md), [../05_quantization/04_fp8_inference_h100.md](../05_quantization/04_fp8_inference_h100.md)
> **Must-Read Papers:** NVIDIA Hopper (2022) & Blackwell (2024) architecture whitepapers; Micikevicius et al. (2022, arXiv) "FP8 Formats for Deep Learning"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- **H100 SXM5:** 80 GB HBM3, 3.35 TB/s, ~989 TFLOP/s FP16 / ~1979 FP8, NVLink 4 (900 GB/s). **FP8 Transformer Engine is the inference unlock** (≈2× over FP16).
- **H200:** same compute as H100 but **141 GB HBM3e @ 4.8 TB/s** — a pure memory/bandwidth upgrade, materially better for decode and long context.
- **B200 (Blackwell):** dual-die, 192 GB HBM3e (~8 TB/s), **FP4/FP6 support**, 2nd-gen Transformer Engine, NVLink 5 (1.8 TB/s). **GB200** pairs 2× B200 with a Grace CPU.
- **NVL72:** 72 Blackwell GPUs in one NVLink domain (~130 TB/s aggregate) — a single coherent memory fabric for serving very large / MoE models.
- Each precision step (FP16→FP8→FP4) roughly doubles Tensor-Core throughput but **moves the roofline ridge right**, so decode gains come mainly from fewer bytes, not more FLOPs.

***

## Overview
NVIDIA's datacenter line defines the inference cost/performance frontier, and knowing the specs cold is table stakes for these interviews. The **Hopper** generation (H100) introduced the **Transformer Engine** with native **FP8** (E4M3/E5M2) Tensor Cores, doubling matmul throughput over FP16 and halving weight/activation bytes — directly attacking both the compute and bandwidth bounds. The **H200** is a Hopper refresh that keeps H100 compute but swaps in larger, faster **HBM3e** (141 GB, 4.8 TB/s); since decode and long-context are bandwidth/capacity-bound, H200 often delivers the better *inference* dollar despite identical FLOPs.

**Blackwell** (B200/GB200) is a generational leap: a dual-reticle design presenting as one GPU with 192 GB HBM3e (~8 TB/s), a 2nd-gen Transformer Engine adding **FP4/FP6** micro-scaling formats, and NVLink 5 doubling interconnect bandwidth. The headline system is **GB200 NVL72** — 72 Blackwell GPUs in a single NVLink domain behaving like one giant accelerator with a unified high-bandwidth memory space, designed for trillion-parameter and large-MoE serving where model + KV must span many GPUs with cheap all-to-all/all-reduce.

The recurring inference lesson across these parts: **lower precision raises peak FLOPs but not HBM bandwidth**, shifting the roofline ridge rightward ([§00](../00_fundamentals/04_roofline_model_for_llm.md)). Compute-bound prefill rides the FLOP gains; bandwidth-bound decode benefits mainly from the *byte reduction* (FP8 weights = half the HBM traffic of FP16) and from raw bandwidth/capacity (H200, B200). Choose the part for the *binding* constraint of your workload.

***

## Core Concepts & Mechanics

### Spec comparison
| Spec | A100 80GB | H100 SXM5 | H200 | B200 | GB200 (per GPU) |
|---|---|---|---|---|---|
| HBM | 80 GB HBM2e | 80 GB HBM3 | 141 GB HBM3e | 192 GB HBM3e | 192 GB HBM3e |
| Bandwidth | 2.0 TB/s | 3.35 TB/s | 4.8 TB/s | ~8 TB/s | ~8 TB/s |
| FP16/BF16 | 312 TFLOP/s | 989 | 989 | ~2.2 PF | ~2.2 PF |
| FP8 | — | 1979 | 1979 | ~4.5 PF | ~4.5 PF |
| FP4 | — | — | — | ~9 PF | ~9 PF |
| NVLink | 600 GB/s (3.0) | 900 GB/s (4.0) | 900 GB/s | 1.8 TB/s (5.0) | 1.8 TB/s |
| TDP | 400 W | 700 W | 700 W | ~1000 W | ~1200 W (w/ Grace) |

(Dense, non-sparse Tensor-Core figures; Blackwell numbers SKU-dependent. Sparsity roughly doubles the headline FLOPs but is rarely realized for LLMs.)

### Why FP8 is the H100 inference unlock
The Transformer Engine performs FP8 GEMM with **FP16/BF16 accumulation** and **per-tensor (or finer) scaling** to manage FP8's limited range. Effect: ~2× matmul throughput (prefill) and **half the weight bytes** (decode bandwidth). E4M3 (4-exp, 3-mantissa, range ±448) for forward weights/activations; E5M2 (5-exp, range ±57344) for wider dynamic range. See [§05](../05_quantization/04_fp8_inference_h100.md) for scaling details.

### Blackwell FP4 and micro-scaling
Blackwell adds **FP4 (E2M1)** with **micro-scaling (MX) block formats** — a shared scale per small block (e.g., 32 elements) to retain accuracy at 4 bits. FP4 ~doubles throughput again and quarters bytes vs FP16, but quality at FP4 is still being characterized (see watchlist). 📐 Ridge point keeps moving right: H100 FP8 ridge ≈ 591 FLOPs/byte; FP4 on B200 is higher still relative to its bandwidth.

### NVL72 and the unified domain
NVLink 5 + NVSwitch create a 72-GPU coherent domain (~130 TB/s aggregate NVLink bandwidth) so a model's weights, experts, and KV can be sharded across 72 GPUs with all-to-all/all-reduce at NVLink (not InfiniBand) speeds — pivotal for large MoE (expert parallelism, [§04](../04_parallelism/04_expert_parallelism_MoE.md)) and very long context (context parallelism, [§10](../10_long_context/02_ring_attention_and_context_parallelism.md)).

***

## Key Challenges
1. **Power and cooling.** H100 700 W, B200 ~1000 W; NVL72 racks need liquid cooling and ~120 kW/rack — a real deployment constraint affecting TCO ([§01](06_tco_and_cost_modeling.md)).
2. **Precision quality cliffs.** FP8 is usually near-lossless with good scaling; FP4 can degrade quality on sensitive layers/tasks and needs careful per-block scaling and calibration.
3. **The ridge keeps moving right.** Each precision doubling helps compute-bound prefill more than bandwidth-bound decode, so naive "2× FLOPs = 2× faster" is wrong for decode.
4. **Supply, cost, and SKU sprawl.** PCIe vs SXM, NVL vs HGX, B200 vs B100 vs GB200 — differing bandwidth/power change the right choice and the math.

***

## Solutions & Current Best Practices
- **Use FP8 by default on Hopper/Blackwell** for weights (and often activations) — near-lossless, ~2× ([§05](../05_quantization/04_fp8_inference_h100.md)).
- **Prefer H200 over H100 for decode/long-context**-heavy workloads (bandwidth + capacity) ([§01](05_hardware_selection_decision_framework.md)).
- **Reserve NVL72/GB200 for very large or MoE models** where the NVLink domain eliminates cross-node bottlenecks ([§04](../04_parallelism/05_parallelism_strategy_selection.md)).
- **Validate FP4 per-task** before production; keep sensitive layers at FP8.

***

## Implementation Notes
- TensorRT-LLM and vLLM/SGLang expose FP8 paths via Transformer Engine / native kernels; enable per-tensor or per-channel scaling and calibrate on representative data.
- On NVL72, set TP/EP degrees to exploit the full NVLink domain; cross-node IB only when exceeding 72 GPUs.
- Account for **TDP in throughput/W** comparisons — B200's higher tok/s comes with higher power; compare tok/s/W and $/token, not raw tok/s.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **"FP8 doubled prefill but decode barely moved."** Expected: decode is bandwidth-bound; the win is the halved weight bytes (~2× if weights are the dominant read), not the FLOP doubling.
- **H100→H200 gave more decode speedup than H100→B200 per dollar for some workloads** — pure bandwidth/capacity (H200) can beat FLOP-heavy B200 for decode-bound serving; always match part to bottleneck.
- **FP4 looked fine on perplexity but failed on a reasoning benchmark** — low-precision damage is task-dependent and concentrated in sensitive layers; perplexity is an insufficient gate.
- **NVL72 assumed but only 8-GPU NVLink available** — many deployments are HGX 8-GPU nodes; cross-node falls back to InfiniBand, changing parallelism math entirely ([§03](03_interconnects_nvlink_infiniband.md)).

***

## Performance Numbers & Benchmarks
| Hardware | Model | Config | Result (order-of-magnitude) |
|---|---|---|---|
| 1× H100 | 8B FP8 | decode b=1 | ~250–350 tok/s |
| 1× H200 | 8B FP8 | decode b=1 | ~1.4× H100 (bandwidth) |
| 8× H100 | 70B FP8 | high batch | 10k+ aggregate tok/s |
| GB200 NVL72 | large MoE / 405B | served as one domain | multi-× over 8-GPU nodes for big models |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Compare H100 vs H200 for an LLM serving workload."* — Expected: identical compute; H200 wins decode/long-context via bandwidth+capacity.
- *"Why does FP8 give ~2× on prefill but less on decode?"* — Expected: compute roof vs bandwidth bound; ridge shift.
- *"What does NVL72 enable that 8-GPU HGX doesn't?"* — Expected: 72-GPU NVLink domain for large MoE / long context without IB bottleneck.
- *"What are E4M3 vs E5M2 and when to use each?"* — Expected: range/precision tradeoff; E4M3 fwd, E5M2 wider range.

***

## Open Problems & Active Research (2025–2026)
- **FP4/MX-format quality at scale** — which layers tolerate 4-bit, robust per-block scaling, calibration (watchlist item).
- **Production Blackwell inference benchmarks** — public, reproducible numbers still maturing.
- **Cooling/power-constrained deployment** economics for 1 kW+ parts and NVL72 racks ([§01](06_tco_and_cost_modeling.md)).

***

## References
- NVIDIA (2022). "H100 Tensor Core GPU Architecture" whitepaper.
- NVIDIA (2024). "Blackwell Architecture" whitepaper; GB200 NVL72 materials.
- Micikevicius, P., et al. (2022). "FP8 Formats for Deep Learning." arXiv:2209.05433.
- Open Compute Project (2023). "Microscaling (MX) Formats" specification.
