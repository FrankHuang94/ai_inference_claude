# Tensor Parallelism for Inference

> **Section:** 04_parallelism
> **Last Updated:** June 2026
> **Related Files:** [01_pipeline_parallelism.md](01_pipeline_parallelism.md), [05_parallelism_strategy_selection.md](05_parallelism_strategy_selection.md), [../01_hardware/03_interconnects_nvlink_infiniband.md](../01_hardware/03_interconnects_nvlink_infiniband.md)
> **Must-Read Papers:** Shoeybi et al. (2019, arXiv) "Megatron-LM"; Pope et al. (2022, MLSys) "Efficiently Scaling Transformer Inference"; Narayanan et al. (2021, SC) "Megatron tensor+pipeline"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- Tensor parallelism (TP) splits each weight matrix **across GPUs**; every GPU computes a slice of each layer, combined via **all-reduce**.
- **Column-parallel** then **row-parallel** layers pair up so each transformer block needs **2 all-reduces** (attention out, FFN out) per forward pass.
- TP reduces per-GPU weights *and* per-GPU memory bandwidth pressure → lower latency, but adds **communication on the critical path**.
- 📐 Communication is latency-sensitive; keep **TP within the NVLink domain** (TP ≤ 8 on HGX; up to 72 on NVL72). Cross-node TP usually destroys decode ITL.
- TP is the primary way to fit large models and cut decode latency; choose degree by memory fit and the latency/communication tradeoff.

***

## Overview
A model too large for one GPU, or one whose single-GPU decode latency is too high, is split across GPUs. **Tensor parallelism** (Megatron-LM, Shoeybi et al. 2019) partitions the *weights of each layer* across GPUs so they cooperate on every token, as opposed to pipeline parallelism (different layers on different GPUs) or data parallelism (full replicas). TP is the workhorse for inference because it (a) lets a model's weights fit by dividing them by the TP degree, and (b) divides the per-GPU **memory bandwidth** load per token — directly attacking the decode bottleneck ([§00](../00_fundamentals/03_memory_bandwidth_bound_compute.md)). With TP=8, each GPU streams ~1/8 of the weights per token, so decode can be ~8× faster (bandwidth permitting), at the cost of inter-GPU communication.

The Megatron scheme is elegant: a linear layer `Y = XW` is split **column-wise** (`W = [W₁ W₂]`, each GPU computes `XWᵢ`, outputs concatenated) or **row-wise** (`W = [W₁; W₂]`, inputs split, partial outputs **all-reduced**). By making the QKV/up/gate projections column-parallel and the output/down projections row-parallel, the partial results align so that only **one all-reduce after attention and one after the FFN** are needed per layer — communication is minimized to two collectives per block. Attention heads divide naturally across GPUs (each GPU owns a subset of heads); GQA requires `n_kv_heads ≥ TP` or KV replication ([§02](../02_kv_cache/00_kv_cache_fundamentals.md)).

The catch is that these all-reduces sit **on the per-layer critical path** of every forward pass, so their latency directly adds to both prefill and decode time. All-reduce latency is dominated by the slowest link, so TP demands a fast, low-latency fabric — NVLink/NVSwitch. Crossing nodes (InfiniBand, ~µs latency, ~20× lower bandwidth) for TP multiplies hundreds of collectives by IB latency and wrecks ITL. Hence the iron rule: **keep TP within the NVLink domain**, and use pipeline/data parallelism to scale beyond it ([§01](01_pipeline_parallelism.md), [§05](05_parallelism_strategy_selection.md)).

***

## Core Concepts & Mechanics

### Column- and row-parallel layers
- **Column-parallel** `Y=[XW₁, XW₂]`: weights split by output dim; each GPU produces part of the output; **no communication** for the forward (outputs are independent columns), but the *next* layer must consume them.
- **Row-parallel** `Y = X₁W₁ + X₂W₂`: weights split by input dim; each GPU computes a partial sum; results **all-reduced** to form Y.
- **Megatron block**: QKV (column) → attention (heads split) → output proj (row, all-reduce) → LayerNorm → FFN gate/up (column) → activation → down (row, all-reduce). → **2 all-reduces/block**.

### Communication volume
📐 Per all-reduce, ring algorithm moves `2·(t−1)/t · M` bytes where `M = tokens × d × bytes` (activation size) and `t = TP degree`. Per block: 2 all-reduces. Over L layers and `tokens` (1 per decode step × batch): decode communication ≈ `L × 2 × 2(t−1)/t × batch × d × bytes`. This is *latency-bound* at decode (small messages) — why fabric latency matters more than bandwidth here.

### Why TP cuts decode latency
Per-GPU weight bytes/token = `weights/t`. Decode time ≈ `weights/(t × bandwidth) + comm`. As long as comm (on NVLink) ≪ the weight-read savings, latency drops ~linearly in t. Cross-node, comm dominates and the savings vanish.

### TP degree guidelines
- TP ∈ {2,4,8} within an 8-GPU NVLink node; up to 72 in NVL72.
- Choose smallest TP that (a) fits weights+KV and (b) meets latency SLO; larger TP adds communication overhead and can hurt throughput per GPU.

***

## Key Challenges
1. **Communication on the critical path.** Two all-reduces per layer add latency to every token; at decode (small messages) this is latency-bound and sensitive to fabric.
2. **Cross-node TP is toxic.** IB latency × hundreds of collectives destroys ITL; TP must stay within NVLink.
3. **GQA/head divisibility.** Heads and KV heads must divide evenly by TP; `TP > n_kv_heads` forces KV replication, raising memory.
4. **Diminishing returns / throughput cost.** Beyond the latency need, higher TP lowers per-GPU throughput (more comm, smaller matmuls) — TP is a latency tool, not a throughput multiplier.

***

## Solutions & Current Best Practices
- **TP within NVLink domain only**; pipeline/data parallel across nodes ([§05](05_parallelism_strategy_selection.md)).
- **Overlap communication with compute** where possible (sequence-parallel + async all-reduce, fused comm).
- **Pick minimal TP** that fits memory and meets latency; use TP for latency, DP for throughput.
- **Sequence parallelism** to also shard LayerNorm/activations and reduce activation memory ([§02](02_sequence_parallelism.md)).

***

## Implementation Notes
- vLLM/SGLang/TRT-LLM set TP via `tensor_parallel_size`; ensure it divides heads and (for GQA) consider KV-head replication.
- Verify NCCL uses NVLink intra-node; cross-node TP should be avoided in config ([§01](../01_hardware/03_interconnects_nvlink_infiniband.md)).
- Combine with FP8 to halve communicated activation bytes and weight memory ([§05](../05_quantization/04_fp8_inference_h100.md)).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **TP=16 across two nodes tanked ITL** — all-reduce crossed IB; latency exploded. Keep TP ≤ NVLink size.
- **TP=8 with 8 KV heads fine; TP=16 forced KV replication** — exceeding n_kv_heads duplicates KV, raising memory unexpectedly.
- **Higher TP lowered throughput-per-GPU** — more communication and smaller per-GPU matmuls; TP helps latency, not aggregate throughput.
- **NCCL fell back to PCIe** — misconfigured topology halved effective bandwidth; verify with NCCL tests.

***

## Performance Numbers & Benchmarks
| TP degree | Per-GPU weights | Decode latency | Comm | Notes |
|---|---|---|---|---|
| 1 | full | high | none | single GPU |
| 2–8 (NVLink) | /TP | ~/TP (comm small) | 2 all-reduce/layer | sweet spot |
| 16+ cross-node | /TP | worse (comm dominates) | IB latency | avoid |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Explain Megatron column/row parallelism and why only 2 all-reduces per block."* — Expected: column then row pairing aligns partials.
- *"Why keep TP within a node?"* — Expected: all-reduce on critical path; NVLink vs IB latency.
- *"How does TP help decode latency, and where does it stop helping?"* — Expected: weights/TP per token; comm overhead and diminishing returns.
- *"How does GQA interact with TP degree?"* — Expected: n_kv_heads ≥ TP or replicate KV.

***

## Open Problems & Active Research (2025–2026)
- **Comm-compute overlap** to hide all-reduce latency at decode (async/fused collectives).
- **Larger NVLink domains (NVL72+)** enabling higher TP without cross-node penalty.
- **TP for MLA/MoE** architectures with different sharding patterns ([§04](04_expert_parallelism_MoE.md)).

***

## References
- Shoeybi, M., et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." arXiv:1909.08053.
- Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference." *MLSys 2023*. arXiv:2211.05102.
- Narayanan, D., et al. (2021). "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM." *SC 2021*. arXiv:2104.04473.
