# Parallelism Strategy Selection

> **Section:** 04_parallelism
> **Last Updated:** June 2026
> **Related Files:** [00_tensor_parallelism.md](00_tensor_parallelism.md), [01_pipeline_parallelism.md](01_pipeline_parallelism.md), [04_expert_parallelism_MoE.md](04_expert_parallelism_MoE.md)
> **Must-Read Papers:** Narayanan et al. (2021, SC) "Megatron 3D parallelism"; Pope et al. (2022, MLSys) "Efficiently Scaling Transformer Inference"; Zheng et al. (2022, OSDI) "Alpa"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- **Roles**: TP = latency + per-GPU memory fit (NVLink only); PP = cross-node scale/capacity; DP = throughput/availability (replicas); EP = MoE experts; SP/CP = long context.
- **Recipe**: TP within a node, PP across nodes, DP for replicas, EP(+TP) for MoE, CP for very long context. Total GPUs = `DP × PP × TP` (× CP, with EP overlapping experts).
- **Fit first** (weights+peak KV must fit), then **latency** (raise TP), then **throughput** (raise DP), then **scale** (PP/EP across nodes).
- Keep **communication-heavy** parallelism (TP, EP) **inside the NVLink domain**; **communication-light** (PP, DP, KV transfer) can cross nodes.
- Over-parallelizing wastes GPUs: minimal TP/PP that meets SLO, then scale with DP.

***

## Overview
Real deployments combine multiple parallelism axes ("3D"/"4D" parallelism), and choosing the combination is a constrained optimization with a fairly deterministic priority order. The axes have distinct, mostly-orthogonal roles: **tensor parallelism** shrinks per-GPU weights and decode latency but is communication-heavy (NVLink-only); **pipeline parallelism** splits the model across nodes with light, latency-tolerant communication (the cross-node scaler); **data parallelism** replicates for throughput and availability with no inter-replica communication; **expert parallelism** distributes MoE experts (all-to-all, NVLink-favoring); and **sequence/context parallelism** distributes long sequences. The skill is mapping each axis to the constraint it solves and to the fabric tier it can tolerate.

The decision proceeds by priority. **First, fit**: weights at your precision plus *peak* KV must fit across the GPUs of one replica — this sets the minimum TP (×PP) needed. **Second, latency**: if single-replica decode latency exceeds the ITL SLO, raise TP (within the NVLink domain) to divide per-token weight bandwidth. **Third, throughput**: scale out with DP replicas to hit QPS, sized via Little's Law. **Fourth, scale beyond a node**: use PP (and EP for MoE) across nodes, keeping the communication-heavy collectives on NVLink. The total GPU count is the product of the degrees, so over-parallelizing on TP/PP (which have communication overhead and diminishing returns) wastes hardware — the rule is **minimal TP/PP to meet fit+latency, then scale with cheap DP**.

This priority interacts with hardware ([§01](../01_hardware/05_hardware_selection_decision_framework.md)) and the fabric hierarchy ([§01](../01_hardware/03_interconnects_nvlink_infiniband.md)): the NVLink domain size (8 on HGX, 72 on NVL72) caps how much TP/EP you can do before paying inter-node penalties. Automated planners (Alpa, Zheng et al. 2022) search the parallelism space, but for inference the heuristics above usually suffice. This file gives the decision procedure and the common configurations.

***

## Core Concepts & Mechanics

### The selection procedure
```
1. FIT: min TP×PP so (weights@precision + peak_KV) fits per replica.
        Prefer TP within NVLink; add PP across nodes if model spans nodes.
2. LATENCY: if decode ITL > SLO, raise TP (≤ NVLink domain) to cut per-token bandwidth.
3. THROUGHPUT: set DP (replicas) for target QPS via Little's Law + margin.
4. MoE: use EP (+TP per expert) within NVLink domain; size so all experts fit.
5. LONG CONTEXT: add CP/SP (Ring/Ulysses) for 100k–1M tokens.
6. Minimize TP/PP (overhead, diminishing returns); scale with DP (near-linear).
```

### Fabric mapping
| Axis | Comm | Fabric tier | Crosses nodes? |
|---|---|---|---|
| TP | all-reduce (heavy, latency-sensitive) | NVLink | no |
| EP | all-to-all (heavy) | NVLink | prefer no |
| PP | point-to-point (light) | IB OK | yes |
| DP | none (routing) | any | yes |
| CP/SP | ring/all-to-all | NVLink (or overlapped ring on IB) | sometimes |

### Worked configs
- **70B, real-time chat, HGX**: TP=8 (fit + latency), DP=N replicas for QPS. No PP.
- **405B dense across 2 nodes**: TP=8 intra-node, PP=2 across nodes, DP for throughput.
- **DeepSeek-V3 MoE, NVL72**: EP across the domain + TP per expert + MLA; DP for replicas.
- **1M-context serving**: TP=8 + CP (Ring Attention) across more GPUs; DP for throughput.

***

## Key Challenges
1. **NVLink-domain ceiling.** TP/EP can't exceed the NVLink domain without IB penalties; large models on small domains force PP and careful placement.
2. **Over-parallelization waste.** Excess TP/PP adds communication and shrinks per-GPU matmuls, lowering throughput-per-GPU; easy to over-shard.
3. **Joint optimization complexity.** TP×PP×DP×EP×CP is a large space; interactions (e.g., PP bubbles needing DP concurrency) are non-obvious.
4. **Heterogeneous hardware.** Mixed fleets and disaggregation complicate uniform parallelism assumptions ([§09](../09_distributed_inference/04_heterogeneous_cluster_inference.md)).

***

## Solutions & Current Best Practices
- **Heuristic priority** (fit → latency → throughput → scale) covers most cases; reserve auto-search (Alpa-style) for unusual constraints.
- **TP=NVLink-domain-bounded, DP for the rest** — the default scaling pattern.
- **EP within NVLink (NVL72) for MoE**; **CP for long context**.
- **Benchmark the chosen config** on real traffic; verify comm isn't the bottleneck (NCCL tests, Nsight Systems).

***

## Implementation Notes
- Set `tensor_parallel_size`, `pipeline_parallel_size` (and EP/CP) so TP×PP fits one replica; replicate replicas for DP behind a router.
- Align PP stage boundaries to node boundaries; keep TP/EP within nodes.
- Re-evaluate when moving generations (NVLink domain size, bandwidth/FLOP ratios change the optimum).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Over-sharded with TP=16 cross-node "to be safe"** — IB all-reduce tanked latency *and* throughput; use TP=8 + PP/DP.
- **PP with too few requests bubbled** — added PP for capacity but low concurrency left stages idle; PP needs DP/concurrency to pay off.
- **EP across nodes for a big MoE** — all-to-all on IB dominated; needed NVL72 or EP within node.
- **Maxed TP for throughput** — TP is a latency/fit tool; throughput comes from DP. Wrong axis.

***

## Performance Numbers & Benchmarks
| Goal | Lever | Scaling |
|---|---|---|
| Fit large model | TP(×PP) | divides weights |
| Lower ITL | TP (NVLink) | ~linear until comm dominates |
| More QPS | DP | near-linear |
| Cross-node scale | PP | capacity |
| MoE | EP(+TP) | distributes experts |
| Long context | CP/SP | distributes sequence |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Pick a parallelism config for 70B real-time chat on 8-GPU nodes."* — Expected: TP=8 for fit+latency, DP replicas for QPS, no PP.
- *"How do you scale to 405B across nodes?"* — Expected: TP intra-node + PP across nodes + DP; keep TP on NVLink.
- *"Which axes can cross nodes and why?"* — Expected: PP/DP (light comm) yes; TP/EP (heavy comm) no.
- *"Why not just crank TP for more throughput?"* — Expected: TP is latency/fit; throughput from DP; TP has comm overhead/diminishing returns.

***

## Open Problems & Active Research (2025–2026)
- **Automated inference-time parallelism planners** that fold in SLOs, fabric, and disaggregation.
- **Elastic parallelism** that reconfigures TP/DP with load.
- **Optimal placement for MoE+long-context** on large NVLink domains.

***

## References
- Narayanan, D., et al. (2021). "Efficient Large-Scale LM Training on GPU Clusters" (3D parallelism). *SC 2021*. arXiv:2104.04473.
- Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference." *MLSys 2023*. arXiv:2211.05102.
- Zheng, L., et al. (2022). "Alpa: Automating Inter- and Intra-Operator Parallelism." *OSDI 2022*. arXiv:2201.12023.
