# Interconnects: NVLink, NVSwitch, InfiniBand, and RoCE

> **Section:** 01_hardware
> **Last Updated:** June 2026
> **Related Files:** [02_nvidia_h100_b200_architecture.md](02_nvidia_h100_b200_architecture.md), [../04_parallelism/00_tensor_parallelism.md](../04_parallelism/00_tensor_parallelism.md), [../09_distributed_inference/03_kv_cache_migration_and_transfer.md](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)
> **Must-Read Papers:** NVIDIA NVSwitch/NVLink whitepapers; Patel et al. (2024, ISCA) "Splitwise"; Qin et al. (2024, arXiv) "Mooncake"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- **NVLink/NVSwitch** = intra-node GPU fabric: H100 900 GB/s, Blackwell 1.8 TB/s per GPU; NVSwitch makes all-to-all within a node (or NVL72 domain) full-bandwidth.
- **InfiniBand/RoCE** = inter-node: NDR InfiniBand ~400 Gb/s (50 GB/s) per NIC; latency ~1–2 µs. ~20× slower than NVLink.
- **Tensor parallelism wants NVLink** (frequent all-reduces); crossing nodes for TP is usually a mistake — latency dominates.
- **Disaggregation and KV transfer** are bottlenecked by inter-node bandwidth; a single H100 can generate tens of GB/s of KV, so transfers need RDMA (IB/RoCE) and overlapping.
- Collective cost, not just point-to-point bandwidth, determines parallelism feasibility: know all-reduce/all-to-all volume formulas.

***

## Overview
Once a model or workload spans multiple GPUs, the **interconnect** becomes a first-class performance variable. There are two tiers: **intra-node** (NVLink + NVSwitch), which is fast enough to treat several GPUs almost like one, and **inter-node** (InfiniBand or RoCE over Ethernet), which is an order of magnitude slower in bandwidth and far higher in latency. The gap between these tiers dictates how you partition a model: communication-heavy strategies (tensor parallelism, expert all-to-all) belong *inside* an NVLink domain, while looser couplings (pipeline parallelism, replica-level data parallelism, KV transfer in disaggregation) can tolerate crossing nodes.

NVLink provides direct GPU-to-GPU links; **NVSwitch** is a crossbar that connects all GPUs in a node (or, in NVL72, 72 GPUs) at full per-GPU bandwidth, enabling non-blocking all-to-all. This is why an 8-GPU HGX node can run TP=8 efficiently and why GB200 NVL72 can serve enormous MoE models — the expensive collectives stay on a fast fabric. Cross-node, **InfiniBand** (NDR ~400 Gb/s) with **GPUDirect RDMA** lets NICs DMA directly to/from GPU memory, and **RoCE** brings RDMA semantics to Ethernet; both are essential for multi-node training and for KV-cache migration in disaggregated serving.

The practical skill is reasoning about **collective communication volume** and mapping it onto this bandwidth hierarchy. Tensor parallelism injects an all-reduce per layer whose volume scales with hidden size and batch; expert parallelism injects all-to-all whose volume scales with tokens routed; disaggregation injects a KV transfer per request whose volume scales with prompt length. Each must fit within the available fabric without becoming the critical path.

***

## Core Concepts & Mechanics

### Bandwidth/latency hierarchy
| Link | Bandwidth (per GPU/NIC) | Latency | Scope |
|---|---|---|---|
| HBM3 (reference) | 3.35 TB/s | ~hundreds ns | on-package |
| NVLink 4 (H100) | 900 GB/s | ~hundreds ns–µs | intra-node / NVL |
| NVLink 5 (Blackwell) | 1.8 TB/s | ~µs | intra-node / NVL72 |
| NDR InfiniBand | ~50 GB/s (400 Gb/s) | ~1–2 µs | inter-node |
| RoCE (Ethernet) | ~25–50 GB/s | ~2–5 µs | inter-node |
| PCIe 5 | ~64 GB/s | ~µs | host↔GPU |

### Collective volume formulas
📐 For **tensor parallelism** degree `t`, hidden size `d`, batch·seq tokens `T`, per layer there are typically 2 all-reduces (attention out, FFN out). Ring all-reduce communicates `2·(t−1)/t · message_size` bytes; message_size ≈ `T·d·bytes`. Over `L` layers this is substantial and latency-sensitive — hence NVLink-only.

For **expert parallelism**, an **all-to-all** routes each token's hidden vector to its expert's GPU and back: volume ≈ `2·T·d·bytes` per MoE layer, spread across the EP group. All-to-all is latency- and bisection-bandwidth-sensitive (NVSwitch shines).

For **KV transfer** (disaggregation), volume per request ≈ KV-cache size = `2·n_layers·n_kv_heads·head_dim·P·bytes`. A busy prefill GPU emitting many requests/s can need tens of GB/s — must use RDMA and overlap with decode start ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).

### GPUDirect RDMA
NICs read/write GPU HBM directly (bypassing host memory and CPU), critical for low-latency cross-node TP and KV transfer. Without it, transfers stage through host RAM over PCIe, adding latency and halving effective bandwidth.

***

## Key Challenges
1. **The 20× intra/inter gap.** Strategies that are free on NVLink become bottlenecks across nodes; mis-mapping parallelism to fabric is the most common scaling error.
2. **Latency, not just bandwidth, kills TP across nodes.** All-reduce is on the per-layer critical path; IB's µs latency × hundreds of collectives/token destroys decode ITL.
3. **All-to-all scaling for MoE.** As EP degree and token count grow, all-to-all volume and contention rise; cross-node EP needs careful topology and overlap.
4. **KV transfer bandwidth in disaggregation.** Under-provisioned interconnect makes transfer dominate TTFT, negating disaggregation's benefits ([§03](../00_fundamentals/02_prefill_vs_decode_phases.md)).

***

## Solutions & Current Best Practices
- **Keep TP within the NVLink domain** (TP ≤ 8 on HGX, up to 72 on NVL72); use pipeline/data parallelism across nodes ([§04](../04_parallelism/05_parallelism_strategy_selection.md)).
- **Overlap communication with compute** (async collectives, double-buffering) to hide latency.
- **Use GPUDirect RDMA** for all cross-node GPU traffic (TP, EP, KV transfer).
- **Topology-aware placement**: co-locate TP/EP groups on the same switch; place pipeline stages to minimize cross-switch hops.

***

## Implementation Notes
- NCCL handles collectives; set `NCCL_*` env (topology, IB HCA selection) and verify it's using NVLink intra-node and IB inter-node, not falling back to PCIe.
- For disaggregation, libraries like **NIXL/LMCache** manage RDMA KV transfer; budget transfer time into the TTFT SLO ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).
- Measure **bus bandwidth utilization** (NCCL tests, `nvbandwidth`) to confirm you reach ~80%+ of link peak before blaming the model.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **TP=16 across two 8-GPU nodes tanked ITL.** TP all-reduce crossed IB; latency on the per-layer critical path exploded. Keep TP ≤ NVLink domain; go pipeline/replica across nodes.
- **NCCL silently used PCIe instead of NVLink.** Misconfigured topology/visibility halves bandwidth; always verify with NCCL tests.
- **Disaggregation raised TTFT.** KV transfer wasn't overlapped or IB was undersized; transfer time added directly to first-token latency.
- **MoE all-to-all stalled at high EP across nodes.** Bisection bandwidth/contention; needs NVSwitch domain or careful token-grouping/overlap.

***

## Performance Numbers & Benchmarks
| Scenario | Fabric | Implication |
|---|---|---|
| TP=8, 70B, H100 | NVLink/NVSwitch | efficient; all-reduce hidden |
| TP across nodes | InfiniBand | ITL regression; avoid |
| KV transfer (disagg) | NDR IB + RDMA | feasible if overlapped & ≤ link BW |
| MoE EP=64 | NVL72 NVSwitch | all-to-all stays on NVLink |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why keep tensor parallelism within a node?"* — Expected: per-layer all-reduce latency; NVLink vs IB 20× gap.
- *"Estimate the interconnect bandwidth needed for KV transfer in a disaggregated system."* — Expected: KV-size × request rate; compare to IB ~50 GB/s; overlap requirement.
- *"What is GPUDirect RDMA and why does it matter?"* — Expected: NIC↔GPU DMA bypassing host; latency/bandwidth.
- *"Which parallelism maps to which fabric tier?"* — Expected: TP/EP→NVLink; PP/DP/KV-transfer→inter-node OK.

***

## Open Problems & Active Research (2025–2026)
- **Scaling NVLink domains** (beyond NVL72) vs better inter-node fabrics (Ethernet/UEC, optical) for large MoE serving.
- **KV-transfer protocols and compression** to make disaggregation cheap over commodity Ethernet ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).
- **Topology-aware schedulers** that place TP/EP/PP groups optimally on heterogeneous fabrics.

***

## References
- NVIDIA (2022/2024). NVLink, NVSwitch, NVL72 whitepapers.
- Patel, P., et al. (2024). "Splitwise." *ISCA 2024*. arXiv:2311.18677.
- Qin, R., et al. (2024). "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving." arXiv:2407.00079.
- NVIDIA NCCL documentation.
