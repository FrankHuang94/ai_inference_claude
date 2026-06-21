# Multi-Node Serving

> **Section:** 09_distributed_inference
> **Last Updated:** June 2026
> **Related Files:** [../04_parallelism/05_parallelism_strategy_selection.md](../04_parallelism/05_parallelism_strategy_selection.md), [01_load_balancing_strategies.md](01_load_balancing_strategies.md), [../01_hardware/03_interconnects_nvlink_infiniband.md](../01_hardware/03_interconnects_nvlink_infiniband.md)
> **Must-Read Papers:** Narayanan et al. (2021, SC) "Megatron 3D"; Pope et al. (2022, MLSys); Qin et al. (2024) "Mooncake"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- Multi-node serving is needed when a model (or its KV/throughput needs) exceeds one node's GPUs — combine **TP within node + PP/EP across nodes + DP replicas**.
- The governing constraint is the **fabric hierarchy**: NVLink intra-node (fast) vs InfiniBand inter-node (~20× slower, higher latency) — keep heavy collectives on NVLink ([§01](../01_hardware/03_interconnects_nvlink_infiniband.md)).
- **Topology-aware placement** (TP/EP groups within nodes, PP across) is essential; mis-placement wrecks latency.
- Adds **failure modes** (node loss, network partitions) and **coordination** overhead vs single-node.
- Large MoE / very long context / huge dense models are the drivers; NVL72 pushes the "single node" boundary to 72 GPUs.

***

## Overview
When a model's weights plus KV cache won't fit on one node, or when throughput/availability demands exceed one node's capacity, serving spans multiple nodes. The architecture composes the parallelism axes ([§04](../04_parallelism/05_parallelism_strategy_selection.md)) with the fabric hierarchy as the hard constraint: **tensor parallelism and expert parallelism are communication-heavy and latency-sensitive, so they must stay within the fast NVLink/NVSwitch domain of a single node** (8 GPUs on HGX, up to 72 on NVL72); **pipeline parallelism and data-parallel replicas tolerate the slower inter-node InfiniBand** and are the tools for crossing node boundaries. A typical large deployment is therefore TP=8 within each node, PP across nodes to span the model, and DP replicas for throughput — total GPUs = DP×PP×TP.

The dominant design rule is **topology-aware placement**: align parallelism groups to the physical fabric. TP and EP groups must be co-located on the same NVSwitch; PP stage boundaries should fall on node boundaries so only the light activation hand-offs cross InfiniBand; DP replicas can be anywhere behind the router. Getting this wrong — e.g., a TP group split across two nodes — forces latency-critical all-reduces over InfiniBand and destroys decode ITL ([§04](../04_parallelism/00_tensor_parallelism.md)). NVL72 substantially eases this by making 72 GPUs a single NVLink domain, so TP/EP for large MoE and long-context can stay "intra-node" at much larger scale.

Multi-node also introduces **distributed-systems concerns** absent single-node: node and link failures, network partitions, coordination/consensus for the cluster state, and the operational complexity of orchestrating many GPUs. These are covered in fault tolerance ([§05](05_fault_tolerance_in_serving.md)) and routing ([§01](01_load_balancing_strategies.md)); here the focus is the parallelism×fabric mapping and placement. The drivers for going multi-node are large MoE models (expert parallelism across many GPUs), very long context (context parallelism), and very large dense models — all of which push beyond single-node capacity. This file covers the composition, placement, and the constraints.

***

## Core Concepts & Mechanics

### Composition across nodes
📐 Total GPUs = `DP × PP × TP` (× CP/EP as applicable). Mapping:
- **TP**: within node (NVLink), ≤ domain size.
- **EP**: within NVLink domain (all-to-all) ([§04](../04_parallelism/04_expert_parallelism_MoE.md)).
- **PP**: across nodes (point-to-point activations over IB).
- **CP**: long context, within/across as fabric allows ([§04](../04_parallelism/02_sequence_parallelism.md)).
- **DP**: replicas anywhere behind router.

### Topology-aware placement
- Co-locate TP/EP groups on one NVSwitch; place PP boundaries at node boundaries; verify NCCL uses NVLink intra-node, IB inter-node.
- 📐 Cross-node TP all-reduce latency (IB µs × layers) ≫ NVLink → avoid.

### NVL72 effect
72-GPU NVLink domain → TP/EP that previously needed multi-node IB now stays on NVLink; pushes the multi-node boundary out dramatically for large MoE/long context ([§01](../01_hardware/02_nvidia_h100_b200_architecture.md)).

### Coordination
A controller/cluster manager tracks instances, routes requests, handles scaling and failures; KV-centric designs (Mooncake) add a global KV pool across nodes ([§03](03_kv_cache_migration_and_transfer.md)).

***

## Key Challenges
1. **Fabric-aware placement.** Mis-mapping TP/EP across nodes (onto IB) destroys latency; placement must match topology.
2. **Inter-node latency for PP/CP.** Even light cross-node comm adds latency; pipeline bubbles and CP exchanges must be managed.
3. **Failure surface.** More nodes/links → more failure modes (node loss, partitions); needs fault tolerance ([§05](05_fault_tolerance_in_serving.md)).
4. **Coordination overhead.** Cluster state, routing, and scaling across nodes add operational complexity.

***

## Solutions & Current Best Practices
- **TP/EP within NVLink domain; PP/DP across nodes**; topology-aware placement ([§04](../04_parallelism/05_parallelism_strategy_selection.md)).
- **Use NVL72** to keep large-MoE/long-context collectives on NVLink.
- **Verify NCCL fabric usage** (NVLink intra, IB inter) via NCCL tests.
- **Global KV pool + cache-aware routing** for disaggregated multi-node ([§03](03_kv_cache_migration_and_transfer.md), [§01](01_load_balancing_strategies.md)).

***

## Implementation Notes
- Set TP/PP/EP/DP degrees to match node sizes; align PP boundaries to nodes.
- Configure NCCL topology and IB HCAs; confirm intra-node NVLink and inter-node IB.
- Plan for failures: replication, retries, draining ([§05](05_fault_tolerance_in_serving.md)).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **TP group split across nodes** — all-reduce over IB tanked ITL; keep TP within a node.
- **NCCL used IB intra-node** — misconfigured topology; verify NVLink is used inside nodes.
- **PP boundary mid-node** — heavy traffic crossed where it shouldn't; align boundaries to nodes.
- **No failure plan** — a single node loss took down a whole TP/PP group; need replication/recovery ([§05](05_fault_tolerance_in_serving.md)).

***

## Performance Numbers & Benchmarks
| Config | Placement rule |
|---|---|
| 70B, 1 node | TP=8 intra-node |
| 405B, 2 nodes | TP=8 intra + PP=2 across |
| DeepSeek MoE | EP+TP in NVL72 domain |
| 1M context | CP across GPUs + TP |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How do you serve a model across multiple nodes?"* — Expected: TP intra-node, PP/DP across, topology-aware placement.
- *"Why can't TP cross nodes?"* — Expected: all-reduce latency over IB; NVLink vs IB gap.
- *"What does NVL72 change for multi-node serving?"* — Expected: 72-GPU NVLink domain; large MoE/long context stay intra-domain.
- *"What new failure modes appear multi-node?"* — Expected: node/link loss, partitions; replication/recovery.

***

## Open Problems & Active Research (2025–2026)
- **Larger NVLink domains / better inter-node fabrics** (optical, UEC) for big MoE.
- **Topology-aware auto-placement** and elastic reconfiguration.
- **Global KV pools** spanning nodes as standard ([§03](03_kv_cache_migration_and_transfer.md)).

***

## References
- Narayanan, D., et al. (2021). "Megatron 3D parallelism." *SC 2021*. arXiv:2104.04473.
- Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference." *MLSys 2023*. arXiv:2211.05102.
- Qin, R., et al. (2024). "Mooncake." arXiv:2407.00079.
