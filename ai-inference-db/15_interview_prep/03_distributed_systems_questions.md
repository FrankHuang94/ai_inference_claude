# Distributed Systems Questions (Q&A)

> **Section:** 15_interview_prep
> **Last Updated:** June 2026
> **Related Files:** [../04_parallelism/05_parallelism_strategy_selection.md](../04_parallelism/05_parallelism_strategy_selection.md), [../09_distributed_inference/02_disaggregated_prefill_decode_systems.md](../09_distributed_inference/02_disaggregated_prefill_decode_systems.md), [../01_hardware/03_interconnects_nvlink_infiniband.md](../01_hardware/03_interconnects_nvlink_infiniband.md)
> **Must-Read Papers:** Shoeybi et al. (2019) "Megatron-LM"; Zhong et al. (2024) "DistServe"; Dean & Barroso (2013) "Tail at Scale"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- Anchor in the **fabric hierarchy** (NVLink intra-node ~20× faster than IB inter-node) and **map parallelism to it**: TP/EP intra-node, PP/DP cross-node.
- Know **collective costs** (all-reduce for TP, all-to-all for EP, KV transfer for disaggregation) and why they constrain placement.
- Be ready on **disaggregation** (P/D ratio, KV transfer, goodput), **routing** (cache-aware vs load), and **fault tolerance** (group-level failure, tail latency).
- Quantify: KV transfer time, all-reduce volume, replica sizing (Little's Law).

***

## Q1: Why must tensor parallelism stay within a node?
TP injects ~2 all-reduces per layer **on the per-token critical path**. All-reduce latency is dominated by the slowest link. NVLink (~900 GB/s, sub-µs) vs InfiniBand (~50 GB/s, ~1–2µs latency) → cross-node TP multiplies hundreds of collectives by IB latency, destroying decode ITL. So keep TP ≤ NVLink domain (8 on HGX, 72 on NVL72); scale across nodes with PP (point-to-point, latency-tolerant) and DP (no comm). ([§04](../04_parallelism/00_tensor_parallelism.md), [§01](../01_hardware/03_interconnects_nvlink_infiniband.md))

## Q2: How do TP, PP, DP, EP compose, and how do you choose degrees?
**TP** (latency/fit, NVLink), **PP** (cross-node scale, light comm), **DP** (throughput/availability, no comm), **EP** (MoE experts, all-to-all, NVLink). Total GPUs = DP×PP×TP. Choose by priority: **fit** (min TP×PP for weights+peak KV) → **latency** (raise TP within NVLink) → **throughput** (DP replicas via Little's Law) → **scale** (PP across nodes; EP for MoE). Minimize TP/PP (overhead); scale with cheap DP. ([§04](../04_parallelism/05_parallelism_strategy_selection.md))

## Q3: Estimate interconnect bandwidth for KV transfer in disaggregation.
Per-request KV (70B@4k) ≈ 1.25GB. Over NDR IB (~50 GB/s): ~25ms — comparable to a long prefill, so must **overlap** (stream layer-by-layer) or it adds to TTFT. Aggregate = req/s × KV_bytes; a busy prefill GPU can emit tens of GB/s → needs **GPUDirect RDMA** and possibly KV compression. If undersized/un-overlapped, disaggregation *raises* TTFT and loses to chunked-prefill colocation. ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md))

## Q4: When does disaggregation beat colocation, and when not?
**Beats** at scale, with good interconnect, and **skewed P/D ratios** (long prompts/short outputs or long outputs/reasoning) + tight dual SLOs — DistServe reports up to ~4–7× goodput by independently optimizing each phase. **Loses** for short prompts (little interference, transfer overhead not worth it), limited interconnect, or small scale — there **chunked-prefill colocation** is simpler and competitive. ([§09](../09_distributed_inference/02_disaggregated_prefill_decode_systems.md), [§03](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md))

## Q5: Design load balancing for a DP fleet with prefix caching.
LLM routing must be **cache-aware** (route shared prefixes to the instance holding their KV → prefix-cache hits), not just load-aware. But pure cache routing creates **hot spots**. Solution: **hybrid** — prefer the prefix-owning instance if its load < threshold, else least-loaded; **replicate hot prefixes** across instances. Use **in-flight tokens** as the load metric (not connections — output length varies). Add **session affinity** for multi-turn. ([§09](../09_distributed_inference/01_load_balancing_strategies.md))

## Q6: What fails when a GPU dies in a TP group, and how do you handle it?
A TP/PP group spans multiple GPUs cooperating per token, so losing **one** GPU kills the **whole group/replica** — you lose a replica, not 1/8 of it. Handle via: **DP redundancy** (other replicas absorb load) with **capacity headroom** (ρ<1) to avoid cascading overload; **health checks + draining**; **idempotent retries** (controlled RNG) on another replica; **fast group recovery**. For long reasoning requests, **KV checkpoint/migration** to avoid losing minutes of work. ([§09](../09_distributed_inference/05_fault_tolerance_in_serving.md))

## Q7: How do you bound P99 latency against slow instances?
"Tail at scale" (Dean & Barroso): with many instances, the probability some are slow grows. Use **hedged/redundant requests** (send to a 2nd replica, take the first response) and tight timeouts to bound P99, at extra compute cost. Provision **headroom** so failures don't overload survivors. Kill **prefill-decode interference** (chunked prefill/disaggregation) — the top P99 ITL cause. ([§09](../09_distributed_inference/05_fault_tolerance_in_serving.md), [§00](../00_fundamentals/02_prefill_vs_decode_phases.md))

***

## Interview Angles
> 💡 **Delivery tips:**
- Lead with the **fabric hierarchy** and **map parallelism to it**.
- Quantify collective/transfer costs; show why placement matters.
- For disaggregation, give **both** the win conditions and when colocation is better.
- For routing/fault-tolerance, name the **LLM-specific** twist (KV locality; group-level failure).

***

## Complexity Gotchas
> ⚠️ **Common mistakes:**
- Crossing nodes with TP (IB latency kills ITL).
- Assuming disaggregation always wins (it doesn't for short prompts/limited fabric).
- Round-robin routing (destroys prefix-cache hits).
- Treating per-GPU failure as fractional (it's group-level).

***

## References
- Shoeybi et al. (2019) "Megatron-LM" arXiv:1909.08053; Zhong et al. (2024) "DistServe" arXiv:2401.09670.
- Dean & Barroso (2013) "The Tail at Scale" *CACM* 56(2); Qin et al. (2024) "Mooncake" arXiv:2407.00079.
