# Data Parallelism for Serving (Replica-Level Scaling)

> **Section:** 04_parallelism
> **Last Updated:** June 2026
> **Related Files:** [00_tensor_parallelism.md](00_tensor_parallelism.md), [05_parallelism_strategy_selection.md](05_parallelism_strategy_selection.md), [../09_distributed_inference/01_load_balancing_strategies.md](../09_distributed_inference/01_load_balancing_strategies.md)
> **Must-Read Papers:** Crankshaw et al. (2017, NSDI) "Clipper"; Pope et al. (2022, MLSys) "Efficiently Scaling Transformer Inference"; industry serving practice
> **Estimated Study Time:** 20 minutes

***

## TL;DR
- Data parallelism (DP) for serving = run **multiple independent replicas** of the model, each handling a share of requests; scale throughput by adding replicas.
- Unlike training DP, **no gradient all-reduce** — replicas are fully independent at inference, so DP scales near-linearly and needs only a **load balancer**, not a fast fabric.
- DP is the **throughput/capacity** lever; TP is the **latency/memory-fit** lever. Combine: each replica is a TP (×PP) group.
- The interesting problems move to the **router/load balancer**: cache-aware routing, least-loaded, session affinity ([§09](../09_distributed_inference/01_load_balancing_strategies.md)).
- Expert/data parallelism for MoE is different (see [§04](04_expert_parallelism_MoE.md)).

***

## Overview
The simplest way to serve more requests per second is to run **more copies of the model** and spread traffic across them — replica-level **data parallelism**. At inference there are no gradients to synchronize, so unlike training DP (which all-reduces gradients every step), serving replicas are **completely independent**: each is a self-contained engine (possibly itself sharded with TP/PP) that processes its assigned requests with no inter-replica communication. This makes DP the cleanest, most linearly-scaling axis — double the replicas, roughly double the throughput — limited only by the load balancer and shared backing services (KV stores, routers).

Because replicas are independent, the engineering challenge shifts from the GPUs to the **router/load balancer** in front of them. Naive round-robin works for stateless, uniform traffic, but real LLM serving benefits enormously from smarter routing: **cache-aware routing** sends requests with shared prefixes to the replica holding that KV (preserving prefix-cache hit rates, [§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)); **least-loaded / least-outstanding** routing avoids piling onto a busy replica; **session affinity** keeps a multi-turn conversation on the replica caching its history. These policies are covered in [§09](../09_distributed_inference/01_load_balancing_strategies.md); here the point is that DP makes the *system* design a routing problem.

The clean mental model: **TP/PP determine how one replica is built** (to fit the model and hit latency targets), and **DP determines how many replicas you run** (to hit throughput/capacity and availability targets). A deployment is typically `DP × TP × PP` GPUs: e.g., 4 replicas × TP=8 = 32 GPUs serving a 70B model, each replica an 8-GPU NVLink group. Autoscaling adjusts the DP degree with load ([§13](../13_production_systems/02_autoscaling_and_capacity_planning.md)).

***

## Core Concepts & Mechanics

### DP vs TP/PP roles
- **DP (replicas)**: scales throughput/QPS and provides redundancy; near-linear; needs only routing.
- **TP**: scales down per-GPU memory and latency within a replica; needs NVLink.
- **PP**: scales model across nodes; capacity.
📐 Aggregate throughput ≈ `DP × per_replica_throughput`; total GPUs = `DP × TP × PP`.

### No gradient sync
Inference DP has **zero** inter-replica communication (no all-reduce), unlike training. The only shared state is optional (global KV store, prefix cache index, metrics) — none on the request critical path.

### The router
- **Round-robin**: simple, ignores load/cache.
- **Least-outstanding-requests**: send to the replica with fewest in-flight — good for heterogeneous request costs.
- **Cache-aware (prefix hash)**: route shared prefixes to the same replica for cache hits.
- **Session affinity**: sticky routing for multi-turn KV reuse.
The router must also handle health checks, retries, and replica draining ([§09](../09_distributed_inference/05_fault_tolerance_in_serving.md)).

### Capacity planning
Using Little's Law ([§00](../00_fundamentals/06_key_metrics_latency_throughput_cost.md)): in-flight `L = λW`; size DP so total concurrent capacity ≥ L with utilization margin (ρ≈0.5–0.7) for P99 protection.

***

## Key Challenges
1. **Routing vs load balance vs cache locality.** Cache-aware routing concentrates hot prefixes on one replica (hot spot), conflicting with even load balancing — must hybridize/replicate hot prefixes.
2. **Replica memory duplication.** Each replica holds a full model copy; DP multiplies weight memory (no sharing), so it's throughput-not-memory scaling.
3. **Autoscaling lag / cold start.** Adding replicas under load is gated by model load time (30–60s for 70B) — scaling can't react instantly ([§13](../13_production_systems/05_cold_start_and_model_loading.md)).
4. **Tail latency from imbalance.** Poor routing creates stragglers; least-loaded/affinity policies and headroom are needed.

***

## Solutions & Current Best Practices
- **DP for throughput/availability**, each replica a TP(×PP) group sized for latency/memory.
- **Cache-aware + least-loaded hybrid routing**; replicate hot prefixes to avoid hot spots ([§09](../09_distributed_inference/01_load_balancing_strategies.md)).
- **Autoscale DP degree** on load, pre-warming replicas to hide cold start ([§13](../13_production_systems/02_autoscaling_and_capacity_planning.md)).
- **Session affinity** for multi-turn KV reuse.

***

## Implementation Notes
- Front replicas with a router (e.g., a gateway, or SGLang/vLLM router) that supports cache-aware and least-loaded policies.
- Provision DP for **peak with margin**; use Little's Law for sizing.
- Keep replicas stateless except for transient KV; persist nothing per-replica that prevents draining/replacement.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Round-robin killed prefix-cache hit rate** — shared prefixes scattered across replicas; each rebuilt cache. Use cache-aware routing.
- **Cache-aware routing created a hot replica** — all hot-prefix traffic piled onto one; replicate hot prefixes or hybridize with load.
- **Autoscaling couldn't keep up with a spike** — 60s cold start meant new replicas arrived late; pre-warm / keep warm pool.
- **Counted DP as memory scaling** — DP duplicates weights; it scales throughput, not model size. Use TP/PP for memory.

***

## Performance Numbers & Benchmarks
| Axis | Scales | Comm | Memory effect |
|---|---|---|---|
| DP (replicas) | throughput/QPS, availability | none (just routing) | duplicates weights |
| TP | latency, per-GPU memory | all-reduce (NVLink) | shards weights |
| PP | model size/capacity | point-to-point | shards layers |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How does inference data parallelism differ from training DP?"* — Expected: no gradient all-reduce; independent replicas; near-linear.
- *"Design routing for a DP fleet with prefix caching."* — Expected: cache-aware + least-loaded hybrid; hot-prefix replication; session affinity.
- *"How do DP, TP, PP combine in a deployment?"* — Expected: DP×TP×PP GPUs; roles (throughput/latency/capacity).
- *"How do you size the number of replicas?"* — Expected: Little's Law, utilization margin, peak provisioning.

***

## Open Problems & Active Research (2025–2026)
- **Optimal hybrid routing** balancing cache locality and load at scale.
- **Fast autoscaling** that defeats cold start (snapshotting, fast weight loading) ([§13](../13_production_systems/05_cold_start_and_model_loading.md)).
- **Disaggregated DP** where prefill/decode replicas scale independently ([§03](03_prefill_decode_disaggregation.md)).

***

## References
- Crankshaw, D., et al. (2017). "Clipper: A Low-Latency Online Prediction Serving System." *NSDI 2017*.
- Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference." *MLSys 2023*. arXiv:2211.05102.
- Zheng, L., et al. (2024). "SGLang" (router/cache-aware). arXiv:2312.07104.
