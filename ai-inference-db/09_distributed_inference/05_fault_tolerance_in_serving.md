# Fault Tolerance in Serving

> **Section:** 09_distributed_inference
> **Last Updated:** June 2026
> **Related Files:** [00_multi_node_serving.md](00_multi_node_serving.md), [01_load_balancing_strategies.md](01_load_balancing_strategies.md), [../13_production_systems/00_production_serving_architecture.md](../13_production_systems/00_production_serving_architecture.md)
> **Must-Read Papers:** classic distributed-systems (Lamport; Dean & Barroso 2013 "The Tail at Scale"); industry SRE practice
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- Inference is **stateful within a request** (KV cache) but **stateless across requests** — so fault tolerance focuses on **request-level retry/replication** and **fast instance recovery**, not durable per-request state.
- GPU/node failures, network partitions, and OOM are the main faults; **TP/PP groups fail as a unit** (one GPU loss kills the group).
- Techniques: **health checks + draining**, **request retries** (idempotent), **replica redundancy (DP)**, **request migration/replay**, **checkpointed KV** for long requests.
- **Tail-at-scale**: redundant/ hedged requests bound P99 against slow/failing instances.
- Long reasoning requests (minutes, huge KV) raise the stakes — losing one wastes large compute ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

***

## Overview
Distributed serving inherits the failure modes of any large GPU system — GPUs die, nodes reboot, links partition, processes OOM — and must keep meeting SLOs through them. The favorable property of inference is that it's **stateless across requests**: each request's only state is its transient KV cache, and there's no durable per-request data to protect. So fault tolerance is mostly about **gracefully handling in-flight requests** when an instance fails and **keeping the fleet serving** via redundancy, rather than the heavy state-replication machinery of databases. The unit of failure to reason about is the **parallel group**: a TP (or PP) group spans multiple GPUs cooperating on each token, so losing *one* GPU kills the *whole group* — you don't lose 1/8 of capacity, you lose the replica.

The standard toolkit: **health checks** detect failing instances; **draining** stops routing new requests to an instance being removed/replaced while letting in-flight ones finish; **request retries** re-run failed requests on another replica (safe because generation can be replayed from the prompt, though sampling RNG/determinism needs care); **data-parallel redundancy** ensures other replicas absorb load when one fails; and **request migration/replay** moves or restarts a request elsewhere. For very long requests (reasoning models generating for minutes with hundreds of GB of KV), losing the request wastes enormous compute, motivating **KV checkpointing** or migration so a node failure doesn't restart from scratch.

A related reliability concern is **tail latency under partial failure** — Dean & Barroso's "tail at scale": a few slow or degraded instances can dominate P99 even without hard failures. **Hedged/redundant requests** (send to two replicas, take the first response) and aggressive timeouts bound the tail at the cost of extra compute. Capacity planning must also leave **headroom** so a failure doesn't immediately overload survivors (cascading failure). This file covers the failure modes, the recovery techniques, and tail-tolerance.

***

## Core Concepts & Mechanics

### What's stateful
- **Within a request**: KV cache (transient, in HBM). Lost on instance failure → restart/replay the request.
- **Across requests**: stateless (model weights are static, loaded at startup). Caches (prefix/KV) are best-effort, rebuildable.

### Failure modes
- **GPU/node loss**: kills the TP/PP **group** (replica) it's part of.
- **Network partition**: isolates instances; router must detect and reroute.
- **OOM**: KV over-admission; triggers preemption or crash ([§02](../02_kv_cache/01_paged_attention_vllm.md)).
- **Slow/degraded instance**: tail-latency contributor without hard failure.

### Recovery techniques
- **Health checks + draining**: detect, stop new traffic, finish in-flight, replace.
- **Retries**: idempotent re-run on another replica (handle RNG/determinism).
- **DP redundancy + headroom**: survivors absorb load (provision ρ<1).
- **Request migration/replay** and **KV checkpointing** for long requests (avoid restarting minutes of work).

### Tail tolerance (Dean & Barroso)
📐 With N instances and per-instance slow-probability p, P(some slow) ≈ 1−(1−p)^N grows with N. **Hedged requests** (duplicate to a 2nd replica, take first) and tight timeouts bound P99 at extra compute cost.

***

## Key Challenges
1. **Group-level failure.** TP/PP groups fail as a unit; one GPU loss = whole replica down; recovery must restart/replace the group.
2. **Losing long-request work.** Reasoning requests (minutes, huge KV) are expensive to lose; checkpointing/migration is hard (KV size) ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).
3. **Cascading overload.** A failure shifts load to survivors; without headroom they overload, cascading.
4. **Retry determinism.** Replaying a request may produce different output (sampling RNG); user-visible inconsistency.

***

## Solutions & Current Best Practices
- **Health checks + draining + DP redundancy** with capacity **headroom** (ρ≈0.5–0.7) to absorb failures ([§13](../13_production_systems/02_autoscaling_and_capacity_planning.md)).
- **Idempotent retries** with controlled RNG; **hedged requests** for P99 tails (Dean & Barroso).
- **KV checkpointing / migration** for long reasoning requests where restart is costly.
- **Fast instance recovery** (quick model reload, warm pools) to restore lost groups ([§13](../13_production_systems/05_cold_start_and_model_loading.md)).

***

## Implementation Notes
- Router: health-check instances, drain on removal, retry on failure to another replica.
- Provision N+1 (or more) replicas so a loss doesn't breach SLO; alert on reduced redundancy.
- For long requests, consider periodic KV checkpoint or migration support; weigh against KV transfer cost ([§03](03_kv_cache_migration_and_transfer.md)).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **One GPU died, a whole replica went down** — TP/PP group fails as a unit; plan replica-level redundancy, not per-GPU.
- **Failure cascaded** — no headroom; survivors overloaded and fell over in turn. Provision ρ<1.
- **Retry produced a different answer** — sampling RNG not pinned; user saw inconsistency. Control determinism on retry.
- **Lost a 5-minute reasoning request** — no checkpoint; restarted from scratch, wasting compute and breaching SLO.

***

## Performance Numbers & Benchmarks
| Technique | Protects against | Cost |
|---|---|---|
| Health check + drain | bad instances | minimal |
| DP redundancy + headroom | instance loss | extra capacity |
| Hedged requests | tail latency | extra compute |
| KV checkpoint/migration | long-request loss | KV transfer/storage |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"What state must you protect in LLM serving, and what not?"* — Expected: transient KV within request; stateless across; caches rebuildable.
- *"What happens when one GPU in a TP group fails?"* — Expected: whole group/replica down; restart/replace, DP redundancy.
- *"How do you bound P99 against slow instances?"* — Expected: hedged/redundant requests, timeouts (tail at scale).
- *"How do you avoid cascading failures?"* — Expected: capacity headroom (ρ<1), draining, retries.

***

## Open Problems & Active Research (2025–2026)
- **Efficient long-request fault tolerance** (cheap KV checkpoint/migration) for reasoning/agentic workloads.
- **Fast group recovery** (sub-second replica restoration) via snapshotting/fast weight load.
- **Deterministic retries** preserving exact outputs across replicas.

***

## References
- Dean, J., Barroso, L. (2013). "The Tail at Scale." *CACM* 56(2).
- Lamport, L. Classic distributed-systems foundations (consensus, ordering).
- Industry SRE / production serving practice (health checks, draining, hedging).
