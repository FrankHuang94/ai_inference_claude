# Cold Start and Model Loading

> **Section:** 13_production_systems
> **Last Updated:** June 2026
> **Related Files:** [02_autoscaling_and_capacity_planning.md](02_autoscaling_and_capacity_planning.md), [00_production_serving_architecture.md](00_production_serving_architecture.md), [../14_company_deep_dives/04_modal_and_serverless_inference.md](../14_company_deep_dives/04_modal_and_serverless_inference.md)
> **Must-Read Papers:** industry serverless-GPU practice; Modal/Replicate engineering writeups; safetensors/loading docs
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- **Cold start** = time to make a new replica serve: download weights + load into GPU memory + warm up (CUDA graphs, compile) — **30–60+s for large models**.
- It defeats reactive autoscaling and is the core barrier to **serverless/scale-to-zero** GPU inference ([§02](02_autoscaling_and_capacity_planning.md)).
- Mitigations: **warm pools**, **fast weight loading** (safetensors, mmap, parallel/streamed load), **local SSD caching** of weights, **snapshotting** (memory/CUDA state), **scale-to-low not zero**.
- The breakdown: weight **download** (network) + **host→GPU** load + **warmup** (graphs/compile) — each optimizable.
- Serverless platforms (Modal, Replicate) live or die on cold-start engineering ([§14](../14_company_deep_dives/04_modal_and_serverless_inference.md)).

***

## Overview
Cold start is the latency to bring a new model replica from nothing to serving traffic, and for large models it's dominated by **moving and loading weights**. The phases: **download** the weights (tens to hundreds of GB) from object storage over the network; **load** them from host into GPU HBM; and **warm up** (allocate KV pools, capture CUDA graphs, JIT/compile kernels, run a dummy forward). For a 70B model this totals 30–60+ seconds, and it's the single biggest obstacle to elastic and serverless GPU serving — you simply cannot conjure capacity fast enough to absorb a traffic spike if each new replica needs a minute to start ([§02](02_autoscaling_and_capacity_planning.md)).

This shapes architecture profoundly. **Autoscaling must be predictive** (pre-warm before the spike) and use **warm pools** (spare loaded replicas ready to activate instantly), trading idle cost for responsiveness. **Serverless/scale-to-zero** GPU inference — appealing for cost (pay only when serving) — is fundamentally limited by cold start: the first request after scale-to-zero eats the full 30–60s, unacceptable for interactive use. Platforms like Modal and Replicate ([§14](../14_company_deep_dives/04_modal_and_serverless_inference.md)) have built their businesses substantially around **engineering cold start down** (and many compromise to "scale-to-low" rather than true zero for hot models).

The mitigations attack each phase. **Download**: cache weights on local NVMe (avoid re-downloading from object storage), use fast formats (**safetensors** with mmap), and parallel/streamed download. **Host→GPU load**: memory-map and stream weights directly to GPU (GPUDirect Storage), parallelize across shards, overlap with download. **Warmup**: pre-capture CUDA graphs and pre-compile (or use AOT-compiled engines like TRT-LLM, though those have their own build cost). **Architectural**: keep warm pools, scale-to-low not zero for hot models, and snapshot loaded process/memory state for fast restore. This file covers the cold-start breakdown, the mitigations, and the serverless tension.

***

## Core Concepts & Mechanics

### The phases (and where time goes)
```
download weights (network, object store → node)   [tens–hundreds GB]
   → load host → GPU HBM                            [PCIe/loading]
   → warmup: KV pool alloc, CUDA graph capture, JIT/compile, dummy forward
```
📐 70B FP16 ≈ 140 GB. Download at, say, ~5 GB/s ≈ 28s; load + warmup adds more → 30–60s+ total.

### Mitigations by phase
- **Download**: local NVMe weight cache, safetensors + mmap, parallel/streamed download, regional weight stores.
- **Load**: memory-map / stream to GPU, GPUDirect Storage, parallel shard load, overlap with download.
- **Warmup**: pre-capture CUDA graphs, pre-compile kernels (or AOT engine), cache compiled artifacts.

### Architectural strategies
- **Warm pools**: spare loaded replicas ready instantly (idle cost vs responsiveness).
- **Scale-to-low, not zero**: keep ≥1 hot replica for popular models; scale-to-zero only for cold/rare ones.
- **Snapshotting**: checkpoint loaded process/CUDA memory state for fast restore (advanced).
- **Predictive scaling**: pre-warm ahead of forecast load ([§02](02_autoscaling_and_capacity_planning.md)).

### Serverless tension
True scale-to-zero saves cost but the first post-zero request pays full cold start → unacceptable interactivity. Hence scale-to-low for hot models; serverless best for batch/rare/latency-tolerant ([§14](../14_company_deep_dives/04_modal_and_serverless_inference.md)).

***

## Key Challenges
1. **Magnitude.** 30–60s+ is far longer than any interactive SLO; can't be hidden reactively.
2. **Download dominance.** For large models, moving tens–hundreds of GB dominates; network/storage is the bottleneck.
3. **Serverless vs interactivity.** Scale-to-zero's cost appeal conflicts with cold-start latency for interactive traffic.
4. **Warmup correctness.** Skipping warmup (CUDA graphs/compile) means the *first real* requests are slow/variable.

***

## Solutions & Current Best Practices
- **Warm pools + predictive scaling**; **scale-to-low not zero** for hot models ([§02](02_autoscaling_and_capacity_planning.md)).
- **Local NVMe weight caching + safetensors/mmap + parallel/streamed load** (and GPUDirect Storage).
- **Pre-capture CUDA graphs / pre-compile**; cache compiled artifacts.
- **Snapshotting** for fast restore where supported.

***

## Implementation Notes
- Cache weights on node-local NVMe; avoid re-pulling from object storage per start.
- Use safetensors (zero-copy mmap) and parallel shard loading across the model's files.
- Pre-warm CUDA graphs for expected batch sizes before marking the replica ready.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Scale-to-zero caused 60s first-request latency** — cold start; use scale-to-low/warm pool for interactive.
- **Re-downloaded weights every start** — no local cache; pull from object store each time. Cache on NVMe.
- **First requests slow after "ready"** — warmup (CUDA graphs/compile) skipped; pre-warm before serving.
- **Autoscaler assumed instant capacity** — ignored cold start; predictive + warm pool needed ([§02](02_autoscaling_and_capacity_planning.md)).

***

## Performance Numbers & Benchmarks
| Phase | 70B model (approx) | Mitigation |
|---|---|---|
| Download (no cache) | ~30s+ | local NVMe cache |
| Host→GPU load | ~seconds–10s | mmap, parallel, GPUDirect |
| Warmup | ~seconds | pre-capture graphs/compile |
| Total cold start | 30–60s+ | warm pool / scale-to-low |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why is cold start a big deal in LLM serving, and what dominates it?"* — Expected: 30–60s; weight download/load dominates; defeats reactive scaling.
- *"How do you make a serverless GPU platform feel fast?"* — Expected: warm pools, scale-to-low, fast loading, snapshotting.
- *"How would you cut model load time?"* — Expected: local NVMe cache, safetensors/mmap, parallel/GPUDirect load, pre-warm graphs.
- *"Why can't you just scale-to-zero?"* — Expected: first-request cold-start latency; OK only for batch/rare/latency-tolerant.

***

## Open Problems & Active Research (2025–2026)
- **Sub-second cold start** (snapshot/restore, streaming load) for true elasticity.
- **Shared/streamed weight serving** across replicas to cut per-replica download.
- **Cold-start-aware autoscaling** that models load time in scaling decisions ([§02](02_autoscaling_and_capacity_planning.md)).

***

## References
- Modal / Replicate engineering writeups on cold start and serverless GPU.
- safetensors and GPUDirect Storage documentation.
- Industry serverless-GPU practice.
