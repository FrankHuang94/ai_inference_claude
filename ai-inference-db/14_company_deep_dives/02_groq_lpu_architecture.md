# Groq LPU Architecture

> **Section:** 14_company_deep_dives
> **Last Updated:** June 2026
> **Related Files:** [../01_hardware/04_alternative_accelerators.md](../01_hardware/04_alternative_accelerators.md), [07_company_comparison_matrix.md](07_company_comparison_matrix.md), [../00_fundamentals/03_memory_bandwidth_bound_compute.md](../00_fundamentals/03_memory_bandwidth_bound_compute.md)
> **Must-Read Papers:** Abts et al. (2020, ISCA) "Think Fast: A Tensor Streaming Processor"; Abts et al. (2022) "A Software-defined Tensor Streaming Multiprocessor"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- Groq's **LPU** (Language Processing Unit), built on the **TSP** (Tensor Streaming Processor), is **SRAM-only (no HBM)** with **deterministic, compiler-scheduled execution**.
- This eliminates the memory-bandwidth bottleneck ([§00](../00_fundamentals/03_memory_bandwidth_bound_compute.md)) and memory stalls → **record low latency / high single-stream tokens/s** (hundreds of tok/s, sub-ms/token).
- Catch: **SRAM is small** (~230 MB/chip) → a model needs **many chips**, constraining model/batch size and making **throughput/$ for large models harder**.
- The compiler **statically schedules every instruction and data movement** (no dynamic caches/branch prediction) → predictable, stall-free timing.
- Optimal for **latency-critical, smaller-model, small-batch** serving; a different point in the design space than GPUs.

***

## Overview
Groq's LPU is the most architecturally distinct accelerator in mainstream inference, and a favorite interview contrast to GPUs. Its core bet is to **eliminate the memory-bandwidth bottleneck** that dominates GPU decode ([§00](../00_fundamentals/03_memory_bandwidth_bound_compute.md)) by having **no HBM at all** — the model lives entirely in **on-chip SRAM**, distributed across many chips. Because SRAM bandwidth is enormous (orders of magnitude above HBM), weights stream to the compute units without the memory stalls that leave GPUs idle during decode, yielding **record single-stream latency** (sub-millisecond per token, hundreds of tokens/second) on supported models.

The second pillar is the **Tensor Streaming Processor (TSP)** architecture and its **deterministic, compiler-scheduled** execution model (Abts et al. 2020). Unlike a GPU's dynamic SIMT scheduling, caches, and branch prediction, the TSP has **no dynamic dispatch**: the **compiler statically schedules every instruction and every data movement cycle-by-cycle** at build time, so execution is fully deterministic with no stalls, no cache misses, and no contention. This determinism is what delivers Groq's signature predictable low latency — there's simply nothing to stall on. The architecture streams tensors through a spatial array of functional units in a precisely choreographed dataflow.

The fundamental **tradeoff** is capacity. SRAM is small (~230 MB/chip), so a large model must be partitioned across **many LPUs** (a rack-scale system for a 70B model), which complicates the economics: Groq excels at **latency** but **throughput-per-dollar for large models/high batch** is harder than HBM-based GPUs, because you're amortizing many expensive SRAM chips. So Groq occupies a specific niche — **latency-critical applications with smaller/medium models and small batches** — rather than being a general GPU replacement. This file covers the SRAM-only/deterministic design, why it's fast, the capacity tradeoff, and the niche; see [§01](../01_hardware/04_alternative_accelerators.md) for the broader accelerator context.

***

## Core Concepts & Mechanics

### SRAM-only design
- No HBM; model weights live in **on-chip SRAM** (~230 MB/chip) distributed across many chips. On-chip bandwidth ~PB/s-class → no memory-bandwidth bottleneck.
- 📐 Recall GPU decode ≈ `HBM_BW / weight_bytes` ([§00](../00_fundamentals/01_autoregressive_decoding.md)); Groq replaces HBM_BW with vastly higher SRAM bandwidth → far higher single-stream tok/s.

### TSP / deterministic execution
- **Compiler statically schedules** all instructions and data movement cycle-by-cycle; no dynamic dispatch, caches, or branch prediction.
- Result: **deterministic, stall-free** timing → predictable ultra-low latency. (Abts et al. 2020, 2022.)
- Spatial dataflow: tensors stream through functional units in a choreographed pipeline.

### Why fast (latency)
No memory stalls (SRAM) + no scheduling stalls (deterministic) → sub-ms/token, hundreds of tok/s single-stream — beating GPUs on latency for supported models.

### Capacity tradeoff
📐 ~230 MB/chip SRAM ≪ a 70B model (140 GB) → **needs ~hundreds of chips**. Many expensive chips per model → **throughput/$** for large models/high batch is harder than HBM GPUs. Latency play, not throughput/cost play.

### Products
GroqCloud API; supported models (Llama-3, Mixtral, etc.); per-token pricing emphasizing speed.

***

## Key Challenges
1. **SRAM capacity.** Small per-chip SRAM forces many chips per model, constraining size/batch and complicating $/token for large models.
2. **Throughput/$ for large models.** Latency is excellent, but cost-per-token at scale/high-batch is harder than HBM GPUs.
3. **Compiler dependence.** Performance hinges on the static compiler; dynamic shapes (variable seq lengths, continuous batching) are handled differently than GPUs.
4. **Model support breadth.** Bringing up new/large models on the deterministic compiler is more involved than on CUDA.

***

## Solutions & Current Best Practices (where Groq fits)
- **Latency-critical, smaller/medium models, small batch** (real-time voice, low-latency agents).
- **Premium latency tier** in a heterogeneous fleet (route latency-sensitive traffic to Groq) ([§09](../09_distributed_inference/04_heterogeneous_cluster_inference.md)).
- Not the default for **throughput/cost-optimized large-model** serving (use HBM GPUs there).

***

## Implementation Notes (for interview prep)
- Frame Groq via the bandwidth bound: SRAM removes the HBM bottleneck → latency win; capacity is the cost.
- Contrast deterministic compiler scheduling vs GPU dynamic SIMT/caches.
- Know it's a **latency** architecture, not a throughput/$ one for large models.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Amazing latency ≠ cheap throughput** — many chips per large model; throughput/$ can trail GPUs. Latency play.
- **Large models need a rack** — SRAM capacity; not a drop-in single-GPU replacement.
- **Dynamic batching differs** — deterministic compiler model handles variable workloads differently than GPU continuous batching.
- **New-model bring-up slower** — compiler-centric vs CUDA ecosystem.

***

## Performance Numbers & Benchmarks
| Aspect | Groq LPU |
|---|---|
| Memory | SRAM-only (~230 MB/chip), no HBM |
| Execution | deterministic, compiler-scheduled (TSP) |
| Latency | sub-ms/token, hundreds tok/s single-stream |
| Tradeoff | many chips/large model → throughput/$ harder |
| Niche | latency-critical, smaller models, small batch |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why can Groq achieve sub-millisecond per-token latency?"* — Expected: SRAM-only (no memory-bandwidth bottleneck) + deterministic compiler scheduling (no stalls).
- *"What's the catch with Groq's design?"* — Expected: SRAM capacity → many chips/large model → throughput/$ harder; latency play.
- *"Contrast TSP deterministic execution with GPU SIMT."* — Expected: static cycle-level scheduling vs dynamic dispatch/caches.
- *"When would you deploy Groq?"* — Expected: latency-critical small/medium-model serving; premium latency tier.

***

## Open Problems & Active Research (2025–2026)
- **Scaling SRAM-centric designs** to large MoE / long context economically.
- **Throughput/$ competitiveness** vs HBM GPUs for large models.
- **Compiler/ecosystem maturity** for rapid new-model support ([§01](../01_hardware/04_alternative_accelerators.md)).

***

## References
- Abts, D., et al. (2020). "Think Fast: A Tensor Streaming Processor (TSP)." *ISCA 2020*.
- Abts, D., et al. (2022). "A Software-defined Tensor Streaming Multiprocessor for Large-scale ML." *ISCA 2022*.
- Groq technical materials / GroqCloud documentation.
