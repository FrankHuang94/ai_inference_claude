# Kernel Profiling and Benchmarking

> **Section:** 06_kernel_optimization
> **Last Updated:** June 2026
> **Related Files:** [00_cuda_kernel_basics_for_inference.md](00_cuda_kernel_basics_for_inference.md), [../00_fundamentals/04_roofline_model_for_llm.md](../00_fundamentals/04_roofline_model_for_llm.md), [../13_production_systems/01_observability_and_profiling.md](../13_production_systems/01_observability_and_profiling.md)
> **Must-Read Papers:** NVIDIA Nsight Compute/Systems docs; Williams et al. (2009) "Roofline"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- **Nsight Systems** = timeline/whole-app view (kernel gaps, CPU-GPU overlap, launch overhead, comm); **Nsight Compute** = single-kernel deep dive (roofline, stalls, occupancy).
- First question for any kernel: **memory-bound or compute-bound?** Read Memory vs SM throughput / roofline overlay.
- Key Nsight Compute metrics: **achieved memory throughput, SM throughput, L2 hit rate, achieved occupancy, warp stall reasons**.
- Benchmark correctly: **warm up, fix clocks (lock GPU frequency), measure steady state, account for CUDA-graph/launch effects**.
- Profile **end-to-end** (Nsight Systems) to find gaps/comm, then **drill into** the hot kernel (Nsight Compute).

***

## Overview
Optimization without measurement is guesswork; the two NVIDIA tools structure the work. **Nsight Systems** gives a system-wide **timeline**: which kernels run when, gaps between them (launch overhead, CPU bottlenecks), CPU↔GPU overlap, memory transfers, and NCCL communication. It's how you discover *system-level* problems — the GPU idling between kernels (need CUDA graphs/overlap, [§02](02_fused_kernels_and_operator_fusion.md)), a scheduler not feeding the GPU ([§03](../03_batching_and_scheduling/01_iteration_level_scheduling.md)), or all-reduce dominating ([§04](../04_parallelism/00_tensor_parallelism.md)). **Nsight Compute** then drills into a single kernel: it reports whether the kernel is memory- or compute-bound, overlays it on the **roofline**, and breaks down warp stall reasons, occupancy, and cache behavior — telling you *why* a kernel is slow and what to fix.

The disciplined workflow is **top-down**: profile end-to-end with Nsight Systems to find where time goes (which kernels, what gaps, what comm), then use Nsight Compute on the hottest kernel(s) to diagnose the bottleneck, classify it on the roofline ([§00](../00_fundamentals/04_roofline_model_for_llm.md)), and target the binding resource. A memory-bound kernel near its bandwidth roof is already optimal — chasing FLOPs there is wasted effort; a compute-bound kernel far below the compute roof has tiling/occupancy headroom. This classification-first habit, repeatedly tested in interviews, prevents the common mistake of optimizing the wrong dimension.

Benchmarking correctly is its own skill. GPUs boost/throttle clocks, so unlocked-clock measurements are noisy; you should **lock clocks**, **warm up** (first runs include compilation/cache-cold effects), measure **steady state** over many iterations, and be aware of **CUDA-graph vs eager** differences and launch overhead at small batch. For serving, also benchmark at realistic **batch sizes and sequence lengths** — a kernel optimal at batch=1 may be wrong at batch=64, and decode vs prefill are different regimes. This file covers the tools, key metrics, the roofline workflow, and benchmarking methodology.

***

## Core Concepts & Mechanics

### Nsight Systems (timeline)
- Shows kernel sequence, durations, **gaps** (idle GPU), CPU threads, memcpys, NCCL.
- Diagnoses: launch-overhead gaps → CUDA graphs; CPU-bound scheduler → overlap/async; comm-dominated → parallelism/fabric ([§01](../01_hardware/03_interconnects_nvlink_infiniband.md)).

### Nsight Compute (kernel)
Key sections/metrics:
- **GPU Speed Of Light**: Memory throughput % vs SM (compute) throughput % — whichever is near 100% is the bound.
- **Roofline**: overlays the kernel's AI and achieved perf vs the roofs ([§00](../00_fundamentals/04_roofline_model_for_llm.md)).
- **Occupancy**: achieved vs theoretical; warp scheduler utilization.
- **Memory chart**: L1/L2 hit rates, HBM throughput, bank conflicts.
- **Warp State / Stall reasons**: why warps stall (long-scoreboard = memory wait, barrier, etc.).

### The classification workflow
📐 1) Is Memory or SM throughput saturated? 2) Plot on roofline: left-of-ridge → memory-bound (reduce bytes: fusion, quantization, coalescing); right-of-ridge → compute-bound (better tiles, FP8, occupancy). 3) Check stalls to confirm (memory stalls vs compute/issue stalls).

### Benchmarking methodology
- **Lock clocks** (`nvidia-smi -lgc`) for repeatability.
- **Warm up** (discard first iters: JIT, cache, autotune).
- **Steady state** over many iterations; report median + percentiles.
- **Realistic shapes**: benchmark decode (small M) and prefill (large M) separately; multiple batch sizes.
- **Account for CUDA graphs**: eager vs graph timing differ; measure the deployed path.

***

## Key Challenges
1. **Misclassifying the bound.** Optimizing FLOPs on a memory-bound kernel (or vice versa) wastes effort; the roofline/SOL section must guide.
2. **Noisy measurements.** Unlocked clocks, cold caches, and launch overhead corrupt benchmarks; methodology matters.
3. **Microbenchmark ≠ production.** A kernel fast in isolation may behave differently under real batching, memory pressure, and L2 contention.
4. **Overhead of profiling.** Nsight Compute serializes/instruments kernels (slow); profile representative kernels, not the whole run, at full detail.

***

## Solutions & Current Best Practices
- **Top-down**: Nsight Systems for the timeline, then Nsight Compute for the hot kernel.
- **Classify on the roofline first**, then optimize the binding resource.
- **Rigorous benchmarking**: lock clocks, warm up, steady state, realistic shapes, deployed path (graphs).
- **Re-profile per generation/precision** — the bound can move (A100→H100, FP16→FP8) ([§01](../01_hardware/01_memory_hierarchy_HBM_SRAM.md)).

***

## Implementation Notes
- Use `ncu` (Nsight Compute CLI) on specific kernels with `--set roofline`; `nsys` for timelines.
- Compare against theoretical: achieved HBM throughput vs ~70–90% of peak tells you coalescing headroom.
- For serving, profile under **load** (real batch/QPS), not a single request.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **"Kernel at 25% of peak FLOPs, must be slow"** — if memory-bound (left of ridge), it may be at 100% of the bandwidth roof and optimal. Read the roofline, not just FLOP%.
- **Benchmark numbers unrepeatable** — clocks unlocked / no warmup; lock clocks, discard warmup, measure steady state.
- **Microbenchmark fast, prod slow** — L2 contention and memory pressure under real batching changed behavior; profile under load.
- **Bound changed after FP8** — re-profile per precision; FP8 moves the ridge ([§05](../05_quantization/04_fp8_inference_h100.md)).

***

## Performance Numbers & Benchmarks
| Tool | Scope | Use |
|---|---|---|
| Nsight Systems (`nsys`) | timeline/app | gaps, overlap, comm, launch overhead |
| Nsight Compute (`ncu`) | single kernel | roofline, stalls, occupancy, cache |
| `nvidia-smi`/DCGM | coarse | utilization, power, clocks |
| NCCL tests | collectives | verify NVLink/IB bandwidth |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How do you tell if a kernel is memory- or compute-bound in Nsight Compute?"* — Expected: SOL Memory vs SM throughput, roofline overlay, stall reasons.
- *"Walk through profiling a slow vLLM serving config."* — Expected: nsys timeline for gaps/comm/scheduler, then ncu on the hot kernel, classify, fix.
- *"How do you benchmark a kernel correctly?"* — Expected: lock clocks, warmup, steady state, realistic shapes, deployed path.
- *"A kernel is at 30% of peak TFLOPS — optimized or not?"* — Expected: depends on AI vs ridge; could be bandwidth-saturated.

***

## Open Problems & Active Research (2025–2026)
- **Automated bottleneck diagnosis** and optimization suggestion from profiles.
- **Production-accurate profiling under load** without heavy instrumentation overhead.
- **Cross-generation roofline tooling** for FP8/FP4 and disaggregated systems.

***

## References
- NVIDIA. "Nsight Compute" and "Nsight Systems" documentation.
- Williams, S., et al. (2009). "Roofline." *CACM* 52(4).
- Yuan, Z., et al. (2024). "LLM Inference Unveiled: Roofline." arXiv:2402.16363.
