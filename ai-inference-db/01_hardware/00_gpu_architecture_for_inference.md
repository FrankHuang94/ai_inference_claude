# GPU Architecture for Inference

> **Section:** 01_hardware
> **Last Updated:** June 2026
> **Related Files:** [01_memory_hierarchy_HBM_SRAM.md](01_memory_hierarchy_HBM_SRAM.md), [02_nvidia_h100_b200_architecture.md](02_nvidia_h100_b200_architecture.md), [../00_fundamentals/03_memory_bandwidth_bound_compute.md](../00_fundamentals/03_memory_bandwidth_bound_compute.md)
> **Must-Read Papers:** NVIDIA Hopper/Blackwell whitepapers; Jouppi et al. (2017, ISCA) "TPU" (for contrast); Dao et al. (2022, NeurIPS) "FlashAttention"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- A GPU is a throughput machine: **many SMs (streaming multiprocessors)**, each running **warps of 32 threads** in SIMT lock-step, fed by a deep memory hierarchy.
- **Tensor Cores** (matrix-multiply units, FP16/BF16/FP8/FP4) provide the bulk of FLOPs; **CUDA cores** handle general/elementwise work. Tensor Cores are the throughput engine.
- **High SM utilization ≠ efficient.** A decode kernel can show ~100% SM-busy while doing almost no useful FLOPs because it's stalled on HBM — **MFU** is the real efficiency measure.
- **More VRAM ≠ faster inference**; capacity lets you hold bigger models/batches/KV, but **bandwidth** sets decode latency.
- Inference engineering is about **feeding Tensor Cores** (prefill) and **not starving on HBM** (decode).

***

## Overview
NVIDIA datacenter GPUs are organized as an array of **streaming multiprocessors (SMs)** — 132 on an H100 SXM5 — each containing CUDA cores, Tensor Cores, register file, L1/shared memory, and warp schedulers. Work is dispatched as a **grid of thread blocks**; each block is assigned to an SM and executed as **warps** of 32 threads in **SIMT** (single-instruction, multiple-thread) fashion. Threads in a warp ideally execute the same instruction on different data; when they diverge (different branches), the warp serializes the paths (**warp divergence**), wasting throughput. Memory accesses by a warp should be **coalesced** into contiguous transactions to use HBM efficiently.

For inference the two structures that matter most are **Tensor Cores** and the **memory hierarchy**. Tensor Cores execute small matrix-multiply-accumulate (MMA) operations natively and deliver nearly all the model's FLOPs; they support progressively lower precisions (FP16/BF16 → FP8 on Hopper → FP4 on Blackwell), each doubling peak throughput. But as established in [§00](../00_fundamentals/03_memory_bandwidth_bound_compute.md), decode rarely reaches the Tensor Cores' potential because it is bandwidth-bound — the SMs spend their time waiting for weights to arrive from HBM.

The crucial mental correction for newcomers from training: **GPU "utilization" as reported by `nvidia-smi` (SM busy %) is not efficiency.** A decode kernel issuing loads and waiting can read as fully utilized while achieving <10% MFU. The right lens is the roofline and MFU, and the right inference goals are: keep Tensor Cores fed during compute-bound prefill, and minimize/maximize-bandwidth-of HBM traffic during memory-bound decode.

***

## Core Concepts & Mechanics

### The SM and the SIMT model
- **Warp** = 32 threads, the scheduling unit. Each SM has multiple warp schedulers issuing instructions per cycle, hiding latency by switching among many resident warps (**latency hiding** via high occupancy).
- **Occupancy** = resident warps / max warps per SM. Higher occupancy hides memory latency but is limited by registers/shared memory per thread. For memory-bound decode, occupancy helps hide HBM latency; for compute-bound GEMM, register-heavy tiles may *lower* occupancy yet still be optimal.
- **Warp divergence**: branches (e.g., attention masking, variable sequence handling) that split a warp serialize execution — relevant in attention/sampling kernels.

### Tensor Cores vs CUDA cores
- **Tensor Cores** perform `D = A·B + C` on small tiles (e.g., 16×16) per instruction (WMMA/`wgmma` on Hopper). They are the FLOP engine: H100 ~989 TFLOP/s FP16, ~1979 FP8.
- **CUDA cores** do scalar/vector FP32/INT ops: norms, activations, RoPE, sampling, dequantization.
- Inference kernels route the big matmuls (QKV, FFN, LM head) to Tensor Cores and the glue to CUDA cores; fusion keeps the glue from round-tripping HBM ([§06](../06_kernel_optimization/02_fused_kernels_and_operator_fusion.md)).

### Utilization vs MFU
📐 `MFU = achieved_FLOP/s / peak_FLOP/s = (2·N·tokens/s) / peak_FLOP/s` (inference forward). SM-busy% measures whether SMs are *issuing/stalled*, not whether they're doing useful matmul. Decode: SM-busy can be high (issuing loads, waiting) while MFU <10%. **Use MFU + tokens/s/GPU, not SM%.**

### Memory capacity vs bandwidth
- **Capacity** (80 GB H100, 141 GB H200, 192 GB MI300X) determines whether weights + KV + activations fit, hence max model size and batch.
- **Bandwidth** (3.35 / 4.8 / 5.3 TB/s) determines decode latency.
- A bigger-VRAM GPU at the same bandwidth lets you batch more (improving throughput) but does not lower single-stream decode latency. H200 helps decode because it adds *both* capacity and bandwidth.

***

## Key Challenges
1. **Feeding Tensor Cores during decode is impossible at small batch.** The hardware's FLOP capacity vastly exceeds what bandwidth can supply per token, so most of the chip is idle — a structural inefficiency, not a tuning bug.
2. **Occupancy/register tradeoffs are non-obvious.** High-performance GEMM uses large register tiles that reduce occupancy; naive "maximize occupancy" advice can hurt compute-bound kernels.
3. **Warp divergence and uncoalesced access in attention** kernels (masking, paged/non-contiguous KV) erode achievable bandwidth.
4. **Precision support varies by generation.** FP8 needs Hopper+, FP4 needs Blackwell; portable kernels must fall back, complicating deployment across mixed fleets ([§09](../09_distributed_inference/04_heterogeneous_cluster_inference.md)).

***

## Solutions & Current Best Practices
- **FlashAttention-class kernels** to keep attention tiles in SRAM and coalesce KV reads ([§06](../06_kernel_optimization/01_flash_attention_deep_dive.md)).
- **CUDA graphs** to remove per-kernel launch overhead at small-batch decode ([§06](../06_kernel_optimization/02_fused_kernels_and_operator_fusion.md)).
- **FP8 GEMM via Transformer Engine** to double prefill throughput on Hopper/Blackwell ([§05](../05_quantization/04_fp8_inference_h100.md)).
- **Profile with Nsight Compute's roofline** to know whether to chase bandwidth or FLOPs ([§06](../06_kernel_optimization/05_kernel_profiling_and_benchmarking.md)).

***

## Implementation Notes
- For decode, target **CUDA-graph-captured, fused** steps; launch overhead can be 10–30% of a sub-millisecond step otherwise.
- Choose **TP degree ≤ NVLink domain** so all-reduces stay on NVLink, not PCIe/IB ([§04](../04_parallelism/00_tensor_parallelism.md)).
- Check that paged-attention block size yields **coalesced** KV reads; tiny blocks fragment memory transactions.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **`nvidia-smi` shows 100% util but throughput is low** — the chip is stalled on HBM, not computing. Diagnose with MFU/roofline, not SM%.
- **Bought a 192 GB GPU expecting lower latency, got the same ITL** — capacity ≠ bandwidth; you gained batch headroom, not single-stream speed.
- **A kernel got slower at "higher occupancy"** — you spilled registers to local memory or reduced tile size; compute-bound GEMM prefers fat tiles over max occupancy.
- **FP8 kernel silently fell back to FP16 on an A100** — A100 has no FP8 Tensor Cores; portability bugs cause perf regressions on mixed fleets.

***

## Performance Numbers & Benchmarks
| GPU | SMs | FP16 TFLOP/s | FP8 TFLOP/s | HBM | BW |
|---|---|---|---|---|---|
| A100 80GB | 108 | 312 | — (no FP8) | 80 GB HBM2e | 2.0 TB/s |
| H100 SXM5 | 132 | 989 | 1979 | 80 GB HBM3 | 3.35 TB/s |
| H200 | 132 | 989 | 1979 | 141 GB HBM3e | 4.8 TB/s |
| B200 | — | ~2.2 PF (FP16-eq via FP8 path varies) | ~4.5 PF FP8 / ~9 PF FP4 | 192 GB HBM3e | ~8 TB/s |

(Dense, non-sparse figures; Blackwell numbers vary by SKU and sparsity.)

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"What's the difference between SM utilization and MFU, and why does it matter for decode?"* — Expected: busy≠useful; decode MFU intrinsically low.
- *"Why doesn't more VRAM make a single request decode faster?"* — Expected: bandwidth, not capacity, sets decode latency.
- *"What is warp divergence and where does it show up in attention/sampling?"* — Expected: branch serialization; masking, variable-length, top-p.
- *"How do Tensor Cores differ from CUDA cores and which ops use which?"* — Expected: MMA tiles for GEMM; CUDA cores for norms/activations/sampling.

***

## Open Problems & Active Research (2025–2026)
- **Keeping Blackwell's huge FLOP capacity fed** for decode — the FLOP/bandwidth gap widens each generation, deepening the memory-bound problem.
- **Kernel portability across precisions/generations** (FP8/FP4 fallbacks) in heterogeneous fleets.
- **Architectural support for the decode regime** (larger on-chip SRAM, smarter prefetch) vs. relying on HBM scaling.

***

## References
- NVIDIA (2022). "NVIDIA H100 Tensor Core GPU Architecture" whitepaper.
- NVIDIA (2024). "NVIDIA Blackwell Architecture" whitepaper.
- Jouppi, N., et al. (2017). "In-Datacenter Performance Analysis of a Tensor Processing Unit." *ISCA 2017*.
- Dao, T., et al. (2022). "FlashAttention." *NeurIPS 2022*. arXiv:2205.14135.
