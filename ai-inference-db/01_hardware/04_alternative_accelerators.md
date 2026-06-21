# Alternative Accelerators: Groq, TPU, Trainium, Gaudi, MI300X, Cerebras

> **Section:** 01_hardware
> **Last Updated:** June 2026
> **Related Files:** [02_nvidia_h100_b200_architecture.md](02_nvidia_h100_b200_architecture.md), [../14_company_deep_dives/02_groq_lpu_architecture.md](../14_company_deep_dives/02_groq_lpu_architecture.md), [../00_fundamentals/03_memory_bandwidth_bound_compute.md](../00_fundamentals/03_memory_bandwidth_bound_compute.md)
> **Must-Read Papers:** Abts et al. (2020, ISCA) "Think Fast: A Tensor Streaming Processor" (Groq); Jouppi et al. (2017/2021, ISCA) "TPU"/"Ten Lessons"; AMD CDNA3/MI300 disclosures
> **Estimated Study Time:** 35 minutes

***

## TL;DR
- **Groq LPU**: SRAM-only, compiler-scheduled, deterministic execution → record **low latency / high single-stream tok/s** for smaller models; **limited by SRAM capacity** (many chips per model, throughput/$ harder).
- **Google TPU (v5e/v5p/v6 Trillium)**: systolic-array, great at large-batch dense matmul, tight XLA integration; different programming model than CUDA.
- **AWS Trainium2/Inferentia2**: cost-focused AWS-native silicon; Neuron SDK; strong for committed AWS workloads.
- **AMD MI300X**: 192 GB HBM3 @ ~5.3 TB/s — **more memory/bandwidth than H100**, competitive price; ROCm maturity is the gating factor.
- **Intel Gaudi 3 / Cerebras WSE**: Gaudi targets price/perf with built-in Ethernet; Cerebras wafer-scale keeps the whole model on-wafer SRAM for extreme latency.
- The **"NVIDIA moat"** is software (CUDA/ecosystem) more than silicon; challengers compete on $/token and niche latency.

***

## Overview
NVIDIA dominates, but a credible inference candidate must reason about alternatives because (a) interviewers probe architectural understanding via contrast, and (b) inference clouds increasingly run heterogeneous fleets to optimize $/token. The alternatives split into a few archetypes: **SRAM-centric deterministic** designs (Groq LPU, Cerebras WSE) that attack the bandwidth wall by eliminating HBM; **systolic-array** designs (Google TPU) optimized for dense matmul throughput at large batch; **GPU-like** designs with more memory (AMD MI300X) competing head-on with CUDA via ROCm; and **cloud-captive** silicon (AWS Trainium/Inferentia, and Google's TPU) optimized for a provider's economics.

The unifying analytical frame is the same roofline/bandwidth lens from [§00](../00_fundamentals/03_memory_bandwidth_bound_compute.md). Decode is bandwidth-bound; an architecture wins decode either by **raising effective bandwidth** (MI300X's 5.3 TB/s, or Groq's all-SRAM at PB/s-class on-chip bandwidth) or by **reducing bytes moved**. Groq's radical bet is to put the *entire* model in on-chip SRAM across many chips, so weights stream at SRAM speed with deterministic, compiler-scheduled timing — yielding sub-millisecond per-token latency, at the cost of needing many chips (SRAM is small) which complicates throughput economics.

The most important strategic point: **NVIDIA's moat is overwhelmingly software** — CUDA, cuDNN, NCCL, Triton, TensorRT-LLM, and the fact that every framework targets it first. Competing silicon can match or exceed raw specs (MI300X has more HBM than H100) yet lose because kernels, quantization, and serving frameworks are immature on the platform. Whether that moat holds through 2026 is an active debate; the bet for challengers is that inference (more standardized than training) is where the ecosystem gap is smallest.

***

## Core Concepts & Mechanics

### Groq LPU (Tensor Streaming Processor)
- **SRAM-only, no HBM.** ~230 MB SRAM/chip; the model is partitioned across *many* LPUs so all weights live in fast on-chip memory. On-chip bandwidth is enormous (~80 TB/s class), eliminating the HBM bottleneck entirely.
- **Deterministic, compiler-scheduled.** No dynamic caches or branch prediction; the compiler statically schedules every memory access and instruction cycle-by-cycle → predictable, low latency (no stalls). This is the **TSP** architecture (Abts et al. 2020).
- **Tradeoff:** SRAM capacity limits model/batch size; you need a *rack* of LPUs for a 70B model. Excellent **latency** (sub-ms/token, hundreds of tok/s single-stream) but **throughput/$** for large models/batches is harder than GPUs. See [§14](../14_company_deep_dives/02_groq_lpu_architecture.md).

### Google TPU
- **Systolic array** (Matrix Multiply Unit): a 2D grid of MACs streaming operands — extremely efficient for large dense GEMM at high batch; less flexible than SIMT.
- **Programming via XLA**; tight integration with JAX/TF. v5e (cost-optimized inference), v5p (training/large), v6 "Trillium" (next-gen). Large HBM and ICI (inter-chip interconnect) pods.
- Excels at **large-batch serving** where the systolic array stays full; SIMT GPUs are more flexible for dynamic/small-batch.

### AMD MI300X (CDNA3)
- **192 GB HBM3 @ ~5.3 TB/s** — fits bigger models per GPU and more KV; bandwidth between H100 and H200. Competitive list price.
- **ROCm + HIP** ecosystem; vLLM, SGLang, and PyTorch increasingly support it, but kernel maturity (FlashAttention, FP8, custom GEMM) lags CUDA. The gating factor is software, not silicon.

### AWS Trainium2 / Inferentia2
- Provider-captive accelerators with the **Neuron SDK**; compelling $/token for committed AWS users. Compiler-centric like TPU; ecosystem narrower.

### Intel Gaudi 3 / Cerebras WSE
- **Gaudi 3**: HBM + integrated RoCE Ethernet (cheap scale-out), price/perf play; SynapseAI/oneAPI software.
- **Cerebras WSE-3**: wafer-scale engine, ~44 GB on-wafer SRAM, entire model on one wafer for extreme latency; niche, specialized software.

***

## Key Challenges
1. **Software ecosystem maturity.** Kernels, quantization, serving frameworks, and debugging are CUDA-first; non-NVIDIA platforms require porting and often leave performance on the table.
2. **SRAM-only economics.** Groq/Cerebras need many chips per large model; latency is superb but $/token at scale and for big models is harder to win.
3. **Programming-model mismatch.** Systolic/compiler-scheduled designs (TPU, Trainium, Groq) don't run CUDA; teams must adopt XLA/Neuron/Groq compilers, raising switching cost.
4. **Driver/stability and supply.** ROCm stability, vendor support, and availability vary; production teams weigh this heavily ([§01](05_hardware_selection_decision_framework.md)).

***

## Solutions & Current Best Practices
- **Match architecture to workload**: Groq/Cerebras for latency-critical small/medium models; TPU/large-batch for throughput; MI300X for memory-bound large models if ROCm supports your stack; NVIDIA as the safe default.
- **Validate the full serving path** (quantization, attention kernels, continuous batching) on the target platform before committing — specs don't predict end-to-end performance.
- **Use heterogeneous fleets** where economically justified (e.g., cheaper accelerators for prefill or batch jobs) ([§09](../09_distributed_inference/04_heterogeneous_cluster_inference.md)).

***

## Implementation Notes
- On MI300X, confirm FlashAttention/FP8 kernel availability and parity; benchmark your exact model, not MLPerf headline numbers.
- Groq/TPU require model compilation; iteration is slower and dynamic shapes (variable sequence lengths) are handled differently — verify continuous-batching support.
- Track **tok/s/$ and tok/s/W**, not peak FLOPs, when comparing across vendors.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Groq's amazing single-stream latency doesn't imply cheap throughput** — many chips per model means throughput/$ can trail GPUs for large models/high batch; it's a latency play.
- **MI300X has more HBM than H100 but a kernel was 2× slower** — ROCm kernel immaturity, not silicon; always benchmark the real path.
- **TPU loves big batches and hates tiny dynamic ones** — systolic arrays underutilize at small/irregular batch; continuous-batching dynamics differ from GPUs.
- **"Cheaper per hour" ≠ cheaper per token** — lower utilization or weaker kernels can make a nominally cheaper accelerator more expensive per token.

***

## Performance Numbers & Benchmarks
| Accelerator | Memory | BW | Inference sweet spot |
|---|---|---|---|
| NVIDIA H100/H200 | 80/141 GB | 3.35/4.8 TB/s | general default |
| Groq LPU | ~230 MB SRAM/chip | ~80 TB/s on-chip | ultra-low latency, small/med models |
| Google TPU v5e/v6 | tens of GB HBM | high | large-batch dense throughput |
| AMD MI300X | 192 GB HBM3 | ~5.3 TB/s | memory-bound large models (if ROCm ready) |
| Cerebras WSE-3 | ~44 GB SRAM | wafer-scale | extreme latency, niche |
| AWS Inferentia2/Trn2 | HBM | — | AWS-committed cost optimization |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why can Groq achieve sub-millisecond per-token latency, and what's the catch?"* — Expected: SRAM-only + deterministic compiler scheduling; capacity/throughput-per-dollar tradeoff.
- *"MI300X has more HBM and bandwidth than H100 — why isn't it the obvious choice?"* — Expected: ROCm/kernel ecosystem maturity; software moat.
- *"Contrast a systolic array (TPU) with SIMT (GPU) for serving."* — Expected: dense large-batch efficiency vs flexibility/small-batch.
- *"Will NVIDIA's moat hold? Argue both sides."* — Expected: software ecosystem vs commoditizing inference + challenger specs.

***

## Open Problems & Active Research (2025–2026)
- **Closing the software gap** (ROCm, XLA, Neuron, Groq compiler) so challenger silicon realizes its specs in production serving.
- **SRAM-centric scaling** — can deterministic/SRAM designs win throughput/$ for large MoE, or stay latency-niche?
- **Heterogeneous serving** that economically mixes accelerators by phase/workload ([§09](../09_distributed_inference/04_heterogeneous_cluster_inference.md)).

***

## References
- Abts, D., et al. (2020). "Think Fast: A Tensor Streaming Processor (TSP)." *ISCA 2020* (Groq).
- Jouppi, N., et al. (2017). "In-Datacenter Performance Analysis of a TPU." *ISCA 2017*; (2021) "Ten Lessons From Three Generations of TPUs." *ISCA 2021*.
- AMD (2023). CDNA3 / Instinct MI300 architecture disclosures.
- Cerebras (2024). WSE-3 technical materials.
