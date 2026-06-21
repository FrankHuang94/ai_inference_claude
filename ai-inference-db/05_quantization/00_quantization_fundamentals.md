# Quantization Fundamentals

> **Section:** 05_quantization
> **Last Updated:** June 2026
> **Related Files:** [01_post_training_quantization.md](01_post_training_quantization.md), [04_fp8_inference_h100.md](04_fp8_inference_h100.md), [../00_fundamentals/03_memory_bandwidth_bound_compute.md](../00_fundamentals/03_memory_bandwidth_bound_compute.md)
> **Must-Read Papers:** Micikevicius et al. (2022) "FP8 Formats"; Dettmers et al. (2022, NeurIPS) "LLM.int8()"; Frantar et al. (2023, ICLR) "GPTQ"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- Quantization reduces the **bits per value** (FP16→FP8→INT4→FP4), cutting **memory and bandwidth** — the direct lever on memory-bound decode ([§00](../00_fundamentals/03_memory_bandwidth_bound_compute.md)).
- Formats trade **range vs precision**: floating (FP8 E4M3/E5M2, FP4) keep dynamic range; integer (INT8/INT4) need explicit scale/zero-point.
- **Granularity** (per-tensor < per-channel < per-group) trades accuracy for overhead; finer = more accurate, more metadata.
- **Symmetric vs asymmetric** (zero-point) and **calibration** data quality govern error.
- Decode wins are near-linear in byte reduction; **prefill (compute-bound)** wins only if the matmul itself runs in low precision (Tensor Cores).

***

## Overview
Quantization maps high-precision values (FP16/BF16 weights and activations) to fewer bits, reducing memory footprint and — crucially for inference — the **bytes moved per token**. Since decode is memory-bandwidth-bound, halving the bits of the weights roughly doubles decode throughput, making quantization the single most cost-effective inference optimization after batching. The design space splits along three axes: the **numeric format** (how bits encode values), the **granularity** (how many values share a scaling factor), and *what* you quantize (weights only vs weights+activations vs KV cache, covered in later files).

The format choice is a range-vs-precision tradeoff. **Floating-point** low-precision formats (FP8 E4M3/E5M2, FP4 E2M1) keep an exponent so they handle wide dynamic ranges gracefully — important because LLM activations have outliers spanning orders of magnitude. **Integer** formats (INT8, INT4) have uniform spacing and need an explicit **scale** (and optional **zero-point** for asymmetric ranges) to map the float range onto the integer grid: `x ≈ scale × (q − zero_point)`. Integer is great for weights (well-behaved distributions) but struggles with activation outliers, motivating the activation-specific techniques in [§03](03_activation_quantization_challenges.md).

Granularity governs accuracy. **Per-tensor** quantization (one scale for the whole tensor) is cheapest but suffers when value ranges vary across channels; **per-channel** (one scale per output channel) and **per-group** (a scale per small block, e.g., 64–128 weights) capture local range much better at the cost of more metadata and slightly more complex kernels. The empirical lesson is that modern LLM quantization (GPTQ, AWQ) leans on fine granularity plus careful handling of outliers to hit INT4 weights with near-zero quality loss. This file establishes the vocabulary; the rest of the section covers the specific methods.

***

## Core Concepts & Mechanics

### Numeric formats
| Format | Bits | Structure | Range/precision | Use |
|---|---|---|---|---|
| FP32 | 32 | E8M23 | huge range, high precision | reference |
| FP16 | 16 | E5M10 | moderate range | baseline inference |
| BF16 | 16 | E8M7 | FP32 range, low precision | training/inference |
| FP8 E4M3 | 8 | E4M3 (±448) | balanced | weights/activations (Hopper+) |
| FP8 E5M2 | 8 | E5M2 (±57344) | wide range | gradients/wide dynamic |
| INT8 | 8 | integer + scale | uniform | weights/activations |
| INT4 | 4 | integer + scale | uniform, coarse | weight-only |
| FP4 (E2M1) | 4 | float + block scale (MX) | range w/ micro-scaling | Blackwell weights |

### The quantization map
📐 Affine quant: `q = round(x/scale) + zero_point`, clamped to the integer range; dequant: `x̂ = scale × (q − zero_point)`.
- **Symmetric**: zero_point=0, `scale = max(|x|)/(2^{b-1}−1)`. Simpler, good for weights.
- **Asymmetric**: nonzero zero_point maps `[min,max]` exactly; better for skewed activation ranges.
Error sources: **rounding** (granularity-dependent) and **clipping** (range-dependent; outliers force large scale → big rounding error elsewhere).

### Granularity
- **Per-tensor**: 1 scale; cheap; sensitive to inter-channel range variance.
- **Per-channel**: 1 scale/output-channel; standard for weights.
- **Per-group/block**: 1 scale per N (e.g., 128) values; best accuracy at 4-bit; used by GPTQ/AWQ and MX-FP4.
📐 Finer granularity → smaller per-block range → less clipping/rounding error, at the cost of more scales stored and applied.

### Calibration
PTQ methods estimate scales (and clipping thresholds) from a small **calibration set** of representative inputs. Poor/unrepresentative calibration → bad scales → quality loss, especially for activations. Data quality matters more than quantity.

### Where the speedup comes from
- **Decode (memory-bound)**: speedup ≈ byte reduction (FP16→INT4 ≈ 4× weight bytes → up to ~4× decode), even if matmul dequantizes to FP16.
- **Prefill (compute-bound)**: speedup only if the **matmul runs in low precision** on Tensor Cores (FP8/INT8 GEMM); weight-only INT4 that dequantizes to FP16 barely speeds prefill ([§02](02_weight_only_quantization.md)).

***

## Key Challenges
1. **Activation outliers.** A few large activation values force a large scale, crushing precision for the rest; the core obstacle to low-bit activation quantization ([§03](03_activation_quantization_challenges.md)).
2. **Range vs precision per format.** Choosing FP8 vs INT8 vs INT4 and symmetric/asymmetric per tensor type requires understanding each distribution.
3. **Granularity/kernel overhead.** Fine granularity improves accuracy but complicates kernels (per-group dequant) and adds metadata bandwidth.
4. **Task-dependent quality loss.** Perplexity may look fine while reasoning/long-context degrades; evaluation must be task-aware.

***

## Solutions & Current Best Practices
- **FP8 (E4M3) weights+activations on Hopper/Blackwell** — near-lossless, ~2× compute, half bytes; the production default ([§04](04_fp8_inference_h100.md)).
- **INT4 weight-only (GPTQ/AWQ)** for memory-bound decode where compute precision can stay FP16 ([§01](01_post_training_quantization.md), [§02](02_weight_only_quantization.md)).
- **Per-group granularity** (128) for 4-bit; per-channel for 8-bit.
- **Representative calibration**; task-aware evaluation (not just perplexity).

***

## Implementation Notes
- Match format to tensor: weights tolerate INT4/FP8 well; activations prefer FP8 (range) or need outlier handling for INT8.
- Store and apply per-group scales efficiently; fused dequant kernels avoid extra HBM round-trips ([§06](../06_kernel_optimization/02_fused_kernels_and_operator_fusion.md)).
- Validate on downstream tasks (MMLU, GSM8K, long-context) and the target domain, not just PPL.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **INT4 weight-only gave ~4× decode but ~0 prefill speedup** — matmul still FP16 after dequant; only bandwidth-bound decode benefits.
- **Per-tensor INT8 activations tanked accuracy** — outliers forced a huge scale; need per-channel/SmoothQuant or FP8.
- **Great perplexity, broken reasoning** — low-precision damage concentrates in sensitive layers/tasks; PPL hides it.
- **Bad calibration set** — unrepresentative data → wrong scales → silent quality loss. Calibrate on real traffic-like data.

***

## Performance Numbers & Benchmarks
| Scheme | Bytes vs FP16 | Decode speedup | Quality (typical) |
|---|---|---|---|
| FP8 W+A | 0.5× | ~2× (prefill too) | near-lossless |
| INT8 W+A (SmoothQuant) | 0.5× | ~2× | small loss |
| INT4 weight-only (GPTQ/AWQ) | 0.25× weights | up to ~4× decode | small loss |
| FP4 (MX) | 0.25× | ~4× | task-dependent (emerging) |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why does weight quantization speed up decode but not prefill?"* — Expected: decode bandwidth-bound (bytes), prefill compute-bound (needs low-precision matmul).
- *"Explain per-tensor vs per-channel vs per-group and the accuracy tradeoff."* — Expected: range locality vs metadata/kernel overhead.
- *"Why are activations harder to quantize than weights?"* — Expected: outliers force large scales; floating formats / SmoothQuant.
- *"FP8 vs INT8 — when each?"* — Expected: FP8 keeps range (activations), INT8 fine for weights with per-channel scaling.

***

## Open Problems & Active Research (2025–2026)
- **Robust 4-bit weights+activations** (W4A4) without quality loss.
- **FP4/MX accuracy** characterization and which layers tolerate it ([§04](04_fp8_inference_h100.md)).
- **Outlier-free architectures** (trained to be quantization-friendly).

***

## References
- Micikevicius, P., et al. (2022). "FP8 Formats for Deep Learning." arXiv:2209.05433.
- Dettmers, T., et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *NeurIPS 2022*. arXiv:2208.07339.
- Frantar, E., et al. (2023). "GPTQ." *ICLR 2023*. arXiv:2210.17323.
- Open Compute Project (2023). "Microscaling (MX) Formats."
