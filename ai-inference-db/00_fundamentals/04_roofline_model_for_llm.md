# The Roofline Model Applied to LLM Inference

> **Section:** 00_fundamentals
> **Last Updated:** June 2026
> **Related Files:** [03_memory_bandwidth_bound_compute.md](03_memory_bandwidth_bound_compute.md), [05_transformer_inference_walkthrough.md](05_transformer_inference_walkthrough.md), [../06_kernel_optimization/05_kernel_profiling_and_benchmarking.md](../06_kernel_optimization/05_kernel_profiling_and_benchmarking.md)
> **Must-Read Papers:** Williams, Waterman & Patterson (2009, CACM) "Roofline"; Yuan et al. (2024, arXiv) "LLM Inference Unveiled: Survey and Roofline"; Dao et al. (2022, NeurIPS) "FlashAttention"
> **Estimated Study Time:** 40 minutes

***

## TL;DR
- The roofline plots **achievable performance (TFLOPS, y) vs arithmetic intensity (FLOPs/byte, x)**; the "roof" is `min(peak_compute, AI × peak_bandwidth)`.
- The **ridge point** `AI_ridge = peak_compute / peak_bandwidth` separates the memory-bound slope (left) from the compute-bound ceiling (right).
- For LLM ops: **decode matmuls have AI ≈ batch** (left of ridge); **prefill matmuls have AI ≈ prompt length** (near/right of ridge); **attention** AI depends on the algorithm (standard ≈ low; FlashAttention raises it).
- Batching **slides operations rightward** along the x-axis; quantization **raises the achievable y** and **lowers required x**.
- It's the single most-used analytical tool in inference interviews — be able to draw it and place each operation.

***

## Overview
The roofline model is a deliberately simple performance model: any kernel's achievable throughput is capped by either the compute peak or what bandwidth allows at its arithmetic intensity, whichever is lower. Plotted on log-log axes, this produces a diagonal "bandwidth roof" (`P = AI × BW`) rising until it meets the flat "compute roof" (`P = peak_FLOPs`) at the ridge point. Every kernel is a point under these roofs; the gap between the point and the roof is your optimization headroom.

For LLM inference the roofline is powerful because the two phases land in different regions and respond to different interventions. Decode kernels sit on the steep bandwidth slope far left of the ridge; the only ways to move up are to increase AI (batch more, speculate) or move the roof (faster memory, fewer bytes via quantization). Prefill kernels sit near the compute ceiling; there the win is achieving a higher *fraction* of peak (better tiling, fusion, FP8 Tensor Cores). This is why "the same optimization helps prefill but not decode" — they live in different roofline regions.

A subtle and interview-relevant point: switching precision (FP16→FP8) *raises the compute roof* (more peak FLOPs) but leaves the bandwidth roof unchanged, **moving the ridge point right**. So lowering precision can push a previously compute-bound op back into the memory-bound region unless you also reduce bytes moved. The roofline makes this visible at a glance.

***

## Core Concepts & Mechanics

### Constructing the roofline
📐 Achievable performance:
```
P(AI) = min( peak_compute , AI × peak_bandwidth )
ridge: AI_ridge = peak_compute / peak_bandwidth
```
For H100 FP16: peak ≈ 989 TFLOP/s, BW ≈ 3.35 TB/s ⇒ ridge ≈ 295 FLOPs/byte. Left of 295 you're on the slope `P = AI × 3.35`; right of 295 you're capped at 989.

### Arithmetic intensity of LLM operations
For a dense linear `y = Wx`, `W ∈ R^{m×k}`, batch B, FP16:
```
FLOPs = 2·m·k·B ;  bytes ≈ 2·m·k (weights, read once)  ⇒  AI = B
```
- **Decode** (B small): AI ≈ B → deep in memory-bound region.
- **Prefill** (effective batch = B·P positions): AI ≈ B·P → can reach/exceed ridge.

For **attention** at one decode step (query len 1, context S, head dim d, h heads):
```
FLOPs ≈ 4·h·S·d (QK^T + softmax·V)
bytes ≈ 2·h·S·d (read K and V from HBM)   ⇒  AI ≈ 2 FLOPs/byte
```
Attention decode is **intensely memory-bound** (AI ~constant, independent of batch in the per-sequence KV term) — which is exactly why KV-cache bandwidth and FlashAttention matter. FlashAttention doesn't change the FLOPs much but slashes the *bytes* (no N×N materialization), raising AI for the prefill/full-attention case.

### How batching and quantization move points
- **Batching:** AI(matmul) = B, so increasing B slides the matmul point **right** along the slope toward the ridge — more achieved TFLOPS per byte. Diminishing once you pass the ridge.
- **Quantization (weights):** halving bytes moved **doubles AI** (point moves right) *and* on Tensor Cores raises the compute roof — both effects raise achieved performance for memory-bound decode.
- **FP8 compute:** raises the flat roof (989→1979) but moves the ridge right (295→591); good for compute-bound prefill, neutral-to-tricky for memory-bound decode.

### Worked examples
**7B model, H100, FP16, decode batch=1:**
- Matmul AI ≈ 1 ⇒ P ≈ 1 × 3.35 TFLOP/s = 3.35 TFLOP/s achieved (vs 989 peak ⇒ ~0.3% of peak compute). Throughput ≈ bandwidth/weights ≈ 3.35e12/14e9 ≈ 239 tok/s. Memory-bound, far left.

**7B model, H100, FP16, batch=64 decode:**
- Matmul AI ≈ 64 ⇒ P ≈ 64 × 3.35 = 214 TFLOP/s (still < 989 ridge value 295×3.35; just left of ridge). ~22% of peak — 60× better utilization than batch=1.

**70B model, H100×2 (TP2), prefill 2048 tokens, batch=1:**
- Effective positions = 2048 ⇒ AI ≈ 2048 ≫ 295 ⇒ compute-bound, P ≈ peak (× MFU). Prefill runs near the compute roof.

**70B, A100, decode batch=1:** AI≈1 ⇒ P ≈ 2.0 TFLOP/s; throughput ≈ 2.0e12/140e9 ≈ 14 tok/s. H100 → ~24 tok/s (bandwidth ratio). Roofline predicts the ratio exactly.

***

## Key Challenges
1. **Roofline ignores latency and overhead.** It models steady-state throughput, not kernel-launch latency, queueing, or pipeline bubbles — which dominate small-batch decode and must be modeled separately (CUDA graphs, etc.).
2. **A single "AI" hides per-operator variance.** A transformer layer mixes high-AI matmuls and low-AI attention/elementwise; the layer roofline point is a weighted blend, so you must analyze operators individually.
3. **Achieved peaks ≠ datasheet peaks.** Real bandwidth ~70–90% of peak, real FLOPs gated by MFU; use measured roofs or your predictions will be optimistic.
4. **Precision changes the model itself.** FP8/FP4 move both roofs/ridge; you must re-draw per precision, a step people forget.

***

## Solutions & Current Best Practices
- Use the roofline to **decide the optimization**: left-of-ridge → batch more, quantize, fuse to cut HBM traffic; right-of-ridge → better tiling, FP8 Tensor Cores, higher MFU.
- **Nsight Compute has a built-in roofline view** — overlay your kernel and read the headroom directly ([§06](../06_kernel_optimization/05_kernel_profiling_and_benchmarking.md)).
- For **attention specifically**, prefer FlashAttention-class kernels to push the operation up off the bandwidth slope ([§06](../06_kernel_optimization/01_flash_attention_deep_dive.md)).
- Model **per-operator AI** for a target model+hardware before buying GPUs or choosing TP degree.

***

## Implementation Notes
- Compute AI per operator using exact shape/byte accounting from [05_transformer_inference_walkthrough.md](05_transformer_inference_walkthrough.md); don't hand-wave "the layer."
- Include **KV-read bytes** in the attention AI — at long context this term dominates and pulls the layer leftward (more memory-bound).
- When comparing frameworks, plot both on the same roofline; the one closer to the roof at the same AI is better-engineered, independent of headline tok/s.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **"My kernel is at 30% of peak FLOPs — it's slow."** If it's left of the ridge, 30% of the *compute* roof may already be 100% of the *bandwidth* roof; you're optimal and should reduce bytes, not chase FLOPs.
- **FP8 moved a prefill kernel back into memory-bound.** Raising the roof moved the ridge right; if the kernel's AI didn't also rise, it's now on the slope. Re-draw the roofline per precision.
- **Roofline says batch=256 is optimal but you OOM at batch=32.** Roofline ignores KV-cache capacity; the *memory budget*, not the roofline, sets the achievable batch for large models.

***

## Performance Numbers & Benchmarks
| Operation | AI (FP16) | Region (H100) | Implication |
|---|---|---|---|
| Decode matmul, B=1 | ~1 | far left, memory-bound | bandwidth/weights-limited |
| Decode matmul, B=64 | ~64 | left, memory-bound | ~60× better than B=1 |
| Prefill matmul, P=2048 | ~2048 | right, compute-bound | near peak × MFU |
| Attention decode (per KV) | ~2 | far left | KV-bandwidth-limited |
| FlashAttention prefill | high | near ridge | SRAM tiling removes HBM traffic |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Draw the roofline for an H100 and place prefill, decode-batch-1, and decode-batch-64."* — Expected: ridge ~295, decode on slope, prefill at ceiling, batching slides right.
- *"Derive the arithmetic intensity of a dense matmul and of attention at decode."* — Expected: AI=B for matmul; AI≈2 for attention KV read.
- *"How does going to FP8 change the roofline, and why might decode not speed up?"* — Expected: ridge moves right; decode bandwidth-bound, only byte reduction helps.
- *"A kernel is at 25% of peak TFLOPS — is it well-optimized?"* — Expected: depends on its AI vs ridge; could be bandwidth-saturated and optimal.

***

## Open Problems & Active Research (2025–2026)
- **Roofline extensions for disaggregated/multi-GPU** serving where interconnect bandwidth adds a third roof.
- **Per-operator rooflines for MoE and linear attention**, whose byte/FLOP profiles differ sharply from dense transformers.
- **Latency-aware rooflines** that fold in launch overhead and pipeline bubbles for small-batch decode (where steady-state roofline is misleading).

***

## References
- Williams, S., Waterman, A., Patterson, D. (2009). "Roofline." *CACM* 52(4).
- Yuan, Z., et al. (2024). "LLM Inference Unveiled: Survey and Roofline Model Insights." arXiv:2402.16363.
- Dao, T., et al. (2022). "FlashAttention." *NeurIPS 2022*. arXiv:2205.14135.
- Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference." *MLSys 2023*. arXiv:2211.05102.
