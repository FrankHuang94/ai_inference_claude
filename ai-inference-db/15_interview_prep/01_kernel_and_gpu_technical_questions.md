# Kernel and GPU Technical Questions (Q&A)

> **Section:** 15_interview_prep
> **Last Updated:** June 2026
> **Related Files:** [../06_kernel_optimization/01_flash_attention_deep_dive.md](../06_kernel_optimization/01_flash_attention_deep_dive.md), [../00_fundamentals/04_roofline_model_for_llm.md](../00_fundamentals/04_roofline_model_for_llm.md), [04_coding_exercises.md](04_coding_exercises.md)
> **Must-Read Papers:** Dao et al. (2022) "FlashAttention"; Williams et al. (2009) "Roofline"; Kwon et al. (2023) "vLLM"
> **Estimated Study Time:** 35 minutes

***

## TL;DR
- This file gives **model answers** to the most common GPU/kernel technical questions.
- Anchor every answer in: **arithmetic intensity, roofline, memory hierarchy, the bandwidth bound**.
- Be ready to **derive** (AI at batch=1, fragmentation savings, FlashAttention IO complexity, FP8 speedup) — not just recite.
- Know **Nsight** workflow (classify bound → roofline → optimize).

***

## Q1: Why is LLM decode memory-bandwidth-bound?
At batch=1, a dense matmul `y=Wx` reads `m·k` weights (2 bytes FP16) and does `2·m·k·B` FLOPs → **AI = B ≈ 1** FLOP/byte. The H100 ridge is `989 TFLOP/s ÷ 3.35 TB/s ≈ 295` FLOPs/byte, so at AI≈1 you're ~300× left of the ridge — the Tensor Cores idle while you stream weights. Throughput ≈ `bandwidth / weight_bytes` (e.g., 70B FP16 → ~24 tok/s). Decode produces one token per full weight read; it's bound by HBM bandwidth, not compute. ([§00](../00_fundamentals/03_memory_bandwidth_bound_compute.md))

## Q2: Calculate PagedAttention's fragmentation savings for lengths uniform in [64, 4096]
Naive contiguous allocation reserves `max_len = 4096` per request; avg used = `(64+4096)/2 = 2080` → utilization ≈ `2080/4096 ≈ 51%`, i.e., ~49% wasted (plus external fragmentation, pushing real waste to 60–80%). PagedAttention allocates in blocks (size 16) on demand: waste ≤ one partial block/sequence = `≤15/2080 ≈ 0.7%`. So fragmentation drops from ~49–80% to <4% → far larger batches → 2–4× throughput. ([§02](../02_kv_cache/01_paged_attention_vllm.md))

## Q3: FlashAttention's IO complexity and how it reduces HBM usage
Standard attention materializes the N×N score matrix in HBM: O(N²) memory and O(N²) HBM traffic (write S, read for softmax, read for ·V). FlashAttention tiles Q,K,V into SRAM blocks and computes softmax **online** (running max/sum), never materializing S in HBM — only the O(N·d) output is written. HBM accesses drop to **Θ(N²·d/M)** (M = SRAM size), memory to **O(N)**. It's the *same FLOPs* but far less HBM traffic; since attention is bandwidth-bound, that's the speedup. ([§06](../06_kernel_optimization/01_flash_attention_deep_dive.md))

## Q4: What is the roofline model and how do you use it for an LLM kernel?
Plot achievable perf = `min(peak_FLOPs, AI × peak_bandwidth)` vs arithmetic intensity. The **ridge** = peak_FLOPs/peak_bandwidth separates memory-bound (left, on the bandwidth slope) from compute-bound (right, at the compute ceiling). For an LLM kernel: compute its AI (FLOPs/bytes), place it; **left of ridge → reduce bytes** (fusion, quantization, coalescing); **right → improve FLOP efficiency** (tiling, FP8, occupancy). Decode matmuls (AI≈batch) sit far left; prefill (AI≈P) near the ceiling. ([§00](../00_fundamentals/04_roofline_model_for_llm.md))

## Q5: Why does FP8 inference on H100 ~double throughput vs FP16?
H100 FP8 Tensor Cores do ~1979 TFLOP/s vs ~989 FP16 (2×), with FP16/BF16 accumulation, **and** FP8 halves bytes moved. For **compute-bound prefill**, the 2× FLOPs ~doubles throughput. For **memory-bound decode**, the win is the **halved weight bytes** (~2× if weights dominate the read), not the FLOPs — because the FP8 ridge moves right (591 FLOPs/byte) and decode stays bandwidth-bound. So "doubles throughput" is most true for prefill. ([§05](../05_quantization/04_fp8_inference_h100.md))

## Q6: Explain CUDA graph optimization — when does it help, when not?
Each kernel launch costs ~µs of CPU/GPU overhead. A decode step launches dozens of kernels; at batch=1 (sub-ms steps) launch overhead and CPU-feed gaps are a real fraction of runtime. **CUDA graphs** capture the kernel sequence once and replay it as a single launch, removing per-kernel overhead and GPU idle gaps. **Helps**: small-batch decode with fixed shapes. **Doesn't help / inapplicable**: dynamic shapes (must capture per shape and pad), large compute-bound prefill (launch overhead negligible vs compute). ([§06](../06_kernel_optimization/02_fused_kernels_and_operator_fusion.md))

## Q7: How would you write a fused LayerNorm + Linear kernel in Triton?
In Triton: one program instance per row; `tl.load` the row tile; compute RMS/mean and normalize in registers; loop over K-tiles doing `tl.dot(normalized_tile, weight_tile)` accumulating; `tl.store` the output — all in one kernel so the normalized activations never round-trip HBM. Add `@triton.autotune` over `BLOCK_M/BLOCK_K/num_warps/num_stages` keyed on shapes. The win: fuses the memory-bound norm with the matmul, eliminating an HBM round-trip and a launch. ([§06](../06_kernel_optimization/04_triton_for_inference.md), code in [04_coding_exercises.md](04_coding_exercises.md))

## Q8: A kernel is at 25% of peak TFLOPS — is it well-optimized?
Depends on its **AI vs the ridge**. If it's left of the ridge (memory-bound), 25% of the compute roof may be **100% of the bandwidth roof** — it's optimal and you should reduce bytes, not chase FLOPs. If it's right of the ridge (compute-bound), 25% means real headroom (better tiling/occupancy/FP8). Always read the roofline/Nsight SOL (Memory vs SM throughput), not FLOP% alone. ([§00](../00_fundamentals/04_roofline_model_for_llm.md), [§06](../06_kernel_optimization/05_kernel_profiling_and_benchmarking.md))

***

## Interview Angles
> 💡 **Delivery tips:**
- **Derive, don't recite** — show the AI/ridge/IO-complexity math.
- Anchor in the **bandwidth bound** and **roofline**.
- For "is X optimized?", **classify the bound first**.
- Mention the **tool** (Nsight SOL/roofline) for diagnosis.

***

## Complexity Gotchas
> ⚠️ **Common mistakes:**
- Saying FP8 doubles *decode* throughput (it ~halves bytes; the 2× is prefill).
- Treating low FLOP% as "slow" without checking the bound.
- Forgetting CUDA graphs need static shapes.
- Confusing FlashAttention's memory reduction (O(N)) with compute reduction (still O(N²)).

***

## References
- Dao, T., et al. (2022). "FlashAttention." arXiv:2205.14135.
- Williams, S., et al. (2009). "Roofline." *CACM* 52(4).
- Kwon, W., et al. (2023). "vLLM." arXiv:2309.06180.
