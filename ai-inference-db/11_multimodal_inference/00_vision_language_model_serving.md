# Vision-Language Model Serving

> **Section:** 11_multimodal_inference
> **Last Updated:** June 2026
> **Related Files:** [01_image_encoding_pipeline.md](01_image_encoding_pipeline.md), [02_multimodal_batching_challenges.md](02_multimodal_batching_challenges.md), [../02_kv_cache/00_kv_cache_fundamentals.md](../02_kv_cache/00_kv_cache_fundamentals.md)
> **Must-Read Papers:** Liu et al. (2023, NeurIPS) "LLaVA"; Alayrac et al. (2022) "Flamingo"; Radford et al. (2021) "CLIP"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- VLMs serve **image (or video/audio) + text → text**: a **vision encoder** turns images into **visual tokens** that are fed alongside text tokens into the LLM.
- Architecture: **encoder (e.g., CLIP/ViT) → projector (MLP/cross-attn) → LLM decoder**; the LLM does standard autoregressive decode over combined tokens.
- Serving challenge: images become **many tokens** (hundreds to thousands each), inflating prefill cost and KV, with **variable token counts** per request (heterogeneous batching).
- The **encoder is a separate compute stage** (compute-bound, like a mini-prefill) that can be disaggregated/cached.
- Prefill (encode + long combined prompt) dominates VLM latency; decode is standard.

***

## Overview
Vision-language models (VLMs) such as LLaVA, Qwen-VL, and GPT-4V-style systems extend an LLM to accept images by **encoding images into tokens** the LLM can attend to. The canonical architecture (LLaVA, Liu et al. 2023) is: a pretrained **vision encoder** (a ViT, often CLIP's) processes the image into a grid of patch embeddings; a **projector** (an MLP, or cross-attention as in Flamingo) maps those into the LLM's embedding space as **visual tokens**; these visual tokens are concatenated with the text tokens and the **LLM decoder** runs as usual, autoregressively generating text. From the LLM's perspective, the image is just a (large) span of input tokens — which is the key to understanding VLM serving cost.

The dominant serving consequence is **token inflation**. A single image typically becomes **hundreds to thousands of visual tokens** (e.g., 576 for LLaVA's 24×24 grid, far more for high-resolution or tiling schemes), so a multi-image prompt can have many thousands of tokens before any text — making **prefill** (encoding + processing the long combined prompt) the dominant cost and KV consumer, while decode (generating the text answer) is standard and comparatively cheap. The vision **encoder** is a distinct compute stage with its own characteristics (compute-bound, fixed-shape per image), which can be **batched, cached, and even disaggregated** from the LLM ([§01](01_image_encoding_pipeline.md)).

The serving challenges that follow: **heterogeneous batching** (requests have different numbers of images → wildly different token counts, complicating batch formation, [§02](02_multimodal_batching_challenges.md)); **encoder/LLM pipeline** management (two stages with different resource profiles); and **caching** opportunities (the same image across requests, or the encoder output, can be cached). This file covers the VLM architecture and its serving implications; the rest of the section drills into the encoder pipeline, batching, and video/audio.

***

## Core Concepts & Mechanics

### Architecture
```
Image → Vision Encoder (ViT/CLIP) → patch embeddings
      → Projector (MLP / cross-attention) → visual tokens (in LLM embedding space)
Text  → text tokens
[visual tokens; text tokens] → LLM decoder → autoregressive text output
```
- **Encoder**: compute-bound, fixed-shape per image; produces a fixed number of patch embeddings.
- **Projector**: small; aligns modalities.
- **LLM**: standard prefill (over combined tokens) + decode.

### Token inflation
📐 Visual tokens per image = grid size (e.g., 24×24 = 576) × (tiles for high-res). A 4-image high-res prompt can be 10k+ tokens before text → prefill and KV dominated by vision. This is *the* VLM serving cost driver.

### Two-stage pipeline
- **Stage 1 (encode)**: run images through the encoder (batchable across images, compute-bound).
- **Stage 2 (LLM)**: prefill combined tokens + decode.
- Stages have different resource profiles → can be separately batched/scaled/disaggregated; encoder outputs can be **cached** (same image reused).

### Cross-attention vs concatenation
- **Concatenation** (LLaVA): visual tokens prepended to text; simple, but adds to sequence length/KV.
- **Cross-attention** (Flamingo): LLM cross-attends to visual features via gated layers; keeps the text sequence shorter but adds architectural complexity.

***

## Key Challenges
1. **Token inflation → prefill/KV cost.** Images become many tokens, making prefill and KV the bottleneck; high-res/multi-image is expensive.
2. **Heterogeneous batching.** Variable image counts → variable token counts; hard to batch efficiently ([§02](02_multimodal_batching_challenges.md)).
3. **Two-stage resource mismatch.** Encoder (compute-bound, fixed shape) vs LLM (prefill+decode) want different handling/scaling.
4. **Caching complexity.** Image/encoder-output caching helps but needs content-addressing and integration with prefix caching.

***

## Solutions & Current Best Practices
- **Cache encoder outputs / image tokens** for repeated images; **prefix-cache** the visual-token span when reused ([§02](../02_kv_cache/02_prefix_caching_and_radix_attention.md)).
- **Batch the encoder** across images separately from the LLM; consider **disaggregating** the encoder stage.
- **Limit visual tokens** (resolution/tiling/token-merging) to control prefill cost where quality permits.
- **Handle heterogeneous batches** with padding-aware or grouped scheduling ([§02](02_multimodal_batching_challenges.md)).

***

## Implementation Notes
- vLLM/SGLang support multimodal models; configure image token handling and encoder batching.
- Cache encoder outputs keyed on image hash; reuse across requests with the same image.
- Account for visual tokens in KV budgeting — they inflate KV like any prompt tokens.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **High-res images blew up prefill/KV** — tiling produced thousands of tokens/image; cap resolution or merge tokens.
- **Batching stalled on heterogeneous image counts** — variable token counts wrecked batch efficiency; group by similar sizes.
- **Re-encoded the same image every request** — no encoder-output cache; wasted compute. Cache by image hash.
- **Forgot visual tokens in KV budget** — OOM under multi-image load; count them.

***

## Performance Numbers & Benchmarks
| Model | Visual tokens/image | Serving note |
|---|---|---|
| LLaVA (336px) | 576 | concatenated, prefill-heavy |
| High-res / tiled | thousands | major prefill/KV cost |
| Flamingo (cross-attn) | features (no seq inflation) | shorter text seq, complex arch |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How does a VLM process an image, and what does it cost at serving?"* — Expected: encoder→projector→visual tokens; token inflation → prefill/KV cost.
- *"Why is the vision encoder a separate serving concern?"* — Expected: compute-bound fixed-shape stage; batch/cache/disaggregate separately.
- *"How do you reduce VLM prefill cost?"* — Expected: cache encoder outputs, limit visual tokens, prefix-cache repeated images.
- *"Concatenation vs cross-attention for visual tokens?"* — Expected: sequence inflation vs architectural complexity.

***

## Open Problems & Active Research (2025–2026)
- **Visual token reduction** (merging, adaptive resolution) without quality loss.
- **Disaggregated/cached encoder pipelines** at scale.
- **Efficient any-resolution / many-image** serving.

***

## References
- Liu, H., et al. (2023). "Visual Instruction Tuning (LLaVA)." *NeurIPS 2023*. arXiv:2304.08485.
- Alayrac, J.-B., et al. (2022). "Flamingo." *NeurIPS 2022*. arXiv:2204.14198.
- Radford, A., et al. (2021). "CLIP." *ICML 2021*. arXiv:2103.00020.
