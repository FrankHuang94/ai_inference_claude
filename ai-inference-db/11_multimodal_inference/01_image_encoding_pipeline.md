# Image Encoding Pipeline

> **Section:** 11_multimodal_inference
> **Last Updated:** June 2026
> **Related Files:** [00_vision_language_model_serving.md](00_vision_language_model_serving.md), [02_multimodal_batching_challenges.md](02_multimodal_batching_challenges.md), [../03_batching_and_scheduling/03_prefill_decode_disaggregation.md](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md)
> **Must-Read Papers:** Dosovitskiy et al. (2021) "ViT"; Radford et al. (2021) "CLIP"; Bolya et al. (2023) "Token Merging (ToMe)"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- The image encoder (ViT) is a **compute-bound, fixed-shape** stage: image → patches → transformer → patch embeddings → projector → visual tokens.
- It behaves like a **mini-prefill** (high arithmetic intensity, parallel over patches) — batchable across images, cacheable by image content.
- **Token count** is the key lever: resolution and tiling determine visual tokens (→ downstream prefill/KV cost); **token merging/pruning** (ToMe) reduces it.
- High-resolution schemes (tiling, AnyRes) multiply tokens; a quality/cost knob.
- Best practices: **batch the encoder**, **cache encoder outputs by image hash**, optionally **disaggregate** the encoder stage.

***

## Overview
The image encoding pipeline is the front half of VLM serving ([§00](00_vision_language_model_serving.md)) and has distinct performance characteristics worth treating separately. An image is split into fixed-size **patches** (e.g., 14×14 or 16×16 pixels), each linearly embedded; a **Vision Transformer** (ViT, Dosovitskiy et al. 2021) processes all patches in parallel through self-attention; the resulting patch embeddings are mapped by a **projector** into the LLM's token space as visual tokens. Because all patches are processed at once, the encoder is **compute-bound with high arithmetic intensity** — it looks like a prefill/training forward pass, not like decode — and it has a **fixed shape** per image resolution, which makes it cleanly batchable and amenable to CUDA graphs.

The pivotal serving decision is **how many visual tokens** the encoder produces, because that count flows directly into the LLM's prefill cost and KV footprint. Base resolution gives a fixed grid (e.g., 24×24=576 tokens for 336px). To handle **high-resolution** images (needed for OCR, charts, fine detail), models use **tiling / AnyRes** schemes that split the image into multiple tiles, each encoded separately — multiplying the token count several-fold and substantially raising downstream cost. The countervailing technique is **token reduction**: **token merging (ToMe)** and pruning combine/drop redundant patch tokens, cutting visual tokens (and thus LLM cost) with modest quality impact — an important efficiency lever.

Operationally, the encoder's fixed-shape, compute-bound, cacheable nature suggests three optimizations: **batch** many images through the encoder together (it loves large batches, unlike memory-bound decode); **cache** encoder outputs keyed by image content hash (the same image across requests — a product screenshot, a logo — need only be encoded once); and optionally **disaggregate** the encoder onto separate (compute-dense) hardware from the LLM decode pool, analogous to prefill/decode disaggregation ([§03](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md)). This file covers the pipeline, the token-count lever, and these optimizations.

***

## Core Concepts & Mechanics

### The pipeline
```
Image → patchify (e.g., 14×14 patches) → linear embed → +pos
      → ViT (self-attention over patches, parallel) → patch embeddings
      → projector (MLP/cross-attn) → visual tokens (LLM space)
```
- Compute-bound, high AI (parallel over patches), fixed shape per resolution → batchable, CUDA-graphable.

### Token count lever
📐 Visual tokens = `(image_size/patch_size)² × tiles`. 336px/14 = 24×24 = 576 tokens; high-res tiling (e.g., 4–6 tiles) → 2k–3k+ tokens/image. This count drives LLM prefill FLOPs and KV → the main cost knob.

### Token reduction
- **ToMe** (Bolya et al. 2023): merge similar tokens between ViT layers → fewer tokens, small quality cost.
- **Pruning / adaptive resolution**: drop low-information patches; use lower resolution when detail isn't needed.

### Optimizations
- **Batch the encoder**: large image batches are efficient (compute-bound); separate from LLM batching.
- **Cache encoder outputs**: key on image hash; reuse across requests — big win for repeated images.
- **Disaggregate encoder**: run on compute-dense GPUs, stream visual tokens to the LLM pool ([§03](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md)).

***

## Key Challenges
1. **Token count vs quality.** High-res tiling improves detail (OCR/charts) but multiplies tokens and cost; reduction risks quality.
2. **Encoder/LLM resource mismatch.** Compute-bound fixed-shape encoder vs the LLM's prefill+decode; co-scheduling is awkward → disaggregation appeal.
3. **Caching correctness.** Content-hash caching of encoder outputs must handle preprocessing variations (resize, crop) consistently.
4. **Variable resolution batching.** Different image sizes/tile counts complicate encoder batch formation ([§02](02_multimodal_batching_challenges.md)).

***

## Solutions & Current Best Practices
- **Batch the encoder** separately; use CUDA graphs (fixed shapes).
- **Cache encoder outputs by image hash**; reuse across requests.
- **Token merging/pruning (ToMe)** and adaptive resolution to control token count.
- **Disaggregate the encoder** onto compute-dense hardware at scale.

***

## Implementation Notes
- Normalize preprocessing (resize/crop/normalize) deterministically so image-hash caching is valid.
- Choose resolution/tiling per task (high-res for OCR, low for general) to balance cost/quality.
- Account for visual token count in LLM KV budgeting and batching.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **AnyRes tiling 5×'d the tokens** — high-res produced thousands of tokens/image; downstream prefill/KV exploded. Use adaptive resolution/ToMe.
- **Encoder-output cache missed due to preprocessing drift** — non-deterministic resize changed the hash; normalize preprocessing.
- **Encoder under-batched** — ran images one at a time; compute-bound encoder loves big batches.
- **Variable image sizes broke encoder batching** — pad/group by resolution.

***

## Performance Numbers & Benchmarks
| Setting | Visual tokens/image | Downstream cost |
|---|---|---|
| 336px base | 576 | moderate |
| High-res tiled (AnyRes) | 2k–3k+ | high |
| + ToMe merging | reduced | lower |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"What are the performance characteristics of the vision encoder vs the LLM?"* — Expected: compute-bound fixed-shape (encoder) vs prefill+decode (LLM); batch/cache/disaggregate encoder.
- *"How do you control VLM serving cost via the encoder?"* — Expected: token count (resolution/tiling), ToMe/pruning, caching.
- *"Why cache encoder outputs and how?"* — Expected: repeated images; key by content hash, deterministic preprocessing.
- *"Why disaggregate the encoder?"* — Expected: different resource profile; compute-dense hardware, separate scaling.

***

## Open Problems & Active Research (2025–2026)
- **Adaptive visual tokenization** (spend tokens where detail matters) at quality parity.
- **Standardized encoder caching/disaggregation** in serving frameworks.
- **Unified any-resolution** encoders minimizing token blowup.

***

## References
- Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words (ViT)." *ICLR 2021*. arXiv:2010.11929.
- Radford, A., et al. (2021). "CLIP." *ICML 2021*. arXiv:2103.00020.
- Bolya, D., et al. (2023). "Token Merging (ToMe)." *ICLR 2023*. arXiv:2210.09461.
