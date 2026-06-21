# Multimodal Batching Challenges

> **Section:** 11_multimodal_inference
> **Last Updated:** June 2026
> **Related Files:** [00_vision_language_model_serving.md](00_vision_language_model_serving.md), [01_image_encoding_pipeline.md](01_image_encoding_pipeline.md), [../03_batching_and_scheduling/00_static_vs_continuous_batching.md](../03_batching_and_scheduling/00_static_vs_continuous_batching.md)
> **Must-Read Papers:** Kwon et al. (2023) "vLLM"; Liu et al. (2023) "LLaVA"; Agrawal et al. (2024) "Sarathi-Serve" (chunked prefill)
> **Estimated Study Time:** 20 minutes

***

## TL;DR
- Multimodal requests are **highly heterogeneous**: 0–N images, different resolutions/tile counts → wildly different token counts and a **two-stage** (encode → LLM) pipeline.
- This breaks the uniformity continuous batching assumes: an image-heavy request has a huge prefill that **interferes** with text-only decodes (worse than text prefill-decode interference).
- The **encoder stage** (compute-bound, fixed-shape) and **LLM stage** (prefill+decode) have different batching needs → batch them **separately**.
- Mitigations: **separate encoder batching**, **chunked prefill** for huge visual prompts, **disaggregation** of encode/prefill/decode, **grouping** by similar token counts.
- KV budgeting must account for variable, often large, visual-token counts.

***

## Overview
Multimodal serving stresses the batching machinery ([§03](../03_batching_and_scheduling/00_static_vs_continuous_batching.md)) far more than text-only serving because requests are **structurally heterogeneous**. One request is text-only (short prefill); another has four high-resolution images (10k+ visual tokens, huge prefill); a third is a single low-res image. Continuous batching assumes it can mix requests and process one token per sequence per iteration efficiently, but multimodal requests differ in (a) **whether they need the encoder stage at all**, (b) **how many tokens** their prefill is (orders of magnitude apart), and (c) **the two-stage pipeline** (encode then LLM). This heterogeneity makes naive batching inefficient and amplifies the prefill-decode interference problem.

The interference is worse than in text. A text long-prefill already inflates iteration time and spikes co-batched decodes' ITL ([§00](../00_fundamentals/02_prefill_vs_decode_phases.md)); a **multi-image visual prefill** is an even larger, lumpier prefill that can stall the batch badly. And the **encoder** is a wholly separate compute stage with its own (compute-bound, fixed-shape) batching profile — running it inline with the LLM iteration loop mixes two very different workloads. The clean architectural response is to **decouple the stages**: batch the encoder separately (it loves large image batches and CUDA graphs, [§01](01_image_encoding_pipeline.md)), then feed the resulting visual tokens into the LLM's continuous-batching loop, ideally with **chunked prefill** so the large visual prompt is ingested in bounded chunks that don't starve text decodes.

The further step, at scale, is **disaggregation** of encode / prefill / decode into separate pools — extending prefill-decode disaggregation ([§03](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md)) to three stages — each batched and scaled for its profile, with visual tokens/KV transferred between them. Practical mitigations also include **grouping requests by similar token counts** to reduce padding waste, and **KV budgeting** that accounts for the large, variable visual-token contribution. This file covers the heterogeneity problem and these mitigations.

***

## Core Concepts & Mechanics

### Sources of heterogeneity
- **Modality presence**: text-only vs image vs multi-image vs video → encoder needed or not.
- **Token count**: 0 to 10k+ visual tokens per request (resolution/tiling) → vastly different prefill/KV.
- **Two stages**: encoder (compute-bound, fixed-shape) + LLM (prefill+decode) — different batching needs.

### Why it breaks batching
- Mixing a 10k-token visual prefill with text decodes inflates iteration time → severe ITL spikes (interference, amplified vs text).
- Running the encoder inline mixes a compute-bound fixed-shape workload with the memory-bound decode loop.
- Variable token counts cause padding waste if batched naively.

### Mitigations
- **Separate encoder batching**: batch images through the encoder (large batch, CUDA graphs), independent of LLM ([§01](01_image_encoding_pipeline.md)).
- **Chunked prefill** for large visual prompts: ingest in bounded chunks to protect text decode ITL ([§03](../03_batching_and_scheduling/02_chunked_prefill.md)).
- **Three-stage disaggregation**: encode / prefill / decode pools, each scaled for its profile ([§03](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md)).
- **Group by token count** to reduce padding/imbalance.
- **KV budget** including visual tokens.

***

## Key Challenges
1. **Amplified interference.** Large lumpy visual prefills stall co-batched decodes worse than text prefills.
2. **Two-stage scheduling.** Coordinating encoder and LLM stages with different profiles in one engine is awkward.
3. **Padding/imbalance.** Wildly varying token counts waste compute if not grouped.
4. **KV budgeting.** Variable, large visual-token KV complicates admission/memory planning ([§02](../02_kv_cache/01_paged_attention_vllm.md)).

***

## Solutions & Current Best Practices
- **Decouple encoder and LLM batching**; batch the encoder large + CUDA graphs.
- **Chunked prefill** for big visual prompts to protect ITL ([§03](../03_batching_and_scheduling/02_chunked_prefill.md)).
- **Disaggregate encode/prefill/decode** at scale ([§03](../03_batching_and_scheduling/03_prefill_decode_disaggregation.md)).
- **Cache encoder outputs** and **group similar-size** requests; budget KV for visual tokens.

***

## Implementation Notes
- vLLM/SGLang multimodal paths handle visual tokens; enable chunked prefill so visual prefills don't starve text decodes.
- Run the encoder as a separate batched stage; cache by image hash ([§01](01_image_encoding_pipeline.md)).
- Monitor P99 ITL under mixed text/image load — the canary for visual-prefill interference.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **A multi-image request spiked everyone's ITL** — huge visual prefill in the batch; chunk it / disaggregate.
- **Encoder ran inline, stalled the decode loop** — mixed compute-bound fixed-shape with memory-bound decode; batch the encoder separately.
- **OOM from unbudgeted visual tokens** — admission ignored visual KV; count visual tokens in the KV budget.
- **Padding waste from mixed sizes** — group requests by token count.

***

## Performance Numbers & Benchmarks
| Strategy | Effect |
|---|---|
| Inline encoder + naive batch | ITL spikes, low efficiency |
| Separate encoder batch + chunked prefill | protected ITL, efficient encode |
| Encode/prefill/decode disaggregation | each stage optimized/scaled |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why is batching harder for multimodal than text?"* — Expected: heterogeneous token counts, two-stage pipeline, amplified interference.
- *"How do you keep a multi-image prefill from starving text decodes?"* — Expected: chunked prefill / disaggregation; separate encoder batching.
- *"How would you architect multimodal serving at scale?"* — Expected: encode/prefill/decode disaggregation, encoder caching, KV budgeting.
- *"What KV consideration is unique to VLMs?"* — Expected: large variable visual-token KV in the budget.

***

## Open Problems & Active Research (2025–2026)
- **Three-stage (encode/prefill/decode) disaggregation** as a standard pattern.
- **Adaptive visual tokenization** to bound interference.
- **Schedulers aware of multimodal heterogeneity** and two-stage pipelines.

***

## References
- Kwon, W., et al. (2023). "vLLM." *SOSP 2023*. arXiv:2309.06180.
- Liu, H., et al. (2023). "LLaVA." *NeurIPS 2023*. arXiv:2304.08485.
- Agrawal, A., et al. (2024). "Sarathi-Serve." *OSDI 2024*. arXiv:2403.02310.
