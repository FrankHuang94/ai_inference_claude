# Video and Audio Inference

> **Section:** 11_multimodal_inference
> **Last Updated:** June 2026
> **Related Files:** [00_vision_language_model_serving.md](00_vision_language_model_serving.md), [01_image_encoding_pipeline.md](01_image_encoding_pipeline.md), [../10_long_context/00_long_context_challenges.md](../10_long_context/00_long_context_challenges.md)
> **Must-Read Papers:** Radford et al. (2022) "Whisper"; Zhang et al. (2023) "Video-LLaMA"; Gemini/long-video technical reports
> **Estimated Study Time:** 20 minutes

***

## TL;DR
- **Video** = many frames → an explosion of visual tokens (frames × tokens/frame), making it an extreme **long-context** problem; aggressive token reduction and sampling are mandatory.
- **Audio** (speech): encode waveform → features (mel-spectrogram) → tokens; **streaming** ASR/audio-LLMs need low-latency incremental processing.
- Both add a **temporal** dimension: frame/clip sampling, temporal pooling, and streaming change the serving profile vs static images.
- Token budgets dominate: a few minutes of video can be hundreds of thousands of tokens → KV/prefill blowup (ties to [§10](../10_long_context/00_long_context_challenges.md)).
- Streaming use cases (live transcription, real-time video) demand incremental encoders and low TTFT.

***

## Overview
Video and audio extend multimodal serving with a **temporal** axis, which compounds the token-inflation problem of images ([§00](00_vision_language_model_serving.md)) and pushes serving into long-context territory ([§10](../10_long_context/00_long_context_challenges.md)). **Video** is the harder case: a video is a sequence of frames, each of which (via a ViT encoder) becomes hundreds of visual tokens, so even modest clips produce enormous token counts — a few minutes at a few frames/second can be hundreds of thousands of tokens. This makes video understanding fundamentally a **long-context + token-reduction** problem: you cannot feed every frame at full resolution, so systems aggressively **sample frames** (uniform or keyframe-based), **pool tokens temporally**, and **merge/prune** visual tokens to fit a tractable budget — trading temporal/spatial detail for feasibility.

**Audio** divides into understanding (speech-to-text or audio-LLM input) and generation (TTS). For input, the pipeline is waveform → **feature extraction** (mel-spectrogram) → audio encoder → tokens, as in Whisper (Radford et al. 2022) for ASR and audio-LLMs that feed audio tokens to an LLM. Audio token rates are lower than video but the defining serving concern is often **streaming**: live transcription and real-time voice assistants need **incremental** encoding and decoding with low latency, processing audio in chunks as it arrives rather than waiting for a complete utterance — a different serving pattern (continuous streaming with low TTFT) than batch image/text.

The unifying serving themes: **token-budget management** (video especially is a long-context problem — apply CP, KV compression, frame/token reduction from [§10](../10_long_context/05_kv_cache_management_at_1M_tokens.md)), **temporal sampling/pooling** as the primary cost lever, and **streaming** for real-time audio/video (incremental encoders, chunked processing, low-latency decode). This file covers the video token-explosion problem, the audio/streaming pipeline, and their serving implications. It's the lightest file in the section, reflecting that video/audio LLM serving is newer and rapidly evolving (a watchlist area).

***

## Core Concepts & Mechanics

### Video token explosion
📐 Tokens ≈ `frames × tokens_per_frame`. At 1 fps for 3 min (180 frames) × 256 tokens/frame ≈ 46k tokens; higher fps/resolution → hundreds of thousands. → long-context KV/prefill problem ([§10](../10_long_context/00_long_context_challenges.md)).
- **Mitigations**: frame sampling (uniform/keyframe), temporal pooling, token merging (ToMe), lower per-frame resolution.

### Audio pipeline
```
waveform → mel-spectrogram (feature extraction) → audio encoder → audio tokens → LLM/decoder
```
- **Whisper**: encoder-decoder ASR; ~lower token rate than video.
- **Streaming**: process audio in chunks (e.g., 30s windows or smaller), incremental decode, low TTFT for live transcription.

### Temporal dimension
- **Sampling**: which frames/clips to include (cost vs coverage).
- **Pooling**: aggregate tokens across time to reduce count.
- **Streaming**: incremental processing for real-time; bounded latency per chunk.

### Serving profile
- Video: long-context-dominated; apply CP, KV compression, token reduction.
- Audio streaming: latency-dominated; incremental encoder/decoder, small chunks.

***

## Key Challenges
1. **Video token explosion.** Frames × tokens/frame quickly exceeds practical context; sampling/pooling/merging are mandatory and lossy.
2. **Long-context cost.** Video inherits all long-context problems (O(N²) prefill, huge KV) ([§10](../10_long_context/00_long_context_challenges.md)).
3. **Streaming latency (audio/live video).** Incremental processing with low TTFT requires different serving than batch.
4. **Temporal sampling tradeoffs.** Aggressive frame sampling loses events; choosing what to keep is hard and task-dependent.

***

## Solutions & Current Best Practices
- **Frame sampling + temporal pooling + token merging** to bound video tokens.
- **Long-context techniques** (CP, KV compression/quantization) for video ([§10](../10_long_context/05_kv_cache_management_at_1M_tokens.md)).
- **Streaming pipelines** (chunked, incremental) for live audio/video with low TTFT.
- **Cache encoder outputs** for repeated clips/frames where applicable ([§01](01_image_encoding_pipeline.md)).

***

## Implementation Notes
- Choose frame rate/sampling per task (dense for action, sparse for summary) to control tokens.
- For streaming audio, process fixed windows incrementally; manage overlapping context across chunks.
- Budget KV for the (large) temporal token counts; apply CP/quantization for long video.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Full-frame-rate video OOM'd / timed out** — token explosion; sample frames and merge tokens.
- **Sparse frame sampling missed a key event** — temporal undersampling lost information; task-aware sampling.
- **Streaming audio had high TTFT** — waited for full utterance; process incremental chunks.
- **Chunk boundaries broke audio context** — overlapping windows / carried state needed across chunks.

***

## Performance Numbers & Benchmarks
| Modality | Token driver | Serving regime |
|---|---|---|
| Image | resolution/tiling | prefill-heavy |
| Video | frames × tokens/frame | long-context (huge KV) |
| Audio (batch) | duration × rate | moderate |
| Audio (streaming) | incremental chunks | low-latency streaming |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why is video understanding a long-context problem?"* — Expected: frames × tokens/frame → huge token counts → KV/prefill blowup.
- *"How do you make video serving tractable?"* — Expected: frame sampling, temporal pooling, token merging, CP/KV compression.
- *"What's special about streaming audio serving?"* — Expected: incremental/chunked processing, low TTFT, context across chunks.
- *"What's the main cost lever for video?"* — Expected: temporal sampling / token count.

***

## Open Problems & Active Research (2025–2026)
- **Efficient long-video** understanding (hours) without prohibitive tokens.
- **Adaptive temporal tokenization** (spend tokens on salient moments).
- **Low-latency streaming multimodal** (real-time video + audio + text) serving.

***

## References
- Radford, A., et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)." arXiv:2212.04356.
- Zhang, H., et al. (2023). "Video-LLaMA." arXiv:2306.02858.
- Google DeepMind (2024). Gemini long-video / multimodal technical reports.
