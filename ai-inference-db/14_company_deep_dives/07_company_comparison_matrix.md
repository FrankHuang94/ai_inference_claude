# Company Comparison Matrix (Inference Cloud Landscape)

> **Section:** 14_company_deep_dives
> **Last Updated:** June 2026
> **Related Files:** [00_fireworks_ai.md](00_fireworks_ai.md), [02_groq_lpu_architecture.md](02_groq_lpu_architecture.md), [../01_hardware/06_tco_and_cost_modeling.md](../01_hardware/06_tco_and_cost_modeling.md)
> **Must-Read Papers:** company engineering blogs; SemiAnalysis market analyses; this database's §00–§13
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- The **inference cloud** landscape exists because closed-lab APIs are expensive and lock-in-prone, and self-hosting is hard — these firms serve open (and custom) models cheaply/fast.
- **Differentiators**: Fireworks/Together (serving speed + kernels/research), Groq (latency/LPU), Anyscale (orchestration/Ray), Modal/Replicate/Baseten (serverless/DX), NVIDIA (integrated stack).
- Compete on: **$/token, latency (TTFT/ITL), model breadth, custom-model/LoRA support, dedicated vs shared, SLAs, DX, differentiator**.
- The market is consolidating around **a few performance leaders + serverless DX players + the NVIDIA stack**, with reasoning-model serving as the new frontier.
- Always compare on **$/token at SLO for your workload**, not headline numbers.

***

## Overview
This file synthesizes the deep-dives into a comparison and the market narrative. The **inference cloud** category exists for a clear reason: the closed frontier labs (OpenAI, Anthropic, Google) offer powerful but **expensive, proprietary, lock-in-prone** APIs, while **self-hosting** open models requires deep GPU/serving expertise most companies lack. Inference clouds fill the gap — hosting open-weight (and custom) models behind fast, cheap, OpenAI-compatible APIs, monetizing the **serving efficiency** (high utilization + kernel/quantization optimization) that this database describes ([§01](../01_hardware/06_tco_and_cost_modeling.md)). They compete to deliver the best $/token and latency on the same open models.

The players differentiate along distinct axes. **Fireworks** and **Together** compete on **raw serving performance and cost** via custom kernels and (for Together) research depth (FlashAttention lineage). **Groq** competes on **latency** with its SRAM-only LPU — a different hardware bet ([§02](02_groq_lpu_architecture.md)). **Anyscale/Ray** competes on **orchestration** of complex pipelines rather than raw serving ([§03](03_anyscale_and_ray_serve.md)). **Modal/Replicate/Baseten/Cerebrium** compete on **serverless and developer experience** for bursty/custom workloads ([§04](04_modal_and_serverless_inference.md)). **NVIDIA** offers the **integrated stack** (TRT-LLM/Triton/NIM) leveraging its hardware+CUDA moat ([§05](05_nvidia_triton_ecosystem.md)). And the closed labs (OpenAI, Anthropic, Google) serve their own frontier models at scale ([§06](06_openai_inference_platform.md)).

The market dynamics: rapid commoditization of core serving techniques (everyone has continuous batching/paged KV/quantization) pushes competition toward **kernel/latency leadership, DX, model freshness, and unit economics (utilization)**, with **reasoning-model serving** as the emerging differentiator (it's expensive and not yet solved, [§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)). The landscape is consolidating around a few performance leaders, serverless DX players, and the NVIDIA stack. The enduring advice: compare on **$/token at your SLO for your workload**, since headline benchmarks are workload- and version-specific. This file gives the matrix and the narrative.

***

## Core Concepts & Mechanics

### Comparison matrix
| Company | Differentiator | Pricing model | Latency | Custom/LoRA | Best for |
|---|---|---|---|---|---|
| **Fireworks** | FireAttention kernels, speed/cost | per-token | low | fine-tune, LoRA | fast/cheap open-model API |
| **Together** | research-to-prod (FlashAttention), train+serve | per-token + clusters | low | fine-tune, LoRA, clusters | open-model + training |
| **Groq** | LPU (SRAM, deterministic) latency | per-token | **lowest** | limited models | latency-critical, smaller models |
| **Anyscale (Ray)** | orchestration (Ray Serve) | compute/cluster | pipeline-dep | via composition | complex pipelines/agents |
| **Modal** | serverless GPU, DX | per-use (scale-to-zero) | cold-start-dep | any/custom | bursty/batch/custom |
| **Replicate** | model sharing + serverless | per-use | cold-start-dep | any/custom | community models, custom |
| **Baseten** | production serverless (Truss) | per-use/dedicated | tunable | any/custom | productionizing custom models |
| **NVIDIA (NIM/TRT-LLM)** | integrated stack, latency | license/cloud | **lowest fixed-shape** | via build | NVIDIA-optimized enterprise |
| **OpenAI/Anthropic/Google** | frontier proprietary models | per-token | tuned | limited | best-quality closed models |

### Why the category exists
Closed APIs: expensive, proprietary, lock-in. Self-host: hard. Inference clouds: cheap/fast open + custom models, monetizing serving efficiency ([§01](../01_hardware/06_tco_and_cost_modeling.md)).

### Competitive axes
$/token, TTFT/ITL latency, model breadth/freshness, custom-model & LoRA, dedicated vs shared, SLAs/enterprise, DX, technical differentiator, reasoning-model support.

***

## Key Challenges (market-wide)
1. **Commoditization.** Core techniques converged; differentiation must come from kernels/latency/DX/economics/reasoning ([§08](../08_serving_frameworks/06_framework_selection_matrix.md)).
2. **Unit economics.** Price competition demands high utilization + efficiency; thin margins ([§01](../01_hardware/06_tco_and_cost_modeling.md)).
3. **Reasoning-model cost.** The expensive new frontier; whoever serves long-CoT cheaply wins ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).
4. **NVIDIA dependence.** Most depend on NVIDIA GPUs (cost/supply); the moat debate ([§01](../01_hardware/04_alternative_accelerators.md)).

***

## Solutions & Current Best Practices (how to choose)
- **Match workload to differentiator**: speed/cost → Fireworks/Together; latency → Groq; pipelines → Anyscale; bursty/custom → Modal/Replicate/Baseten; NVIDIA-enterprise → NIM; best quality → closed labs.
- **Benchmark $/token at your SLO** on your workload, not headline numbers.
- **Consider reasoning-model support** if that's your use case ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

***

## Implementation Notes (for interview prep)
- Know each company's one-line differentiator and ideal workload.
- Be able to recommend a provider given a workload's latency/cost/custom/burstiness profile.
- Frame the "why inference clouds exist" market thesis.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Chose on headline tok/s** — workload/version-specific; benchmark your own at SLO.
- **Serverless for steady high volume** — cold start + economics; dedicated/inference-cloud wins ([§04](04_modal_and_serverless_inference.md)).
- **Groq for large-model throughput** — it's a latency play; capacity limits ([§02](02_groq_lpu_architecture.md)).
- **Ignored reasoning-model cost** — if serving o-series-style models, it dominates; check support ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

***

## Performance Numbers & Benchmarks
| Axis | Leaders |
|---|---|
| Lowest latency (small models) | Groq |
| Speed/cost on open models | Fireworks, Together |
| Serverless/DX | Modal, Replicate, Baseten |
| Orchestration/pipelines | Anyscale (Ray) |
| Integrated NVIDIA stack | NVIDIA (NIM/TRT-LLM) |
| Frontier quality | OpenAI/Anthropic/Google |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why do inference clouds exist, and how do they compete?"* — Expected: closed-API cost/lock-in + self-host difficulty; compete on $/token, latency, DX, differentiator.
- *"Recommend a provider for [workload]."* — Expected: match latency/cost/custom/burstiness to differentiator.
- *"What's each major player's differentiator?"* — Expected: the matrix (FireAttention, FlashAttention/train+serve, LPU latency, Ray orchestration, serverless DX, NVIDIA stack).
- *"Where is the market heading?"* — Expected: commoditization → kernel/latency/DX/economics; reasoning-model serving frontier.

***

## Open Problems & Active Research (2025–2026)
- **Reasoning-model serving** as the key competitive frontier ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).
- **Differentiation under commoditization** (custom silicon, DX, economics).
- **NVIDIA-alternative** viability and its market effect ([§01](../01_hardware/04_alternative_accelerators.md)).

***

## References
- Company engineering blogs (Fireworks, Together, Groq, Modal, Anyscale, NVIDIA).
- SemiAnalysis and industry market analyses (2023–2025).
- This database, §00–§13.
