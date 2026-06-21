# AI Inference Systems & Optimization — Master Index

> **A PhD-level, interview-ready knowledge database for AI inference engineering.**
> **Last Updated:** June 2026
> **Audience:** 5th-year CS PhD candidate targeting inference-focused roles (Fireworks AI, Together AI, Groq, Anyscale/Ray, Modal, Replicate, Baseten, OctoAI, NVIDIA, and inference teams at OpenAI / Anthropic / Google DeepMind / Meta AI).
> **Total Sections:** 16 folders, 100 files.

---

## How To Use This Database

This is a **file-based, cross-linked knowledge graph**. Each `.md` file is standalone and follows a fixed template (TL;DR → Overview → Core Concepts → Challenges → Solutions → Implementation → Gotchas → Benchmarks → Interview Angles → Open Problems → References).

- **Sequential learners**: follow the folder numbering 00 → 15.
- **Interview crammers**: jump to the [Top 10 Read-First](#top-10-read-first-for-interviews) list below, then the [2-Week Study Plan](#2-week-study-plan).
- **Topic lookup**: use the [Full File Map](#full-file-map) with study-time estimates.

**Convention:** ⚠️ marks production gotchas, 💡 marks interview angles, 📐 marks a required derivation. Hardware shorthand: A100 = A100 80GB SXM (HBM2e, ~2.0 TB/s); H100 = H100 SXM5 (HBM3, ~3.35 TB/s); H200 = (HBM3e, ~4.8 TB/s); B200 = Blackwell (HBM3e, FP4).

---

## The Mental Model (Read This First)

Everything in this database hangs off **one central fact**:

> **LLM decode is memory-bandwidth-bound at the batch sizes that matter; prefill is compute-bound.**

The entire field of inference optimization is the project of (a) raising the arithmetic intensity of decode so it stops wasting FLOPs, and (b) keeping prefill from starving decode. From this single tension, every technique in this database is derivable:

- **Batching** (§03) raises decode arithmetic intensity by reusing one weight load across many sequences.
- **KV cache management** (§02) is what makes large batches *fit in memory* so batching is possible.
- **Quantization** (§05) shrinks the bytes-moved-per-token, directly attacking the bandwidth bound.
- **Speculative decoding** (§07) converts bandwidth-bound decode into compute-bound verification.
- **Disaggregation** (§03, §09) physically separates the compute-bound and bandwidth-bound phases.
- **Kernel work** (§06) maximizes the fraction of peak bandwidth/FLOPs you actually achieve.

If you can re-derive each technique from the bandwidth bound in an interview, you are operating at the level these firms hire for.

---

## Top 10 Read-First (For Interviews)

In recommended order — this is the critical path:

1. `00_fundamentals/02_prefill_vs_decode_phases.md` — the single most-probed distinction.
2. `00_fundamentals/03_memory_bandwidth_bound_compute.md` — why decode is slow.
3. `00_fundamentals/04_roofline_model_for_llm.md` — the analytical framework everything uses.
4. `02_kv_cache/00_kv_cache_fundamentals.md` — the central data structure.
5. `02_kv_cache/01_paged_attention_vllm.md` — *the* foundational serving paper.
6. `03_batching_and_scheduling/00_static_vs_continuous_batching.md` — how throughput is actually won.
7. `06_kernel_optimization/01_flash_attention_deep_dive.md` — the most important kernel paper.
8. `05_quantization/04_fp8_inference_h100.md` — the dominant production quantization story.
9. `07_speculative_decoding/00_speculative_decoding_fundamentals.md` — the latency lever.
10. `03_batching_and_scheduling/03_prefill_decode_disaggregation.md` — the 2024–2026 architectural trend.

---

## Full File Map

### 00_fundamentals — *the bedrock; everything else assumes it* (≈3.5 hrs total)
| File | What it covers | Study |
|------|----------------|-------|
| [00_inference_vs_training.md](00_fundamentals/00_inference_vs_training.md) | Compute-bound vs bandwidth-bound; arithmetic intensity; why inference is a different discipline | 30 min |
| [01_autoregressive_decoding.md](00_fundamentals/01_autoregressive_decoding.md) | The token-by-token loop; sampling mechanics; decode throughput equation | 30 min |
| [02_prefill_vs_decode_phases.md](00_fundamentals/02_prefill_vs_decode_phases.md) | The core phase distinction; TTFT/ITL; prefill-decode interference | 35 min |
| [03_memory_bandwidth_bound_compute.md](00_fundamentals/03_memory_bandwidth_bound_compute.md) | Memory hierarchy bandwidths; batch breakeven point | 35 min |
| [04_roofline_model_for_llm.md](00_fundamentals/04_roofline_model_for_llm.md) | Roofline construction; worked 7B/70B examples | 40 min |
| [05_transformer_inference_walkthrough.md](00_fundamentals/05_transformer_inference_walkthrough.md) | Tensor-by-tensor forward pass with shapes and FLOPs | 35 min |
| [06_key_metrics_latency_throughput_cost.md](00_fundamentals/06_key_metrics_latency_throughput_cost.md) | TTFT/ITL/TPOT/goodput/MFU; Little's Law; SLOs | 30 min |

### 01_hardware — *the substrate* (≈3 hrs)
| File | What it covers | Study |
|------|----------------|-------|
| [00_gpu_architecture_for_inference.md](01_hardware/00_gpu_architecture_for_inference.md) | SMs, warps, Tensor Cores, MFU vs utilization | 30 min |
| [01_memory_hierarchy_HBM_SRAM.md](01_hardware/01_memory_hierarchy_HBM_SRAM.md) | HBM/L2/SRAM/registers; bandwidth & capacity tradeoffs | 25 min |
| [02_nvidia_h100_b200_architecture.md](01_hardware/02_nvidia_h100_b200_architecture.md) | H100/H200/B200/GB200 specs; FP8/FP4; NVL72 | 30 min |
| [03_interconnects_nvlink_infiniband.md](01_hardware/03_interconnects_nvlink_infiniband.md) | NVLink/NVSwitch/InfiniBand/RoCE; collective costs | 25 min |
| [04_alternative_accelerators.md](01_hardware/04_alternative_accelerators.md) | Groq LPU, TPU, Trainium, Gaudi, MI300X; NVIDIA moat | 35 min |
| [05_hardware_selection_decision_framework.md](01_hardware/05_hardware_selection_decision_framework.md) | Choosing hardware by workload | 25 min |
| [06_tco_and_cost_modeling.md](01_hardware/06_tco_and_cost_modeling.md) | Capex/opex, cost-per-token, utilization economics | 30 min |

### 02_kv_cache — *the most important section* (≈3 hrs)
| File | What it covers | Study |
|------|----------------|-------|
| [00_kv_cache_fundamentals.md](02_kv_cache/00_kv_cache_fundamentals.md) | Footprint formula; MQA/GQA; the central constraint | 30 min |
| [01_paged_attention_vllm.md](02_kv_cache/01_paged_attention_vllm.md) | Virtual-memory analogy; block tables; vLLM scheduler | 40 min |
| [02_prefix_caching_and_radix_attention.md](02_kv_cache/02_prefix_caching_and_radix_attention.md) | Exact prefix caching; SGLang RadixAttention trie | 30 min |
| [03_kv_cache_compression.md](02_kv_cache/03_kv_cache_compression.md) | StreamingLLM, H2O, SnapKV, PyramidKV | 30 min |
| [04_kv_cache_offloading.md](02_kv_cache/04_kv_cache_offloading.md) | CPU/NVMe offload, tiered KV, LMCache | 25 min |
| [05_cross_request_kv_sharing.md](02_kv_cache/05_cross_request_kv_sharing.md) | Shared prefixes across requests; privacy | 25 min |

### 03_batching_and_scheduling — *how throughput is won* (≈3 hrs)
| File | What it covers | Study |
|------|----------------|-------|
| [00_static_vs_continuous_batching.md](03_batching_and_scheduling/00_static_vs_continuous_batching.md) | Orca; iteration-level scheduling | 30 min |
| [01_iteration_level_scheduling.md](03_batching_and_scheduling/01_iteration_level_scheduling.md) | Scheduler internals; admission control | 30 min |
| [02_chunked_prefill.md](03_batching_and_scheduling/02_chunked_prefill.md) | Sarathi-Serve; chunk size tuning | 30 min |
| [03_prefill_decode_disaggregation.md](03_batching_and_scheduling/03_prefill_decode_disaggregation.md) | Splitwise/DistServe; P/D ratio | 35 min |
| [04_request_scheduling_policies.md](03_batching_and_scheduling/04_request_scheduling_policies.md) | FCFS/SJF/priority; HOL blocking; length prediction | 30 min |
| [05_multi_priority_and_SLA_scheduling.md](03_batching_and_scheduling/05_multi_priority_and_SLA_scheduling.md) | Tiered SLAs; preemption; fairness | 25 min |

### 04_parallelism — *scaling across devices* (≈2.5 hrs)
| File | What it covers | Study |
|------|----------------|-------|
| [00_tensor_parallelism.md](04_parallelism/00_tensor_parallelism.md) | Megatron TP; all-reduce volume; TP degree | 30 min |
| [01_pipeline_parallelism.md](04_parallelism/01_pipeline_parallelism.md) | Micro-batching; bubbles; PP for serving | 25 min |
| [02_sequence_parallelism.md](04_parallelism/02_sequence_parallelism.md) | SP for activations/long context | 20 min |
| [03_data_parallelism_for_serving.md](04_parallelism/03_data_parallelism_for_serving.md) | Replica-level DP; routing | 20 min |
| [04_expert_parallelism_MoE.md](04_parallelism/04_expert_parallelism_MoE.md) | MoE routing; all-to-all; DeepSeek-V3 | 35 min |
| [05_parallelism_strategy_selection.md](04_parallelism/05_parallelism_strategy_selection.md) | Combining TP/PP/EP/DP | 25 min |

### 05_quantization — *attacking bytes-per-token* (≈3.5 hrs)
| File | What it covers | Study |
|------|----------------|-------|
| [00_quantization_fundamentals.md](05_quantization/00_quantization_fundamentals.md) | Formats; granularity; error sources | 30 min |
| [01_post_training_quantization.md](05_quantization/01_post_training_quantization.md) | GPTQ, AWQ, SmoothQuant | 30 min |
| [02_weight_only_quantization.md](05_quantization/02_weight_only_quantization.md) | INT4/INT3 weight-only; dequant kernels | 25 min |
| [03_activation_quantization_challenges.md](05_quantization/03_activation_quantization_challenges.md) | Outliers; SmoothQuant migration | 25 min |
| [04_fp8_inference_h100.md](05_quantization/04_fp8_inference_h100.md) | E4M3/E5M2; Transformer Engine; ~2x throughput | 35 min |
| [05_kv_cache_quantization.md](05_quantization/05_kv_cache_quantization.md) | INT8/INT4/FP8 KV; KIVI; sensitivity | 30 min |
| [06_quantization_aware_training.md](05_quantization/06_quantization_aware_training.md) | QAT; STE; when PTQ is insufficient | 25 min |

### 06_kernel_optimization — *achieving peak* (≈3 hrs)
| File | What it covers | Study |
|------|----------------|-------|
| [00_cuda_kernel_basics_for_inference.md](06_kernel_optimization/00_cuda_kernel_basics_for_inference.md) | Execution model; Nsight; coalescing | 30 min |
| [01_flash_attention_deep_dive.md](06_kernel_optimization/01_flash_attention_deep_dive.md) | Online softmax; IO complexity; FA-2/FA-3; MLA | 40 min |
| [02_fused_kernels_and_operator_fusion.md](06_kernel_optimization/02_fused_kernels_and_operator_fusion.md) | Fusion; CUDA graphs; launch overhead | 25 min |
| [03_custom_cuda_kernels_gemm.md](06_kernel_optimization/03_custom_cuda_kernels_gemm.md) | GEMM tiling; CUTLASS; tensor-core MMA | 30 min |
| [04_triton_for_inference.md](06_kernel_optimization/04_triton_for_inference.md) | Triton DSL; autotuning; framework usage | 25 min |
| [05_kernel_profiling_and_benchmarking.md](06_kernel_optimization/05_kernel_profiling_and_benchmarking.md) | Nsight Compute/Systems; methodology | 25 min |

### 07_speculative_decoding — *the latency lever* (≈2.5 hrs)
| File | What it covers | Study |
|------|----------------|-------|
| [00_speculative_decoding_fundamentals.md](07_speculative_decoding/00_speculative_decoding_fundamentals.md) | Draft-verify; rejection sampling; speedup formula | 35 min |
| [01_draft_model_selection.md](07_speculative_decoding/01_draft_model_selection.md) | Choosing/distilling drafts; acceptance rate | 25 min |
| [02_speculative_decoding_variants.md](07_speculative_decoding/02_speculative_decoding_variants.md) | SpecInfer, EAGLE-2, REST | 30 min |
| [03_medusa_and_hydra_heads.md](07_speculative_decoding/03_medusa_and_hydra_heads.md) | Self-drafting heads; tree attention | 25 min |
| [04_lookahead_and_jacobi_decoding.md](07_speculative_decoding/04_lookahead_and_jacobi_decoding.md) | Jacobi iteration; lookahead n-grams | 25 min |
| [05_when_speculative_decoding_fails.md](07_speculative_decoding/05_when_speculative_decoding_fails.md) | Batch-size threshold; net-negative cases | 25 min |

### 08_serving_frameworks — *the tools* (≈3 hrs)
| File | What it covers | Study |
|------|----------------|-------|
| [00_serving_system_architecture.md](08_serving_frameworks/00_serving_system_architecture.md) | Engine/scheduler/worker anatomy | 25 min |
| [01_vllm_deep_dive.md](08_serving_frameworks/01_vllm_deep_dive.md) | Block manager; scheduler; LoRA; weaknesses | 35 min |
| [02_sglang_deep_dive.md](08_serving_frameworks/02_sglang_deep_dive.md) | RadixAttention; zero-overhead scheduler | 30 min |
| [03_tensorrt_llm_deep_dive.md](08_serving_frameworks/03_tensorrt_llm_deep_dive.md) | Compiled engines; in-flight batching | 30 min |
| [04_triton_inference_server.md](08_serving_frameworks/04_triton_inference_server.md) | Model serving runtime; backends | 20 min |
| [05_tgi_and_lmdeploy.md](08_serving_frameworks/05_tgi_and_lmdeploy.md) | TGI; LMDeploy/TurboMind | 20 min |
| [06_framework_selection_matrix.md](08_serving_frameworks/06_framework_selection_matrix.md) | Decision tree & comparison table | 30 min |

### 09_distributed_inference — *multi-node* (≈2.5 hrs)
| File | What it covers | Study |
|------|----------------|-------|
| [00_multi_node_serving.md](09_distributed_inference/00_multi_node_serving.md) | Cross-node TP/PP; topology | 25 min |
| [01_load_balancing_strategies.md](09_distributed_inference/01_load_balancing_strategies.md) | Prefix-aware & cache-aware routing | 25 min |
| [02_disaggregated_prefill_decode_systems.md](09_distributed_inference/02_disaggregated_prefill_decode_systems.md) | Splitwise/DistServe/Mooncake | 35 min |
| [03_kv_cache_migration_and_transfer.md](09_distributed_inference/03_kv_cache_migration_and_transfer.md) | RDMA KV transfer; LMCache | 30 min |
| [04_heterogeneous_cluster_inference.md](09_distributed_inference/04_heterogeneous_cluster_inference.md) | Mixed GPU fleets | 20 min |
| [05_fault_tolerance_in_serving.md](09_distributed_inference/05_fault_tolerance_in_serving.md) | Failures; retries; replication | 25 min |

### 10_long_context — *the N² problem* (≈2.5 hrs)
| File | What it covers | Study |
|------|----------------|-------|
| [00_long_context_challenges.md](10_long_context/00_long_context_challenges.md) | Quadratic cost; lost-in-the-middle; RoPE limits | 30 min |
| [01_attention_complexity_solutions.md](10_long_context/01_attention_complexity_solutions.md) | Survey of sub-quadratic approaches | 25 min |
| [02_ring_attention_and_context_parallelism.md](10_long_context/02_ring_attention_and_context_parallelism.md) | Ring attention; Ulysses SP | 30 min |
| [03_sparse_and_linear_attention.md](10_long_context/03_sparse_and_linear_attention.md) | Sparse/linear/SSM (Mamba) | 30 min |
| [04_retrieval_augmented_generation_vs_long_context.md](10_long_context/04_retrieval_augmented_generation_vs_long_context.md) | RAG vs long context tradeoffs | 25 min |
| [05_kv_cache_management_at_1M_tokens.md](10_long_context/05_kv_cache_management_at_1M_tokens.md) | 1M-token serving | 25 min |

### 11_multimodal_inference (≈1.5 hrs)
| File | What it covers | Study |
|------|----------------|-------|
| [00_vision_language_model_serving.md](11_multimodal_inference/00_vision_language_model_serving.md) | VLM architecture & serving | 25 min |
| [01_image_encoding_pipeline.md](11_multimodal_inference/01_image_encoding_pipeline.md) | Vision encoder; token explosion | 25 min |
| [02_multimodal_batching_challenges.md](11_multimodal_inference/02_multimodal_batching_challenges.md) | Heterogeneous-modality batching | 20 min |
| [03_video_and_audio_inference.md](11_multimodal_inference/03_video_and_audio_inference.md) | Video/audio streaming inference | 20 min |

### 12_reasoning_model_inference — *the 2025–2026 frontier* (≈2 hrs)
| File | What it covers | Study |
|------|----------------|-------|
| [00_long_chain_of_thought_serving.md](12_reasoning_model_inference/00_long_chain_of_thought_serving.md) | o1/R1-style long CoT serving | 30 min |
| [01_process_reward_model_serving.md](12_reasoning_model_inference/01_process_reward_model_serving.md) | PRM-guided search serving | 25 min |
| [02_tree_search_and_mcts_serving.md](12_reasoning_model_inference/02_tree_search_and_mcts_serving.md) | MCTS/tree search inference | 25 min |
| [03_dynamic_compute_allocation.md](12_reasoning_model_inference/03_dynamic_compute_allocation.md) | Adaptive compute per request | 25 min |
| [04_thinking_budget_and_inference_time_scaling.md](12_reasoning_model_inference/04_thinking_budget_and_inference_time_scaling.md) | Test-time scaling; compute-optimal inference | 30 min |

### 13_production_systems — *the full stack* (≈3 hrs)
| File | What it covers | Study |
|------|----------------|-------|
| [00_production_serving_architecture.md](13_production_systems/00_production_serving_architecture.md) | End-to-end request journey | 30 min |
| [01_observability_and_profiling.md](13_production_systems/01_observability_and_profiling.md) | Metrics, tracing, dashboards | 25 min |
| [02_autoscaling_and_capacity_planning.md](13_production_systems/02_autoscaling_and_capacity_planning.md) | Scaling policies; capacity math | 30 min |
| [03_model_routing_and_cascading.md](13_production_systems/03_model_routing_and_cascading.md) | RouteLLM; cascades | 25 min |
| [04_multi_tenant_serving.md](13_production_systems/04_multi_tenant_serving.md) | Isolation; S-LoRA multiplexing | 30 min |
| [05_cold_start_and_model_loading.md](13_production_systems/05_cold_start_and_model_loading.md) | Cold start; fast weight loading | 25 min |
| [06_SLO_definition_and_enforcement.md](13_production_systems/06_SLO_definition_and_enforcement.md) | SLO design; goodput | 25 min |

### 14_company_deep_dives — *know your interviewer* (≈3 hrs)
| File | What it covers | Study |
|------|----------------|-------|
| [00_fireworks_ai.md](14_company_deep_dives/00_fireworks_ai.md) | FireAttention; compound AI | 25 min |
| [01_together_ai.md](14_company_deep_dives/01_together_ai.md) | Together Inference Engine; research | 25 min |
| [02_groq_lpu_architecture.md](14_company_deep_dives/02_groq_lpu_architecture.md) | LPU/TSP deterministic execution | 30 min |
| [03_anyscale_and_ray_serve.md](14_company_deep_dives/03_anyscale_and_ray_serve.md) | Ray Serve; orchestration | 25 min |
| [04_modal_and_serverless_inference.md](14_company_deep_dives/04_modal_and_serverless_inference.md) | Serverless GPU; cold start | 25 min |
| [05_nvidia_triton_ecosystem.md](14_company_deep_dives/05_nvidia_triton_ecosystem.md) | Triton/TensorRT-LLM/NIM | 25 min |
| [06_openai_inference_platform.md](14_company_deep_dives/06_openai_inference_platform.md) | Inferred platform architecture | 25 min |
| [07_company_comparison_matrix.md](14_company_deep_dives/07_company_comparison_matrix.md) | Cross-company comparison | 30 min |

### 15_interview_prep — *the payoff* (≈4 hrs)
| File | What it covers | Study |
|------|----------------|-------|
| [00_inference_system_design_questions.md](15_interview_prep/00_inference_system_design_questions.md) | Worked system-design answers | 45 min |
| [01_kernel_and_gpu_technical_questions.md](15_interview_prep/01_kernel_and_gpu_technical_questions.md) | GPU/kernel Q&A | 35 min |
| [02_quantization_and_efficiency_questions.md](15_interview_prep/02_quantization_and_efficiency_questions.md) | Quantization Q&A | 30 min |
| [03_distributed_systems_questions.md](15_interview_prep/03_distributed_systems_questions.md) | Distributed serving Q&A | 30 min |
| [04_coding_exercises.md](15_interview_prep/04_coding_exercises.md) | Implementation exercises + solutions | 50 min |
| [05_paper_deep_dives.md](15_interview_prep/05_paper_deep_dives.md) | 10 must-know papers dissected | 45 min |
| [06_behavioral_and_research_framing.md](15_interview_prep/06_behavioral_and_research_framing.md) | Research narrative & behavioral | 25 min |

---

## 2-Week Study Plan

Assumes ~3 hrs/day. Adjust to taste; weekends are lighter for review.

**Week 1 — Foundations & Core Systems**
- **Day 1** — §00 fundamentals 00–03 (inference vs training, autoregressive, prefill/decode, bandwidth).
- **Day 2** — §00 fundamentals 04–06 (roofline, walkthrough, metrics). *You now own the mental model.*
- **Day 3** — §01 hardware 00–03 (GPU arch, memory hierarchy, H100/B200, interconnects).
- **Day 4** — §01 hardware 04–06 + §02 kv_cache 00 (accelerators, selection, TCO, KV fundamentals).
- **Day 5** — §02 kv_cache 01–05 (PagedAttention, RadixAttention, compression, offload, sharing). *Critical day.*
- **Day 6** — §03 batching 00–05 (continuous batching, chunked prefill, disaggregation, scheduling).
- **Day 7 (review)** — §04 parallelism 00–05 + re-read Top-10 #1–5.

**Week 2 — Optimization, Production & Interview**
- **Day 8** — §05 quantization 00–06 (PTQ, weight-only, activations, FP8, KV quant, QAT).
- **Day 9** — §06 kernels 00–05 (CUDA basics, FlashAttention, fusion, GEMM, Triton, profiling).
- **Day 10** — §07 speculative 00–05 + §08 frameworks 01–02 (vLLM, SGLang).
- **Day 11** — §08 frameworks 03–06 + §09 distributed 02–03 (TRT-LLM, disaggregated systems, KV transfer).
- **Day 12** — §10 long context + §12 reasoning (test-time scaling). §11 multimodal skim.
- **Day 13** — §13 production 00–06 + §14 company deep dives (focus on your target firms).
- **Day 14** — §15 interview_prep 00–06 (system design, coding, papers, behavioral). *Mock interviews.*

---

## Maintenance / Future-Update Watchlist

Topics that move fastest and should be refreshed when new papers/hardware land:
- **Blackwell (B200/GB200) real-world inference numbers** — specs are public, production benchmarks still maturing.
- **FP4 inference quality** — NVFP4/MXFP4 accuracy at scale is actively being characterized.
- **Disaggregation in production** — Mooncake, vLLM/SGLang PD support evolving monthly.
- **Reasoning-model serving economics** — long-CoT cost models are new and unsettled.
- **Speculative decoding SOTA** — EAGLE-3 and successors; acceptance-rate frontier.
- **Linear attention / SSM hybrids** — Mamba-2, hybrid architectures in production.
- **Framework leaderboard** — vLLM vs SGLang vs TensorRT-LLM ranking shifts release-to-release.

---

*This index is the entry point. Start with the [Mental Model](#the-mental-model-read-this-first), then the [Top 10](#top-10-read-first-for-interviews).*
