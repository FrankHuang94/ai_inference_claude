# Anyscale and Ray Serve

> **Section:** 14_company_deep_dives
> **Last Updated:** June 2026
> **Related Files:** [07_company_comparison_matrix.md](07_company_comparison_matrix.md), [../13_production_systems/00_production_serving_architecture.md](../13_production_systems/00_production_serving_architecture.md), [../09_distributed_inference/00_multi_node_serving.md](../09_distributed_inference/00_multi_node_serving.md)
> **Must-Read Papers:** Moritz et al. (2018, OSDI) "Ray: A Distributed Framework for Emerging AI Applications"; Ray Serve documentation
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- Anyscale is the company behind **Ray**, a general distributed-computing framework; **Ray Serve** is its model-serving library — an **orchestration layer**, not an LLM engine itself.
- Strength: **orchestrating complex, multi-stage, multi-model pipelines** (RAG, agents, compound AI, preprocessing) and **autoscaling** across a cluster — composing vLLM/other engines as components.
- Ray underpins **distributed training, data processing, RL, and serving** in one framework — appealing for end-to-end AI infrastructure.
- For pure single-LLM serving, a dedicated engine (vLLM/SGLang) is simpler; Ray Serve shines for **heterogeneous pipelines and scale orchestration**.
- **Interview focus**: distributed systems, orchestration, autoscaling, pipeline composition — more systems/infra than kernels.

***

## Overview
Anyscale commercializes **Ray**, the open-source distributed-computing framework (Moritz et al. 2018) widely used for distributed training, hyperparameter tuning, RL, data processing, and serving. **Ray Serve** is Ray's model-serving library, and the key framing for this database is that it's an **orchestration layer** rather than an LLM inference engine: Ray Serve doesn't replace vLLM/SGLang/TRT-LLM — it **composes and scales** them (and other components) into deployable applications. It excels where serving is more than "one model behind an API": **multi-stage pipelines** (retrieval → rerank → LLM → postprocess), **multi-model** applications, **agents/compound AI**, and **autoscaling heterogeneous components** across a cluster.

Ray's value proposition is **one framework for the whole AI lifecycle**: the same cluster and abstractions handle distributed data preprocessing, training, and serving, with Ray's actor/task model providing distributed orchestration, fault tolerance, and autoscaling. For organizations building complex AI systems (not just exposing a single model), this end-to-end coherence is compelling — you orchestrate a RAG or agentic pipeline as Ray Serve deployments, each independently scaled, with vLLM as the LLM-serving component inside.

The tradeoff is that for **pure single-LLM serving**, Ray Serve adds an orchestration layer over what a dedicated engine already does well — so a team just serving one model often uses vLLM/SGLang directly. Ray Serve earns its place when the application is a **heterogeneous, multi-component, autoscaled pipeline**. For candidates, Anyscale/Ray roles emphasize **distributed systems, orchestration, autoscaling, and pipeline composition** (this section + §09 + §13) more than low-level kernel work — it's the infrastructure/distributed-systems end of inference. This file covers Ray/Ray Serve's role, strengths, the orchestration-vs-engine distinction, and interview emphasis.

***

## Core Concepts & Mechanics

### Ray and Ray Serve
- **Ray**: distributed framework (actors/tasks) for training, tuning, RL, data, serving — one cluster, one API.
- **Ray Serve**: serving library on Ray; **deployments** (scalable units) composed into **applications** (pipelines/graphs). Orchestrates, autoscales, and connects components.
- **Not an LLM engine**: composes vLLM/SGLang/etc. as the LLM-serving component.

### What it's good at
- **Multi-stage pipelines**: retrieval → rerank → LLM → postprocess as composed deployments.
- **Multi-model / compound AI / agents**: orchestrate many models/tools.
- **Autoscaling heterogeneous components** independently across a cluster.
- **End-to-end lifecycle**: train + data + serve in one framework.

### Engine vs orchestration
- **Engine** (vLLM/SGLang/TRT-LLM): the per-model continuous-batching/KV serving runtime ([§08](../08_serving_frameworks/00_serving_system_architecture.md)).
- **Orchestration** (Ray Serve): composes/scales engines + other stages into applications.
- They're complementary: Ray Serve + vLLM is a common stack.

### Interview emphasis
Distributed systems, orchestration, autoscaling, fault tolerance, pipeline composition — infra/distributed-systems end ([§09](../09_distributed_inference/00_multi_node_serving.md), [§13](../13_production_systems/00_production_serving_architecture.md)).

***

## Key Challenges
1. **Orchestration overhead.** Adds a layer over the engine; for single-model serving it may be unnecessary complexity.
2. **Pipeline performance.** Multi-stage pipelines have cumulative latency; each stage and the orchestration must be efficient.
3. **Autoscaling heterogeneous components.** Different stages (retrieval, LLM, rerank) have different scaling profiles to coordinate.
4. **Complexity.** Ray's generality is powerful but has a learning curve and operational surface.

***

## Solutions & Current Best Practices (where it fits)
- **Ray Serve for complex multi-component/agentic pipelines** with autoscaling; **vLLM as the LLM component** inside.
- **Dedicated engine directly** for simple single-model serving.
- **Leverage Ray's lifecycle** (train + data + serve) when building end-to-end.

***

## Implementation Notes (for interview prep)
- Understand Ray's actor/task model and Ray Serve deployments/applications.
- Know how to compose vLLM inside Ray Serve and autoscale stages independently.
- Frame the engine-vs-orchestration distinction clearly.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Used Ray Serve for a single model** — added orchestration complexity over vLLM with little benefit; use the engine directly.
- **Pipeline latency stacked up** — multi-stage cumulative latency; optimize each stage + orchestration overhead.
- **Confused Ray Serve with an LLM engine** — it orchestrates engines, doesn't replace them.
- **Autoscaling stages uncoordinated** — heterogeneous profiles; coordinate scaling.

***

## Performance Numbers & Benchmarks
| Use case | Fit |
|---|---|
| Single LLM behind API | vLLM/SGLang directly |
| Multi-stage RAG/agent pipeline | Ray Serve (+ vLLM) |
| End-to-end (train+data+serve) | Ray |
| Autoscaled heterogeneous app | Ray Serve |

***

## Interview Angles
> 💡 **What Anyscale/Ray (and similar infra) ask:**
- *"Ray Serve vs vLLM — what's the difference?"* — Expected: orchestration layer vs LLM engine; complementary (Ray Serve + vLLM).
- *"How would you serve a multi-stage RAG/agent pipeline?"* — Expected: compose stages as deployments, autoscale independently, vLLM as LLM stage.
- *"When is Ray Serve overkill?"* — Expected: simple single-model serving.
- *"How does Ray provide fault tolerance/autoscaling?"* — Expected: actor/task model, supervision, autoscaler ([§09](../09_distributed_inference/05_fault_tolerance_in_serving.md)).

***

## Open Problems & Active Research (2025–2026)
- **Efficient agentic/compound-AI orchestration** (low-latency multi-stage).
- **Tighter engine integration** (KV-aware routing across pipeline stages).
- **Autoscaling complex heterogeneous pipelines** ([§13](../13_production_systems/02_autoscaling_and_capacity_planning.md)).

***

## References
- Moritz, P., et al. (2018). "Ray: A Distributed Framework for Emerging AI Applications." *OSDI 2018*. arXiv:1712.05889.
- Ray / Ray Serve documentation (Anyscale).
- Kwon, W., et al. (2023). "vLLM" (composed component). arXiv:2309.06180.
