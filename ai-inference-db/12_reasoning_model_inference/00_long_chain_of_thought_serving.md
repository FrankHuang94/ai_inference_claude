# Long Chain-of-Thought Serving

> **Section:** 12_reasoning_model_inference
> **Last Updated:** June 2026
> **Related Files:** [04_thinking_budget_and_inference_time_scaling.md](04_thinking_budget_and_inference_time_scaling.md), [03_dynamic_compute_allocation.md](03_dynamic_compute_allocation.md), [../00_fundamentals/06_key_metrics_latency_throughput_cost.md](../00_fundamentals/06_key_metrics_latency_throughput_cost.md)
> **Must-Read Papers:** OpenAI (2024) "Learning to Reason (o1)"; DeepSeek-AI (2025) "DeepSeek-R1"; Snell et al. (2024) "Scaling Test-Time Compute"
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- Reasoning models (o1/o3, DeepSeek-R1, Gemini Thinking) generate very long **chain-of-thought** — 10k–100k+ tokens — before the final answer, inverting serving economics.
- **Decode dominates massively**: output (thinking) tokens ≫ input; TTFT becomes irrelevant, **cost/throughput per reasoning token** is the metric.
- **KV explosion**: a single request's KV reaches **tens to hundreds of GB** (long-context problem per request) ([§10](../10_long_context/05_kv_cache_management_at_1M_tokens.md)).
- **Unpredictable length**: thinking length varies wildly → hard scheduling, batching, and capacity planning.
- Serving strategies: dedicated reasoning clusters, **thinking-budget control**, streaming vs withholding thinking, cost-aware routing.

***

## Overview
Reasoning models trained to "think before answering" (OpenAI o1/o3, DeepSeek-R1, Gemini Thinking) generate extended internal chain-of-thought before emitting a final answer, and that CoT can be **10,000 to 100,000+ tokens**. This fundamentally inverts the serving profile assumed throughout this database. For normal chat, prompts are often long and outputs short, so prefill matters and TTFT is a key SLO. For reasoning, the **output dwarfs the input** — the model decodes for a very long time — so **decode utterly dominates** wall-clock and cost, TTFT becomes nearly irrelevant (the user waits for the thinking regardless), and the meaningful metric becomes **cost and throughput per reasoning token** (or per solved task), not first-token latency.

This creates three acute serving problems. First, **KV explosion**: because KV grows linearly with sequence length ([§02](../02_kv_cache/00_kv_cache_fundamentals.md)), a single reasoning request generating 100k tokens accumulates a KV cache of tens to hundreds of GB — the long-context problem ([§10](../10_long_context/05_kv_cache_management_at_1M_tokens.md)) now hits *every* reasoning request, not just long-prompt ones. Second, **unpredictable length**: the model decides how long to think, so output length is highly variable and unknown a priori, wrecking the output-length prediction that scheduling ([§03](../03_batching_and_scheduling/04_request_scheduling_policies.md)) and capacity planning rely on — requests in a batch finish at wildly different times, and a few very long thinkers dominate resource consumption. Third, **cost variance**: a hard query might cost 100× a easy one in tokens, so flat per-request economics break.

The serving strategies that follow: **dedicated reasoning clusters** (decode-optimized hardware — H200/MI300X for bandwidth/capacity — since decode dominates); **thinking-budget control** (cap CoT length to bound cost/latency, [§04](04_thinking_budget_and_inference_time_scaling.md)); **streaming vs withholding** the thinking tokens (UX/product choice, and whether to bill them); **dynamic compute allocation** (spend more thinking on hard queries, [§03](03_dynamic_compute_allocation.md)); and **cost-aware routing** (only invoke the expensive reasoning model when needed, [§13](../13_production_systems/03_model_routing_and_cascading.md)). This file frames the inverted economics and the core challenges; the rest of the section covers PRM/search serving and test-time scaling.

***

## Core Concepts & Mechanics

### Inverted economics
📐 Normal chat: E2E ≈ TTFT + (short output)×ITL → prefill/TTFT matters. Reasoning: E2E ≈ TTFT + (10k–100k tokens)×ITL → **decode dominates**; TTFT negligible; cost ≈ output_tokens × decode_cost. Metric: **$/reasoning-token** or **$/solved-task**.

### KV explosion per request
📐 100k tokens × ~320 KB/token (70B GQA) ≈ **32 GB KV for one request**; larger models / 100k+ tokens → hundreds of GB. Every reasoning request is a long-context request → apply CP, KV quantization/compression ([§10](../10_long_context/05_kv_cache_management_at_1M_tokens.md)).

### Unpredictable length
Output length is model-decided and highly variable (easy vs hard queries differ 100×). Breaks:
- **Length prediction** for scheduling/SJF ([§03](../03_batching_and_scheduling/04_request_scheduling_policies.md)).
- **Batching**: requests finish at very different times; long thinkers hold KV/slots.
- **Capacity planning**: variance dominates; provision for the tail.

### Serving strategies
- **Dedicated reasoning clusters**: decode-optimized hardware (bandwidth/capacity).
- **Thinking-budget control**: cap CoT tokens ([§04](04_thinking_budget_and_inference_time_scaling.md)).
- **Stream vs withhold thinking**: product/UX + billing choice.
- **Cost-aware routing**: invoke reasoning only when needed ([§13](../13_production_systems/03_model_routing_and_cascading.md)).

***

## Key Challenges
1. **KV per request is huge.** Every reasoning request is long-context; KV memory caps batch hard → low throughput without aggressive KV management.
2. **Length unpredictability.** Wildly variable thinking length breaks scheduling, batching, and capacity planning; long thinkers monopolize resources.
3. **Cost variance & billing.** 100× cost spread per query; whether to bill hidden thinking tokens is a product/economics question.
4. **Throughput vs depth.** Longer thinking improves quality but costs more; the tradeoff is per-query and dynamic ([§04](04_thinking_budget_and_inference_time_scaling.md)).

***

## Solutions & Current Best Practices
- **Decode-optimized dedicated clusters** (H200/MI300X) + aggressive **KV quantization/compression/CP** ([§10](../10_long_context/05_kv_cache_management_at_1M_tokens.md)).
- **Thinking-budget caps** and **dynamic compute** allocation ([§04](04_thinking_budget_and_inference_time_scaling.md), [§03](03_dynamic_compute_allocation.md)).
- **Cost-aware routing**: cheap model first, escalate to reasoning only when needed ([§13](../13_production_systems/03_model_routing_and_cascading.md)).
- **Provision for the length tail**; track $/solved-task, not $/request.

***

## Implementation Notes
- Optimize for decode throughput (batching across many reasoning requests) and KV efficiency; TTFT tuning is wasted effort here.
- Decide streaming-thinking vs answer-only (UX + whether thinking is billed/exposed).
- Use long-context KV management per request; budget for the worst-case thinker.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Throughput collapsed serving R1** — each request's 50k-token KV capped batch to a handful; needs KV quant/compression/CP.
- **Optimized TTFT for a reasoning model** — wasted effort; decode dominates, TTFT is noise.
- **A few hard queries monopolized the cluster** — 100× length variance; thinking-budget caps + tail provisioning.
- **Flat per-request pricing lost money** — huge cost variance; price per token / per solved task.

***

## Performance Numbers & Benchmarks
| Aspect | Normal chat | Reasoning (long CoT) |
|---|---|---|
| Output tokens | ~10²–10³ | 10⁴–10⁵+ |
| Dominant cost | mixed | decode |
| KV/request | ~GB | tens–hundreds of GB |
| Key metric | TTFT/ITL | $/reasoning-token, $/solved-task |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How does serving a reasoning model differ from a chat model?"* — Expected: decode-dominated, KV explosion, length unpredictability, TTFT irrelevant.
- *"How would you reduce the cost of serving long-CoT models by 50%?"* — Expected: KV quantization/compression, thinking-budget control, speculative decoding, routing ([§15](../15_interview_prep/00_inference_system_design_questions.md)).
- *"Why does length unpredictability hurt scheduling?"* — Expected: breaks length prediction/batching; long thinkers monopolize.
- *"What's the right cost metric for reasoning?"* — Expected: $/reasoning-token or $/solved-task.

***

## Open Problems & Active Research (2025–2026)
- **Cost models and pricing** for reasoning/agentic workloads (watchlist).
- **KV management for 100k+ per-request CoT** at high batch.
- **Adaptive thinking budgets** that match compute to query difficulty ([§03](03_dynamic_compute_allocation.md), [§04](04_thinking_budget_and_inference_time_scaling.md)).

***

## References
- OpenAI (2024). "Learning to Reason with LLMs (o1)."
- DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL." arXiv:2501.12948.
- Snell, C., et al. (2024). "Scaling LLM Test-Time Compute Optimally." arXiv:2408.03314.
