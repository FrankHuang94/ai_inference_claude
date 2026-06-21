# Model Routing and Cascading

> **Section:** 13_production_systems
> **Last Updated:** June 2026
> **Related Files:** [04_multi_tenant_serving.md](04_multi_tenant_serving.md), [../12_reasoning_model_inference/03_dynamic_compute_allocation.md](../12_reasoning_model_inference/03_dynamic_compute_allocation.md), [00_production_serving_architecture.md](00_production_serving_architecture.md)
> **Must-Read Papers:** Ong et al. (2024) "RouteLLM"; Chen et al. (2023) "FrugalGPT"; Snell et al. (2024) "Test-Time Compute"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- **Model routing**: pick which model serves each request — simple → small/cheap, hard → large/expensive — to cut cost at fixed quality.
- **Cascading**: try a small model first; **escalate** to a larger one only if confidence is low (FrugalGPT) — pay for the big model only when needed.
- **RouteLLM** (Ong et al. 2024): learn the router from preference data, tracing an **accuracy-cost Pareto frontier**.
- Key tradeoff: **router accuracy vs savings** — a perfect router approaches big-model quality at small-model average cost; a bad router wastes escalations or misroutes.
- Central pattern at inference clouds and for reasoning-model cost control ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

***

## Overview
Most queries don't need your most expensive model: a large fraction are simple enough for a small, cheap model to answer correctly, and only a minority require a frontier (or reasoning) model. **Model routing** exploits this by deciding, per request, which model to use — sending easy queries to cheap models and hard ones to expensive models — cutting average cost while preserving quality. **Cascading** is the sequential form: run the cheap model first, and **escalate** to a bigger model only when a confidence/quality signal says the cheap answer is inadequate (FrugalGPT, Chen et al. 2023). Both are central cost levers at inference clouds (where margins depend on serving cheaply) and for reasoning models (invoke expensive long-CoT only when needed, [§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

The defining artifact is the **accuracy-cost Pareto frontier**. **RouteLLM** (Ong et al. 2024) learns a router from human-preference data that, for any target quality, picks the cheapest model mix achieving it — empirically recovering most of GPT-4-level quality at a fraction of the cost by routing only the hard queries to the strong model. The router's job is essentially **difficulty/quality prediction** (will the cheap model suffice?), the same estimation problem as dynamic compute allocation ([§12](../12_reasoning_model_inference/03_dynamic_compute_allocation.md)) — and its quality determines how close you get to the frontier.

The tradeoffs and costs: **router overhead** (a routing call/classifier adds latency and a little cost — must be cheap relative to savings), **misrouting** (sending a hard query to the small model hurts quality; sending easy ones to the big model wastes money), and for cascades, **double-cost on escalation** (you pay the small model *plus* the large one when escalating, so escalation rate must be low enough to net save). A well-tuned router/cascade delivers large savings; a poorly-calibrated one can cost more than just using the big model. This file covers routing vs cascading, RouteLLM, the difficulty-estimation core, and the tradeoffs.

***

## Core Concepts & Mechanics

### Routing vs cascading
- **Routing** (parallel decision): a router classifies the query up front → send to the chosen model once. One model runs.
- **Cascading** (sequential): run cheap model → check confidence → escalate to bigger model if low. 📐 Cost = `c_small + P(escalate)·c_large`; nets savings only if `P(escalate)` is low enough.

### RouteLLM and the Pareto frontier
Learn a router (from preference/win-rate data) that, per target quality q, routes the minimal fraction to the strong model. Traces the **accuracy-cost frontier**; e.g., match strong-model quality on most queries by routing ~the hard fraction to it.

### The core: difficulty/quality estimation
Router predicts "will the cheap model suffice?" via: query features/classifier, the cheap model's confidence (cascade), or a small judge. Same estimation problem as dynamic compute ([§12](../12_reasoning_model_inference/03_dynamic_compute_allocation.md)); accuracy bounds the savings.

### Other routing axes
- **Task-type** (code → code model, [§00](00_production_serving_architecture.md)).
- **Cost/latency-SLO-based** (premium tier → faster/bigger).
- **Tenant/tier** ([§04](04_multi_tenant_serving.md)).

***

## Key Challenges
1. **Router accuracy vs savings.** Misrouting hard→small hurts quality; easy→big wastes money. Router quality bounds the achievable frontier.
2. **Escalation double-cost (cascade).** Escalating pays both models; needs low escalation rate to net save; confidence signal must be calibrated.
3. **Router overhead.** A routing classifier/call adds latency/cost; must be cheap relative to savings.
4. **Distribution drift.** Query mix shifts over time; a router trained on old data misroutes; needs monitoring/retraining.

***

## Solutions & Current Best Practices
- **Learned routers (RouteLLM)** on the accuracy-cost frontier for target quality.
- **Cascades (FrugalGPT)** with calibrated confidence/escalation for cost control.
- **Cheap routers** (small classifiers / cheap-model confidence) to keep overhead < savings.
- **Monitor routing quality & escalation rate**; retrain on drift; combine with dynamic compute for reasoning ([§12](../12_reasoning_model_inference/03_dynamic_compute_allocation.md)).

***

## Implementation Notes
- Calibrate the confidence/escalation threshold on validation data (quality vs escalation-rate curve).
- Keep the router lightweight (small model/classifier) and on a fast path; cache routing decisions for repeated queries.
- Track per-route quality and cost; alert on rising misroute/escalation rates (drift).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Cascade cost more than the big model** — escalation rate too high (poor confidence signal); pay small+large too often. Calibrate.
- **Router misrouted hard queries to the small model** — difficulty estimation off; quality dropped. Improve estimator / fall back.
- **Router overhead ate the savings** — heavy routing model; use a cheap classifier.
- **Quality drifted after a traffic shift** — stale router; monitor and retrain.

***

## Performance Numbers & Benchmarks
| Approach | Cost | Quality |
|---|---|---|
| Always big model | high | high |
| Always small model | low | lower |
| RouteLLM (learned) | medium | near-big (best accuracy/$) |
| FrugalGPT cascade | low–medium | near-big (low escalation) |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How would you cut serving cost with routing/cascading?"* — Expected: easy→small, hard→large; cascade escalate-on-low-confidence; accuracy-cost frontier.
- *"What's the cost formula for a cascade and when does it save?"* — Expected: c_small + P(escalate)·c_large; low escalation rate.
- *"What's the core ML problem in routing?"* — Expected: difficulty/quality estimation (will cheap model suffice?).
- *"What can make routing net-negative?"* — Expected: misrouting, high escalation, router overhead, drift.

***

## Open Problems & Active Research (2025–2026)
- **Better, cheaper difficulty/confidence estimators** for routing.
- **Drift-robust / online-updating routers**.
- **Joint routing + dynamic compute + reasoning** allocation ([§12](../12_reasoning_model_inference/03_dynamic_compute_allocation.md)).

***

## References
- Ong, I., et al. (2024). "RouteLLM: Learning to Route LLMs with Preference Data." arXiv:2406.18665.
- Chen, L., et al. (2023). "FrugalGPT: How to Use LLMs While Reducing Cost." arXiv:2305.05176.
- Snell, C., et al. (2024). "Scaling Test-Time Compute." arXiv:2408.03314.
