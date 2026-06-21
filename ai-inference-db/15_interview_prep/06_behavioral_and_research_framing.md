# Behavioral and Research Framing

> **Section:** 15_interview_prep
> **Last Updated:** June 2026
> **Related Files:** [00_inference_system_design_questions.md](00_inference_system_design_questions.md), [05_paper_deep_dives.md](05_paper_deep_dives.md), [../14_company_deep_dives/07_company_comparison_matrix.md](../14_company_deep_dives/07_company_comparison_matrix.md)
> **Must-Read Papers:** (n/a — this is framing/communication guidance)
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- As a PhD candidate, your edge is **depth + research framing** — translate your research into systems impact and show you can both go deep and ship.
- Use **STAR** (Situation-Task-Action-Result) with **quantified results**; for research, frame as **problem → insight → method → impact → limitation**.
- Tailor to the firm: Fireworks/Together (kernels/systems), Groq (architecture), Anyscale (distributed systems), labs (frontier-scale/research) ([§14](../14_company_deep_dives/07_company_comparison_matrix.md)).
- Show **bandwidth-bound thinking** even in behavioral answers — it signals you internalize the field's core.
- Have **smart questions** that reveal you understand inference deeply.

***

## Overview
Behavioral and research-framing rounds matter more for PhD candidates than for junior hires: firms are assessing whether you can convert deep expertise into shipped impact, collaborate, and pick the right problems. Your differentiator is **depth + the ability to frame research as engineering impact** — connecting your dissertation/projects to the systems problems these firms care about (the bandwidth bound, serving efficiency, novel kernels/algorithms). The goal across these rounds is to demonstrate: technical depth, research taste (you work on problems that matter), execution (you ship, not just publish), and communication (you can explain complex systems clearly to varied audiences).

For **technical-experience** questions, use **STAR** with **quantified results** ("reduced decode latency 2.3× by writing a fused W4A16 kernel, measured on H100"), and always include the **why** (the bandwidth-bound reasoning that motivated it) and a **limitation** (shows maturity). For **research** questions, frame as **problem → key insight → method → impact → honest limitation**, and explicitly bridge to the firm's work ("my work on X relates to your serving challenge Y because..."). Demonstrating **bandwidth-bound / roofline thinking** even informally signals you've internalized the field's organizing principle ([§00](../00_fundamentals/00_inference_vs_training.md)).

**Tailor** to the firm's emphasis ([§14](../14_company_deep_dives/07_company_comparison_matrix.md)): kernel/systems shops (Fireworks/Together) want depth in CUDA/attention/quantization and shipping; Groq wants architectural reasoning; Anyscale/distributed-infra wants distributed-systems judgment; frontier labs want research depth + frontier-scale systems thinking. And prepare **questions that reveal expertise** — about their disaggregation strategy, reasoning-model serving, kernel roadmap, or how they measure goodput. This file covers framing your research, behavioral structure, firm tailoring, and good questions.

***

## Core Concepts & Mechanics

### Framing your research
**problem → insight → method → impact → limitation**. Bridge explicitly to the firm:
- *Problem*: what was hard and why it matters.
- *Insight*: the non-obvious idea (ideally tied to a systems principle).
- *Method*: what you built/proved.
- *Impact*: quantified (speedup, accuracy, cost).
- *Limitation*: honest scope (shows research maturity).
- *Bridge*: "this relates to your inference work because…"

### STAR for engineering stories
**Situation-Task-Action-Result**, quantified. Include the **reasoning** (bandwidth bound / roofline) and a **what-I'd-do-differently**. Avoid vague ("made it faster") — give numbers and hardware context.

### Firm tailoring
| Firm type | Emphasize |
|---|---|
| Fireworks/Together | CUDA/attention/quantization depth, shipping, profiling |
| Groq | architecture reasoning (SRAM/deterministic), latency |
| Anyscale/infra | distributed systems, orchestration, autoscaling |
| Frontier labs | research depth + frontier-scale serving, reasoning models |

### Questions that reveal expertise
- "How do you handle prefill-decode interference — chunked prefill or disaggregation?"
- "What's your reasoning-model serving cost strategy?" ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md))
- "How do you measure and optimize goodput vs raw throughput?" ([§13](../13_production_systems/06_SLO_definition_and_enforcement.md))
- "Where's your kernel roadmap heading on Blackwell/FP4?"

***

## Key Challenges
1. **Translating research to impact.** Academics over-index on novelty; firms want impact/shipping — frame for both.
2. **Depth vs breadth balance.** Show deep expertise *and* breadth across the stack (this database).
3. **Honest limitations.** Owning limitations signals maturity; hiding them backfires with expert interviewers.
4. **Communication.** Explaining complex systems clearly (to a non-specialist or a skeptic) is itself evaluated.

***

## Solutions & Current Best Practices
- **STAR + quantified results + reasoning + limitation** for every story.
- **problem→insight→method→impact→limitation→bridge** for research.
- **Tailor** depth to the firm; **demonstrate bandwidth-bound thinking**.
- **Prepare expert-signaling questions**; research the firm's recent work/blog.

***

## Implementation Notes (prep checklist)
- Have 3–5 STAR stories (a kernel/systems win, a research result, a debugging saga, a collaboration, a failure-and-learning).
- Quantify everything; know your numbers and hardware context.
- Practice the 2-minute research pitch with the firm bridge.
- Prepare 3–4 expert questions per firm.

***

## Complexity Gotchas
> ⚠️ **Common mistakes:**
- Vague results ("improved performance") — always quantify with hardware context.
- Pure-novelty research pitch with no impact/shipping bridge.
- No limitations stated — reads as naive to expert interviewers.
- Generic questions — ask things that show you understand inference deeply.

***

## Interview Angles
> 💡 **What firms assess behaviorally:**
- *"Tell me about a hard systems problem you solved."* — STAR + numbers + bandwidth-bound reasoning + limitation.
- *"Describe your research and why it matters here."* — problem→insight→method→impact→limitation→bridge.
- *"A time you were wrong / a project failed."* — honest, with the lesson.
- *"Why this company?"* — tie to their specific differentiator ([§14](../14_company_deep_dives/07_company_comparison_matrix.md)).

***

## Open Problems & Active Research (to mention as interests)
- Reasoning-model serving economics ([§12](../12_reasoning_model_inference/04_thinking_budget_and_inference_time_scaling.md)).
- Sub-quadratic / hybrid architectures in production ([§10](../10_long_context/03_sparse_and_linear_attention.md)).
- Disaggregation at scale and cheap KV transfer ([§09](../09_distributed_inference/03_kv_cache_migration_and_transfer.md)).
- FP4/Blackwell quality and kernels ([§05](../05_quantization/04_fp8_inference_h100.md)).

***

## References
- (Communication/framing guidance; technical depth from §00–§14 of this database.)
- Firm engineering blogs and recent papers (research before each interview).
