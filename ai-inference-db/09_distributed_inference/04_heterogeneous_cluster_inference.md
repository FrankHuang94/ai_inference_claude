# Heterogeneous Cluster Inference

> **Section:** 09_distributed_inference
> **Last Updated:** June 2026
> **Related Files:** [02_disaggregated_prefill_decode_systems.md](02_disaggregated_prefill_decode_systems.md), [../01_hardware/05_hardware_selection_decision_framework.md](../01_hardware/05_hardware_selection_decision_framework.md), [../01_hardware/04_alternative_accelerators.md](../01_hardware/04_alternative_accelerators.md)
> **Must-Read Papers:** Patel et al. (2024, ISCA) "Splitwise" (heterogeneous); Jiang et al. (2024) "HexGen"; Zhong et al. (2024) "DistServe"
> **Estimated Study Time:** 20 minutes

***

## TL;DR
- Heterogeneous serving mixes **different GPU types** (or accelerators) in one deployment to optimize $/token by matching hardware to task.
- Natural fit with **disaggregation**: compute-dense GPUs (H100/B200) for **prefill**, bandwidth/capacity GPUs (H200/MI300X) for **decode** ([§02](02_disaggregated_prefill_decode_systems.md)).
- Other uses: old + new generations coexisting, cheaper GPUs for batch/offline, latency accelerators (Groq) for premium latency tiers.
- Challenges: **load balancing across unequal capacities**, parallelism that assumes homogeneity, kernel/precision portability, scheduling complexity.
- HexGen and similar systems search placement over heterogeneous resources.

***

## Overview
Most serving assumes a homogeneous fleet, but real organizations accumulate **mixed hardware** (multiple GPU generations from successive purchases, different accelerators for different needs) and can exploit heterogeneity to lower $/token. The clearest opportunity is **phase-matched disaggregation**: since prefill is compute-bound and decode is bandwidth/capacity-bound ([§00](../00_fundamentals/02_prefill_vs_decode_phases.md)), you can run prefill on FLOP-dense GPUs (H100/B200) and decode on bandwidth/capacity-rich GPUs (H200/MI300X), each phase on its ideal (and possibly cheaper) hardware — exactly Splitwise's heterogeneous phase-splitting. Beyond phases, heterogeneity lets you route batch/offline traffic to cheaper/older GPUs, reserve latency accelerators (Groq) for a premium low-latency tier, and keep older generations productive instead of retiring them.

The difficulties stem from breaking the homogeneity assumption baked into most serving logic. **Load balancing** must account for unequal instance capacities (a router sending equal shares to an H100 and an A100 overloads the A100). **Parallelism** strategies (TP/PP) often assume identical GPUs; splitting a model across mixed GPUs creates stragglers gated by the slowest. **Kernels and precision** differ by generation (FP8 needs Hopper+, FP4 needs Blackwell, ROCm vs CUDA), so a single binary/engine may not run optimally everywhere — portability and fallbacks matter ([§01](../01_hardware/04_alternative_accelerators.md)). And the **scheduling/placement** search space (which model/phase/tier on which hardware) is large; systems like **HexGen** formalize searching placements over heterogeneous, even geo-distributed, resources.

The practical posture: heterogeneity is most cleanly exploited via **disaggregation** (independent per-phase hardware) and **tiered routing** (batch→cheap, premium→fast), with capacity-aware load balancing and careful precision/kernel portability. For tightly-coupled parallelism (a single model split with TP), homogeneity within the parallel group is strongly preferred — mix *across* groups/phases, not *within* a TP group. This file covers the use cases, challenges, and best practices.

***

## Core Concepts & Mechanics

### Use cases
- **Phase-matched disaggregation**: FLOP-dense prefill GPUs + bandwidth/capacity decode GPUs (Splitwise).
- **Generation mixing**: keep A100s for batch/cheap traffic, H100/H200 for interactive.
- **Tiered routing**: premium latency → fast accelerators (Groq/H200); batch → cheapest.
- **Cost arbitrage**: route to whatever hardware gives best $/token for that request class.

### Capacity-aware load balancing
📐 Route shares proportional to per-instance capacity `C_i` (in-flight-token capacity), not equally: `share_i ∝ C_i`. Equal routing overloads weaker GPUs and underuses stronger ones ([§01](01_load_balancing_strategies.md)).

### Parallelism and homogeneity
- **Within a TP/PP group**: prefer identical GPUs; the slowest gates the group (straggler).
- **Across groups/phases/replicas**: heterogeneity is fine and exploitable.
- HexGen-style placement search assigns model shards/phases to heterogeneous resources to optimize throughput/cost.

### Portability
FP8 (Hopper+), FP4 (Blackwell), CUDA vs ROCm (MI300X) → kernels/precision differ; need fallbacks or per-hardware builds (esp. TRT-LLM compiled per GPU) ([§08](../08_serving_frameworks/03_tensorrt_llm_deep_dive.md)).

***

## Key Challenges
1. **Unequal capacities.** Load balancing and parallelism assume homogeneity; mixing creates stragglers/overload without capacity-awareness.
2. **Straggler in coupled parallelism.** Mixing GPUs within a TP/PP group is gated by the slowest; avoid.
3. **Kernel/precision portability.** Different generations/vendors need different kernels/precisions; portability and fallback complexity.
4. **Scheduling/placement search.** Assigning models/phases/tiers to heterogeneous hardware is a large optimization.

***

## Solutions & Current Best Practices
- **Exploit heterogeneity via disaggregation** (per-phase hardware) and **tiered routing**, not within tightly-coupled parallel groups ([§02](02_disaggregated_prefill_decode_systems.md)).
- **Capacity-aware load balancing** (shares ∝ capacity) ([§01](01_load_balancing_strategies.md)).
- **Keep TP/PP groups homogeneous**; mix across groups/phases/replicas.
- **Handle precision/kernel portability** with per-hardware builds/fallbacks; benchmark $/token per hardware class.

***

## Implementation Notes
- Tag instances with hardware class + capacity; router uses capacity-aware policy.
- Build/compile per-GPU (TRT-LLM) or use portable kernels (vLLM/SGLang) with FP8/FP4 fallbacks.
- For disaggregation, assign prefill/decode pools to the best-fit hardware ([§02](02_disaggregated_prefill_decode_systems.md)).

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Equal routing overloaded the A100s** — capacity-blind LB; route proportional to capacity.
- **Mixed GPUs in one TP group** — slowest GPU gated the group; keep groups homogeneous.
- **FP8 path crashed on A100** — no FP8 Tensor Cores; needs fallback or homogeneous Hopper pool.
- **Compiled engine didn't run on the other generation** — TRT-LLM is per-GPU; build per class.

***

## Performance Numbers & Benchmarks
| Pattern | Benefit |
|---|---|
| Prefill on H100 + decode on H200 | $/token + throughput/W (Splitwise) |
| Batch on A100, interactive on H100 | cost arbitrage |
| Premium latency on Groq | latency tier |
| HexGen placement | throughput/$ over heterogeneous fleet |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"How would you use a mixed GPU fleet efficiently?"* — Expected: phase-matched disaggregation, tiered routing, capacity-aware LB; homogeneous parallel groups.
- *"Why not mix GPUs within a tensor-parallel group?"* — Expected: straggler gated by slowest.
- *"How does load balancing change with heterogeneous capacities?"* — Expected: shares ∝ capacity, not equal.
- *"What portability issues arise across generations/vendors?"* — Expected: FP8/FP4/ROCm kernels; fallbacks/per-hardware builds.

***

## Open Problems & Active Research (2025–2026)
- **Automated heterogeneous placement** (HexGen-style) at production scale, incl. geo-distributed.
- **Portable kernels** across generations/vendors with graceful precision fallback.
- **Economic schedulers** optimizing $/token over heterogeneous + spot resources ([§01](../01_hardware/06_tco_and_cost_modeling.md)).

***

## References
- Patel, P., et al. (2024). "Splitwise" (heterogeneous). *ISCA 2024*. arXiv:2311.18677.
- Jiang, Y., et al. (2024). "HexGen: Generative Inference over Heterogeneous Environment." *ICML 2024*. arXiv:2311.11514.
- Zhong, Y., et al. (2024). "DistServe." *OSDI 2024*. arXiv:2401.09670.
