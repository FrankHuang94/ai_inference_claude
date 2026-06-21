# Expert Parallelism and MoE Inference

> **Section:** 04_parallelism
> **Last Updated:** June 2026
> **Related Files:** [00_tensor_parallelism.md](00_tensor_parallelism.md), [05_parallelism_strategy_selection.md](05_parallelism_strategy_selection.md), [../01_hardware/03_interconnects_nvlink_infiniband.md](../01_hardware/03_interconnects_nvlink_infiniband.md)
> **Must-Read Papers:** Fedus et al. (2022, JMLR) "Switch Transformer"; Lepikhin et al. (2021, ICLR) "GShard"; DeepSeek-AI (2024) "DeepSeek-V2/V3"
> **Estimated Study Time:** 35 minutes

***

## TL;DR
- MoE replaces the dense FFN with **many experts**; a router selects **top-k** (usually k=2) per token, so only a fraction of parameters activate → **high capacity, low active FLOPs/bytes**.
- **Expert parallelism (EP)**: experts distributed across GPUs; tokens are routed to their experts via **all-to-all** communication (twice per MoE layer).
- MoE raises **effective arithmetic intensity** for decode (few active params/token), but the **all-to-all is communication-intensive** and **load imbalance** (hot experts) is the central problem.
- **DeepSeek-V3** (fine-grained experts + shared experts + MLA) is the reference modern MoE; **Mixtral 8×7B** is the canonical open MoE.
- Key optimizations: expert capacity buffers, token dropping vs drop-free, expert affinity routing, EP+TP composition.

***

## Overview
Mixture-of-Experts decouples parameter *count* from per-token *compute*. Instead of one large dense FFN, an MoE layer has `E` expert FFNs and a lightweight **router (gate)** that, for each token, selects the **top-k** experts (commonly k=2) to process it. Only those k experts run, so a model with, say, 256 experts and k=8 (DeepSeek-V3 scale) has enormous total parameters but activates only a small fraction per token. For inference this is a double win: **memory bandwidth per token** drops (you only read the active experts' weights, not all of them — directly attacking the decode bound, [§00](../00_fundamentals/03_memory_bandwidth_bound_compute.md)), and **arithmetic intensity rises** because the active compute is concentrated. This is why frontier models increasingly use MoE.

The cost is a fundamentally different parallelism and communication pattern. With **expert parallelism**, experts are spread across GPUs (e.g., 8 experts per GPU on an 8-GPU node), so a token routed to an expert on another GPU must be **sent there and its result sent back** — an **all-to-all** collective, twice per MoE layer (dispatch and combine). All-to-all is bisection-bandwidth- and latency-sensitive; it's the defining performance challenge of MoE serving and a major reason large NVLink domains (NVL72) matter for MoE ([§01](../01_hardware/03_interconnects_nvlink_infiniband.md)). On top of this, **load imbalance** is endemic: routing is data-dependent, so some experts become "hot" (receive far more tokens) while others idle, creating stragglers that gate the whole layer.

Managing imbalance and all-to-all is the art of MoE inference. **Expert capacity** caps tokens per expert (dropping or rerouting overflow — the token-dropping problem); **drop-free** schemes pad instead. **Expert affinity** routing tries to keep correlated tokens/requests on the same GPU to reduce cross-GPU traffic and improve KV/cache locality. **DeepSeek-V2/V3** advanced the design with fine-grained experts (more, smaller experts for better specialization), shared "always-on" experts, and MLA attention to slash KV — making large MoE practical to serve. This file covers the mechanics, EP, and these optimizations.

***

## Core Concepts & Mechanics

### Routing and top-k
For token hidden `h`, gate computes scores `g = softmax(h·W_gate) ∈ R^E`; pick top-k experts; output = `Σ_{i∈topk} g_i · Expert_i(h)`. 📐 Active params/token ≈ `k/E × FFN_params + shared`. For 256 experts, k=8: ~3% of expert params active.

### Expert parallelism + all-to-all
- Experts partitioned across `p` GPUs (EP degree). Each MoE layer:
  1. **Dispatch all-to-all**: send each token to the GPU(s) hosting its top-k experts.
  2. Local expert compute.
  3. **Combine all-to-all**: send results back to the token's origin GPU.
- 📐 All-to-all volume ≈ `2 × k × T × d × bytes` per MoE layer (T=tokens). Latency/bisection-bandwidth bound → favors NVSwitch domains.

### Load imbalance and capacity
- Routing is data-dependent → **hot experts**. The slowest (most-loaded) expert gates the layer.
- **Expert capacity** `C = capacity_factor × (T·k/E)` caps tokens/expert. Overflow is **dropped** (token-dropping, quality loss) or rerouted; **drop-free** pads to capacity (wasted compute). Training uses **load-balancing loss** to spread tokens; inference inherits the resulting (im)balance.

### Optimizations
- **Expert affinity / locality routing**: route similar tokens/requests to the same GPU to cut cross-GPU all-to-all and improve cache reuse.
- **EP + TP composition**: shard each expert with TP *and* distribute experts with EP for very large MoE (DeepSeek-V3 on NVL72).
- **Shared experts** (DeepSeek): a few always-active experts capture common computation, reducing routing variance.
- **MLA**: low-rank KV, orthogonal but critical for serving large MoE long-context ([§02](../02_kv_cache/00_kv_cache_fundamentals.md)).

***

## Key Challenges
1. **All-to-all communication.** Two all-to-alls per MoE layer are latency/bandwidth-intensive; cross-node EP can dominate runtime without NVSwitch-class fabric.
2. **Load imbalance / hot experts.** Data-dependent routing creates stragglers; the most-loaded expert gates the layer, and imbalance varies per batch.
3. **Token dropping vs padding.** Capacity limits force a choice between dropping tokens (quality) and padding (wasted compute); both hurt.
4. **Memory for all experts.** Although few activate per token, **all** experts' weights must reside in (distributed) HBM — huge total memory, requiring EP across many GPUs.

***

## Solutions & Current Best Practices
- **EP within a large NVLink domain** (NVL72) to keep all-to-all fast; EP+TP for the largest models ([§05](05_parallelism_strategy_selection.md)).
- **Expert-affinity routing** and request grouping to reduce cross-GPU traffic.
- **Tuned capacity factor** balancing drop rate vs padding waste; monitor per-expert load.
- **Fine-grained + shared experts (DeepSeek-V3 design)** and **MLA** for serviceable large MoE.
- **Overlap all-to-all with compute** to hide communication.

***

## Implementation Notes
- vLLM/SGLang/TRT-LLM support EP for Mixtral/DeepSeek; set EP degree and ensure all experts fit across the EP group's HBM.
- Monitor **per-expert token counts** and **all-to-all time**; rising imbalance or comm time are the top regressions.
- For decode (1 token/step × batch), all-to-all messages are small → latency-bound; co-locate EP on NVSwitch.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **All-to-all dominated decode across nodes** — cross-node EP on IB; latency killed ITL. Keep EP in NVLink domain.
- **A hot expert gated the whole layer** — imbalanced routing; the busiest GPU stalls others every MoE layer. Affinity routing / capacity tuning.
- **Token dropping silently degraded quality** — capacity factor too low under bursty routing; overflow tokens dropped. Tune capacity, monitor drop rate.
- **Forgot all experts must fit in HBM** — only k activate per token, but every expert's weights reside in memory; under-provisioned EP OOMs.

***

## Performance Numbers & Benchmarks
| Model | Experts / k | Active params | Serving note |
|---|---|---|---|
| Mixtral 8×7B | 8 / 2 | ~13B of 47B | EP across 2–8 GPUs |
| DeepSeek-V3 | 256 (+shared) / 8 | ~37B of 671B | EP+TP, NVL72, MLA |
| Switch Transformer | up to thousands / 1 | tiny fraction | research scale |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Why is MoE inference fundamentally different from dense inference?"* — Expected: top-k activation, all-to-all comm, load imbalance, all-experts-in-memory.
- *"Walk through the all-to-all pattern in expert parallelism."* — Expected: dispatch + combine per MoE layer; volume; NVSwitch dependence.
- *"What is the token-dropping problem and how do you manage it?"* — Expected: capacity factor, drop vs pad, load-balancing.
- *"How does DeepSeek-V3 make large MoE serviceable?"* — Expected: fine-grained + shared experts, MLA KV reduction, EP+TP on NVL72.

***

## Open Problems & Active Research (2025–2026)
- **Dynamic load balancing** of hot experts at inference without retraining.
- **Cheap cross-node EP** (compression, topology-aware all-to-all) to escape the NVLink-domain constraint.
- **Expert caching / offloading** for serving MoE larger than aggregate HBM.
- **Inference-time routing** that improves locality and reduces all-to-all.

***

## References
- Fedus, W., Zoph, B., Shazeer, N. (2022). "Switch Transformers." *JMLR*. arXiv:2101.03961.
- Lepikhin, D., et al. (2021). "GShard." *ICLR 2021*. arXiv:2006.16668.
- DeepSeek-AI (2024). "DeepSeek-V2" arXiv:2405.04434; "DeepSeek-V3" arXiv:2412.19437.
- Jiang, A., et al. (2024). "Mixtral of Experts." arXiv:2401.04088.
