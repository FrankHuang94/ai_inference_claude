# TCO and Cost Modeling for Inference Infrastructure

> **Section:** 01_hardware
> **Last Updated:** June 2026
> **Related Files:** [05_hardware_selection_decision_framework.md](05_hardware_selection_decision_framework.md), [../00_fundamentals/06_key_metrics_latency_throughput_cost.md](../00_fundamentals/06_key_metrics_latency_throughput_cost.md), [../14_company_deep_dives/07_company_comparison_matrix.md](../14_company_deep_dives/07_company_comparison_matrix.md)
> **Must-Read Papers:** Patel et al. (2024, ISCA) "Splitwise" (cost/power analysis); industry cost analyses (SemiAnalysis); Pope et al. (2022, MLSys)
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- **$/token** = `GPU_hourly_cost / tokens_per_hour`. Everything in TCO funnels into the denominator (throughput) and numerator (capex+opex per hour).
- **Capex**: H100 ~$25–35k, B200 ~$60–70k; amortize over ~3–4 yr. **Opex**: power (H100 700 W), cooling, colocation (~$0.10–0.15/kWh), networking, staff.
- **Utilization is the killer variable**: at 30% utilization your $/token roughly triples vs 90%. Idle GPUs burn capex.
- **Output tokens cost far more than input tokens** (decode vs prefill); price and model them separately.
- Inference clouds (Fireworks/Together) win on **utilization aggregation + kernel efficiency**; their margin is the gap between their $/token and yours.

***

## Overview
Total Cost of Ownership translates hardware and operations into the only number that matters commercially: **cost per token** (split by input/output). The model is simple in form — amortized hourly cost divided by tokens served per hour — but every serving optimization in this database is ultimately an attempt to move one of its terms: raise throughput (continuous batching, quantization, speculative decoding), reduce bytes/energy (quantization, fusion), or raise utilization (autoscaling, multi-tenancy, routing). A candidate who can connect a systems technique to its $/token impact stands out.

The dominant, often-underappreciated lever is **utilization**. A GPU costs the same whether it runs at 5% or 95%; cost-per-token scales inversely with how busy it is with *useful, SLO-meeting* work (goodput). This is why the inference-cloud business model exists: by aggregating many customers' bursty demand onto shared fleets, providers achieve high utilization that an individual team's spiky workload cannot, and they pass part of that efficiency on as lower prices while keeping a margin. It's also why **30% utilization kills profitability** — a recurring interview talking point.

Finally, cost must be **phase-aware**. Prefill (input tokens) is compute-bound and amortized across the prompt; decode (output tokens) is bandwidth-bound and serial — far more expensive per token. This is why APIs price output tokens 2–5× input tokens and why reasoning models (massive output) have radically different unit economics ([§12](../12_reasoning_model_inference/00_long_chain_of_thought_serving.md)).

***

## Core Concepts & Mechanics

### The cost-per-token formula
📐
```
$/token = hourly_cost / tokens_per_hour
hourly_cost = capex_amortized_hourly + power_cost + cooling + colocation + network + staff_overhead
tokens_per_hour = throughput(tok/s) × 3600 × utilization
```
Split for input/output: `$/input_token` uses prefill throughput; `$/output_token` uses decode throughput (much lower → higher cost).

### Capex amortization
📐 `capex_hourly = purchase_price / (lifetime_years × 8760 × availability)`. Example: H100 at $30k over 3 years at 95% availability ≈ `30000 / (3×8760×0.95) ≈ $1.20/hr` just for the card (before server, networking, opex). Cloud on-demand H100 rents ~$3–10/hr, reflecting provider capex+opex+margin.

### Opex
- **Power**: H100 700 W → `0.7 kW × $0.12/kWh ≈ $0.084/hr` for the GPU alone; full server (CPU, NICs, cooling overhead, PUE ~1.2–1.5) multiplies this ~2–3×.
- **Cooling/PUE**, **colocation/space**, **InfiniBand fabric** (NICs + switches amortized), **staff/ops**.

### Utilization economics
📐 Since `tokens_per_hour ∝ utilization`, `$/token ∝ 1/utilization`. Going from 90%→30% utilization ≈ **3× higher $/token**. Burstiness, over-provisioning for P99, and cold capacity all erode utilization. This is the single biggest determinant of inference profitability.

### Build vs buy vs cloud
- **Buy/colocate**: lowest $/token at high, steady utilization and scale; high capex risk, depreciation, ops burden.
- **Cloud reserved/committed**: predictable, moderate cost; good for steady baseline.
- **Cloud on-demand/spot**: flexible/cheap-spot but eviction risk; good for burst and batch.
- **Inference cloud API** (Fireworks/Together/etc.): no infra; you pay their $/token + margin; best for variable demand and time-to-market.

***

## Key Challenges
1. **Utilization vs SLO tension.** Protecting P99 requires headroom (ρ<1), which lowers utilization and raises $/token; the optimization is goodput-per-dollar, not raw utilization.
2. **Spiky, unpredictable demand.** Real traffic is bursty; provisioning for peaks wastes capex off-peak. Autoscaling helps but cold start limits responsiveness ([§13](../13_production_systems/05_cold_start_and_model_loading.md)).
3. **Phase-asymmetric cost.** Output tokens dominate cost for generation-heavy workloads; flat per-token models mis-price them, especially reasoning models.
4. **Depreciation and obsolescence.** GPUs depreciate fast as new generations (H200, B200) reset the price/performance frontier; capex timing risk is real.

***

## Solutions & Current Best Practices
- **Maximize goodput/GPU** via continuous batching, FP8, prefix caching, speculative decoding — directly lowers $/token ([§03](../03_batching_and_scheduling/00_static_vs_continuous_batching.md), [§05](../05_quantization/04_fp8_inference_h100.md)).
- **Raise utilization** via multi-tenancy, autoscaling, and routing batch/latency traffic to fill troughs ([§13](../13_production_systems/02_autoscaling_and_capacity_planning.md), [§13](../13_production_systems/04_multi_tenant_serving.md)).
- **Use spot/committed mixes**: committed for baseline, spot for burst/batch.
- **Model $/token at SLO** per workload and choose hardware accordingly ([§01](05_hardware_selection_decision_framework.md)).

***

## Implementation Notes
- Track **$/output token** and **$/input token** separately in cost dashboards; tie to goodput, not raw throughput.
- Include **full-server and PUE overhead**, not just the GPU card, in hourly cost.
- Model **utilization distributions** (not averages) to size committed vs burst capacity.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **Great per-GPU throughput, terrible unit economics** — low utilization (over-provisioned for P99) tripled $/token. Optimize goodput-per-dollar, not peak throughput.
- **Forgot PUE/server overhead** — the GPU is ~30–50% of true power/cost; "$0.08/hr power" becomes $0.20+ at the wall.
- **Priced reasoning workloads flat** — 50k output tokens at decode cost wrecks margins; price output tokens at their true (higher) cost.
- **Bought at the top of a generation** — H200/B200 reset price/perf; depreciation hit harder than planned. Time capex with the roadmap.

***

## Performance Numbers & Benchmarks
| Item | Figure (approx, 2025–26) |
|---|---|
| H100 purchase | ~$25–35k |
| B200 purchase | ~$60–70k |
| H100 capex/hr (3yr) | ~$1.0–1.3 |
| H100 cloud on-demand | ~$3–10/hr |
| Power (GPU only) | 700 W → ~$0.08/hr @ $0.12/kWh |
| Full-server power w/ PUE | ~1.5–2.5× GPU-only |
| Utilization impact | 90%→30% ≈ 3× $/token |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Derive $/token for serving 70B on H100s. What dominates?"* — Expected: hourly cost / (throughput×utilization); utilization and decode throughput dominate.
- *"Why does 30% utilization kill an inference business?"* — Expected: $/token ∝ 1/utilization; idle GPUs burn fixed capex.
- *"Why do APIs charge more for output than input tokens?"* — Expected: decode (bandwidth-bound, serial) costs more than prefill.
- *"How does an inference cloud achieve lower $/token than a customer self-hosting?"* — Expected: utilization aggregation + kernel efficiency at scale.

***

## Open Problems & Active Research (2025–2026)
- **Cost models for reasoning/agentic workloads** with huge, variable output ([§12](../12_reasoning_model_inference/04_thinking_budget_and_inference_time_scaling.md)).
- **Carbon/energy-aware cost modeling** as power becomes the binding datacenter constraint.
- **Dynamic pricing tied to real-time utilization/goodput** at inference clouds.

***

## References
- Patel, P., et al. (2024). "Splitwise" (power/cost analysis). *ISCA 2024*. arXiv:2311.18677.
- Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference." *MLSys 2023*. arXiv:2211.05102.
- SemiAnalysis and industry cost teardowns (2023–2025).
