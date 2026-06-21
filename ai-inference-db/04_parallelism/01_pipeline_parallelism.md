# Pipeline Parallelism for Inference

> **Section:** 04_parallelism
> **Last Updated:** June 2026
> **Related Files:** [00_tensor_parallelism.md](00_tensor_parallelism.md), [05_parallelism_strategy_selection.md](05_parallelism_strategy_selection.md), [../09_distributed_inference/00_multi_node_serving.md](../09_distributed_inference/00_multi_node_serving.md)
> **Must-Read Papers:** Huang et al. (2019, NeurIPS) "GPipe"; Narayanan et al. (2019, SOSP) "PipeDream"; Narayanan et al. (2021, SC) "Megatron pipeline"
> **Estimated Study Time:** 25 minutes

***

## TL;DR
- Pipeline parallelism (PP) splits the model **by layers** across GPUs/nodes: GPU0 holds layers 0–k, GPU1 holds k+1–2k, etc.; activations pass between stages.
- Communication is **point-to-point and low-volume** (just activations between adjacent stages), so PP **tolerates slower interconnects** (cross-node IB) — unlike TP.
- The cost is the **pipeline bubble**: stages idle while the pipeline fills/drains; mitigated by micro-batching, but decode (1 token/step) makes bubbles harder to amortize.
- Best for **scaling across nodes** and fitting very large models; usually combined with TP within nodes.
- For serving, PP improves **throughput/capacity** more than single-request latency (which it can slightly worsen via stage hops).

***

## Overview
Pipeline parallelism partitions a model along its **depth**: contiguous groups of layers ("stages") live on different GPUs, and a request's activations flow stage-to-stage like an assembly line. Because only the activation tensor at each stage boundary crosses the interconnect — a small, point-to-point transfer, not a collective all-reduce — PP's communication is far lighter and more latency-tolerant than tensor parallelism. This is precisely why PP is the tool for scaling **across nodes**: where TP would choke on InfiniBand latency, PP's occasional activation hand-offs are cheap enough to cross node boundaries. The standard recipe is **TP within a node** (fast NVLink for the heavy collectives) and **PP across nodes** (cheap activation passing over IB).

The defining cost of PP is the **pipeline bubble**: when you push a single batch through, stage 1 works while stages 2..N idle (fill), and at the end stage N works while others idle (drain). The classic fix from training (GPipe, PipeDream) is **micro-batching** — split the batch into many micro-batches so all stages stay busy on different micro-batches simultaneously, shrinking the bubble fraction to `(N−1)/(m + N−1)` for m micro-batches. Training has large batches to split, so bubbles amortize well. **Inference decode is harder**: each step produces one token per sequence, so the natural "micro-batch" is small, and keeping all pipeline stages busy requires enough concurrent requests in flight. Continuous batching helps by supplying a steady stream of work to fill the pipeline.

In serving, PP mainly buys **capacity and the ability to fit/scale very large models across many nodes**, and raises aggregate throughput when enough requests keep the pipeline full. It does not reduce single-request latency the way TP does — in fact each stage hop adds a little latency. So the mental model is: **TP for latency and within-node memory fit; PP for cross-node scale and capacity**; combine them ([§05](05_parallelism_strategy_selection.md)).

***

## Core Concepts & Mechanics

### Stage partitioning
Model's L layers split into N stages of ~L/N layers each. Stage boundaries pass the hidden-state activation `∈ R^{batch×d}` (decode) or `∈ R^{batch×seq×d}` (prefill) to the next stage. Balance stages to equalize compute (uneven stages create stragglers).

### The bubble
📐 With N stages and m micro-batches, bubble fraction ≈ `(N−1)/(m + N−1)`. Large m → small bubble. Prefill (many tokens) splits naturally; **decode** has effectively m≈(in-flight requests batched), so PP efficiency depends on having enough concurrent requests to fill N stages.

### Communication cost
Per stage boundary per step: send `batch×d×bytes` (decode) — small, point-to-point. 📐 Total ≈ `(N−1) × batch × d × bytes` per token, far less than TP's all-reduces and latency-tolerant → node-crossing OK.

### Scheduling (1F1B etc.)
Training uses schedules like **1F1B** (one-forward-one-backward) to bound activation memory. Inference is forward-only, so scheduling is simpler: stream micro-batches/requests to keep stages busy; the continuous-batching scheduler feeds the pipeline.

### Interaction with continuous batching
To keep all N stages busy at decode, you need ≥N "waves" of requests in flight. Too few concurrent requests → bubbles dominate → poor utilization. PP thus pairs with high request concurrency.

***

## Key Challenges
1. **Decode bubbles.** One token/step limits natural micro-batching; without enough concurrent requests, stages idle and PP efficiency drops.
2. **Stage balancing.** Uneven layer/compute distribution creates stragglers; the slowest stage gates throughput.
3. **Added per-request latency.** Each stage hop adds transfer + scheduling latency; PP can slightly worsen single-request ITL vs TP.
4. **KV cache locality.** Each stage holds KV only for its layers; a request's KV is spread across stages — fine, but memory accounting and migration (disaggregation) get more complex.

***

## Solutions & Current Best Practices
- **TP within node + PP across nodes** as the standard large-model recipe ([§05](05_parallelism_strategy_selection.md)).
- **Keep many requests in flight** (continuous batching) to fill the pipeline and amortize bubbles.
- **Balance stages** by compute, not just layer count (attention vs FFN cost varies).
- **Use PP for capacity/scale**, TP for latency.

***

## Implementation Notes
- vLLM/TRT-LLM expose `pipeline_parallel_size`; combine with `tensor_parallel_size` (total GPUs = TP×PP).
- Place PP stage boundaries at node boundaries so heavy TP collectives stay on NVLink and only light activations cross IB.
- Ensure enough concurrency (`max_num_seqs`) to fill N stages, or PP underperforms.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **PP with low concurrency wasted GPUs** — too few in-flight requests left stages idle (bubbles); needs high request rate to pay off.
- **Imbalanced stages bottlenecked on one GPU** — the heaviest stage gated throughput; balance by compute.
- **Expected PP to cut latency, it didn't** — PP adds stage-hop latency; use TP for latency, PP for capacity.
- **Cross-node TP instead of PP** — someone scaled with TP across nodes and hit IB latency; PP is the node-crossing tool.

***

## Performance Numbers & Benchmarks
| Strategy | Interconnect need | Latency | Throughput/scale |
|---|---|---|---|
| TP | NVLink (high BW, low lat) | improves | within-node fit |
| PP | IB OK (low volume) | slightly worse | cross-node scale, capacity |
| TP×PP | NVLink intra + IB inter | balanced | very large models |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"When use pipeline vs tensor parallelism?"* — Expected: TP intra-node/latency; PP cross-node/capacity; combine.
- *"What is the pipeline bubble and why is it worse for decode?"* — Expected: fill/drain idle; decode's 1 token/step limits micro-batching; need concurrency.
- *"Why can PP cross nodes but TP can't?"* — Expected: PP point-to-point small activations vs TP latency-sensitive all-reduces.
- *"How do you keep a pipeline full during decode?"* — Expected: enough concurrent requests via continuous batching.

***

## Open Problems & Active Research (2025–2026)
- **Bubble-free decode pipelines** via better request interleaving and micro-batch scheduling.
- **Auto-balancing stages** accounting for attention/FFN/MoE compute heterogeneity.
- **PP + disaggregation** interactions for very large models across many nodes ([§09](../09_distributed_inference/00_multi_node_serving.md)).

***

## References
- Huang, Y., et al. (2019). "GPipe: Efficient Training of Giant Neural Networks Using Pipeline Parallelism." *NeurIPS 2019*. arXiv:1811.06965.
- Narayanan, D., et al. (2019). "PipeDream: Generalized Pipeline Parallelism for DNN Training." *SOSP 2019*.
- Narayanan, D., et al. (2021). "Efficient Large-Scale LM Training on GPU Clusters." *SC 2021*. arXiv:2104.04473.
