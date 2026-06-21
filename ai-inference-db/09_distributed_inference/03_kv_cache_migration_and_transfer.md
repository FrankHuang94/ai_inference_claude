# KV Cache Migration and Transfer

> **Section:** 09_distributed_inference
> **Last Updated:** June 2026
> **Related Files:** [02_disaggregated_prefill_decode_systems.md](02_disaggregated_prefill_decode_systems.md), [../02_kv_cache/04_kv_cache_offloading.md](../02_kv_cache/04_kv_cache_offloading.md), [../01_hardware/03_interconnects_nvlink_infiniband.md](../01_hardware/03_interconnects_nvlink_infiniband.md)
> **Must-Read Papers:** Qin et al. (2024) "Mooncake"; Liu et al. (2024) "CacheGen"; NVIDIA NIXL / GPUDirect RDMA docs
> **Estimated Study Time:** 30 minutes

***

## TL;DR
- Disaggregation requires moving a request's **KV cache** from prefill to decode GPU — often **GBs per request** over the interconnect.
- 📐 Transfer time = `KV_bytes / interconnect_BW`; e.g., 1.25 GB over NDR IB (~50 GB/s) ≈ 25 ms — must **overlap** with compute or it inflates TTFT.
- Use **GPUDirect RDMA** (NIC↔GPU DMA, bypass host) over InfiniBand/RoCE; NVLink within a node.
- **Pipeline** the transfer with decode start (layer-by-layer streaming) so decode begins before the full KV arrives.
- **Compress/quantize KV** before transfer (CacheGen) to cut volume; **LMCache/Mooncake** manage tiered, transferable KV.

***

## Overview
KV cache migration is the enabling-and-limiting mechanism of disaggregated serving ([§02](02_disaggregated_prefill_decode_systems.md)). When prefill happens on one instance and decode on another, the prompt's KV cache — the keys and values for every prompt token across every layer — must physically move to the decode GPU before (or as) generation begins. This is a substantial data movement: a single request's KV is ~1.25 GB for a 70B model at 4k context ([§02](../02_kv_cache/00_kv_cache_fundamentals.md)), and a busy prefill instance produces many such transfers per second, so aggregate KV transfer can reach tens of GB/s. If this movement isn't engineered carefully, it dominates TTFT and negates disaggregation's benefits.

The core techniques mirror those for offloading ([§02](../02_kv_cache/04_kv_cache_offloading.md)) but across the network. **GPUDirect RDMA** lets the NIC DMA directly between GPUs' HBM over InfiniBand/RoCE, bypassing host memory and the CPU — essential for low latency and full bandwidth; without it, KV stages through host RAM over PCIe, doubling latency and halving throughput. **Overlap and pipelining** are mandatory: rather than transfer the entire KV then start decode, systems **stream KV layer-by-layer** so the decode instance can begin computing layer 0 as soon as layer 0's KV arrives, hiding most transfer latency behind decode compute. **Compression** (CacheGen quantizes/encodes KV for cheaper transfer/storage, decompressing on arrival) further cuts the volume that must move.

The systems layer treats KV as a first-class, movable, storable object. **LMCache** provides a KV caching/transfer layer for vLLM/SGLang; **Mooncake**'s global KV pool ([§02](02_disaggregated_prefill_decode_systems.md)) and **NIXL** (NVIDIA's transfer abstraction) standardize moving KV across GPUs, nodes, and storage tiers. The **routing** problem — which decode instance should receive a given prefill's KV, ideally one already caching a shared prefix — ties back to load balancing ([§01](01_load_balancing_strategies.md)). This file covers the bandwidth math, RDMA, pipelining/overlap, compression, and the management layers.

***

## Core Concepts & Mechanics

### Transfer bandwidth math
📐 Per-request transfer time `t = KV_bytes / BW`.
- 70B@4k KV ≈ 1.25 GB. NVLink (900 GB/s): ~1.4 ms. NDR IB (~50 GB/s): ~25 ms. PCIe-staged (~32 GB/s effective): ~40 ms.
- Aggregate rate from a prefill GPU = `req/s × KV_bytes`; can be tens of GB/s → needs RDMA, possibly compression.

### GPUDirect RDMA
NIC reads/writes GPU HBM directly over IB/RoCE, bypassing host. Without it, KV bounces through host RAM (PCIe) → 2× latency, half BW. Mandatory for efficient cross-node KV transfer.

### Pipelining / overlap
Stream KV **layer-by-layer**: decode instance starts layer 0 once its KV arrives, while later layers transfer concurrently. 📐 If per-layer transfer < per-layer decode compute, transfer is fully hidden; only the first layer's transfer is exposed in TTFT.

### Compression (CacheGen)
Quantize/encode KV before transfer; decompress on arrival. Cuts transfer volume (and storage in tiered KV) at the cost of (de)compression compute and some quality if lossy ([§05](../05_quantization/05_kv_cache_quantization.md)).

### Management layers
- **LMCache**: KV cache/transfer layer for OSS engines.
- **Mooncake KV pool**: global tiered KV across GPU/CPU/SSD/nodes.
- **NIXL**: NVIDIA inference transfer library abstracting RDMA/NVLink moves.

***

## Key Challenges
1. **TTFT exposure.** Un-overlapped transfer adds directly to first-token latency; pipelining is essential.
2. **Aggregate bandwidth.** Many concurrent transfers can saturate the fabric; RDMA + compression + provisioning needed.
3. **Routing for locality.** Choosing a decode instance that already caches a shared prefix reduces transfer; non-trivial ([§01](01_load_balancing_strategies.md)).
4. **Consistency with paging/quantization.** Transferred blocks must integrate with the decode side's paged allocator and any KV quantization.

***

## Solutions & Current Best Practices
- **GPUDirect RDMA over IB/RoCE**; NVLink intra-node ([§01](../01_hardware/03_interconnects_nvlink_infiniband.md)).
- **Layer-by-layer pipelined transfer** overlapping decode start.
- **KV compression (CacheGen)** for volume; **LMCache/Mooncake/NIXL** for management.
- **Cache-aware KV routing** to minimize transfer and maximize reuse.

***

## Implementation Notes
- Verify GPUDirect RDMA is active (not host-staged); check with transfer benchmarks.
- Budget first-layer transfer into TTFT SLO; confirm overlap in Nsight Systems.
- Provision interconnect for aggregate KV rate (req/s × KV size), not just per-request.

***

## Complexity Gotchas
> ⚠️ **Production surprises and edge cases:**
- **KV staged through host RAM** — GPUDirect RDMA not enabled; 2× latency, half BW. Enable it.
- **Full-KV-then-decode added 25 ms to TTFT** — no pipelining; stream layer-by-layer to overlap.
- **Fabric saturated under load** — aggregate transfer exceeded IB; compress KV and/or provision more fabric.
- **Transferred KV didn't match decode-side quantization** — block format mismatch; align KV quant across pools.

***

## Performance Numbers & Benchmarks
| Path | 1.25 GB KV transfer | Note |
|---|---|---|
| NVLink (intra-node) | ~1.4 ms | best |
| NDR IB + RDMA | ~25 ms | overlap required |
| Host-staged PCIe | ~40 ms | avoid (no RDMA) |
| + CacheGen compression | reduced volume | (de)compress cost |

***

## Interview Angles
> 💡 **What inference-focused firms actually ask:**
- *"Estimate the bandwidth/time to transfer KV in a disaggregated system."* — Expected: KV_bytes/BW; ~25 ms over IB; aggregate = req/s×KV.
- *"How do you keep KV transfer off the TTFT critical path?"* — Expected: GPUDirect RDMA + layer-by-layer pipelined overlap.
- *"How can you reduce KV transfer cost?"* — Expected: compression (CacheGen), cache-aware routing, NVLink where possible.
- *"What is GPUDirect RDMA and why does it matter here?"* — Expected: NIC↔GPU DMA bypass host; latency/bandwidth.

***

## Open Problems & Active Research (2025–2026)
- **Cheap KV transfer over commodity Ethernet** (compression + RoCE) to broaden disaggregation.
- **Standard global KV pools/transfer APIs** (NIXL, LMCache, Mooncake convergence).
- **Quantization-aware transfer** that composes with prefix caching ([§05](../05_quantization/05_kv_cache_quantization.md)).

***

## References
- Qin, R., et al. (2024). "Mooncake." arXiv:2407.00079.
- Liu, Y., et al. (2024). "CacheGen." *SIGCOMM 2024*. arXiv:2310.07240.
- NVIDIA. "GPUDirect RDMA" and "NIXL" documentation.
- LMCache project documentation.
