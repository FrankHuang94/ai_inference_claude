# Coding Exercises (with Solutions)

> **Section:** 15_interview_prep
> **Last Updated:** June 2026
> **Related Files:** [../02_kv_cache/01_paged_attention_vllm.md](../02_kv_cache/01_paged_attention_vllm.md), [../07_speculative_decoding/00_speculative_decoding_fundamentals.md](../07_speculative_decoding/00_speculative_decoding_fundamentals.md), [01_kernel_and_gpu_technical_questions.md](01_kernel_and_gpu_technical_questions.md)
> **Must-Read Papers:** Kwon et al. (2023) "vLLM"; Leviathan et al. (2023) "Speculative Decoding"; Dao et al. (2022) "FlashAttention"
> **Estimated Study Time:** 50 minutes

***

## TL;DR
- Six implementation exercises with **working solutions**: paged block manager, speculative rejection sampling, continuous-batching scheduler, Triton scaled-dot-product attention, Nsight bottleneck analysis, optimal TP degree.
- Focus on the **logic that interviews probe**, not production completeness.
- Each solution maps to a database section for the underlying theory.

***

## Exercise 1: PagedAttention block manager
**Task**: `allocate/free/get_physical_block` with a fixed block pool (theory: [§02](../02_kv_cache/01_paged_attention_vllm.md)).
```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.free = list(range(num_blocks))          # free physical block ids
        self.tables: dict[int, list[int]] = {}        # seq_id -> [physical block ids]
        self.refcount: dict[int, int] = {}            # physical block -> refs (sharing)

    def allocate(self, seq_id: int, num_tokens: int):
        need = (num_tokens + self.block_size - 1) // self.block_size
        have = len(self.tables.get(seq_id, []))
        for _ in range(need - have):
            if not self.free:
                raise MemoryError("no free KV blocks")   # -> trigger preemption
            blk = self.free.pop()
            self.refcount[blk] = 1
            self.tables.setdefault(seq_id, []).append(blk)

    def get_physical_block(self, seq_id: int, token_pos: int) -> int:
        return self.tables[seq_id][token_pos // self.block_size]

    def share(self, src_seq: int, dst_seq: int):         # prefix sharing / CoW
        self.tables[dst_seq] = list(self.tables[src_seq])
        for blk in self.tables[dst_seq]:
            self.refcount[blk] += 1

    def free(self, seq_id: int):
        for blk in self.tables.pop(seq_id, []):
            self.refcount[blk] -= 1
            if self.refcount[blk] == 0:                  # only free when unshared
                self.free.append(blk)
```
**Key points interviewers want**: on-demand block allocation (no max-len reservation), ref-counting for sharing/CoW, OOM → preemption signal.

## Exercise 2: Speculative decoding rejection sampling
**Task**: given draft tokens + target/draft probs, return accepted tokens (theory: [§07](../07_speculative_decoding/00_speculative_decoding_fundamentals.md)).
```python
import torch

def speculative_accept(draft_tokens, p_target, q_draft, bonus_logits):
    # draft_tokens: [k]; p_target,q_draft: [k, V] at each draft position; bonus_logits: [V]
    accepted = []
    for i, tok in enumerate(draft_tokens):
        pt, qd = p_target[i, tok], q_draft[i, tok]
        if torch.rand(()) < min(1.0, (pt / qd).item()):     # accept w.p. min(1, p/q)
            accepted.append(tok.item())
        else:                                               # reject -> resample (p - q)_+
            resid = torch.clamp(p_target[i] - q_draft[i], min=0)
            resid = resid / resid.sum()
            accepted.append(torch.multinomial(resid, 1).item())
            return accepted                                 # stop at first rejection
    # all k accepted -> sample a bonus token from target
    accepted.append(torch.multinomial(torch.softmax(bonus_logits, -1), 1).item())
    return accepted
```
**Key points**: `min(1,p/q)` acceptance; resample from normalized `(p−q)_+` on rejection; bonus token if all accepted; this guarantees the **exact target distribution**.

## Exercise 3: Continuous-batching scheduler (skeleton)
**Task**: admit/evict per iteration within a token budget (theory: [§03](../03_batching_and_scheduling/01_iteration_level_scheduling.md)).
```python
class ContinuousScheduler:
    def __init__(self, max_tokens: int, bm: 'BlockManager'):
        self.waiting, self.running = [], []
        self.max_tokens, self.bm = max_tokens, bm

    def add(self, req): self.waiting.append(req)

    def step(self, model):
        # 1. evict finished, free KV
        done = [r for r in self.running if r.finished]
        for r in done: self.bm.free(r.id); self.running.remove(r)
        # 2. admit within token budget + KV availability (chunked prefill)
        budget = self.max_tokens - len(self.running)            # 1 token/running decode
        while self.waiting and budget > 0:
            r = self.waiting[0]
            chunk = min(r.remaining_prefill, budget)
            try: self.bm.allocate(r.id, r.prefilled + chunk)
            except MemoryError: break                           # no KV -> stop admitting
            r.prefilled += chunk; budget -= chunk
            if r.remaining_prefill == 0: self.running.append(self.waiting.pop(0))
        # 3. one forward pass over running batch (mixed prefill-chunk + decode)
        model.forward_step(self.running)
```
**Key points**: per-iteration evict→admit→forward; **token budget** bounds ITL; chunked prefill; OOM → stop admitting (or preempt).

## Exercise 4: Triton scaled-dot-product attention (sketch)
**Task**: a basic attention kernel (theory: [§06](../06_kernel_optimization/01_flash_attention_deep_dive.md), [§06 Triton](../06_kernel_optimization/04_triton_for_inference.md)).
```python
import triton, triton.language as tl

@triton.jit
def attn_kernel(Q, K, V, Out, scale, N, D: tl.constexpr, BLK: tl.constexpr):
    m = tl.program_id(0)                                  # query row block
    q = tl.load(Q + m*D + tl.arange(0, D))               # [D]
    acc = tl.zeros([D], tl.float32); l = 0.0; mx = -1e30  # online softmax state
    for start in range(0, N, BLK):                        # stream K,V blocks (SRAM)
        k = tl.load(K + (start+tl.arange(0,BLK))[:,None]*D + tl.arange(0,D)[None,:])
        v = tl.load(V + (start+tl.arange(0,BLK))[:,None]*D + tl.arange(0,D)[None,:])
        s = tl.sum(q[None,:]*k, 1) * scale               # [BLK] scores
        blk_mx = tl.max(s); new_mx = tl.maximum(mx, blk_mx)
        p = tl.exp(s - new_mx)                            # rescale (online softmax)
        corr = tl.exp(mx - new_mx)
        l = l*corr + tl.sum(p); acc = acc*corr + tl.sum(p[:,None]*v, 0)
        mx = new_mx
    tl.store(Out + m*D + tl.arange(0,D), acc / l)
```
**Key points**: online-softmax running (max, sum, acc) so the N×N matrix never hits HBM; tiles in SRAM — the FlashAttention idea.

## Exercise 5: Nsight bottleneck analysis
**Task**: given Nsight output, find the bottleneck (theory: [§06](../06_kernel_optimization/05_kernel_profiling_and_benchmarking.md)).
**Approach**: Read **SOL**: if **Memory throughput ~high, SM ~low** → memory-bound (decode-like) → reduce bytes (fusion, quantization, coalescing), raise batch. If **SM ~high, Memory ~low** → compute-bound (prefill) → better tiling/FP8/occupancy. Check **roofline overlay** (AI vs ridge), **L2 hit rate**, **stall reasons** (long-scoreboard = memory wait), **achieved vs peak bandwidth** (coalescing headroom). Don't chase FLOP% if left of ridge.

## Exercise 6: Optimal tensor-parallel degree
**Task**: min TP for a 70B model on H100s meeting a TTFT/latency SLO (theory: [§04](../04_parallelism/00_tensor_parallelism.md)).
**Approach**:
1. **Fit**: weights FP8 ≈ 70GB; +peak KV. On 80GB H100, TP=1 leaves little for KV → TP≥2 for headroom.
2. **Latency**: decode/token ≈ `weights/(TP × bandwidth) + comm`. TP=2 ≈ 70/(2×3.35TB/s) ≈ 10ms weight-read/token; TP=4 ≈ 5ms — pick smallest TP meeting ITL within the NVLink domain.
3. **Constraint**: TP ≤ 8 (NVLink), TP ≤ n_kv_heads or replicate KV; TP must divide heads.
4. **Answer**: typically **TP=2 or 4** for 70B on H100 (FP8) to meet interactive ITL with KV headroom; scale throughput with DP, not more TP.

***

## Interview Angles
> 💡 **Delivery tips:**
- For each, state the **invariant** (e.g., rejection sampling preserves target distribution; paged alloc avoids reservation).
- Handle the **edge cases** (OOM→preempt, all-accepted→bonus token, online-softmax rescale).
- Tie to the **bandwidth bound** when reasoning about performance.

***

## Complexity Gotchas
> ⚠️ **Common mistakes:**
- Speculative: forgetting the bonus token or resampling from raw `p` instead of `(p−q)_+`.
- Block manager: reserving max-len (defeats paging) or freeing shared blocks (refcount bug).
- Triton attention: missing the online-softmax rescale of `acc` and `l`.
- TP: choosing TP for throughput (use DP) or crossing the NVLink domain.

***

## References
- Kwon et al. (2023) "vLLM" arXiv:2309.06180; Leviathan et al. (2023) "Speculative Decoding" arXiv:2211.17192.
- Dao, T., et al. (2022). "FlashAttention." arXiv:2205.14135.
