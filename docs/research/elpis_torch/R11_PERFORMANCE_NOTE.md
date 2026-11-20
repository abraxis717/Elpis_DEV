# Elpis Torch R11 — performance handoff (from R9 and R10; nothing here was changed in R10)

This note hands performance work to a separate phase. R10 is a scientific study, and none of these items was
touched in it: every R10 arm runs the code as merged.

The items come from R9's executed decode trace (`EVIDENCE_R9.md` §3.3, `results/r9/p3_trace.json`) and from R10's
measured training step times. Each is a target to measure and qualify, not a claim that fixing it is safe or
worthwhile.

## 1. The wrapped-index copy (R5 coarse scan)

**Where.** `EvidenceLM._retrieve` (`evidence_model.py`, the "wrapped: two views joined" branch).

**What it does.** Once the history ring has wrapped past capacity, the live descriptor blocks no longer sit in one
contiguous slice. The two views are joined with `torch.cat` before the scan.

**Cost.**
- Every decoded token copies the whole live index and then scans the copy.
- At 1M capacity that is +64 MiB of operand traffic per token (32 MiB copied, then scanned).
- For the `enc256` geometry it more than doubles the compulsory per-token traffic: 55 → 119 MiB.

**Candidate.** Score the two views separately and join only the small score vectors. The scan's exactness and its
tie order must be preserved; R5 and R8 test both.

## 2. The window-attention recompute

**Where.** `ScrollAttention.forward` (`scroll.py`).

**What it does.** Rotary positions are relative to the window, so every call:
- re-rotates every held key;
- concatenates the held window with the chunk;
- expands keys and values for grouped-query attention with `repeat_interleave`.

**Cost.**
- R9's trace: 688 MiB of the 2,050 MiB a float32 `r6_500m` decode step executes at 1M (5 attention layers, a
  4,096-token window). That is 17× the KV bytes themselves.
- R10, measured at d 64, 2 threads, 32-token chunks: a training step takes 0.74 s with a 64-token window in both
  blocks, against 0.33 s without attention.

**Candidates.**
- Cache rotated keys with their absolute positions, so that only the query is rotated per call. This changes
  numerics, so it needs a qualification against the current path.
- Compute grouped-query attention by reshaping queries per KV group instead of copying K and V.
- Keep the window in time order, avoiding the gather and the concatenation.

## 3. The linear descriptor scan

R5's coarse scan reads every live descriptor at every decoded token, by design: 1 KiB per live 32-token block, so
32 MiB per token at 1M. This is the only per-token term that grows with context (R9).

Any sublinear index (hierarchical, quantized, tiled with bounds) must keep R5's exact-selection contract or state
plainly what it gives up.

## 4. Precision on CPU

**What fails.** The chunked ERSA scan's triangular solve has no bfloat16 CPU kernel, so this implementation does not
decode in bfloat16 on CPU.

**Consequence.** R9's bfloat16 rows are projections.

**Candidates.**
- Keep the recurrence in float32 under bfloat16 weights (the mixer already runs the recurrence in the state's
  dtype).
- Add a dedicated one-token decode step that needs no solve.

**Second cost.** With float32 descriptors on CPU, a bfloat16 model would also convert the live index at every read
(`res_keys.to(dtype=...)`): another O(L) copy.

## 5. Owner constraints that bind every item

These come from the owner's architectural direction:
- Normal experts stay resident in RAM or VRAM; no SSD expert paging.
- The PLE remains the primary lookup-addressable capacity.
- ERSA carries the recurrent computation.
- No persistent inference writes, and no inference-time weight mutation.
- No added computational pass without a measured benefit.

A performance change must preserve the tested causality, chunk invariance and exact-selection contracts. It must
be measured on the same configurations as R8 and R9, with performance and correctness labelled separately.
