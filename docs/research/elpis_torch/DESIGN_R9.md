# Elpis Torch R9 — design: Boardy adversarial qualification

R9 is a falsification phase, not a feature phase. It asks whether CGC earns its place in the Elpis PyTorch
architecture, and whether the architecture's computational claims survive complete accounting.

It adds no model feature, changes no model code and touches nothing under `src/` or `native/`.

Evidence and verdicts are in `EVIDENCE_R9.md`. The frozen plan is `PREREGISTRATION_R9.md` with
`results/r9/preregistration.json`.

## 1. Order of work and why

1. **R8 closeout (P0)** against the merged `main` (`CLOSEOUT_R8.md`): the authority every later result builds on.
2. **Claim audit (P1)** (`CLAIM_AUDIT_R9.md`): what each existing record supports, before anything new is claimed.
3. **CGC attribution (P2)**, the decisive experiment, preregistered and frozen before any confirmatory run.
4. **Whole-model memory and bandwidth (P3):** EXACT, MEASURED and PROJECTED kept apart.
5. **Gate A (P4):** projection and transport, audited for an executable seam. There is none, so the gate stops at a
   frozen protocol with HOLD (`GATE_A_PROTOCOL_R9.md`).

## 2. P2 — the attribution experiment

**Why these arms.** Every model since R6 that carried CGC carried it inside a routed expert, beside an FFN, behind a
learnable gate. The only clean question is what the same resources do as ordinary FFN width, so every arm except D
changes the routed channel alone:

- **A:** R7's PLE model itself, so R7's result is the reference.
- **B:** removes CGC and its parameters.
- **C:** spends the CGC's parameters on FFN width.
- **C_mac:** spends its multiply-adds on FFN width.
- **D:** removes CGC's contribution from trained A at evaluation. It measures reliance, never superiority.

**Matching.** Matching is computed before training, on the meta device (`results/r9/accounting.json`):

| Pairing | Matched on | Residual mismatch |
|---|---|---|
| C against A | total parameters, within 0.022% | 2.6% fewer routed multiply-adds |
| C_mac against A | multiply-adds, within 0.26% | 2.2% more parameters per expert |

A must beat both. A CGC "win" that is really a compute advantage therefore fails.

**Why exactly 1,500 steps differs from R7, and how R9 still anchors to R7.** R7's records show 1,432 executed steps:
one epoch, with the 1,500-step schedule cut short.
- R9 runs exactly 1,500 steps, as directed, by continuing into a second seeded epoch.
- It separately reproduces R7's seed 0 under R7's own protocol (`integrity`) to show the pipeline is R7's.

**Pairing.**
- Initialization is a function of seed and parameter name, so every shared parameter starts identical across arms.
- Every arm of a seed sees the same batches in the same order.
- Differences are therefore paired by seed.

**Decision rule.**
- `delta_min` is 0.020 bits/byte, with sign consistency over all three seeds, on both held-out domains.
- Classes and verdicts are fixed in `PREREGISTRATION_R9.md` §5 and applied mechanically by `run_r9 combine`.
- Combine refuses any record whose preregistration or runner digest differs from the committed files.

**Secondary measurements.** Gates, gradient norms, routing, stability, generation mechanics, learning curves against
measured FLOPs, latency and peak RSS. They are recorded in the same runs and decide nothing.

## 3. P3 — the accounting boundary

`experiments/run_r9_p3.py`. Configurations:
- the R9 study arm (d 64, no history head);
- the R8 timing geometry `enc256` (d 256, 4 ERSA blocks, routed experts, PLE, an R5 head with a 256-wide encoder),
  with fine keys reread or stored and the PLE on or off;
- `r6_500m` and `r6_1b` on the meta device (never allocated), and `r6_500m` materialized on CPU for `trace`.

History capacity is 1,048,576 tokens throughout. Lengths are 1K, 8K, 64K, 256K and 1M. One point is added
beyond the preregistered lengths: the steady state of an endless stream (capacity + 4,096 tokens). By then the
R5 ring has wrapped, and the decode path behaves differently there (§3.2). It is an extra accounting point, not
a changed threshold.

### 3.1 Three kinds of number

**EXACT**, from `context_interface(model).fixed_bytes` and parameter shapes:
- **Weights:** resident and active per token.
- **CGC and PLE:** CGC parameters; PLE table and fusion parameters.
- **Recurrent state:** ERSA cells and convolution tails.
- **Attention:** window KV, allocated and live.
- **R5 history:** the token ring, allocated and live; descriptors and ids; R8 fine keys (when stored).
- **Lookup tables:** tables read by row (such as the encoder's `delta` table), which are not matrix weights.

**PROJECTED (`account`).** The **compulsory** traffic of a decoded token: every persistent byte it needs, once.
- Covered: weights, state tiers and gathered rows.
- Not counted: reuse across tokens, and step-local intermediates.
- **Bytes:**
  - matrix weights;
  - ERSA state read and written once;
  - the live window KV;
  - the descriptor scan, plus the wrapped-index copy once the ring has wrapped;
  - token rereads and gathered fine keys;
  - the encoder's gathered `delta` rows and the core-embedding rows it reads.
- **FLOPs:**
  - the core matrix weights and window attention;
  - the open block's encoding, a full 40-token window at every decoded token;
  - R5 re-encoding (64 blocks reread, or 4 with stored keys);
  - the descriptor scan and reranking.
- Routed experts that do not run are not read. They stay resident, and resident bytes are reported separately.

**EXECUTED (`trace`).** The operand bytes of real decode steps. A `TorchDispatchMode` sees every aten op, and each
operand is labelled by the storage it lives in: a parameter, a state tier, a lookup row, a buffer, or a step-local
intermediate.
- **Counting rule:** a dense op reads every operand and writes every result in full; a gather reads only what it
  gathers; views move nothing.
- **Stages** come from the call chain and the module stack. Line boundaries inside `_retrieve` and `_retrieve_tile`
  are located in the source, not hard-coded.
- **Fill:** history is filled synthetically to each length (counters, live range, window position; random contents).
  The fill is validated against a real prefill: total bytes are equal, and the only differing entries are which
  routed experts the different hidden states chose.

These are logical bytes, not DRAM transfers. This VM exposes no hardware performance counters (`trace` records the
event sources), so DRAM traffic is not measured. HBM is not measured either: CUDA is NOT_RUN.

**MEASURED:**
- decode p50 and p95, the R5 read's stage costs, and peak RSS, at every length, from R8's unchanged admission
  measurement (`run_r8 admission`) on `enc256`, with the PLE on and off;
- a large-copy memory bandwidth of this machine (`run_r9_p3 bandwidth`), labelled as this VM, not the DDR3
  scenario;
- RSS and USS growth when a 1M-capacity state is created and filled (`rss`), against its EXACT bytes. The fill
  goes through R8's no-read ingest; the measured decode step reads.

### 3.2 Corrections made before the P3 records were produced

A working draft of `account` reported 614 MiB per decoded token for `r6_500m` at 1M (stored keys, bfloat16 weights).
The trace was written to verify that figure, and found:

1. **A lookup table counted as matrix weights.** The evidence encoder's `delta` table (65,536 × 256; 32 MiB at
   bfloat16) is read by row: 200 rows per decoded token. Corrected: 583 MiB.
2. **The open block's encoding was amortized** as if once per block. The decode path encodes the open block's whole
   40-token window at every token (0.23 GFLOP for `r6_500m`, not 0.007).
3. **Not counted:** the core-embedding rows the encoder reads (0.3 MiB) and the 40-token window width (41 was used).
4. **Not modelled: the wrapped ring.** When the live blocks wrap around the ring, `_retrieve` joins two views with
   `torch.cat`, so every decoded token copies the whole live index (+64 MiB at 1M) before scanning it.
5. **bfloat16 does not run on CPU.** The bfloat16 rows are projections of a precision this implementation cannot
   decode on CPU: the chunked ERSA scan's triangular solve has no bfloat16 CPU kernel. Had it run, the float32
   descriptors would also be converted to the model dtype at every read (`res_keys.to(dtype=...)`), another O(L) copy.

**Roofline (PROJECTED).** The lower bound per token is the larger of bytes / (bandwidth × achieved fraction) and
FLOPs / compute. Because the bytes are compulsory, these ceilings are optimistic. The scenarios are ASSUMED and
frozen in the preregistration:
- **DDR3:** 8 GiB RAM, 10 / 15 / 20 GB/s, 50–200 GFLOP/s, float32 / bfloat16 / int4 weights, batch 1 and 4, 0 or 50%
  of active weights cache-resident.
- **GPU:** 300 / 900 GB/s and 20 / 80 TFLOP/s. CUDA is NOT_RUN.

The 60–100 tokens/s target is compared, never claimed.

**What is not modelled:**
- allocator reserve and fragmentation;
- cache hierarchy effects beyond the stated cache-resident fraction;
- kernel efficiency beyond the achieved fraction;
- the overlap of compute with memory.

## 4. P4 — Gate A

The audit and the frozen protocol are in `GATE_A_PROTOCOL_R9.md`.
- No φ from K1 state to the HCA's relation rows exists. Nor does any re-import, or any task that consumes transported
  K1 information.
- The HCA line keeps CGC `SHADOW_ONLY` and lists frame transport as an unqualified seam.
- Building the seam would be inventing what the gate tests. Disposition: **HOLD**.

## 5. P5 — the workflow controls

| Control | How |
|---|---|
| Base | one branch from the verified green `main` (`4534fbd`); one PR; no stacking |
| Preregistration before data | the preregistration and runner are committed and pushed before the first confirmatory run; records carry both digests; combine refuses mismatches; a test re-checks the digests |
| No reruns, no extra seeds, no threshold changes | stopping rules in the preregistration; every run's status is recorded |
| Adverse results kept | every record is committed whatever it shows; the claim audit corrects imprecise historical wording without editing the records |
| No scope expansion on failure | no larger model in R9 whatever the verdict |
| No canonical change | `src/` and `native/` untouched; no CI runner change |
| No external models or private files | the corpus is this repository's own text at `c05adff`; nothing is downloaded |
| No persistent inference writes, no runtime weight mutation | none of the measured paths writes |
| Separate labels | correctness, scientific usefulness, memory, performance and integration are separate verdicts in `EVIDENCE_R9.md` |
