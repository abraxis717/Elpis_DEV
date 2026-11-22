# Elpis Torch R13 — evidence: the Geodome cognitive substrate admission (P0–P4)

**Question.** Can Geodome v1.0.0 be admitted as a frozen dependency and used as the substrate of a differentiable,
bounded, distributed computation inside the actual EvidenceLM, without changing anything the backbone computes when
the computation is off?

**Answer: yes, mechanically.**
- Every one of the twelve gates passes, on CPU, in float64 reference checks and float32 resource runs.
- The tensor executor reproduces Geodome's own scheduler: equal round records, statuses, rounds and visits, with
  rows within 6.1e-16, in both residencies.
- Gradients reach every kernel through a 5-hop path, stop exactly at the hop bound, and agree with float64
  finite differences end to end through the actual routed final channel.
- With the gate at zero, the refined logits equal the backbone's bit for bit in all 18 EvidenceLM variants.
  Reasoning never moves the external clock.
- **The owner's review hardening is in, and re-qualified** (§9).
  - `release` is all or none and refuses a repeated handle.
  - A synchronous plan beyond the touched-vertex budget is refused before execution.
  - The backbone's own candidate ranking can be captured bit-exactly for P5.
  - Every gate passes again on the corrected sources, and every deterministic result of the original records
    reappears unchanged.
- **Competence is not tested yet.** `DISTRIBUTED_COMPETENCE` is `HOLD`.
  - P5 is frozen (`PREREGISTRATION_R13_P5.md`).
  - Only its DEVELOPMENT-ONLY feasibility pilot has run (`PILOT_R13_P5.md`), and the pilot decides nothing.
  - The pilot found complete retrieval coverage on its primary set (recall 1.000).
  - It also found a chance-level native backbone, and no learning signal within its 200 updates: every arm's
    predictions equal the backbone's.
  - The confirmation needs the owner's further authorization.
  - Nothing here supports a cognitive or geometric advantage claim.

**Records.**
- **The original receipts** are in `research/elpis_torch/results/r13/`: `identity`, `audit`, `parity`,
  `gradients`, `bridge`, `resources` and `verdict`, with `MANIFEST.json`. They were produced by
  `experiments/run_r13.py` on the clean revision `ae2f804`, and are kept unchanged.
- **The supplemental receipts** are in `research/elpis_torch/results/r13_hardening/`: the same seven records plus
  `hardening`, with `MANIFEST.json`. They were produced on the corrected sources at `fffe0cc` (§9).
- **Tests.**
  - `test_r13_records.py` reads JSON only. It recomputes every gate and every disposition of both sets from the
    raw rows, and checks the supplement's lineage against the original.
  - `test_r13.py` runs the mechanics in CI.

## Disposition

| Field | Value | Basis |
|---|---|---|
| `GEODOME_SOURCE_PIN` | HexaFlake `4e5fe56670c61a0d8906770b03033658c3087c3a`, `Geodome/`, Reference API v1.0.0, `SOURCE_PIN` `993429cc94e9e3dcfd42df40502fa5b57175d883`, `PUBLIC.sha256` `e44d288f…d23d31cb8` | §1 |
| `ELPIS_BASE_PIN` | `d73cdde3a5050de50cce098472ee966eec103000` (main after R12) | §1 |
| `SUBSTRATE_IDENTITY_PARITY` | **PASS** | gates 1, 3, 4 |
| `DIFFERENTIABLE_EXECUTION` | **PASS** | gates 5, 6, 7; no host extraction; bounded graph |
| `DISABLED_PATH_PARITY` | **PASS** | gate 2 |
| `CONTEXT_STATE_COHERENCE` | **PASS** | P0 seam; gates 11, 12; training-mode qualification |
| `EVIDENCE_ADMISSION` | **PASS** | causal, canonical, live, exact-reread admission in all 18 variants; candidate capture bit-exact (§9) |
| `SESSION_ISOLATION` | **PASS** | gates 8, 9, 10; all-or-none `release` (§9) |
| `BOUNDED_COMPUTATION` | **PASS** | caps refused atomically; constant graph and gradient scratch; the synchronous touched budget (§9) |
| `DISTRIBUTED_COMPETENCE` | **HOLD** | P5 frozen (plan `094adb45…`); only the DEVELOPMENT-ONLY pilot ran (`PILOT_R13_P5.md`); the confirmation is not authorized yet |
| `LEARNED_ADDRESSING` | **NOT_STARTED** | P6 follows P1–P5 |
| `CGC_ADMISSION` | **NOT_STARTED** | P7; no CGC volatile adaptation in this integration |
| `HCA88_ADMISSION` | **NOT_ADMITTED** | qualified upstream only for C4 two-arc relation windows; not used |
| `COGNITIVE_ADVANTAGE` | **NOT_CLAIMED** | mechanical tests cannot carry it |
| `CI_STATUS` | see §8 | exact-head CI of the pull request |

A mechanical PASS and a learned-competence HOLD are independent outcomes. This phase reports both. Each
mechanical disposition is PASS in the original receipts and again in the supplemental receipts.

## 1. The dependency and its identity (gate 1, `identity.json`)

- **The sources.** All ten vendored files equal their pins and their lines in the vendored `PUBLIC.sha256`, which is
  itself pinned. The committed git blobs are byte-identical too.
  - A damaged tree is refused: an edited, added or missing file (tests).
  - So is a module that shadows a Geodome name, before anything is imported.
  - Every Geodome module resolved inside `upstream/geodome_v1/`, and `sys.path` was restored.
- **The identity.** Geodome's `build_index()` rebuilt the index in 1.1 s, once per process.
  - Counts: 144002 vertices, 432000 edges, 864000 arcs, capacity 823543 at depth 7.
  - The mapping and edge digests equal the pins, and the payload is 11,502,238 bytes.
  - There are 12 degree-5 vertices; vertex 0 is one of them.
- **Round trips.** Addresses, coordinates, arcs (`arc ^ 1` reverses one) and ports round-trip on the sampled
  vertices. A non-edge is refused (`NONEDGE`), and so is an out-of-range arc (`ARC_RANGE`).
- **Prefixes are not regions.** 16 of the 27 examined depth-3 prefix groups (420 vertices each) are physically
  disconnected. Geodome's C1 receipt found 236 of 343. Only `physical_arcs`, built from real neighbours, admits
  arcs.
- **The base is untouched.** No `.py` file under `research/elpis_torch` and no earlier record differs from the base
  `d73cdde`.

## 2. P0: the seam (`audit.json`)

| Measurement | Value |
|---|---|
| captured channel rows of a 70-token chunk; final row equal to `return_pre` | `[1, 70, 32]`; bit-identical |
| tail recomputed over the same T rows against the forward's logits | bit-identical |
| tail recomputed over the final row only (the batch variance the bridge avoids) | max abs 5.6e-17 |
| full-logits forward against `logit_positions=[T-1]` at the last position | max abs 1.1e-16 |
| one 70-token chunk against 35 + 35 | max abs 3.3e-16 |
| evidence `off` against `learned`: earlier rows, recurrent state, final row | bit-identical, bit-identical, the read moves it by up to 0.0063 |
| refusals (oversize chunk, out-of-range token, oracle without labels) | all refused, the state digest unchanged |
| evidence read placement | `read_layer = 1`, the last block |

## 3. The tensor executor against Geodome's scheduler (gates 3, 4, 7; `parity.json`)

The reference is Geodome's own `SparseScheduler` over its own float64 `NodeState`. Its pure callbacks compute the
same kernels. The tensor executor ran the changed-destination schedule (Geodome's semantics) in sparse and in
dense residency.

| Workload | Enrolled / arcs | Status | Rounds | Visits |
|---|---|---|---|---|
| spread from one vertex (degree 5) | 40 / 198 | `ROUND_BUDGET` | 12 | 1880 |
| three sources | 60 / 306 | `ROUND_BUDGET` | 10 | 1790 |
| ball around a degree-5 vertex | 30 / 144 | `ROUND_BUDGET` | 8 | 849 |
| chain, forward arcs only | 8 / 7 | `QUIESCENT` | 8 | 7 |
| frontier without outgoing arcs | 2 / 1 | `QUIESCENT` | 1 | 0 |
| active budget (12 touched per round) | 40 / 198 | `ACTIVE_BUDGET` | 0 | 0 |
| arc budget (40 visits) | 40 / 198 | `ARC_BUDGET` | 2 | 35 |
| round budget (2) | 40 / 198 | `ROUND_BUDGET` | 2 | 35 |

- **Every workload matches in both residencies.** Status, rounds, visits and every round record
  (`status, sources, arcs, destinations, changed, touched`) are equal, and rows agree to at most 6.1e-16.
- **Dense and sparse tensor rows are bit-identical** (gate 4). So are the bridge's latent and logits in the two
  residencies.
- **The synchronous schedule, on its matched class.** On a 50-vertex ball, with a full start and every destination
  changing every round (read from Geodome's records), it equals the scheduler to 4.4e-16, with equal visits.
- **Outside that class it differs, by design.** A one-directional chain differs by 0.87. That difference is
  recorded and never claimed equal.
- **Hop causality** (gate 7, executor). After r synchronous rounds, a perturbation of the first chain vertex has
  moved exactly hops 1 to r, for r = 1, 3 and 6.

## 4. Gradients (gates 5, 6, 7; `gradients.json`)

- **Multi-hop** (gate 5). The plan puts evidence at one end of a 6-vertex chain, the query 3 hops from the readout,
  and the readout at the other end, with 6 rounds and gate 0.5. Every one of the 19 kernel tensors receives a
  finite nonzero gradient (L1 from 1.7e-4 to 8.3):
  - ingress (`query_in`, `evidence_query`, `evidence_in`);
  - message (`message_in`, `port_out`, `port_in`, `message_out`);
  - update (`update_gate`, `update_cand`);
  - readout, injection and gate.
- **The zero gate.** With `g = 0`, only `gate` receives a gradient, and the refined logits equal the backbone's.
- **The hop bound** (gate 7, bridge). The evidence vertex is 5 hops from the readout. With 4 rounds its ingress
  gradient is exactly 0.0; with 5 rounds it is 1.2e-3.
- **Float64 numerical agreement** (gate 6). `torch.autograd.gradcheck` passes:
  - for the executor and readout, over the initial rows and seven weight tensors;
  - end to end on the actual bridge, over nine kernel tensors, through ingress, rounds, readout, injection, and the
    final block's actual routed channel, final norm and head.
- **The bounded graph.** One `reason()` call builds 185 autograd nodes, the same on repeated calls. The working
  state holds no graph, and an external step drops every result.
- **No host extraction on the cognitive path.** The `TorchFunctionMode` detector recorded none across ingress,
  synchronous rounds, readout and injection. Its positive control records `item`, `tolist`, `__bool__`,
  `__float__`, `__int__` and `numpy`. The path's modules import neither NumPy nor ctypes.

## 5. The bridge on actual EvidenceLM (gates 2, 8–12; `bridge.json`)

**The variants.** The full matrix of evidence `off`/`learned`, fine keys `reread`/`stored`, PLE off/on and origin
0/64, plus two window-attention variants: 18 in all. Every stream wraps the 256-token ring and evicts its earliest
blocks: at the query, the live blocks are 20–35 at origin 0 and 24–39 at origin 64 (21–36 and 25–40 with the
window).

- **Disabled path** (gate 2). In every chunk of every variant, the session's logits and state digest equal a plain
  `EvidenceLM.forward` of the same chunk, bit for bit. The digest covers cells, tails, window, PLE history, ring
  tokens, keys, ids, fine keys and positions.
- **Zero gate.** The refined logits equal the bypass bit for bit in all 18 variants.
- **The context clock** (gate 11). `admit_evidence`, `reason` and `readout`, and every refused admission, leave the
  state digest unchanged. The only counters that move are `blocks_reencoded` (+2) and `ring_rereads` (+2), for two
  admitted blocks.
- **Admission.** In every variant, two causal, canonical and live blocks are admitted. Each of these is refused:
  - an evicted address (`StaleAddressError`);
  - the query's own block and a future block (`EVIDENCE_CAUSALITY`);
  - a tampered generation, a tampered span and another epoch (`StaleAddressError`);
  - a duplicate (`EVIDENCE_DUPLICATE`);
  - a non-address (`EVIDENCE_TYPE`).
- **Inference and training.** Observing inside `enable_grad` still runs the backbone without gradients.
  - With stored fine keys and the `learned` read, the backbone's own read used the R8 tier: 32–95 fine-tier reads.
  - The cognitive admission never did: its evidence is the exact reread.
- **Reset** (gate 12). After `reset()`, the same stream gives exactly what a brand-new session gives: logits, state
  digest, cognitive latent and refined logits. The working state is empty. The previous epoch's handle and address
  are refused.
- **Refusals and atomicity** (gates 8, 9). 31 refusals, plus two bypass checks. Each refusal leaves the latent
  rows, the resident count, the epoch, the version, the state digest, the position and every result object
  unchanged.

| Refused | Code |
|---|---|
| an arc leaving the enrolled vertices | `ARC_AUTHORITY` |
| a fabricated non-edge | `NONEDGE` |
| an out-of-range arc; a duplicate arc | `ARC_RANGE`; `DUPLICATE_ARC` |
| a non-resident vertex | `NONRESIDENT` |
| 17 rounds (cap 16); visits below rounds × arcs | `ROUND_RANGE`; `VISIT_BUDGET` |
| two evidence bindings with one admitted item; the query and evidence on one vertex; no readout | `EVIDENCE_BINDING`; `BINDING_OVERLAP`; `READOUT_BINDING` |
| a frontier on a synchronous plan; an unknown schedule | `START`; `SCHEDULE` |
| enrolment beyond capacity; an already-resident vertex | `STATE_CAPACITY`; `ALREADY_RESIDENT` |
| another session's handle | `FOREIGN_OR_STALE_HANDLE` |
| another session's address (same block, slot and generation) | `StaleAddressError` |
| nine evidence items (cap 8) | `EVIDENCE_COUNT` |
| publishing non-finite rows, the wrong shape, the wrong dtype, foreign slots | `WorkingStateError` |
| float tokens; 71 tokens; batch two | `ValueError` |
| a non-finite computation (a NaN weight) | `NONFINITE`; nothing published; the committed answer kept |
| a closed session; serialization; copying | `SessionClosedError`; `TypeError` |
| a backbone revision (an in-place write) | `ModelRevisionError`, sticky until `reset()` |

- **The bypass.** Before a readout, `prediction()` returns the backbone's cached logits; after one, the committed
  refined logits.
- **Immutability** (gate 10). Three SGD steps on the kernels left the backbone's parameter digest unchanged, with
  no backbone gradient, while the kernels did train.

## 6. Resources (`resources.json`; MEASURED on this CPU container, 2 threads, float32)

| Quantity | Value |
|---|---|
| kernel parameters (width 32, hidden 64; 19 tensors) | 14,401 (57,604 bytes) |
| backbone parameters (fixture) | 165,633 |
| backbone context state per session | 4,992 bytes |
| working state, 64 resident slots | 585,292 bytes: latent rows 8,192, Geodome `NodeState` identity metadata 576,844 (an O(V) vertex-to-slot map), unused rows 256 |
| per-session total (exact projection) | 647,888 bytes |
| topology (the frozen index; shared by the process) | 11,502,238 bytes |
| compiled plan (48 vertices, 238 arcs) | 10,072 bytes |
| gradient scratch (saved tensors per training step) | 3,199,388 bytes, constant over steps |
| autograd nodes per training step | 387, constant |
| message visits per `reason()` (8 rounds × 238 arcs) | 1,904 |
| active vertices per round | 48 |
| latency, median (observe 70 tokens; admit 4 blocks; reason; readout; reason + readout + backward) | 7.9; 1.3; 2.1; 1.2; 8.0 ms |
| index build (once per process) | 1.14 s |
| process peak RSS (the whole runner) | 282.1 MB |

The latencies are one container's measurement, not a benchmark. Geodome's own README notes that its generic
scheduler is slower than task-specific traversal. The trainable path here does not use that scheduler.

## 7. Limitations

- **No competence.** In P0–P4 nothing was trained to do anything, and no task was evaluated.
  - Whether the substrate helps any task is P5's question, with its controls: native, equal-budget,
    compute-matched, flat-recurrent, relabel-isomorphic and rewired.
  - That study is frozen, but only its DEVELOPMENT-ONLY pilot has run, and the pilot decides nothing.
- **Plans are host-supplied** (tests use chains and BFS balls). No learned addressing or routing exists (P6).
- **The fixture is a toy.** A two-block, 32-wide EvidenceLM, as in R5–R8 tests. There is no scale, GPU or
  language claim.
- **Batch one, the final position only.**
- **Mechanical, not behavioral.** The no-host-extraction gate covers the cognitive path only. The backbone's routed
  channel keeps R6's integer bookkeeping.
- **Process-level RSS.** It includes the index build and Python. It is not an isolated session measurement.

## 8. Reproduce, and CI

```bash
python -m research.elpis_torch.experiments.run_r13 identity --out research/elpis_torch/results/r13/identity.json
```

The other commands are `audit`, `parity`, `gradients`, `bridge`, `resources`, `verdict` and `manifest`, in that
order. Each takes seconds on one CPU.

**The supplemental receipts** are produced the same way, with `--results research/elpis_torch/results/r13_hardening`,
`--out` into that directory, and the extra `hardening` command before `verdict`.

**The tests** are `tests/research/elpis_torch/test_r13.py` (35 tests, about 10 s) and `test_r13_records.py`
(JSON only, both receipt sets).

**CI_STATUS.** The pull request's exact-head CI is reported in the pull request. The default and forced-AVX2 kernel
paths of the Elpis Torch job both run these tests.

**The first red run.** The first CI run of this branch failed six R13 tests, all with "Numpy is not available".
- **Cause.** The Elpis Torch job installs PyTorch without NumPy (R10 documents the same constraint), and the
  runner's state digest hashed tensors through `.numpy()`.
- **Fix.** `ae2f804` hashes each tensor's dtype, shape and raw bytes through `ctypes`, as `model.parameter_digest`
  does. The cognitive path never used NumPy, and its modules still import neither NumPy nor ctypes.
- **Records.** They were regenerated on `ae2f804`. Every deterministic field is identical to the first records;
  only the provenance and the timing and RSS measurements differ.
- **Verified.** The R13 tests pass locally with NumPy made unimportable (as in CI).

## 9. The owner's review: hardening and supplemental receipts (`results/r13_hardening/`)

**Authority.** The owner's review of the pull request at head `343dc92` accepted the P0–P4 mechanical
qualification and asked for two additive hardening fixes before merge. The executor and its numerical semantics
were not to change. The corrections are in `fffe0cc`, and the supplemental receipts in `c565097`.

**1. `release` is all or none.**
- **The defect, reproduced on the reviewed sources.** `release((h, h))` released the vertex, then failed on the
  second handle as stale. The row was left unzeroed and the version unadvanced: a partial mutation.
- **The fix.** The whole batch is checked first: emptiness, every handle's validity, and repeated vertices.
- **The receipt.** Each invalid batch below is refused with its code. A complete snapshot of the working state is
  equal before and after the refusal: the latent bytes, version, epoch, resident count, Geodome's payload
  accounting, and every vertex's handle (slot, generation, epoch) or its absence.

| Batch | Code |
|---|---|
| `(h, h)`; `(h₁, h₂, h₁)` | `DUPLICATE_HANDLE` |
| empty | `EMPTY_BATCH` |
| a valid handle with another working state's handle; a valid handle with a non-handle; an earlier epoch's handles (after `reset`) | `FOREIGN_OR_STALE_HANDLE` |
| a valid handle with a stale generation's handle | `STALE_HANDLE` |

A valid release makes its vertices non-resident, zeroes their rows, leaves every other row unchanged and advances
the version. The released handle is then refused as stale.

**2. The synchronous touched-vertex budget.**
- **The rule.** A synchronous plan whose touched set (the sources and destinations of its arcs) exceeds
  `max_active` is refused at compile time (`ACTIVE_BUDGET`), before any execution.
- **The receipt.**

| Case | Result |
|---|---|
| a chain touching 5 vertices, `max_active = 5` / `4` | compiles / `ACTIVE_BUDGET` |
| a partial arc set, at its touched count / one below | compiles / `ACTIVE_BUDGET` |
| Geodome's default cap: 1024 touched / 1025 touched | compiles / `ACTIVE_BUDGET` |
| a changed-destination plan with a budget below its touched count | compiles, and stops at run time with Geodome's own `ACTIVE_BUDGET` status, as before |

- **The bridge battery** gains the same refusal through `reason()`, and the duplicate release through the working
  state. That makes 33 coded refusals, each leaving the full state unchanged.

**3. Candidate capture for P5** (the owner's decision on the evidence source).
- **What it captures.** `CognitiveSession(capture_candidates=n)` keeps the backbone's own top-n fine-ranked live
  blocks at the query, from the same forward.
- **Bit-exact.** In four EvidenceLM variants (reread and stored fine keys, PLE on and off, origins 0 and 64,
  window attention on and off), capture leaves every chunk's logits and state digest, and every counter, equal to a
  session without it.
- **Admissible.** The captured addresses are distinct, causal, canonical and live, at most `fine_top`, and admitted
  by the ordinary exact-reread admission.
- **Refusals.** Capture beyond `fine_top`, and capture with the evidence read off, are refused.

**Lineage.**
- **Every gate passes again** on the corrected sources, with the same dispositions.
- **Every deterministic field** of the original records reappears unchanged in the supplement: identity, seam
  measurements, parity rows and records, gradients, bridge results and resource counts. Only three things differ:
  the provenance, the timing and RSS measurements, and the two added refusals.
- **The sources that changed** since the original receipts are exactly the declared four: `plan`, `state`,
  `binding` and the runner.
- **The original receipts** are untouched; their manifest digest `c97841af…` is pinned in `test_r13_records.py`.

**CI.** The exact-head CI of `c565097` passed every job, in both the push and the pull-request runs, including the
default and forced-AVX2 kernel paths of the Elpis Torch job.
