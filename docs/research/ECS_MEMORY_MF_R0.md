# ECS Memory MF-R0: bounded, model-free, typed relational memory (research)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `TRAINING=NONE` · `SEMANTICS=NONE` · `NO_LANGUAGE_CLAIM`

Code: `research/ecs_memory_mf_r0` (native C11). Successor of the PR #47 DEV foundation
(`ae1677b`, formerly `native/ecs_memory_experimental`). This document is the authority for the
mechanism's contract, its census, its experiments and its limits. It grants nothing: the one ECS
remains `native/ECS` (K1), and nothing in this directory is on the cognitive path.

## M1 — Repository and authority census

### Constitutional constraints that bind this work

| Authority | Constraint | Consequence here |
|---|---|---|
| `docs/ELPIS_MISSION.md` | one ECS (`native/ECS`); `native/` and `src/` are canonical, `research/` is not | the experimental memory moved out of `native/` (where PR #47 had placed it, unbuilt) into `research/` |
| `tests/boundary/_mission.py` (`KERNEL_DIGESTS`) | the qualified ECS kernel sources are SHA-256-pinned | `native/ECS` is not modified by this work |
| `tests/boundary/test_mission.py` | `research.native_targets` is pinned to the three DSV4.1 libraries | the memory code is built as test-kind research support (`elpis_add_test_library`), not a declared library |
| `docs/ECS_K1_RUNTIME.md` | the complete cognitive state is `(W, epoch, H, a)`; `ELPISGK1` v1 is the retained-state envelope and its SHA-256 trailer the identity | the memory state is **not** part of the K1 identity; it cannot be added without a new, qualified envelope version |
| `docs/CONTINUITY.md` | continuity is two 176-byte slots holding one K1 digest and the evolution authority; no history | no memory state, log or second digest may be placed in continuity |
| `docs/K1_RECOVERY_R0.md` | checkpoint slots hold exactly one `ELPISGK1` envelope of the admitted shape | the K1 checkpoint owner cannot hold the memory envelope |
| `docs/ECS_MUTABLE_FMS_R0.md`, `ecsg_k1_fms.h` | FMS is generic residency; it never interprets an object | FMS could host memory bytes generically, but no admitted memory exists to host |
| `docs/ARCHITECTURE.md` ("No canonical HACF -> ECS edge") | structural memory reaches ECS only through research (`research/hacf_ecs_bridge_r0`, read-only QUERY) | the HACF adapter here is research-only and confers no truth or authority |
| `docs/CI_POLICY.md` | every test in exactly one lane; NATIVE = ctest; SCIENTIFIC = frozen evidence | native mechanics are ctest targets `research.ecs_mem_*`; evidence tests live under `tests/research/` |

### Ownership of state transitions (who may do what)

| Concern | Owner today | Function / data contract | Used by MF-R0 |
|---|---|---|---|
| authoritative cognitive state, LEARN/CONSOLIDATE transactions | ECS K1 (`native/ECS`) | `elpis_ecsg_k1_txn_*`, `elpis_ecsg_k1_state_digest(state, out[32])` | read only: the K1 digest is an input to the composite identity (M4) |
| managed exclusive mutation | RuntimeCore lease on K1 | `elpis_ecsg_k1_lease_claim(state, lease)`; unmanaged writes refused `LEASED` | mirrored: `ecs_mem_lease_claim` with the same refusal law |
| durable current authority | continuity | `elpis_continuity_commit_cognition(store, before[32], after[32])` (K1 digest only) | **not used** (blocked, below) |
| durable state bytes | K1 Recovery R0 | `k1-checkpoint.{a,b}`, one `ELPISGK1` envelope | **not used** (blocked, below) |
| residency and budgets | FMS ABI v2 (`native/substrate`) | generic objects, leases, tiers | not used in R0; the memory has its own fixed arena and byte budget |
| structural identity, N-ary relations, provenance | structure (`native/structure/semantic`) | `elpis_semantic_hyperedge_v1`, `elpis_semantic_assertion_v1`, `elpis_semantic_hyperedge_identity`, `*_validate` | reused by the typed HACF adapter (validation and identity recomputation are structure's, not ours) |
| retrieval evidence | HACF retrieval bundle (`retrieval_bundle.h`, bridge ABI v2) | `RetrievalBundle` | not consumed; candidate entities may be supplied by a caller that obtained them there |
| hashing | substrate | `elpis_sha256_*` | code derivation and state identity |

### Verified capabilities, unsupported assumptions, duplication

* **Verified:** the D31-B/D31-C lattices compute what their headers state; the D31-B↔D31-C differential
  and the adapter smoke test pass (now in ctest, all native jobs).
* **Not supported by any authority in the repository** (classified **UNDEFINED**):
  * a LEARN authority for a memory other than K1 (RuntimeCore's `LearnAuthority` is K1-specific);
  * a meaning for HACF assertion flag bits (`SEMANTIC_ASSERTION_FLAG_MASK = 0x03` with no named bit):
    observation polarity cannot be read from HACF and must be supplied, and recorded as caller-supplied;
  * a qualified observation → ECS-state map (`ARCHITECTURE.md`: no canonical HACF -> ECS edge);
  * a composite K1 + memory retained-state format, its continuity binding and its checkpoint owner;
  * `docs/ECS_MEMORY_R0_CONTRACT.md` / `ECS_MEMORY_R0_ENVELOPE.md`, cited by the PR #47 note, are not in
    the repository and were not used.
* **Duplication avoided:** no second continuity register, no second checkpoint store, no second
  hashing or identity scheme for structure (HACF identities are recomputed by structure's own functions).

### Audit of the PR #47 lattice (verified by execution, pinned by `research.ecs_mem_lattice_audit`)

| Question | Finding |
|---|---|
| Eviction ordering | **write recency**, not LRU: `stamp` is set only by writes; reads take `const` and never refresh. A cell read most recently is still the next victim. |
| Cost counters | writes only. `index_lookups` counts 9 per successful write; reads perform 9 probes each but are **not counted** (the header comment says otherwise). No counter reports read arithmetic. |
| Admission planning | **O(capacity) per missing cell** (linear free-slot scan, then a linear victim scan), with an O(9) reservation check per step; 3 writes at capacity 18 cost 495 scan steps. |
| Byte accounting | `d31c_allocated_bytes` = struct + cells + direct index; the adapter adds its struct. Excluded: allocator overhead and the 9 × 8 KiB on-stack staging of every write (stated in the README). The differential test's byte assertion is a disjunction that is almost always true (`== ... || > ...`). |
| Failure atomicity | write and read refusals leave arena and output unchanged (tested); `ecs_mf_address` writes `NaN` into its output on an invalid radius (an output mutation on failure). |
| Arithmetic determinism | binary32/binary64 with libm `pow`, `cos`, `sqrt`; contraction is not pinned (`-ffp-contract` default differs between GCC and Clang). Bitwise results across compilers/hosts are **not** defined. |
| Replay | no serialization of any kind; no restore. |
| CI registration | **none**: `native/ecs_memory_experimental` was not added to the root build, so CI never compiled or ran it. |

The donor-derived sources stay byte-identical (they are the PR's frozen D31-B/D31-C handoff) and serve
as differential references and as a competing design in the experiments. Their defects are fixed in
the new mechanism, not in them.

### Integration map

```
HACF/structure (owner of identity + validation)
  elpis_semantic_hyperedge_v1 + elpis_semantic_assertion_v1
    | ecs_mem_hacf_observation(): validate, recompute both identities, refuse overlays/flags/too many participants,
    |                             polarity supplied explicitly (UNDEFINED in HACF)
    v
ecs_mem_observation (typed, N-ary, provenance, tick)            <- structural evidence reference (not truth)
    | ecs_mem_observe(): resident evidence slot + entity codebook (bounded, LRU, O(1) admission)
    v
resident representation (evidence slots)                         <- not authoritative cognition
    | ecs_mem_learn(): explicit, separate call; lease-guarded; net evidence only
    | ecs_mem_consolidate(): separate law (replay into slow banks)
    v
committed memory state (fast + slow gated-delta banks, integer)  <- the recurrent state
    | ecs_mem_query(): read-only, per-channel (LOOKUP | FAST | SLOW)
    v
answer + scores + work                                           <- a claim is derived by the caller
```

Seams reused: `elpis_sha256_*`; `elpis_semantic_hyperedge_validate/identity`,
`elpis_semantic_assertion_validate/identity`; the K1 lease refusal law (reproduced, not shared);
`elpis_ecsg_k1_state_digest` (read only, M4). Seams **blocked by authority** (no code written against
them): `elpis_continuity_commit_cognition`, the `ELPISGK1` envelope, the K1 checkpoint slots,
RuntimeCore's managed turn.
