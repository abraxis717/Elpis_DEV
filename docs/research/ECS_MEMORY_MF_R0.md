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

## M2 — The memory contract (`native/include/elpis/ecs_mem.h`)

The header is normative. One handle is one bounded memory with operator-fixed capacity.

### State, and what is authoritative

| Component | Kind | Written by | In snapshot / identity |
|---|---|---|---|
| evidence slots (S records: relation key, type, payload, canonical participants `(role, ordinal, entity)`, assert/negate/uncertain counts, first/last/last-assert/last-negate ticks, first/last provenance, last structure reference, last authority, learn count) | **resident representation** of observed structure; exact, reconstructible, not truth | OBSERVE (LEARN updates `learn_count` and recency) | yes |
| entity codebook (E identities with first/last tick) | candidate set for associative cleanup | OBSERVE | yes |
| LRU order of slots and of codebook entries | retention order (write recency) | OBSERVE, LEARN (slots) | yes |
| fast banks (B integer D×D, Q16) | **committed memory state**, learned | LEARN only | yes |
| slow banks (B integer D×D, Q16) | **committed memory state**, consolidated | CONSOLIDATE only | yes |
| clock, consolidation cursor | causal order, replay position | OBSERVE/LEARN/CONSOLIDATE | yes |
| hash indexes, restore workspace | derived (rebuilt on restore) | — | no |
| lease | ownership, not state | lease calls | no |
| telemetry (counters, refusals, write work) | non-authoritative | every mutator | no |

Structural evidence itself (HACF hyperedges and assertions) is **not** held: a slot carries the observation's
identity references (`structure_ref`, provenance digests) so that the evidence can be found again in HACF.
Observing a record does not make it true, authorized or learned.

### Operations

| Operation | Law | Refusals (state and outputs unchanged) | Work bound |
|---|---|---|---|
| INITIALIZE `ecs_mem_create` | one checked layout; one arena; `arena <= byte_budget` and `<= 1 GiB` | `INVALID`, `CAPACITY`, `NOMEM` | O(arena) zeroing |
| OBSERVE `ecs_mem_observe` | canonicalize participants; key = SHA-256(type, payload, participants); update or admit the slot; admit/touch distinct entities; `clock = tick` | `INVALID`, `CLOCK` (tick < clock, counter exhausted), `CAPACITY` (REFUSE policy, or no non-participant victim), `LEASED`, `BUSY` | O(P) hash probes (expected), O(P²) planning; no scan |
| LEARN `ecs_mem_learn` | per participant `p` (target) with the others as cue: one DELTA (or HEBBIAN) step into fast bank `bank(T, r_p, o_p)`; target = entity code if net > 0, zero if net < 0 | `NOT_FOUND`, `NO_NET_EVIDENCE`, `NUMERIC` (touched bank beyond 2^24), `CLOCK`, `LEASED`, `BUSY`, `INVALID` | P·(2–3)·D² integer MACs |
| CONSOLIDATE `ecs_mem_consolidate` | law C1: round-robin from the cursor over resident slots, at most `max_items` (≤ 16) with nonzero net evidence, one slow-bank step per direction at `consolidate_rate`, no decay | `INVALID`, `NUMERIC`, `CLOCK`, `LEASED`, `BUSY` | ≤ S slot visits + 128·2·D² MACs |
| QUERY `ecs_mem_query` | LOOKUP: exact scan, partial cues allowed, answer = latest assert among net-positive matches; FAST/SLOW: `y = M c / D`, score each candidate (caller list or codebook) | `INVALID`, `BUSY` | LOOKUP 2S slot scans; FAST/SLOW D² + n·D |
| INSPECT `ecs_mem_inspect`, `slot_read`, `codebook_contains`, `identity` | read only | `INVALID`, `BUSY`, `NOT_FOUND` | O(1) / O(state) for identity |
| SNAPSHOT / RESTORE | ELPISEM1 v1 (below) | `CAPACITY`, `CORRUPT`, `LEASED`, `BUSY`, `INVALID`, `NOMEM` | O(state) |

QUERY never learns, never changes recency and never writes telemetry; it reports its own work in the answer.
Consequently retention follows write use only: a relation that is only ever read is evicted like one never
read. This is deliberate (QUERY must not mutate the authoritative state), and it is a limit.

Every answer is labelled by the channel that produced it. LOOKUP is ordinary lookup over resident evidence;
FAST is the learned recurrent state; SLOW is the consolidated state. A claim (for example "X is the current
object of R for S") is derived by the caller from an answer, its channel, its score, `conflicts`,
`contradicted`, `uncertain` and `signal` (0 means the bank holds nothing along that cue: explicit absence).

### Encoding (deterministic, not semantic)

Codes are bipolar vectors of D bits from SHA-256 in counter mode over domain-separated identifiers
(`elpis.ecs-mem.code.{entity,role,target,type,tie}.v1`). A cue binds (XOR) the majority of the known
`role ⊗ entity` pairs with the relation type and the target role; the bank is
`SHA-256("elpis.ecs-mem.bank.v1" | T | r | o) mod B`. Codes of distinct identifiers are quasi-orthogonal by
construction. Any "generalization" from shared cue components (partial cues) is a property of this fixed
geometry, measured as such in M3; it is not learned similarity.

### Numerical bound

Banks are int32 Q16. Let `c ∈ {±1}^D`. The DELTA step on row `i` is
`row ← row (I − rate c cᵀ/D) + rate t_i cᵀ`; `I − rate c cᵀ/D` has spectral norm ≤ 1 for rate ∈ (0, 1], and
rounding moves each entry by at most 1, so each step adds at most `(65536 + 1)·√D` to the row 2-norm (the
HEBBIAN step likewise). With every `|w| ≤ 2^24` before a call (checked on every touched bank: `NUMERIC`
otherwise) and at most 128 steps per call (LEARN ≤ 8, CONSOLIDATE ≤ 16 × 8), every entry stays below
`√1024 · (2^24 + 128·65537) < 2^30 < 2^31`. Hence no call can overflow, and a refusal happens before any
write. Restore refuses any stored `|w| > 2^30`. The differential test reaches the bound with the Hebbian
law and checks that core and reference refuse at the same step.

### Determinism, resources, concurrency

* Integer arithmetic only; division rounds half away from zero. The state identity of a fixed script is
  pinned (`ECS_MEM_KNOWN_ANSWER` in `test_ecs_mem.c`) and must be reproduced by gcc and clang, Debug and
  Release, ASan+UBSan and TSan builds.
* One allocation (handle + one 64-byte-aligned arena) at create/restore; no heap operation afterwards
  (`research.ecs_mem_alloc`, link-time wrapping of malloc/calloc/realloc/aligned_alloc/free).
  `allocated_bytes = arena + handle`; allocator overhead and bounded stack staging (≤ ~40 KiB: cues, plans,
  readout) are excluded.
* Hash indexes are exact (full 32-byte comparison); collisions cost probes, never identity. Admission is
  O(1) expected through an intrusive LRU list (the PR #47 O(capacity) scan is gone). A deliberately
  adversarial set of identities can still lengthen probe chains up to O(capacity); `index_probes` reports it.
* Guard: one writer XOR many readers, non-blocking; a conflicting entry is `BUSY`. It has **no fairness**:
  continuously overlapping readers starved a writer completely in the first concurrency run (400/400 writes
  refused). Retry policy is the caller's.

### Defects found while building it (fixed before commit, recorded here)

* The first DELTA step divided the update by D twice (readout normalization and outer product), so each LEARN
  moved the readout by `rate/D` of the error. Found by the suppression test; fixed to `rate·(t − y)·cᵀ`, and the
  overflow bound was re-derived for the corrected law.
* Code derivation produced `words/4` SHA-256 chunks, i.e. **none** for D = 64 and D = 128: those codes were
  uninitialized stack memory. The state identity changed between runs under ASLR; valgrind located it. Fixed
  to `ceil(words/4)` chunks; the known-answer identity now pins D = 128 across processes.

## M5 — The ELPISEM1 v1 envelope

Little-endian, explicit layout; everything that determines future behaviour and nothing else.

| Offset | Field |
|---|---|
| 0 | magic `ELPISEM1` |
| 8 | format version (u32) = 1 |
| 12 | ABI (u32) = 1 |
| 16..55 | dim, banks, slot capacity, codebook capacity, slot policy, codebook policy, law, learn rate, retain, consolidate rate (u32 each) |
| 56..63 | reserved, zero |
| 64 | clock (u64) |
| 72 | consolidation cursor (u32) |
| 76 | resident slots `s` (u32) |
| 80 | codebook entries `e` (u32) |
| 84..127 | reserved, zero |
| 128 | fast banks, then slow banks: `2·B·D·D` int32 |
| … | `s` slot records of 544 bytes, in slot-index order |
| … | slot LRU order: `s` u32 indices, least recent first |
| … | `e` codebook records of 48 bytes (identity, first tick, last tick), in index order |
| … | codebook LRU order: `e` u32 indices |
| end | SHA-256 of every preceding byte (the state identity) |

`byte_budget` is not in the envelope: it constrains a host, not the state. Restore validates everything
before touching the state (no allocation in `restore_into`): magic, version, ABI, configuration (equal to the
handle's), reserved fields, exact size, checksum, every weight bound, every slot record (participant count,
canonical order, zero padding, polarity counts, tick order against the clock, recomputed key), duplicate
slot keys and codebook identities, LRU orders as permutations, cursor range. Only then is the state installed
and the indexes rebuilt. Version 1 is the only version; any other is refused (`CORRUPT`) and **no migration
exists**. The checksum gives integrity, not authenticity: a resealed in-bound edit is accepted (tested and
stated, as for `ELPISGK1`).

Tests (`research.ecs_mem_snapshot`): round trip (restore then snapshot is bitwise identical; identity equals
trailer), exact replay of 30 further steps on original and restored handles (identities and SLOW answers
equal at every step), every header and trailer byte and every 13th body byte flipped (all refused, target
unchanged), truncation and trailing bytes, a torn envelope (prefix of one state over another), 25 resealed
semantic violations, configuration mismatch, foreign lease, a budget one byte short, a short output buffer.

## Typed HACF adapter (`native/include/elpis/ecs_mem_hacf.h`)

`ecs_mem_hacf_observation(edge, assertion, polarity, tick, out)` maps one structure hyperedge
(`elpis_semantic_hyperedge_v1`) and one assertion that names it (`elpis_semantic_assertion_v1`) to one
MF-R0 observation. Structure stays the owner of identity and validation: the adapter calls
`elpis_semantic_hyperedge_validate`, `elpis_semantic_assertion_validate`, recomputes both identities with
structure's functions, and requires the assertion to assert exactly this hyperedge. The map is lossless for
what it accepts (type, payload, canonical participants with incidence role, ordinal and node identity,
provenance, authority, hyperedge identity), and `ecs_mem_hacf_hyperedge(slot, out)` inverts it: the rebuilt
hyperedge is byte-identical and its recomputed identity equals the original (tested).

| Refusal | Why |
|---|---|
| `STRUCTURE` | structure's validators refuse (namespace, reserved fields, authority > 3, …) |
| `IDENTITY` | a stored identity is a claim: it must equal the recomputed one; the assertion must assert this hyperedge |
| `NONCANONICAL` | participants not in structure's canonical order (structure's validator does not check it; its identity function assumes it) |
| `UNSUPPORTED` | assertion flags or hyperedge flag 0x01 (meaning **UNDEFINED** in the repository), query-local participants (`SEMANTIC_PARTICIPANT_FLAG_QUERY`), 0 or > 8 participants, two participants sharing `(role, ordinal)` |

Polarity is supplied by the caller because HACF does not define it; it is recorded as given. Mapping and
observing a record never makes it true, authorized or learned: LEARN is a separate call, and a conflict
(one source asserting, another negating the same hyperedge) is retained with both provenances and refuses
LEARN (`NO_NET_EVIDENCE`).

## M3 — Laboratory results (`research/ecs_memory_mf_r0`, preregistered, frozen, QUAL once)

Protocol: `research/ecs_memory_mf_r0/PREREGISTRATION.md` (committed `35fc484`, before the laboratory existed).
Laboratory, runner and scorer `0204eff`; DEV record (seeds 1–4) `a3713c7`; freeze manifest `8b384e9`
(SHA-256 `3cf46ea5…`); QUAL record (seeds 1001–1008, once) `2744ece` (SHA-256 `5454cbf1…`). Binary
`research/ecs_memory_mf_r0/ecs_mem_lab` SHA-256 `545ced3f…`, gcc 13.3.0 Release, x86_64, Intel Xeon @ 2.10 GHz.
Parameters the preregistration left open were fixed in the laboratory before DEV and are echoed in every
record: SLOW consolidation rate 0.5, D31C α = β = 1 and radius 8, S8 intermediate pool of 64. DEV found no
harness defect; nothing was tuned between DEV and QUAL. Guards: `tests/research/ecs_memory_mf_r0`
(seals, recomputed verdicts, freeze binding, chronology, exact replay of seed 1001 on the current build).
Cross-compiler check (descriptive, after QUAL): a clang 18.1.3 Release build reproduced every integer-arm row
of QUAL seed 1004, work counters included, exactly; the binary32 D31C rows also matched on that seed, which the
contract does not guarantee.

### Verdicts (QUAL, pooled over 8 seeds; chance = 1/64 = 0.016)

| Hypothesis | Verdict | Decisive numbers |
|---|---|---|
| H1 acquisition and state causality | **PASS** | FAST 1.000; NONE 0.016, FIXED 0.014, PERM 0.016 |
| H2 updating AB / ABA / ABAB | **PASS** | FAST 0.996 / 1.000 / 1.000; HEBB 0.457 on ABAB |
| H3 partial-cue completion (agent only) | **PASS** | FAST 0.932 — algebraic completion by fixed geometry, as classified in advance |
| H4 byte-matched retention under interference | **NONPASS** | FAST 0.090 / 0.012 / 0.012 at K = 256 / 1024 / 4096; KEYED 1.000 at every K |
| H5 consolidation of rare recurrent relations | **PASS** | SLOW(RECUR) 0.883; FAST(RECUR) 0.023; SLOW(NORECUR) 0.016 |

### Every arm (QUAL accuracy)

| scenario / condition | FAST | SLOW | HEBB | LOOKUP | KEYED | D31C | FIXED | NONE | PERM |
|---|---|---|---|---|---|---|---|---|---|
| S1 base | 1.000 | 0.016 | 1.000 | 1.000 | 1.000 | 0.811 | 0.014 | 0.016 | 0.016 |
| S2 AB | 0.996 | 0.016 | 0.434 | 1.000 | 1.000 | 0.945 | 0.012 | 0.016 | 0.016 |
| S2 ABA | 1.000 | 0.012 | 1.000 | 1.000 | 1.000 | 0.996 | 0.031 | 0.012 | 0.012 |
| S2 ABAB | 1.000 | 0.012 | 0.457 | 1.000 | 1.000 | 0.984 | 0.023 | 0.012 | 0.012 |
| S3 shared roles | 0.557 | 0.000 | 0.454 | 1.000 | 1.000 | 0.902 | 0.010 | 0.000 | 0.034 |
| S4 simultaneous conflict (either object) | 1.000 | 0.027 | 1.000 | 1.000 | 1.000 | 0.895 | 0.043 | 0.027 | 0.027 |
| S5 UNIQUE agent only | 0.932 | 0.018 | 0.977 | 1.000 | 0.000 | 0.018 | 0.010 | 0.018 | 0.013 |
| S5 UNIQUE recipient only | 0.927 | 0.018 | 0.982 | 1.000 | 0.000 | 0.016 | 0.013 | 0.018 | 0.005 |
| S5 UNIQUE both | 0.948 | 0.018 | 0.935 | 1.000 | 1.000 | 0.932 | 0.013 | 0.018 | 0.010 |
| S5 SHARED agent only (either) | 0.964 | 0.036 | 1.000 | 1.000 | 0.000 | 0.031 | 0.026 | 0.036 | 0.036 |
| S5 SHARED recipient only | 0.966 | 0.018 | 0.992 | 1.000 | 0.000 | 0.013 | 0.013 | 0.018 | 0.013 |
| S5 SHARED both | 0.917 | 0.018 | 0.919 | 1.000 | 1.000 | 0.930 | 0.008 | 0.018 | 0.021 |
| S6 K=0 | 1.000 | 0.012 | 1.000 | 1.000 | 1.000 | 0.973 | 0.027 | 0.012 | 0.105 |
| S6 K=256 | 0.090 | 0.020 | 0.965 | 1.000 | 1.000 | 0.020 | 0.008 | 0.020 | 0.020 |
| S6 K=1024 | 0.012 | 0.020 | 0.461 | 0.000 | 1.000 | 0.008 | 0.031 | 0.020 | 0.020 |
| S6 K=4096 | 0.012 | 0.008 | 0.168 | 0.000 | 1.000 | 0.016 | 0.012 | 0.008 | 0.020 |
| S7 RECUR | 0.023 | 0.883 | 0.031 | 1.000 | 1.000 | 0.016 | 0.031 | 0.008 | 0.008 |
| S7 NORECUR | 0.008 | 0.016 | 0.016 | 0.000 | 1.000 | 0.016 | 0.008 | 0.008 | 0.008 |
| S8 two-hop chain | 1.000 | 0.020 | 1.000 | 1.000 | 1.000 | 0.996 | 0.023 | 0.020 | 0.016 |

Budget: every arm's capacity ≈ 562.8 kB (the FAST handle's full ELPISEM1 state: two bank sets of 4 × 128²
int32, 64 slots, 64 codebook entries). KEYED holds 7,035 exact entries in it; LOOKUP ≈ 1,000 evidence
slots; D31C 68 lattice cells. Mean query latency (descriptive, one shared VM): FAST/SLOW/HEBB ≈ 79 µs
(64 candidate codes derived by SHA-256 plus a 128² readout), D31C ≈ 88 µs, LOOKUP ≈ 2 µs, KEYED ≈ 1 µs.
Work per arm (integer MACs, slot scans, SHA blocks, probes) is in the record.

### What the numbers do and do not show

* **Acquisition is state-borne and causal (H1).** Answers follow the learned banks: empty banks, fixed random
  banks and permuted role addressing all fall to chance. This is learning in the narrow sense of a state
  transition driven by observations; it is one-shot storage of quasi-orthogonal associations.
* **Lookup, algebraic completion and learned state are separated, and the separation matters.** LOOKUP
  (ordinary lookup over resident evidence) is exact wherever its evidence is resident, including partial
  cues and two-hop chains (S8, an algebraic inference composed of two lookups by the caller). FAST completes
  partial cues (H3) because a partial-cue code correlates with the stored bundled cue: a property of the fixed
  encoding, not generalization to unseen entities. No arm generalizes to entities it never observed, and the
  laboratory tested none.
* **Byte-matched exact storage dominates retention (H4 NONPASS).** At equal bytes, KEYED retains every target
  at every interference level; FAST keeps roughly its last D associations per bank and loses targets after
  a few hundred interfering relations. LOOKUP fails once its ~1,000 resident slots overflow (K ≥ 1024).
* **The delta residual trades retention for updating.** The Hebbian ablation retains far better under
  interference (0.965 at K = 256) but cannot update (0.43–0.46 on AB / ABAB, where it splits between the old
  and new object), and it reaches the numerical learning bound under heavy traffic (refusals recorded). The
  delta law updates perfectly and forgets by overwriting. Neither law is "better"; the trade-off is measured.
* **Consolidation C1 helps only what recurs (H5).** Replay of resident evidence into the undecayed slow
  banks retains rare relations that recur (0.883) and not those observed once (0.016). It is still below the
  byte-matched exact stores in the same scenario (LOOKUP and KEYED 1.000), so it is not evidence that
  consolidation beats storage, only that the separately specified law retains recurrent structure that
  the decaying fast state loses.
* **Shared roles cost the associative state (S3).** With entities filling several roles across 48 ternary
  relations, FAST answers 0.557 and HEBB 0.454 against 1.000 for the exact stores. Conflicting simultaneous
  assertions (S4) are answered with one of the two objects by every learning arm; only LOOKUP reports the
  conflict (`conflicts`), and only through its retained evidence.
* **The PR #47 lattice (D31C)** is competitive on full-cue recall and updating at this scale (0.81–1.00),
  cannot answer partial cues by construction (hashed cue), and is overwhelmed by interference as FAST is.

## M4 — Composite ECS integration: what exists and what is blocked

Implemented (`integration/`, `ecs_mem_composite.h`, `research.ecs_mem_composite`):

* `ecs_mem_composite_identity(k1_digest, memory_identity)` — the identity a composite retained state would
  need. It binds both components with domain separation and is bound by nothing.
* Against the real native K1 library: MF-R0 transitions leave `elpis_ecsg_k1_state_digest` unchanged. The
  K1 digest is the only cognitive identity continuity binds, so **today continuity can neither detect, order
  nor restore memory state**; a restart would resume K1 and silently pair it with any memory state.
* Generic FMS residency: the ELPISEM1 envelope survives COLD demotion and verified rematerialization as an
  opaque FMS object (FMS interprets nothing), and restores the identical state. This is placement, not
  checkpoint authority.

Blocked, with the exact authority each needs (no code was written against them):

| Seam | Current authority | What integration would require |
|---|---|---|
| retained-state identity | `ELPISGK1` v1, `elpis_ecsg_k1_state_digest`; kernel sources digest-pinned by the mission gate | a new, versioned composite envelope (or K1 envelope v2) qualified under the K1 qualification line, with the mission gate's pins updated by explicit review |
| durable authority | continuity `ELPCONT` v2: one 32-byte K1 digest (`elpis_continuity_commit_cognition`) | a continuity format change (v3, new frozen byte-parity vectors) binding the composite digest, or an explicit decision that memory state is not durable authority |
| durable state | K1 Recovery R0: two slots of exactly one `ELPISGK1` envelope | a checkpoint format carrying the memory envelope, or a separately authorized owner (not created here) |
| managed mutation | RuntimeCore lease and turn over K1 only | RuntimeCore ABI changes to lease, fuel-admit and commit the memory with the K1 turn |
| observation → state map | none (`ARCHITECTURE.md`: no canonical HACF → ECS edge) | a scientifically identified observation map; the HACF adapter here maps structure to a typed record, not to meaning |
| LEARN authority | `LearnAuthority` for K1 only | a defined authority for memory LEARN/CONSOLIDATE; MF-R0 only separates the calls and mirrors the K1 lease |

Continuity integration is **not claimed**.

## M6 — Status

| Question | Answer |
|---|---|
| Artifact class | **experimental research mechanism (RESEARCH_ONLY)**, mechanically tested; not admitted, not canonical, not part of the one ECS |
| Mechanics | contract, refusal atomicity, capacity/eviction, lease, guard, determinism (pinned identity), allocation freedom, differential reference, snapshot faults, HACF adapter, K1/FMS adjacency: all in ctest under gcc/clang Debug/Release, ASan+UBSan and TSan |
| Science | one preregistered DEV → freeze → QUAL in a synthetic regime: H1, H2, H3, H5 PASS; H4 NONPASS (preserved). This is not scientific qualification of a cognitive memory, of semantics, or of generalization |
| Remaining blockers | the six authority seams above; a qualified observation map; LEARN authority; a composite retained-state format; fairness of the guard (writers can starve); read-recency retention is impossible while QUERY is read-only |

### CI note (run 204)

The push run for `8561838` failed one job, Native ASan+UBSan, in `research.hecs_r1_mechanics` (untouched by
this work) with an AddressSanitizer runtime-internal `CHECK failed: sanitizer_thread_registry.cpp:316` inside
a Rust test harness that preloads the ASan runtime. The next head (run 205) passed the same job with the same
H-ECS R1 sources. Per `docs/CI_POLICY.md` rule 3 this is classified NONDETERMINISTIC / ENVIRONMENT-SENSITIVE and
recorded here; it was not re-run to obtain green.

It recurred after the merge: post-merge run 215 (`91516e5`, run id 38451028810) failed the same job in
`research.hecs_r0_mechanics` (SIGSEGV with no sanitizer report, after three tests had passed). The investigation,
classification, repair and its limits are recorded in `docs/CI_INCIDENTS.md` (incident CI-HECS-ASAN-1).

## Non-claims

No semantics, language, meaning or understanding; no learned similarity or generalization to unseen
entities (codes are fixed SHA-256 derivations); no claim that the memory is better than exact storage at
equal bytes (H4 says the opposite); no continual-learning or catastrophic-forgetting solution; no
crash-durable commit, continuity binding or checkpoint authority; no claim beyond the synthetic regime,
budgets and seeds recorded; latency is descriptive; the snapshot checksum is integrity, not authentication.
