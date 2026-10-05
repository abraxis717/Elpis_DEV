# ECS_G native K1 runtime

`SEMANTICS=NONE` · `NO_LANGUAGE_CLAIM`. This document covers what is implemented in
`native/ECS_G/include/elpis/ecsg_k1.h`, `ecsg_k1_fms.h` and `src/elpis/ECS_G/k1.py`. The headers are the normative
ABI.

**Qualification:**

- The law is Retention R3's K1, `OUTCOME_A` (`docs/research/ECS_RETENTION_R3_RESULTS.md`).
- Native qualification v1 is `NOT_QUALIFIED` (`docs/research/ECS_K1_NATIVE_RESULTS.md`, historical and unchanged).
- Native qualification v2 is **QUALIFIED**: all E1–E5, L1, L2, D1_Q and D1_F gates passed on
  the 32 frozen Q worlds and 32 fresh F worlds under `research/ecs_k1_native/PLAN_V2.md`.
  The write-once record is `research/ecs_k1_native/evidence/ecsg-k1-native.v2.qualification.json`
  (digest `fbea9d43…`, clean harness `f720e4a`). Native K1 is canonical under this admitted authority.

The canonical status is in `ELPIS_SYSTEM.json` (ECS_G) and `tests/research/_k1_promotion.py`.

## State

The complete, authoritative cognitive state of one K1 state is `(W, epoch, H, a)`:

| component | shape | meaning |
|---|---|---|
| `W` | `dim x width` binary64 | microscopic state (Runtime R1's W) |
| `epoch` | u64 | learning steps applied |
| `H` | packed upper triangle of `F x F`, `F = |S3(dim)|` (83 at `dim = 6`) | accumulated input statistics |
| `a` | `F` | the S3 anchor of the last consolidation |

Everything else is non-authoritative workspace:

- the K-loop buffers;
- the gradient and the correction;
- the staged consolidation;
- the transaction candidate;
- the admitted rows.

None of it appears in a snapshot, and no operation can expose a state that mixes the old W with the new `(H, a)`,
or the reverse.

## Operations (one native call each)

| operation | law | notes |
|---|---|---|
| QUERY `forward(X)` | `f_W(x) = 1/2 phi(x) . S3(W)` | Reads W only; H and a never enter. Bitwise the Runtime R1 executor's forward map. No allocation. |
| LEARN `learn(X, y, eta, K)` | `W <- G1(W; X, y) - eta J(W)^T u` with `u = 1/2 H (S3(W) - a)` at the pre-step W | K steps in one call: no per-step crossing, no Python loop. With `H = 0` the step is bitwise Runtime R1 G1. Success commits W and `epoch + K`; any refusal commits nothing. |
| CONSOLIDATE `consolidate(X)` | `H <- H + (1/n) sum phi(x) phi(x)^T`, `a <- S3(W)` | Inputs only; no target enters. Closed form. Commits H and a together, and the provenance becomes COMPLETE. |
| RESET `reset()` | `H <- 0`, `a <- 0` | W and epoch are kept. Provenance RESET. |

**Transactions** (`txn_begin`, `txn_learn`, `txn_consolidate`, `txn_forward`, `txn_epoch`, `txn_commit`,
`txn_abort`):

- `txn_begin` stages a candidate copy of the complete state.
- `txn_commit` installs W, epoch, H and a in one copy and advances the generation.
- A direct transition after begin replaces the source, and the candidate is then refused as `STALE`.

**Refusal contract.** The status decides, and native, the FMS adapter and Python implement it identically:

| status | fate of the transaction |
|---|---|
| `INVALID`, `CAPACITY`, `BUSY` | Refused before the candidate is touched. The transaction stays open and unchanged; retry is allowed. |
| `STALE` | Discarded. |
| `NONFINITE` from `txn_learn` / `txn_consolidate` | Discarded (non-finite input or arithmetic). |
| `NONFINITE` from `txn_forward` | Read-only; nothing is discarded. |

- `txn_abort` with nothing open is OK.
- `txn_abort` with a wrong token is `INVALID` and changes nothing.
- No refusal changes the authoritative state.

## Retained-state envelope `ELPISGK1` v1

All integers and binary64 values are little-endian, with an explicit layout:

| offset | field |
|---|---|
| 0 | magic `ELPISGK1` |
| 8 | version 1 (u32) |
| 12 | mechanism 1 = K1 (u32) |
| 16 | dim (u64) |
| 24 | width (u64) |
| 32 | F (u64) |
| 40 | epoch (u64) |
| 48 | provenance (u64) |
| 56 | reserved = 0 (u64) |
| 64 | W, then H packed upper row-major, then a |
| end | SHA-256 of every preceding byte |

`restore` refuses:

- a wrong magic, version, mechanism, F, provenance or reserved field;
- a size that is not exact (truncated or trailing bytes);
- a checksum mismatch;
- any non-finite value.

Serialized dimensions are validated as u64 before any narrowing to `size_t`. A shape beyond the declared byte budgets
is refused with `CAPACITY` before anything is allocated.

The checksum gives integrity, not authenticity: an envelope edited and resealed with a recomputed SHA-256 still
passes every semantic check, or is refused by one.

The format is deterministic and portable. It assumes IEEE-754 binary64, and byte order is explicit. Restore followed
by snapshot is bitwise identical.

## Imports, reset and provenance

| provenance | how a state gets it |
|---|---|
| `COMPLETE` | create; restore of a COMPLETE envelope; every successful consolidation (direct or committed) |
| `RESET` | reset (repeatable) |
| `UNCONSOLIDATED_IMPORT` | `import_w_only` of a Runtime R1 `ELPISG01` W-only snapshot: H = 0, a = 0, never a retained state |

LEARN never changes provenance.

## Bounds and memory

- Image (header, W, H and a): at most `ELPIS_ECSG_K1_MAX_IMAGE_BYTES` = 64 MiB.
- Workspace: at most `ELPIS_ECSG_K1_MAX_WORKSPACE_BYTES` = 256 MiB.
- `dim` is at most 64 (index tables). In practice the byte budgets bind first: `dim = 64` alone would need about
  9 GB for H.
- Every product and sum is checked before any allocation or `memset`; an oversized request returns `CAPACITY`. It
  never depends on OS overcommit.
- `create` and `restore` allocate the object, one 64-byte-aligned arena and the image. `reserve` is the only later
  allocation.
- No query, learn, consolidation, transaction, reset or snapshot allocates (link-time allocator interposition:
  `test_ecsg_k1_alloc`, `test_ecsg_k1_fms_alloc`).

## Concurrency

- SINGLE_WRITER per state: every operation, including queries, enters one guard, and a concurrent entry is refused
  `BUSY`. Queries are therefore not concurrent on the same state.
- The getters `epoch`, `generation`, `provenance_of` and `max_rows` never enter the guard. They are atomic loads of
  values the writer publishes after each committed transition. Each value is current as of some committed
  transition, but together they are not a joint snapshot.
- `dim` and `width` are immutable.
- Distinct states share nothing.
- TSan covers concurrent same-state entry, getters and stats under a writer, and independent FMS states.

## FMS residency (`libelpis_ecsg_k1_fms`)

One generic FMS object per state holds the resident image. FMS never interprets it, and FMS code names no K1
concept (boundary gate `fms_genericity`).

**Warm path:** pin the object WARM, run the native K1 operation directly over the resident bytes, unpin. Queries,
copies and snapshots pin READ; writes pin WRITE, which marks the object dirty and invalidates the cold replica.

- No executor or state is rebuilt per operation.
- No (de)serialization happens per operation.
- On a WARM object nothing is allocated.
- A transaction holds its WRITE pin from begin to commit or abort.

**COLD to WARM** is FMS materialization, the only allocating path:

- a failed cold read leaves the object FAILED, and it is never soft-recovered;
- a corrupted cold replica is refused by its digest.

**Failed release:** if FMS release fails, the adapter keeps the pin FMS still holds. That pin stays consistent and
is retried by the next operation and by close. A held READ pin never serves a write. A committed result is never
reported as refused (`test_ecsg_k1_fms_faults`).

## Python control plane (`elpis.ECS_G.k1`)

`K1State`, `K1Transaction`, `K1FMSRuntime`. Each method admits its arguments, makes one native call whatever K,
and packages the result:

- it does no cognitive mathematics;
- it has no loop on a hot path;
- it uses no NumPy.

`tests/boundary/test_k1_runtime.py` checks this statically, and `tests/ECS_G/test_k1_runtime.py` counts native
crossings.

## Nonclaims

- No claim about language, meaning or general cognition.
- No claim beyond the frozen synthetic regime of Retention R3 (`dim = 6`, `width = 36`, the aliasing-conflict task
  family).
- The checksum is not authentication.
- Long-horizon W coordinates of numerically different implementations of the same law are not claimed identical.
  On some R3 trajectories, rounding-scale differences are amplified by the dynamics itself (K1N-v1). Qualification
  is decided on decision-bearing quantities and same-host exact invariants (K1N-v2).
- Latency numbers in `docs/performance/ECS_K1_RUNTIME.md` are descriptive.
