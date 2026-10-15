# Continuity: minimal durable runtime authority (`native/continuity`, `elpis.continuity`)

Continuity records the current durable lineage and authority that the runtime
needs for safe restart. It is not an ECS, not context memory, not an event
database, not topology, not a message bus and not an audit ledger. It retains
no history: it holds one fixed-size current-authority record.

The authority is implemented in Rust (`native/continuity`: one crate, the
standard library only) and reached through a stable C ABI
(`include/elpis/continuity.h`, ABI v1). Every law below — record format,
digests, slot selection, locking, publication and the evolution transitions —
is the Rust code's. `elpis.continuity` is a thin Python adapter over the ABI
that owns no state, format, transition law or digest. The Rust implementation
reproduces, byte for byte, vectors frozen from the qualified Python
implementation it replaced (`native/continuity/tests/fixtures`).

There is one ECS (`elpis.ECS`, `native/ECS`): the cognitive/dynamical
substrate that owns `W`, epoch, `H`, `a`, recurrence, learning, consolidation,
readout and K1 transactions. Continuity only remembers which retained ECS
state the runtime is entitled to resume from.

## Consumer census (red-team of the retired event-kernel history)

This census decided what survives. Every row asks:

* What decision changes because this capability exists?
* What incorrect state becomes possible if it is removed?
* Must it be synchronous? Must it survive restart? Must it retain historical
  events?

The burden of proof was on retention. A standalone test suite, documentation,
or hypothetical future use did not count as a consumer.

| Capability | Live caller(s) before | Decision it changed | Incorrect state if removed | Sync | Restart | Historical events | Disposition |
|---|---|---|---|---|---|---|---|
| K1 lineage: explicit anchor, expected post-turn K1 identity | `Runtime.anchor_cognition`, `_reconcile_cognition_substrate`, `run_turn` | Whether a supplied K1 state may be mutated by the next managed turn | Resuming from a K1 state the runtime never committed (silent lineage fork); a second implicit anchor | yes (after K1 commit, before return) | yes | no: only the current expected identity | **KEEP AS CONTINUITY**: one 32-byte retained-state digest plus an anchored flag |
| Evolution stale-authority binding (projection digest, history head, final root) | `Runtime.evolve` -> `EvolutionPathGate.reject_reason` | Whether an assertion may execute an attempt | Replaying one assertion more than once, or executing an assertion built before another admitted transition | yes | yes | no: only the current head of admitted transitions | **KEEP AS CONTINUITY**: the evolution authority `(revision, last admitted path-transition receipt digest)`. The gate binds it through a new assertion schema (`elpis.evolution-path-assertion.v1`). The event-history projection is deleted. |
| Ingress proposal receipts | `Runtime.run_ingress`, `admit_context` | None (nothing read them except the projection digest above) | None | - | - | - | **DELETE** (audit only) |
| Retrieval-bundle receipts | `Runtime.admit_retrieval` | None | None | - | - | - | **DELETE** (audit only) |
| Canonical publication receipts | `Runtime.publish_canonical` | None. The pipeline ledger already owns durable publication and idempotent replay. | None | - | - | - | **DELETE**; publication authority stays with the pipeline ledger |
| Context-admission receipts | `Runtime.admit_context` | None | None | - | - | - | **DELETE** (audit only) |
| Evolution transition receipts as history entries | `Runtime.evolve` | Only by moving the head (row 2) | Covered by row 2 | - | - | - | **MOVE TO OWNER**: the gate returns the receipt; continuity keeps only its digest as the head |
| `ReceiptHistory` / `ReceiptRecord` / `RecordedReceipt`, recorder roles, history entity | runtime only | None beyond rows 1-2 | None | - | - | - | **DELETE** |
| Entity registry, lifecycle, mailboxes, sender-bound ports, watermarks, scheduler (v1/v2), event kinds | the receipt history only | None outside the deleted history | None | - | - | - | **DELETE** |
| Event log, replay, state roots, checkpoint markers, compaction checkpoints, generations, segments, retention floors, duplicate index | the receipt history only | None outside the deleted history | None | - | - | - | **DELETE**: a fixed-size register has nothing to compact or replay |
| Context projections over event history (`ContextProjection`, `ProjectionRequest`, retained bindings) | `Runtime.history_projection`, `evolve`, gate | Only row 2 | Covered by row 2 | - | - | - | **DELETE** |
| Topology / topology analysis over event history | none in the runtime; frozen research collision experiment | None | None | - | - | - | **DELETE**. HACF/structure owns structural topology. |
| Structural R0 mutation grammar and sealed authority bytes | no canonical runtime consumer (own tests only; never admitted as transitions) | None | None | - | - | - | **DELETE** from the live package; the beta migration record and git history keep its provenance |
| Native receipt codec, log, runtime session (the retired native history library) | the receipt history only | None | None | - | - | - | **DELETE**: continuity needs no native code |

**Correction (Elpis_DEV, RuntimeCore ABI v4).** The first row's "Sync: yes (after K1 commit, before return)" is
superseded. Continuity still keeps exactly one current expected identity, but it is the last **explicitly
published** one: a managed commit is volatile (it advances the live coordinate only), and the identity is published
by RuntimeCore's durability lane when a caller requests durable publication, recoverable (the state's bytes in the
checkpoint store first) or identity-only (docs/RUNTIME_CORE.md, "Live and durable authority"). Continuity no longer decides when the runtime may think. Nothing else in this census
changes; the retired event-history kernel stays retired, and continuity is not the contextual lane's memory.

## K1 identity rule

The K1 **retained-state identity** is `state_digest()`: the SHA-256 trailer of
the portable `ELPISGK1` envelope, binding `W`, epoch, `H` and `a`. It survives
snapshot and restore exactly.

The K1 `generation` is process-local transaction-staleness metadata. It starts
at 0 on every create or restore and is not part of the envelope. Continuity
therefore never records it as a restart coordinate. RuntimeCore uses it, in memory
only, as the live coordinate of the state it holds under its managed lease (live
authority); continuity holds the durable authority. Turn-local diagnostics
(epochs, generations, token and readout digests) stay on the returned turn
result and are not durable continuity.

## Persisted layout

A continuity directory contains exactly two files:

    continuity.a
    continuity.b

Each holds one record of exactly 176 bytes:

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `ELPCONT\x02` |
| 8 | 2 | format version (2) |
| 10 | 6 | reserved, zero |
| 16 | 8 | generation (monotonic, >= 1) |
| 24 | 1 | cognition state: 0 unanchored, 1 anchored |
| 25 | 7 | reserved, zero |
| 32 | 32 | K1 retained-state digest (zero when unanchored) |
| 64 | 8 | evolution revision |
| 72 | 32 | evolution head: last admitted path-transition receipt digest (zero at revision 0) |
| 104 | 1 | evolution state: 0 idle, 1 pending |
| 105 | 7 | reserved, zero |
| 112 | 32 | reserved assertion digest (zero bytes when idle) |
| 144 | 32 | SHA-256 over `elpis.continuity.register.v2\0` and bytes 0..143 |

A record that is all zero bytes is an empty slot, permitted only as the
never-written second slot. The current authority is the valid record with the
higher generation. Two valid records with the same generation, or no valid
record, fail closed with `CONTINUITY_CORRUPT`. The maximum durable size is
therefore 352 bytes, whatever the runtime lifetime.

### Record v3: composite lineages

A lineage may be anchored at an ECS composite retained-state identity (`ELPISEC1`,
docs/ECS_COMPOSITE_STATE.md) instead of a K1 identity. Such an authority is written as **record v3**: the v2
layout with magic `ELPCONT\x03`, format version 3, byte 25 = 1 (state format: composite), the anchored flag
required, and the checksum domain `elpis.continuity.register.v3\0`. Every K1 or unanchored authority is still
written as v2, byte for byte (the frozen v2 parity vectors are unchanged), so the digest at 32 keeps its v2
meaning wherever v2 is written. A v2-only reader refuses a v3 record (`CONTINUITY_CORRUPT`): it fails closed
rather than misreading a composite identity as a K1 identity. The C snapshot reports the format in
`state_format` (formerly the first reserved byte); the Python continuity adapter binds K1 lineages only and
refuses a composite one (`CONTINUITY_STATE_MISMATCH`).

A lineage keeps its format. `commit_cognition_as(format, ...)` by a committer of another format is
`CONTINUITY_STATE_MISMATCH`; the only transition between formats is the explicit legacy import
`import_cognition(k1, composite)`: refused unless the lineage is K1-format and anchored at exactly `k1`, and
published like any transition (one record, one `fdatasync`; a crash leaves the K1 or the composite authority).
Continuity still holds one 32-byte identity and the evolution authority; no memory content, observation or
history enters it.

## Publication (crash law)

Publication writes the complete next record (generation + 1) into the slot
that does **not** hold the current authority, then calls `fdatasync` on it. The
slot files never change size, so there is no rename, no directory sync and no
log.

* A crash or failure before the write is complete leaves that slot torn or
  stale. Reopen selects the untouched slot: the previous authority.
* A crash after the write reaches disk makes the new record valid with the
  higher generation. Reopen selects it: the next authority.
* A failed write before `fdatasync` returns `CONTINUITY_PUBLICATION_REFUSED`.
  The previous authority is intact and in force.
* A failed `fdatasync` returns `CONTINUITY_PUBLICATION_UNCERTAIN` and closes
  the store. Reopen resolves to exactly one of the two complete records.

Initialization (empty directory) writes `continuity.b.tmp` (empty slot) and
`continuity.a.tmp` (generation 1, unanchored, evolution revision 0) and
fsyncs both. It then renames b, then a, and syncs the directory. The
existence of `continuity.a` is the initialization point: leftover `*.tmp`
files without it are discarded and initialization restarts.

Restart reads exactly the two 176-byte slots: constant work and constant
memory, independent of how many turns or transitions were ever published.

## K1 restart law

```
first managed K1 lineage            -> explicit anchor (no implicit anchor, ever)
successful K1 turn A -> B           -> volatile: the live state is B, continuity still says A
explicit durable publication of B   -> publish expected identity B (durability lane), then ACKNOWLEDGED
restart, durable says A, K1 is B    -> CONTINUITY_STATE_MISMATCH before any K1 mutation
restart, durable says B, K1 is B    -> resume
corrupt / ambiguous register        -> CONTINUITY_CORRUPT
```

A K1 commit is never rolled back because a publication failed. A refused publication resolves its request
`FAILED` (nothing durable changed); an uncertain one resolves it `UNCERTAIN` and faults the durability lane until
reopen; the HOT lane keeps its live state either way. Restart resolves through the law above: a crash loses the
volatile work since the last completed publication, never more, and no missing turn is synthesized. A publication
whose record completed may be the authority even though its caller never received the acknowledgment (the process
died first): a missing acknowledgment proves nothing about durability, and restart reads only what continuity
names. One open `Runtime`
owns one K1 lineage handle; the binding, the live coordinate and the fail-stop are RuntimeCore's
(docs/RUNTIME_CORE.md).

Continuity holds identities, never K1 state. Resuming a lost in-memory state
from its complete envelope is K1 Recovery R0's job (docs/K1_RECOVERY_R0.md), a
distinct, operator-provisioned owner of two fixed slots whose bytes never
authorize themselves: a slot is resumable only when its identity is the one
continuity holds.

## Evolution binding

`EvolutionAuthority(revision, head, pending_assertion)` holds exactly one
current authority. `pending_assertion=None` means EVOLUTION_IDLE; a 64-hex
assertion digest means EVOLUTION_PENDING. Revision and head remain unchanged
during reservation. No attempt payload, result, receipt object or collection
is retained.

Its digest uses `elpis.continuity.evolution-authority.v2\0`, followed by
the big-endian 8-byte revision, 32-byte head, 1-byte state and 32-byte assertion
identity (zeros when idle). State distinguishes idle from a pending zero digest.

The runtime performs these steps:

1. Validate the assertion against the current idle authority and episode.
   The assertion's predecessor claim must equal the trusted authority head.
2. `reserve_evolution_assertion(expected_idle, assertion_digest)` publishes
   pending authority durably before any `advance()` call.
3. Execute the attempt once. Exceptions, cancellation and invalid result
   contracts leave pending authority; they never establish that retry is safe.
4. `commit_evolution_transition(expected_pending, receipt_digest)` compares
   the complete pending authority, then publishes idle revision + 1 with the
   result receipt digest as head.

Both methods return a `ContinuitySnapshot`. Exhausted revision/generation
space is refused before reservation. The public store surface is `open`,
`close`, `snapshot`, `anchor_cognition`, `commit_cognition_transition`,
`reserve_evolution_assertion`, `commit_evolution_transition`.

### Evolution failure law

| Failure point | Executions | Authority after reopen |
|---|---:|---|
| Reservation write refused or torn | 0 | Previous idle |
| Reservation durability uncertain | 0 | Previous idle or exact pending reservation |
| Reservation durable; process dies before execution | 0 | Pending; no automatic retry |
| Attempt runs; final publication refused or torn | 1 | Pending |
| Attempt runs; final durability uncertain | 1 | Pending or next idle |
| Attempt runs; final publication succeeds | 1 | Next idle revision/head |

An uncertain final publication can have reached durable storage despite a
failed sync acknowledgment. Reopening may therefore select the next idle
record. It must never restore the *previous idle* authority after a durable
reservation. Pending refuses all evolution with `CONTINUITY_EVOLUTION_PENDING`;
next idle rejects the old revision with `STALE_EVOLUTION_AUTHORITY`.
The runtime fail-stops after publication errors. Repeated restart never clears
a pending identity. K1 transitions preserve the evolution fields unchanged.

### Explicit reconciliation

Read `store.snapshot().evolution` to inspect pending authority.
`commit_evolution_transition(expected_pending, receipt_digest)` is also the
minimal typed reconciliation primitive: a trusted caller must independently
establish the completed result and supply its receipt digest. It checks the
entire pending authority, advances exactly once and never invokes `advance`.
The caller is responsible for verifying that the result belongs to the
reserved assertion and predecessor; continuity stores only fixed-size identities.

There is no cancellation, reset-to-idle, automatic retry or automatic external
side-effect reconciliation. If the result cannot be established, authority
remains pending. After a runtime fault, close/reopen before explicit
reconciliation. Never erase/reinitialize the directory to bypass pending.

### Path predecessor and persisted schemas

`EvolutionAuthorityBinding(revision, digest, head)` comes from runtime's
continuity authority. The gate refuses a conflicting assertion predecessor
with `PATH_PREDECESSOR_MISMATCH` before reservation or execution, and derives
`PathTransitionReceipt.previous_path_receipt_digest` from that trusted head.

The existing assertion v1 and receipt v0 payloads, digest domains and field
meanings are unchanged. Admission is stricter; unsafe predecessor claims are
refused rather than reinterpreted. Standalone gate callers must provide a
trusted binding and manage reservation themselves; durable at-most-once
execution is the composed Runtime contract.

Assertion v0 (history-projection fields) remains retired. Its identity is
computable, but the gate refuses it with `ASSERTION_SCHEMA_RETIRED`.

## Legacy storage

A directory holding the retired receipt-history layout (`MANIFEST`,
`g*.ckpt`, `g*.seg`, `events.log`, `checkpoint.bin`, `LOCK`) is refused with
`CONTINUITY_LEGACY_STORAGE`. It is never read, replayed or converted by the
live code. New installations never create it. The one-shot conversion is a
documented offline step: run the retired runtime at the last commit that
contained it, read its cognition tip, and start a continuity store with an
explicit anchor of that K1 state. The evolution authority of a migrated store
starts at revision 0, because no `v1` assertion predates continuity.

The previous 136-byte register format v1 is refused with
`CONTINUITY_CORRUPT` (slot size); its bytes are never reinterpreted, resized or
automatically upgraded. A v1 store cannot prove whether a failed unpublished
attempt ran. Migrating existing authority requires explicit offline
reconciliation of external effects; this repair provides no automatic
migration or license to reset authority.

## Hot path

The managed canonical turn is: codec -> RuntimeCore (one native K1 transaction begun at the live coordinate:
experience schedule and readout) -> decode -> RuntimeCore (native commit: volatile). It touches continuity not at
all. The store is embedded in RuntimeCore (native/runtime, docs/RUNTIME_CORE.md) and owned by its durability lane;
`tests/integration/test_runtime_hot_path.py` checks dynamically, on every run, that Python makes no K1 call and
three crossings into RuntimeCore, that RuntimeCore's K1 crossings are the transaction's own (begin, one schedule,
commit) with no publication, that Python performs no file I/O, and that the store performs no I/O at all during the
turn (testing-library I/O counters); `native/runtime/tests/test_runtime_hot_io.c` checks the same at the
system-call boundary. Each explicit durable publication writes one 176-byte record and syncs it once (plus the
checkpoint slot when a store is attached), on the durability lane. `tests/boundary/test_one_ecs.py` checks
statically that the turn, in Python and in RuntimeCore, reaches no receipt, history, event, scheduler, projection
or compaction machinery, and no store, lane, sync or digest.

**Superseded (through RuntimeCore ABI v3):** every managed turn published one 176-byte record and called
`fdatasync` before returning. The measurements below are historical records of that design and its predecessor;
they are not rewritten.

One measurement of a warm managed turn, before (`e05b33a`, receipt history)
and after (the original 136-byte format-v1 design), on the same host (Linux VM, ext4, Release build;
fixture map with one experience and one step; strace over 500 turns, latency
over 1000 turns). Latency figures are single-host observations, not a
performance claim.

| Per warm `Runtime.run_turn` | Before | After |
|---|---|---|
| native K1 crossings | 4 (`max_rows`, `txn_begin`, `txn_run_schedule`, `txn_commit_identity`) | 4 (same) |
| other native crossings | 1 (the retired native history library's record call) | 0 |
| filesystem syscalls | 4 `write`, 2 `fsync`, 2 `fstat` | 1 `pwrite64`, 1 `fdatasync` |
| bytes written | 3731 (mean) | 136 |
| durable store after 503 turns | 1,890,477 bytes, growing to the policy bound | 272 bytes, constant |
| latency p50 / p90 / p99 (us) | 1749 / 2392 / 3487 | 616 / 794 / 1157 |
| bare `run_turn` p50 (us), same process | 445 | 361 |
