# Continuity: minimal durable runtime authority (`elpis.continuity`)

Continuity records the current durable lineage and authority that the runtime
needs for safe restart. It is not an ECS, not context memory, not an event
database, not topology, not a message bus and not an audit ledger. It retains
no history: it holds one fixed-size current-authority record.

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

## K1 identity rule

The K1 **retained-state identity** is `state_digest()`: the SHA-256 trailer of
the portable `ELPISGK1` envelope, binding `W`, epoch, `H` and `a`. It survives
snapshot and restore exactly.

The K1 `generation` is process-local transaction-staleness metadata. It starts
at 0 on every create or restore and is not part of the envelope. Continuity
therefore never records it as a restart coordinate. Turn-local diagnostics
(epochs, generations, token and readout digests) stay on the returned turn
result and are not durable continuity.

## Persisted layout

A continuity directory contains exactly two files:

    continuity.a
    continuity.b

Each holds one record of exactly 136 bytes:

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `ELPCONT\x01` |
| 8 | 2 | format version (1) |
| 10 | 6 | reserved, zero |
| 16 | 8 | generation (monotonic, >= 1) |
| 24 | 1 | cognition state: 0 unanchored, 1 anchored |
| 25 | 7 | reserved, zero |
| 32 | 32 | K1 retained-state digest (zero when unanchored) |
| 64 | 8 | evolution revision |
| 72 | 32 | evolution head: last admitted path-transition receipt digest (zero at revision 0) |
| 104 | 32 | SHA-256 over `elpis.continuity.register.v1\0` and bytes 0..103 |

A record that is all zero bytes is an empty slot, permitted only as the
never-written second slot. The current authority is the valid record with the
higher generation. Two valid records with the same generation, or no valid
record, fail closed with `CONTINUITY_CORRUPT`. The maximum durable size is
therefore 272 bytes, whatever the runtime lifetime.

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

Restart reads exactly the two 136-byte slots: constant work and constant
memory, independent of how many turns or transitions were ever published.

## K1 restart law

```
first managed K1 lineage           -> explicit anchor (no implicit anchor, ever)
successful K1 turn A -> B          -> publish expected identity B
restart, durable says A, K1 is B   -> CONTINUITY_STATE_MISMATCH before any K1 mutation
restart, durable says B, K1 is B   -> resume
corrupt / ambiguous register       -> CONTINUITY_CORRUPT
```

A K1 commit is never rolled back because publication failed. The runtime
fail-stops with the publication code, and restart resolves through the law
above. No missing turn is synthesized. One open `Runtime` owns one K1 lineage
handle.

## Evolution binding

`EvolutionAuthority(revision, head)` is the current head of admitted
evolution path transitions. Its digest is a domain-separated SHA-256 over the
revision and head.

An `elpis.evolution-path-assertion.v1` assertion binds that digest and
revision. The gate refuses any assertion whose authority is not the current
one. After an admitted attempt the runtime publishes revision + 1 with the
transition receipt's digest as the new head. An assertion built against
revision N is therefore rejected once revision N + 1 exists, and no history
projection is involved.

`elpis.evolution-path-assertion.v0` (history-projection fields) is a retired
persisted schema. Its identity is still computable, but the gate refuses it
with `ASSERTION_SCHEMA_RETIRED`.

## Legacy storage

A directory holding the retired receipt-history layout (`MANIFEST`,
`g*.ckpt`, `g*.seg`, `events.log`, `checkpoint.bin`, `LOCK`) is refused with
`CONTINUITY_LEGACY_STORAGE`. It is never read, replayed or converted by the
live code. New installations never create it. The one-shot conversion is a
documented offline step: run the retired runtime at the last commit that
contained it, read its cognition tip, and start a continuity store with an
explicit anchor of that K1 state. The evolution authority of a migrated store
starts at revision 0, because no `v1` assertion predates continuity.

## Hot path

The managed canonical turn is: codec -> one native K1 transaction (experience
schedule and readout) -> decode -> native commit -> one continuity
publication. `tests/integration/test_runtime_hot_path.py` checks dynamically,
on every run, that continuity adds no native crossing to the canonical turn
and that its filesystem work per turn is exactly one 136-byte `pwrite` and
one `fdatasync`. `tests/boundary/test_one_ecs.py` checks statically that the
turn reaches no receipt, history, event, scheduler, projection or compaction
machinery.

One measurement of a warm managed turn, before (`e05b33a`, receipt history)
and after (this design), on the same host (Linux VM, ext4, Release build;
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
