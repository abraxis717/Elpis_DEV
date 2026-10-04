# ECS_G Mutable FMS R0

`MECHANICS_ONLY` · `NO_SCIENTIFIC_CLAIM` · `NO_NEW_LEARNING_LAW`

Mutable FMS R0 is an additive residency adapter for the already-qualified
ECS_G Runtime R1 executor. It changes where an idle logical ECS state is
materialized; it does not change what ECS_G computes.

## Authority

The learned state remains the existing portable ECS_G snapshot:

    ELPISG01 header | d | N | epoch | W (little-endian binary64)

`W` and the epoch retain their existing semantics. `S3` remains derived.
Runtime generation remains process-local concurrency bookkeeping and is not
part of the snapshot or persistent identity.

An FMS object owns the authoritative snapshot bytes between operations. The
adapter does not introduce another cognitive serialization.

## Execution

A direct operation is:

    FMS snapshot object
        -> READ/WRITE lease in WARM
        -> restore the qualified Runtime R1 Executor
        -> query or compute a complete candidate
        -> for successful mutation, serialize the complete new snapshot
        -> publish those bytes while the WRITE lease is held
        -> destroy the transient Executor
        -> release the lease

A transaction retains one restored Executor and one WRITE lease from begin
through commit or abort. Candidate state remains in the existing executor's
transaction buffer. Commit publishes one complete snapshot; abort, refusal or
staleness publishes nothing.

The adapter uses only the public Runtime R1 executor ABI. The measured Runtime
R1 executor source, Python binding and canonical cognitive turn are unchanged.

## Residency

FMS remains generic. It owns placement, budgets, leases, verified mutable cold
replicas and eviction. It does not know G1, S3, learning, tasks or cognitive
semantics.

Logical state identity is independent of the transient FMS slot:

    SHA256("elpis.ecsg.logical.v1\\0" || namespace_key[32] || LE64(d) || LE64(N))

The FMS handle is process-local residency identity only.

Multiple logical ECS states can therefore exceed WARM residency. An idle state
may be demoted to COLD and later rematerialized. A leased state cannot be
moved or unregistered.

## Atomicity

Within one live process:

    failed query/learn/transaction -> authoritative snapshot unchanged
    successful query               -> authoritative snapshot unchanged
    successful learn               -> one complete replacement snapshot
    successful transaction commit  -> one complete replacement snapshot

Publication is a bounded in-memory copy while an exclusive state guard and FMS
WRITE lease are held. R0 does **not** claim crash-durable atomic publication if
the process dies during that copy.

## Concurrency

One logical state is SINGLE_WRITER. Concurrent entry on that state returns
`BUSY`. Independent logical states may proceed independently. An open
transaction pins its FMS object and retains its executor. A direct committed
learn on that same state advances the logical generation and makes the older
transaction stale, matching Runtime R1 semantics.

## Python boundary

`elpis.ECS_G.residency.FMSRuntime` is a control-plane factory. It returns an
actual existing `Executor` object whose ABI table targets the native residency
adapter. `CognitiveCore` and the canonical runtime therefore use their existing
Executor interface unchanged. Python does not execute rows, entities or
recurrent steps.

`elpis.substrate.residency.Context` constructs the generic POSIX FMS context and
transfers ownership to the native adapter.

## Accounting

The adapter reports separately:

- logical state bytes;
- currently resident authoritative bytes;
- active transient/transaction executor workspace;
- WARM/HOT/COLD and physical-domain FMS bytes;
- pins and leases;
- promotions/demotions and cold reads/writes;
- materialization, query, learn, commit and demotion timing.

The existing portable snapshot header is included in logical/resident bytes.
Transient executor workspace is not counted as FMS authoritative residency.

## Performance scope

R0 deliberately restores a qualified executor for each direct operation. This
preserves the closed Runtime R1 implementation and evidence rather than adding
a borrowed-pointer mode to that executor. Warm and cold characterization must
therefore separate materialization/restore overhead from ECS arithmetic.

A later runtime may qualify a lower-copy execution surface, but it must be a
new additive ABI/evidence line rather than retroactively changing Runtime R1.

## Nonclaims

Mutable FMS R0 does not establish:

- a new learning mechanism;
- language or an ECS/DSV semantic map;
- restart-time discovery of logical ECS states;
- a durable catalog of logical identities;
- crash-durable cognitive commits;
- multi-writer semantics;
- device/HOT execution of ECS_G;
- larger or multi-kernel EDEN dynamics;
- HACF-to-ECS cognition;
- ECS_C recording of cognitive transitions.
