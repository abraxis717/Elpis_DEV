# Elpis architecture

`ELPIS_SYSTEM.json` is the authority for which subsystems exist, what they
depend on and what they may mutate. This document explains the design those
facts describe.

## ECS: identity, history and replay (`elpis.ecs`)

The ECS is the system's memory of *what happened*. It is a deterministic,
same-process kernel.

### Entities and messaging

An entity is founded from a canonical founding record. Its `entity_id` is
domain-separated content identity: an identifier, not a credential. The
lifecycle is `FOUNDED -> ACTIVE <-> DORMANT -> TERMINATED`. Terminated
identities are never recycled.

Entity-facing messaging goes through an `EntityPort` that the kernel binds to
one entity and one live kernel epoch. `EntityPort.propose(receiver, payload)`
has no sender parameter: sender identity, per-sender sequence and commit clock
are assigned by the kernel. This is an API-level same-process guarantee, not
isolation from hostile code that already holds the kernel object.

Mailboxes are bounded FIFOs whose capacity is bound into the genesis
authority. Per-sender sequences are monotonic and replay-checked. No
transport-level at-least-once or exactly-once claim is made.

### One authoritative history

The event log is the single authoritative mutable history. Entity, mailbox and
watermark state are projections of committed events. A mutation is linearized
under the kernel lock:

```text
read projection -> clone post-state -> validate/apply -> advance clock
-> construct canonical event -> compute after-root -> append framed event
-> durability boundary (fsync) -> install projection -> return
```

The log is length-framed (8-byte big-endian length + canonical UTF-8 JSON).
Recovery distinguishes a complete valid event, an incomplete trailing crash
frame (recoverably truncated) and complete-but-corrupt history (fails closed).
When a durable outcome cannot be proven, the live kernel is invalidated and
must be reopened. The append is a *recoverable framed append*, not a
syscall-level atomic transaction.

### State roots and replay

The state root (`ecs.state_root.v3`) binds genesis, history digest, logical
clock, next founding index, mailbox capacity, the full entity registry,
mailbox contents, sender watermarks and scheduler inputs. The design rule:
anything that can change a future accepted transition is root-bound.

```text
genesis/configuration authority + ordered committed event bytes
    = exactly one reconstructed kernel state
```

Replay re-applies every transition's preconditions and effects. It does not
trust a stored after-root because its syntax is valid. Checkpoints are local
rollback-floor markers: full replay remains authoritative.

### Projections

* **Topology** (`topology`, `topology_analysis`): interaction-derived,
  read-only structural views of committed history.
* **Context projection** (`projection`): bounded, canonical selection of
  committed events for a consumer. `project_history` independently replays
  the history it is given. `project_kernel_history` reads a live kernel only
  through public surfaces. Projections have no mutation, model, network or
  execution authority, and they are the ECS side of the ECS → inference
  context path.

### Structural R0

`elpis.ecs.structural` is the frozen Structural R0 mutation grammar: a
participation mask over the exact nonzero columns of a frozen 6 × N binary64
sidecar, with ABSTAIN / DISABLE_COLUMN / RESTORE_COLUMN as the only
operations. Its sealed authority bytes are embedded and identified by logical
anchors. The grammar is executable and verified, but **the kernel does not yet
admit Structural R0 mutations as transitions**. That bridge is an incomplete
interface.

### Protocol identifiers

Persisted identifiers keep their historical spelling because replay and data
identity depend on them: `ecs.state_root.v3`, `ecs.event.v1`,
`ecs.genesis.v1`, the genesis protocol revision `ecs.m1a.integration.v3`, both
scheduler protocol strings, and every `elpis.ecs.r0.*` /
`elpis.ecs.structural_r0.*` domain. Beta *phase* names such as M1A were
removed from the code and docs everywhere else.
