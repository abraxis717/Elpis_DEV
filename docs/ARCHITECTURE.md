# Elpis architecture

`ELPIS_SYSTEM.json` is the authority for which subsystems exist, what they
depend on and what they may mutate. This document explains the design those
facts describe.

## Structure: structural memory and representation (`elpis.structure`, `native/structure`)

Structure is the system's memory of *what things are and how they relate*.

### HACF structural store (`native/structure/hacf`)

The hash-addressed cascade fabric is the content-addressed store:

* **identity**: SHA-256 object identity for cascade nodes and graph deltas.
  A digest is integrity/content identity, not a semantic encoding;
* **corpus**: deterministic structural chunking into a SQLite-backed,
  content-addressed corpus with lexical retrieval and namespace/authority
  metadata;
* **vectors**: deterministic embedding profiles and an exact CPU vector index
  whose shards are FMS-resident (substrate residency). It is built with
  `-ffp-contract=off` so scores cannot drift between hosts;
* **hybrid retrieval**: lexical + dense + one-hop context-graph fusion under a
  versioned fusion policy, producing a canonical, digest-bound
  `RetrievalBundle`. Cross-host golden fixtures pin the exact output.

### Semantic core (`native/structure/semantic`)

* a typed **hypergraph** (nodes, hyperedges, incidences) persisted as
  segments and snapshots, with query-local **overlays** that never mutate the
  base graph;
* **embeddings** bound to profiles and snapshots;
* **context-deficit control**: typed requirements, deficit reports and
  retrieval requirements, then re-evaluation with bounded iteration and
  progress guarantees;
* **retrieval materialization**: retrieval epochs bind corpus, index and graph
  and detect drift, so hybrid query plans and bridge receipts never retry
  silently;
* **evidence typing and admission**: typed spans, claim and relation
  candidates, an admission policy, an adjudicator, admission receipts, and
  typed evidence views;
* **bounded semantic views**: seeds, candidate enumeration, bounded selection,
  conflict and provenance closure, and downstream handoffs; metrics are
  non-authoritative;
* **semantic topology IR**: anchors, constellations, addresses, constraints;
* **Grid81 structural packets**: topology → 81-cell capsules and codebook →
  constraint projection → packet → compile receipt → handoff. Also
  **structural observations**: read-only records mapping a topology vertex to
  a cell transition.

The canonical Grid81 digit template is the unique row/column/box
Latin-square codebook (`digit(r,c) = 1 + ((3r + r/3 + c) mod 9)`). It is
structural geometry, not a solver. Its persisted domain keeps the historical
spelling `elpis.semantic.grid81.sudoku_template.v1`.

### Retrieval stage (`elpis.structure.retrieval`)

```text
request -> derive_query (NFKC, bounded, no model inference)
-> hybrid_retrieve via the explicitly loaded native bridge
-> RetrievalBundle -> validate_bundle (schema, query/corpus binding, ranks,
   dedup, frozen text, <= 1 context hop) + check_budget
-> EvidenceEnvelope (ordered references, frozen texts, provenance)
```

The envelope is a **structured observation**. It states retrieval provenance
and never claims truth. The ECS history, the pipeline and inference
(`AddressProposal`) can all consume it.

## Substrate: resources and residency (`elpis.substrate`, `native/substrate`)

The substrate manages bytes, descriptors, residency and native code. It does
not know which model, if any, consumes them.

### FMS residency (native)

FMS ABI v2 separates **logical latency tiers** (HOT, WARM, COLD) from
**physical resource domains** (RAM, device, storage). On an integrated GPU,
HOT and WARM both charge RAM. Objects are registered with a kind, size and
preferred tier. Readers take **leases**, and demotion is forbidden while a
lease or device fence is pending. Budgets and high/low watermarks drive
deterministic eviction. The **platform abstraction layer** (PAL) owns
storage: the POSIX PAL provides file-backed COLD tokens or a RAM-only mode.
There is no silent device emulation, and the tier-collapse policy
(`FOLD_DOWN` or `REJECT`) is explicit.

### File-backed assets (Python + native bridge)

External, immutable files such as weight banks or memory tables are
**admitted**, not copied:

1. A deployment supplies a catalog's bytes and, through a separate trusted
   channel, its SHA-256 (`PinnedAuthority`). A catalog never authorizes itself.
2. `RootCapability` opens one trusted root. Every later open resolves beneath
   that descriptor with `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)`.
   FIFOs, symlinks, traversal and rename races fail closed.
3. The native provider library is copied into a sealed `memfd`, hashed
   against the catalog pin, and only then `dlopen`ed from that descriptor.
4. `FMSFileAssets.register` streams and verifies every page against the
   pinned manifest (size, page size, page map, raw SHA-256). `inspect_asset`
   is observation only.
5. `acquire(asset, offset, length)` materializes the covering pages into
   bounded native FMS residency and returns a `RangeLease`. Evicting a leased
   page is `BUSY`. Released pages are evicted LRU under the WARM budget.

`SyntheticFileAssets` self-authorizes generated test fixtures. The production
constructor rejects its `synthetic-test` provenance.

Nonclaims: the Linux page cache is not charged or bounded (buffered `pread`);
there is no writable COLD replica; the CPU PAL has no accelerator fences.

### Typed contracts

`elpis.substrate.contracts` defines the fail-closed `ContractError(Code, detail)`
used by the substrate and inference. Callers branch on `Code` (`IDENTITY`,
`INTEGRITY`, `LIMIT`, `BUSY`, `STALE`, …), never on message text.

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
