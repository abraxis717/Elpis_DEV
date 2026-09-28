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

### Grid81 representation (`elpis.structure.grid81`)

Grid81 is the bounded 9 × 9 structural representation. Everything in this
package is read-only with respect to canonical state.

```text
source rows -> typed projection (source identity + transition, expansion
               locus, quiescence and rationale views, each with a D4 orbit
               digest)
            -> join (explicit, caller-pinned row count; all five views must
               cover exactly the same rows or nothing is joined)
            -> structural groups (per row: five group evidence records, five
               proposals, one ordering, conflict evidence, row index)
```

* `semantics` defines the D4 dihedral actions on the 81 cells, pair orbits
  and passive structural contracts.
* `typed` and `groups` each keep their own D4 table and canonical-JSON helper.
  Persisted orbit identities are bound to each, so they are not merged; a
  test proves the three D4 tables are the same group.
* `canonical` is the fail-closed reader of published canonical state. It
  reads HEAD first and verifies every hash. It rejects symlinks and
  unexpected files. It holds a shared `flock` on the `Canonical` directory
  inode for the whole multi-file read. `load_grid81_runtime_state` reduces a
  verified state to its runtime projection. A rejected read yields no object.
  The supplied project root is only data: no code is imported from it.

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

## Pipeline: ingress and canonical publication (`elpis.pipeline`, `native/pipeline`)

The pipeline has two paths. **Ingress** turns bytes into zero-authority
proposals. The **canonical writer path** is the only way canonical Grid81 state
changes.

### Bounded ingress (`native/pipeline/ingress`, `elpis.pipeline.ingress`)

```text
task bytes
-> streaming Regex lexer (PCRE2; bounded interval-specification grammar:
   comparisons, bounds, role bindings, coalescence relations and reducers)
   -> lexical evidence + task candidates + composition, or fail-closed on
      ambiguity
-> HACF: lexical corpus search per evidence anchor + one-hop context graph
   (read-only) -> context proposal (elpis.regex-hacf-context-proposal.r1,
   self-digested, all four authority flags false)
-> query-local proposal batch: every candidate materialized as an unadmitted
   node and assertion in one semantic query overlay, or none
```

* **Bounded input.** The whole-input lexer admits input only up to its carry
  profile (default 1024 bytes). Longer input is rejected before any evidence
  exists, because the grammar has unbounded-span expressions and a rolling
  window could retire a negation before the final disposition. The streaming
  lexer accepts unbounded input with bounded memory. Its identity equals the
  whole-input lexer's wherever both accept; matches longer than 4096 bytes use
  the v2 evidence schema.
* **Ambiguity fails closed.** Contradictory evidence is rejected with
  `REJECTED_PRE_BATCH_AMBIGUITY`. No proposal set, segment, overlay or receipt
  exists for it.
* **Python binding.** Python loads one library, `libelpis_ingress_bridge`,
  from an explicit path. A linker version script limits its exports to the
  bridge, the lexer ABI and the query-ingress result ABI. The bridge opens
  only an existing corpus: symlinked, relative or absent roots are refused,
  and it never creates a corpus. The binding re-reads the authority flags and
  refuses any result that claims authority.
* **Parity.** At migration, the bounded composition identities were
  byte-identical to the donor, and they are pinned in
  `test_query_ingress_bounded`. An 18,000-case differential run matched the
  donor lexer.

### Canonical writer path

Each stage is a separate authority boundary and consumes only the typed
output of the stage before it:

```text
structural groups (structure)
-> adjudication      deterministic policy over each row's five proposals ->
                     dispositions, abstention, adjudication record, inert
                     capability review request
-> capability        evaluation input -> authority decision -> one granted,
                     unconsumed structural-influence capability
-> consumption       capability consumed once -> inert structural-influence
                     artifact + receipt + lifecycle transition
-> application       17 guards; artifact applied to shadow capability state;
                     artifact-bound durable SQLite application ledger (v2)
-> promotion         read-only: source chain, 19 gates, advisory decision,
                     non-executable plan
-> canonical.authority  explicit operator approval digest -> one-use
                        promotion capability bound to the live source state
-> canonical.candidate  isolated immediate-successor candidate tree
-> canonical.publisher  atomic publication
```

Proposals are never authority. A grant is inert until it is consumed. An
application changes only shadow state. The plan cannot execute. The
publisher accepts only the exact capability object: bare digests and a
different lock path are both refused.

### Canonical publication protocol

The lock domain is the existing `Canonical` directory inode of the resolved
project root. Writers take `LOCK_EX` and readers take `LOCK_SH`. No lock file
is ever created. Under the exclusive lock the publisher re-reads everything
mutable: live canonical digest, generation and history, promotion bindings,
ledger head, reservation and recovery record. It then walks this state
machine:

```text
NO_RESERVATION -> PREPARED (fsynced recovery record binds the exact receipt,
                  candidate tree digest and ledger identity)
-> RESERVED_NOT_VISIBLE (durable publication-ledger entry; expected head is
                  checked inside the SQLite write transaction)
-> renameat2(RENAME_EXCHANGE) of Canonical/Grid81 + parent fsync
-> VISIBLE_UNVERIFIED -> verified by the production reader -> VISIBLE_VERIFIED
-> old stage cleanup -> CLOSED
```

* **Exact retries are idempotent.** A retry that finds the new state already
  visible never exchanges again. It returns `ALREADY_COMMITTED` with the
  original receipt.
* **Conflicts fail closed.** A same-source retry with a different object, a
  stale source, a future generation, a different ledger database or a
  malformed recovery record is rejected.
* **There is no rollback.** A verification error keeps the new state and the
  recovery material.

After PREPARED, only the exact object can finish. The protocol has no timeout
and no reservation cancellation.

Trust boundary: this is Linux only, with local-filesystem `flock`,
`renameat2` and directory `fsync`. SQLite needs working locks and FULL sync.
Cooperating writers are assumed. A same-user attacker with write access to
the parent directory is out of scope. Process-death tests are not power-cut
tests.

### Ledgers

The v1 durable ledger is the publication ledger, and the publisher requires
it. The v2 ledger (`elpis.grid81.durable-application-ledger.v2`) is the
artifact-bound application ledger. They are distinct persisted identities.
Both verify their chains relative to trusted database/head authority. They
are not signatures.

### Incomplete interfaces

* **Promotion gates need an evidence writer.** Gate evaluation still reads
  the historical phase-evidence directory layout. Application identity can be
  bound in memory (`bind_g53c_application_identity`), but no in-repo stage
  writes the gate evidence yet.
* **The historical markdown disposition is gone.** Human-readable reports
  never supply a phase disposition. The donor did this despite its own test,
  and the migration fixed it, so phase disposition is always unestablished.
* **The only canonical state is historical.** The one canonical generation is
  a historical test fixture, and no in-repo producer creates a genesis state.
* **The ingress library is not digest-pinned.** It is loaded by explicit
  path only, not yet through substrate authority.
* **Query-local overlays are transient.** Each one lives only in memory for
  the duration of a call. It is not persisted into a semantic snapshot or
  recorded in the ECS history.

### Protocol identifiers

Schema and domain strings keep their historical spelling because persisted
digests depend on them. Examples: `capability-review-request.v1`,
`g5.structural-group-proposal.v1`, `g52a-reason-taxonomy.v1`,
`elpis.grid81.canonical-generation.v2`, the promotion phase ids
`G5.3B.1`/`G5.3C`/`G5.3D`, and `source_gate` values inside digested records.
Ingress keeps its schemas (`elpis.regex-*`, `elpis.semantic.query_local_*`),
the digest-bound component value `HACF_R3` inside the context proposal, and
its C ABI symbol names. Phase names were removed from module names,
docstrings and comments.

