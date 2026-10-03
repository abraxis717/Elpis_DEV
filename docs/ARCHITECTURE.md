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

### Native file-asset page service (`native/substrate`, `fms_file_service.h`)

The runtime half of file-backed assets, for native hosts that must not call
back into Python (the DSV4.1 Native Materializer R1 embeds it). It creates no
authority and resolves no path:

1. Cold admission receives an already-open, already-authorized read-only
   regular-file descriptor from `FMSFileAssets.transfer_asset`, together with
   the pinned size, page size, page map and identity stamp recorded by
   `register`. The service duplicates the descriptor (`F_DUPFD_CLOEXEC`) and
   owns only the duplicate; writable, non-regular, changed or mis-mapped
   objects are refused.
2. A page load re-checks the stamp, reads with bounded `pread` (`EINTR`
   retried, a zero-byte read is `IO`), verifies the exact
   `elpis.inference.raw-bytes.r0` page digest, re-checks the stamp, and only
   then registers the verified bytes as one FMS ABI v2 WARM object.
3. Each service owns one private FMS context: FMS ABI v2 remains the residency
   authority. Range leases pin pages; leased pages are never evicted; the
   victim is the least-recently-used unleased page, chosen only when FMS
   refuses a registration — the same policy, counters and failure codes as
   `FMSFileAssets`, verified differentially against it.

The same nonclaims apply: buffered `pread` leaves the Linux page cache
uncharged and unbounded.

Nonclaims: the Linux page cache is not charged or bounded (buffered `pread`);
there is no writable COLD replica; the CPU PAL has no accelerator fences.

### Typed contracts

`elpis.substrate.contracts` defines the fail-closed `ContractError(Code, detail)`
used by the substrate and inference. Callers branch on `Code` (`IDENTITY`,
`INTEGRITY`, `LIMIT`, `BUSY`, `STALE`, …), never on message text.

## ECS_C: identity, continuity, history and replay (`elpis.ECS_C`)

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

`elpis.ECS_C.structural` is the frozen Structural R0 mutation grammar: a
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


## Evolution: heredity, selection and gated promotion (`elpis.evolution`)

Evolution is a set of deterministic, immutable record transitions. Nothing in
it randomizes implicitly, loads a model or touches numerics.

### Heredity and lifecycle

* **Genotypes** are canonical tuples of bounded integer genes.
* **Mutation** is bounded and seeded from content-addressed lineage (parents,
  birth tick and ordinal, world digest), so the same history always
  reproduces the same child.
* **Organisms** carry lineage, genotype, energy, age and a lifecycle state
  (EMBRYO → ALIVE ⇄ REPRODUCTIVE → DYING → DEAD). Death is irreversible.
* **Reproduction** is an atomic, energy-conserving transaction against the
  exact parent revision the caller observed.

### Fitness and selection

A fitness record binds one measured observation to one organism revision and
one fitness policy (exact integer weights). Selection is deterministic
truncation selection over a population revision. It takes no externally
proposed solution: the commit itself verifies every record against the
organism's current revision, recomputes every score, ranks by score and then
organism id, keeps the policy's survivor count and moves every other eligible
organism to DYING. Any stale, missing, duplicate, foreign or misscored record
rejects the whole commit, and the population is unchanged.

The QUBO/solver-proposal selection branch of the beta is retired: its commit
trusted a caller-claimed optimality gap.

### Path gate

An `EvolutionPathAssertion` binds:

* the episode state digest, attempt index and attempt head;
* an edit count within an edit budget;
* a component scope;
* a resource budget and an evaluation contract;
* the exact ECS history the caller reasoned over (projection digest, head
  event digest and final state root of a real `ContextProjection`).

The gate re-checks every binding against the live state. A rejected assertion
executes nothing. An admitted one executes exactly one attempt, which must
return a typed `EvolutionAttempt`. The result is a `PathTransitionReceipt`
chained to the previous receipt by digest. Assertion and receipt records are
byte-identical to the beta gate's.

### Promotion

A candidate workspace is promoted over its incumbent only when all of these
hold:

* its evaluation evidence is bound to the exact parent, candidate, path
  receipt, evaluation contract and all four data partitions (EVOLVE,
  CALIBRATION, HELD_OUT, OOD);
* the correctness, leakage, resource and source-scope checks all pass;
* the held-out gain exceeds the fixed noise envelope;
* the OOD delta stays above the regression floor.

Ties break by held-out gain, then resource cost, then edit count, then
candidate digest. Otherwise the incumbent is retained. Materialization copies
the parent and applies explicit edits in a private stage directory. The child
is renamed into place only if its content map equals the selected manifest.
Symlinks, special files and escaping edit paths are refused. The `elpis.rsi.*`
schema identifiers are historical and carry no self-improvement claim.

### Incomplete interfaces

* **No in-repo fitness environment.** No environment produces fitness
  observations; the beta's Torch lattice ecology is retired.
* **No in-repo evaluator.** Nothing produces promotion evaluation evidence.
* **The caller records receipts.** Transition receipts are returned rather
  than recorded; the runtime composition records them in the ECS history.

## Inference: models behind contracts (`elpis.inference`)

Inference turns proposals into verified target steps. A model is a *driver*
behind a contract: it never owns ECS state, structural memory or canonical
state, and everything it emits outside a committed decode transaction is a
proposal whose authority flags are fixed at zero (`ProposalOnly`).

### Driver contract

`elpis.inference.target` holds the driver-neutral records: tensors, latent
projections and latent inputs, the target configuration, the target state
and the step receipt. It also defines the `Target` protocol, which the
transaction, speculative verification and steering all require of a driver:

* a model identity and numerical profile;
* `initial(context_snapshot)`;
* `step(state, token, expected_state=..., latents=...)`, which returns the next
  state and a step receipt.

The first driver is `drivers/dsv4`. Its `CompactTarget` is a compact
DSV4-shaped target: local and compressed attention, MoE experts executed from
file assets, engram rows and latent channels M/G/X/R. It has no weights of
its own. It is built only from explicitly supplied, digest-verified file
assets registered through `elpis.substrate`, and every bank, expert,
parameter artifact and projection is bound to its model identity.
`drivers/dsv4/fixtures.make_fixture` is an opt-in synthetic fixture intake:
seeded tensors, a synthetic tokenizer, and no training or evaluation data.
Nothing in the base install downloads or ships weights.

### Memory-facing primitives

* **Associative addressing** (`associative`): n-gram address schemes. DSV4.1
  engram and Qwen PLE parameter artifacts are bound by digest to their
  tokenizer and scheme.
* **Rows** (`rows`): bounded row lookup from a bank over a file asset in FMS
  residency.
* **Context** (`context`, `global_context`): immutable context items, lifetimes,
  snapshots, compaction and forks, plus sparse global candidate selection.
* **Experts** (`experts`): expert tensors verified against their manifest
  digests before execution.
* **Prefetch** (`prefetch`): a plan predicts physical byte ranges only. It
  cannot select rows or experts, and replay recomputes it and must match
  exactly.
* **Structural proposals** (`structural`): the edge contract. An
  `AddressProposal` arrives already validated and digest-pinned; inference
  never parses or validates slow-lane artifacts. The adapters that build
  proposals from the pipeline ingress export (`from_regex_hacf`) and from a
  structure retrieval bundle (`from_retrieval_bundle`) live in the runtime
  composition (`elpis.runtime.edges`). `elpis.inference` imports no ECS,
  structure, pipeline, evolution or runtime module, directly or transitively;
  the boundary suite checks both.

### Decode transaction

`InferenceEngine.execute(state, request, expected_state=...)` is the only
model-execution path. It applies a whole request, or nothing:

* A committed base different from `expected_state` rejects as stale.
* A malformed request, proposal or latent returns the original state with a
  typed failure receipt.
* A state this engine has not validated is replayed from the target's initial
  state through every recorded receipt before it is trusted. That covers a
  fresh engine and a state that was tampered with. The validated-state cache
  is bounded.
* `replay(state, request, receipt)` re-executes and requires an identical
  receipt.

Receipts chain request, target steps, latents, structural proposals and
prefetch plans. Determinism holds across processes and hash seeds.

`speculative` drafts tokens with a Markov drafter and verifies them token by
token against the target. The accepted prefix is exactly what greedy decoding
would have produced, and a rejected suffix never enters state or the
validation cache.

### Token lane: sequence transactions

While a sequence is producing tokens, only model execution may make the user
wait. `InferenceEngine.begin` validates the committed state and request once,
admits expert bytes once (`admit_stream`: verified against their manifests,
then never re-hashed), and resumes the driver's stream state. Each
`Sequence.next()` is one `stream_step`: n-gram address arithmetic, row lookup,
local and compressed attention, global selection, expert routing and
execution. The token is returned at once. No canonical JSON, no content or
provenance identity, no `.digest`, no receipt and no replay happen there; the
substrate still verifies any cold page it faults in, and resident pages cost
no hashing. All inputs (tokens or count, latents, proposals, admission) are
frozen at `begin`, and a sequence has no way to take new ones.

`InferenceEngine.finalize` is the commit boundary. The driver's
`finalize_stream` rebuilds, from bounded per-step records and without
re-running the model, exactly the states and step receipts the legacy step
writes. The transaction layer adds prefetch records and the decode receipt.
The result equals `InferenceEngine.execute` for the effective request, so
legacy replay verifies it and no persisted identity changes meaning. A
failure before finalization leaves the committed state untouched; tokens
already streamed are visible but uncommitted. A stop token or an explicit
`stop()` commits what was produced as the effective request (the same request
cut to that length).

### Steering

`steering` is a read-only observer over *completed* decode epochs. It binds
the request, result and receipt of epoch *n*. It then derives a DYN4
observation (hidden and logit statistics, current value and delta) and
proposes one X-channel latent for epoch *n+1*, which expires after *n+2*.

The proposal takes effect only by being applied to a later request as a
`LatentInput`. That application is gated by a host-owned `FastControlState`
with a stall, cycle and hop guard. The transaction records it in the normal
chain: request latents → step receipt latents → decode receipt. There is no
second steering receipt.

Steering never retroacts on its source epoch, replaces tokens, overwrites
logits, mutates weights or context, or touches the ECS. `steered` composes
the engine with this lane: exactly one `execute` per epoch, no retry, and all
recurrence state held in an immutable session value.

The frozen steering contract and guard digests are persisted identities and
are checked at import time. The beta's empirical steering claims are not
carried forward (`docs/NONCLAIMS.md`).

### Incomplete interfaces

* **Synthetic fixture only.** No trained table, tokenizer map or production
  parameter artifact exists here.
* **DSV4-shaped records.** `NeuralState` and `TargetConfig` still carry
  DSV4-shaped fields. A second driver will need them generalized.
* **No vendor accelerator.** The NumPy F32 tower is the reference; a sealed
  CPU-native DSV4.1 backend, the YTS-R0 provider stream over the generic
  execution port, Native Clock R0 and Native Materializer R1 are qualified
  against it using the **test-only** CPU reference provider. No vendor
  accelerator provider and no learned parameter artifact ship (see below).
* **Steering is not in the ECS.** `global_event_fields` exposes what an ECS
  integration would bind, but no steering event is recorded in the ECS
  history yet.
* **Greedy speculative verification only.**

### DSV4.1 native recurrence (`drivers/dsv41`, `native/inference`)

The Python implementation (NumPy tower, YTS host adapter, `RowEngine`,
`TensorStore`, `FMSFileAssets`) remains the oracle. Opt-in native paths:

* **YTS-R0** (`docs/inference/DSV41_PROVIDER_STREAM.md`): one provider stream
  per runtime over the unchanged generic execution port; host-prepared rows
  and bounded expert parts flow in, the provider owns recurrent state.
* **Native Clock R0** (`docs/inference/DSV41_NATIVE_CLOCK.md`): one
  `clock_advance` drives prefill and generation — address hashing, YTS
  codec/submit/take, validation, argmax and stop logic — without entering
  Python.
* **Native Materializer R1** (same document): the production Elpis host
  service behind the clock's materializer table. Cold admission transfers
  pinned descriptors, page maps and bindings from Python; at runtime the
  native file-asset page service, the FP8/E8M0/BF16 row decoder and canonical
  `w1 || w3 || w2` expert staging run natively. With it, the production-shaped
  file-backed recurrence executes no Python inside `clock_advance`
  (qualified with Python traps and a profiler positive control). The
  accelerator provider never receives descriptors, page maps or FMS
  authority.

## Context substrate: HACF, not KV

Regex/HACF is Elpis's context substrate. Keys and values are at most bounded,
ephemeral model scratch inside one sequence; they are never memory, never
cross a subsystem boundary and never persist in principal committed state.

```
historical / user / tool / system content
  -> Regex ingress -> HACF (external store; digest-addressed chunks)
  -> edge adapter (runtime.edges): proposals + chunk claims from the pinned export
  -> object resolution (structure.retrieval.objects): pinned manifest, verified blob, elpis-chunk-v1
  -> ContextAdmission (inference.admission): tokens, digests, budget; frozen
  -> begin -> DSV4 window scratch -> token token ... -> EOS / YIELD -> finalize -> commit
```

All retrieval, resolution and admission happen at the turn boundary, before
`begin`. While a principal sequence produces tokens, only `window_step` runs:
no Regex, HACF, ECS, runtime edge, canonical identity or provenance.

### Disposition of the DSV4 context structures

| Structure | Category | Principal path |
|---|---|---|
| `local_keys` / `local_values` | bounded window, but committed across turns in `NeuralState` | `WindowState.keys/values`: at most `local_window`, sequence-local, dropped at finalize |
| `pending` + `compress` | bounded buffer feeding the pool | removed |
| `global_pool` / `GlobalCandidate` / `StreamCandidate` | persistent, growing compressed-KV context | removed |
| `select_global*` / `IndexResult` / `IndexConfig` | sparse global KV index over the pool | removed |
| `NeuralState.tokens` | token history, grows with all history | removed; per-sequence outputs only, bounded by the sequence budget |
| n-gram `history` tail, `hidden`, `logits` | transient model arithmetic | kept in `WindowState` |
| M projection over associative rows | model-owned addressed memory | kept |
| G / X / R latent channels | alternate context and steering ingress | not accepted on the principal path |
| `NeuralState`, `StepReceipt`, `native-kv-source`, `global-*`, `elpis.sot.*`, `elpis.r3sot.*` | persisted identity | unchanged; legacy transaction and legacy-identical sequence path only |

### What is and is not equivalent

With the same tokens and no latents, the principal kernel and the legacy
kernel agree exactly on keys, values, address rows and the memory term at
every step. Their hidden state and logits agree exactly while the legacy
global selection is empty, and diverge once it is not. The removed term is
exactly the compressed global KV. The principal path is qualified against
an independent reference implementation of its own equations, not against
the legacy kernel.

### Cost

Resident principal state is bounded: 339 canonical bytes of committed state
and at most `local_window` scratch entries in the fixture, at every corpus
size. The price is re-reading: every turn prefills its admitted context
again, and resolution reads and verifies whole document blobs. The
benchmark (`tests/integration/context_scaling_benchmark.py`) reports those
costs separately, per boundary.

## Runtime: one composition over one history (`elpis.runtime`)

There is one runtime composition. The beta's numbered runtime generations
(R0–R4, R3SOT) are retired.

### Receipt history

`ReceiptHistory` is an ordinary ECS kernel history with a fixed genesis
(`elpis.runtime.history.v1`). At genesis it founds, in order, one `history`
entity and one recorder entity for each of `pipeline`, `structure`,
`evolution` and `inference`.

Recording a receipt sends one message from the owning subsystem's recorder
to the history entity. The kernel attributes the sender, so a record cannot
claim another subsystem. The payload is a canonical
`elpis.runtime.receipt-record.v1` record: subsystem, kind, digest and named
bindings.

Recording is idempotent: an equal record returns the existing entry. Opening
replays and verifies the whole event chain, and it refuses a history founded
differently, a record from the wrong recorder or a non-canonical payload.

### Composition

`Runtime` owns only its history and the edge adapters (`elpis.runtime.edges`)
that turn ingress exports and retrieval bundles into inference
`AddressProposal` values before a sequence begins. The caller supplies
everything else explicitly: library paths, corpus roots, file assets, ledgers
and capabilities. Each operation goes through its subsystem's own fail-closed
entry point and is recorded only if that entry point committed:

| Operation | Subsystem entry point | Recorded |
|---|---|---|
| `run_ingress` | `QueryIngress.run` | published zero-authority proposal batch |
| `admit_retrieval` | `validate_bundle` | valid retrieval bundle |
| `publish_canonical` | `publish_candidate` | publication receipt (replay records nothing new) |
| `evolve` | `EvolutionPathGate.execute` over a projection of this history taken at call time | transition receipt of an admitted attempt |
| `decode` | `InferenceEngine.execute` | committed decode receipt |

The evolution gate reasons over the runtime's own history. Because each
recorded transition moves the history head, an assertion built against an
older head is rejected before anything runs.

### Integration suites (`tests/integration`)

Each suite runs over the real native libraries and a real HACF corpus:

* ingress;
* structural memory;
* canonical writer;
* evolution;
* inference on the synthetic DSV4 fixture, where the live ingress export
  becomes structural address proposals for a committed decode.

Each suite also checks that refused operations leave both the history and
the subsystem state unchanged.

### Incomplete interfaces

* **Proposals only, not overlays.** Ingress overlays are recorded by
  identity only and are not persisted.
* **No steering epochs.** Steering epochs are not recorded, and the steered
  engine is composed by the caller.
* **Only publication is recorded.** The application and promotion stages
  before canonical publication are driven by the caller and not recorded.
* **Single process.** There is no cross-process transport. `record()`
  re-reads the event log after each write.
