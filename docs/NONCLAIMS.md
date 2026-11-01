# Non-claims

This file lists what Elpis does **not** claim. `ELPIS_SYSTEM.json` carries the
short machine-readable form (`nonclaims`). A passing test here establishes
mechanics only: an interface does what its contract says, and fails closed
where its contract says it must. It does not establish that a mechanism is
useful, that a model behaves well, or that any scientific hypothesis holds.

## System

* Elpis is not claimed to be generally intelligent. "Geodesic intelligence"
  names a research direction, not a result.
* No release, tag or PyPI publication exists for this development repository.
  The beta's release, sealing and publication authority was not migrated
  (`migration/BETA_MIGRATION.json` records each retired surface).
* Mechanics qualification is not scientific qualification, and it is not
  runtime admission of any learned model. No neural model is authority over
  ECS state, structural memory or canonical Grid81 state.
* **The volatile-native migration is closed, and only its stated scope is claimed** (VOLATILE MIGRATION: COMPLETE;
  `docs/ELPIS_MISSION.md`, Amendment 3; `docs/VOLATILE_RUNTIME_CENSUS.md`, section 8). Phases V0 to V5 are
  implemented:
  - RuntimeCore ABI v8 holds one volatile ECS_C session per open;
  - the native host and the Python reference write nothing;
  - FMS has no COLD writeback, and its COLD tier is an immutable, read-only backing;
  - the H-gram writers, the durable ECS_C authority, the checkpoint store and the durability lane are retired;
  - every runtime writer of the census is retired, replaced with volatile state, or a read-only static resource
    (`elpis.runtime.persistence`).

  The composed runtime recorded zero write-class events on storage in every traced path, unconfined and confined
  (`research/volatile_runtime_v1`, `ZERO_WRITE_PASS`). Tracing shows what those runs did, not every possible input.
  The static gates (the census bound to the code, the import gate, the absence of every retired entry point) and the
  containment bound the rest.

  No crash-persistent identity, resurrection or exactly-once-across-restart guarantee is claimed for a session: a
  restart is a new, empty session.
* **The guarantee ends at Elpis's process-control boundary.** Elpis claims only that its runtime issues no
  persistent-storage mutation:
  - the native host's containment denies every filesystem write;
  - it refuses to start without its environmental prerequisites (no core dumps, the swap policy, no storage
    standard output).

  Services outside Elpis's process-control boundary are not Elpis's: swap configuration, crash collectors,
  journals, backups and the filesystems' own metadata activity. Elpis claims nothing about how much the operating
  system or the machine writes to storage. The census, section 6, lists what the host must configure.

## Inference (`elpis.inference`)

* **No model quality claim.** The DSV4 driver runs only on deterministic
  synthetic fixtures (`tests/fixtures/inference`,
  `elpis.inference.drivers.dsv4.fixtures`). No trained table, tokenizer map or
  production parameter artifact is present. A separately admitted real recipe
  V4.1 tokenizer is supported; this does not change the synthetic target.
  Compatibility with DeepSeek V4.1,
  Qwen or any other production model is **not** established. The upstream
  mechanisms the code follows are listed in `LICENSES/PROVENANCE.md`.
* **The DSV4.1 tower is an architecture qualification, not a trained model.**
  `research/dsv41_tower` (noncanonical; relocated out of the canonical tree) is a
  production-shaped DSV4.1 arithmetic driver
  (local/SWA and compressed sparse attention, Engram, mHC, routed and shared MoE).
  Its mechanics are qualified against pinned DeepSeek V4.1 donor behavior: every
  sublayer, layer residual stream and logit at every position, within F32
  reduction-order tolerance (`docs/research/dsv41_tower/DSV41_TOWER_QUALIFICATION.md`). Only
  deterministic training-free fixtures (`TRAINING=NONE`) are exercised. No learned
  production parameter artifact or trained Engram address table is admitted, so
  generated text has no meaning and no language-quality, reasoning or
  intelligence claim follows. Where the pinned donor's own decode path disagrees
  with its prefill path (shared index keys of a filling compressor group), Elpis
  follows the prefill/documented semantics. The NumPy F32 code is the reference
  implementation, not a native hot-path backend.
* **Frozen address artifacts are synthetic.** The DSV4.1 and Qwen PLE address
  vectors pinned by the tests are synthetic maps and constants. They pin the
  addressing arithmetic, not agreement with a trained model.
* **Speculative decoding** establishes greedy equivalence with the target
  on the tested inputs. No speed-up, acceptance rate or sampling-equivalence
  claim is made.
* **Steering** (`steering`, `steered`) is mechanics only. It proves that a
  steering latent derived from a completed epoch can be delivered into later
  epochs, and only later epochs, under the frozen temporal/control contract
  and through the target's own latent-input path. No reasoning, quality,
  fitness or causal-effect claim is made.
* **Context substrate.** Bounded resident model state is claimed, and it
  is measured by inspecting state geometry, not process RSS. Free retrieval
  is not claimed: prefill re-reads admitted context every turn, and object
  resolution reads and verifies document blobs. Both costs are reported
  separately. Admitted-context rendering supports both synthetic nibbles and
  a digest-bound recipe V4.1 text tokenizer in distinct domains. Text transport
  tests use an explicitly scripted token-emission double, not a trained model.
  Parameter inspection is read-only preflight, not an executable production
  intake: no learned production parameter artifact is admitted. The
  principal path is not numerically equivalent to the legacy compressed-KV
  kernel, and no quality comparison between them is claimed.
* **ECS Runtime R1 is engineering, not cognition.** Its numbers are
  PERFORMANCE_ONLY, NO_SCIENTIFIC_CLAIM, measured on one shared virtual
  machine with one compiler (`docs/performance/ECS_RUNTIME_R1.md`); they do
  not transfer across hosts, and the scaling widths are not supported
  cognition. The registered cross-process Python-overhead estimator failed as
  registered and is reported so. The executor is SINGLE_WRITER: no
  multi-writer or concurrent-reader claim. Bitwise parity is claimed against
  the scalar reference only; no optimized backend with a different
  accumulation order exists.
* **Elpis does not generate text.** No ECS<->DSV semantic codec is defined or
  qualified, so every text path fails closed with `ECS_CODEC_UNQUALIFIED`:
  the native host's `text` operation and, in the offline Python reference,
  every codec operation (the legacy `Runtime.run_turn` included). The only codec maps in the repository are
  `TRAINING=NONE SEMANTICS=NONE` interface fixtures, never presented as
  cognition. A mechanics pass is not a mission pass (`docs/ELPIS_MISSION.md`).
  No geodesic-intelligence claim follows.

### Beta steering claims not carried forward

The beta's steering component carried research conclusions as code
constants. They were removed during migration and are **not** claims of this
repository:

* `BOUNDED_CLOSURE_CLAIM`: an assertion of positive empirical causal-transfer
  support at every X-interface width 4 through 32 under both address schemes
  inside the compact-target testbed.
* `NEGATIVE_CORE12_X_OUTCOME` / `NEGATIVE_CORE12_X_CLOSURE_DIGEST`: the closure
  of a CORE12-to-X adapter line as "no causal steering demonstrated under the
  frozen contract".
* `CONTIGUOUS_WIDTH_CLOSURE_DIGEST` / `CONTIGUOUS_WIDTH_RERUN_DIGEST`: digests of
  the beta's width-sweep closure evidence, which is not in this repository.

The frozen steering contract, width-transport rule and fast-control guard
digests stay as persisted identities: the code checks itself against them at
import time. They identify *which* contract the mechanics implement. They
are not evidence that the contract has any effect.

## Continuity: the volatile session (the durable authorities retired)

The runtime's lineage and authority are bound by the volatile ECS_C session (`docs/ECS_C_SESSION.md`), which
RuntimeCore ABI v8 holds for one session per open.

* **Nothing survives the session.** No persistence, crash survival, resumption or exactly-once behaviour across
  restarts. A crash, a restart or `close` loses the session, by design. There is nothing to recover.
* **Identity within a session, not authentication.** The session's lineage and commit identify its state within
  its life. They authenticate nothing to anyone outside the process. No persistent authentication, rollback
  resistance or cross-restart monotonicity is claimed. Code inside the process can read or corrupt the session's
  memory.
* **Current authority, not history.** The session holds the cognition lineage's anchored identity, the evolution
  authority `(revision, head, pending_assertion)`, the entities and their sequences. Its receipts are a chain head,
  not a log; it keeps no turn log, event history or audit trail.
* **At-most-once within a session, not automatic retry.** The runtime reserves the exact assertion before
  execution. An attempt that cannot be finalized leaves the reservation pending and fail-stops the session, so the
  assertion never executes again in it. A new session starts idle, and whether an earlier attempt's external effects
  happened is for the caller to establish. There is no automatic retry or reconciliation.
* **Not cognitive state.** The session holds a digest of the anchored state, never the state. ECS_G owns `W`,
  epoch, `H` and `a`. The session grants no authority over ECS mathematics.
* **The session is not swap-proof by itself.** The native host and the Python process gate lock the process's
  memory, or refuse to start, when swap is active. Whether the machine swaps is the host's configuration, which
  Elpis never changes.
* **The ECS_C relational view is DEV.** `native/ECS_C/relational` (`docs/ECS_C_RELATIONAL.md`,
  ECSC_DYNAMIC_RELATIONAL_DEV_R0) keeps typed evidence per (source, version, scope) binding and answers only from
  evidence the session's admission law still accepts. Nothing canonical composes it: RuntimeCore's HOT lane admits
  by `source_authority` alone, and memory LOOKUP stays source-blind. It persists nothing and qualifies nothing. Its
  evidence counts are not truth. Its HCA comparison (`research/ecsc_dynamic_relational_r0`) is a DEV measurement,
  not a qualification of HCA or of the view.

**History.** Standalone continuity retired into the durable ECS_C authority at RuntimeCore ABI v7 (Amendment 2), and
the durable ECS_C authority, with K1 Recovery R0's checkpoint store, retired at ABI v8 (Amendment 3). Their
non-claims (authentication only against the T1 adversary, crash consistency under `SIGKILL` and never power loss,
integrity rather than authentication of checkpoint slots) are recorded in `docs/ECS_C_AUTHORITY.md`,
`docs/K1_RECOVERY_R0.md` and `docs/CONTINUITY.md`. Their qualifications bind their commits only: ECS_C Recovery Q0 is
RUN PASS for commit `117b41a` (ABI v6) only, and K1 Recovery R0 v4 for commit `18f2f4d` only. Neither qualifies
anything under Amendment 3.

## Runtime boundary after the red team (`elpis.runtime`, `native/runtime`)

* **Mechanics, not qualification.** Read-only QUERY and authority-gated LEARN,
  independent codec admission, total cognitive fuel, managed K1 lease
  ownership, sealed native admission, the volatile session and the process
  containment are implemented and tested. None of them is formally qualified
  as cognition: the specifications in `docs/qualification` (Cognitive Boundary
  R2 and its ABI v5 successor included) are NOT RUN, so the cognitive boundary
  itself is unqualified. Cognitive Boundary R1's retrospective closure covers
  commit `bec6da7` (ABI v3) only. The recovery qualifications (K1 Recovery R0
  v4 for `18f2f4d`, ECS_C Recovery Q0 for `117b41a`) are historical: their
  subjects are retired.
* **Python is off the production path, not removed.** The production entry
  point is the native host `elpis` (RuntimeCore ABI v8, no interpreter in the
  process). The Python runtime composition is the offline reference interface
  over the same RuntimeCore library. A build needs a Rust toolchain.
* **Zero writes, as traced.** `research/volatile_runtime_v1` is the evidence
  for the composed runtime's zero persistent writes: what its traced runs did
  on one host (Linux 6.18, ext4, no swap). It is not a proof over every input,
  and it says nothing about the host's own services.
* **Pins authenticate bytes, not authors.** The native, codec and evolution
  policy authorities verify exact bytes against SHA-256 pins from trusted
  configuration. They offer no protection against a malicious in-process
  caller, who can construct a catalog and its pin. The codec implementation
  digest covers its defining source file only.
* **The lease is not a credential.** RuntimeCore's K1 lease defends against
  aliasing by cooperating local callers; in-process code can replace it (the
  owner then fail-stops). Reads stay open to anyone.
* **Fuel is not a deadline.** Cognitive fuel is a deterministic integer
  work-unit bound admitted before an operation starts. No wall-clock deadline
  or preemption is claimed: the K1 ABI has no cancellation point.
* **The contextual lane is mechanics, not cognition.** `elpis.runtime.context`
  proves ownership, bounded handoff and isolation of an active context: it
  implements no contextual dynamics law (Hypergraph Cellular Automata are an
  unqualified candidate), infers no relation, encodes no view into the K1
  stimulus and makes no semantic claim from retrieval similarity,
  co-occurrence or interaction. Its tests use synthetic references and the
  deterministic hash-derived fixture embedding, which is not evidence of
  semantic retrieval. It is session-local and never persisted.
* **HOT-lane performance figures are descriptive.** Before/after timings of the
  volatile turn and the query are single-host observations, not a performance
  qualification and not evidence of model-level inference throughput.

## Evolution (`elpis.evolution`)

* The historical `elpis.rsi.*` identifiers on promotion records are persisted
  schema names. They do not claim recursive self-improvement.
* Selection, reproduction and fitness are deterministic record mechanics. No
  adaptive or open-ended evolution claim is made, and no evolutionary process
  is claimed to improve anything.
* **The policy authority confines nothing.** `elpis.evolution.policy` pins and
  measures the evaluator and issues gates, evaluations, selections and grants,
  but its confinement is `TRUSTED_OPERATOR_CALLBACKS`: evaluators and attempt
  callbacks are trusted operator code, not sandboxed. Its issuance registries
  are process-local. The bare promotion law still accepts caller-built evidence.
* The only environment producing fitness, `research/evolution_fitness_r0`, is a
  RESEARCH_ONLY toy corridor; it is unqualified and its objective transfers
  nowhere.

## Structure and pipeline

* HACF retrieval, Grid81 structural representation and the canonical writer
  grant no semantic truth. Retrieval results, structural groups, adjudication
  outcomes and promotion plans are evidence or proposals. Only the explicit,
  one-use promotion authority can change canonical state.
* The historical Grid81 canonical fixture is a regression fixture, not a
  benchmark.
* HACF is not complete. Implemented: A (deployment-pinned, substrate-sealed
  bridge loading) and C v1 (an admitted claim-to-claim relation with PRIMARY
  witnesses projected onto one chunk context edge; scope
  `ADMITTED_CLAIM_TO_CLAIM_PRIMARY_WITNESSES_ONLY`). B (admission and immutable
  snapshot publication) is an explicit, offline operator process, never
  reachable from the runtime. Volatile retrieval epochs are the canonical
  autonomous retrieval path. A retained H-gram is a read-only static resource
  with no learning authority; its writers are retired. There is no arbitrary semantic
  hyperedge projection, no natural-language relation extraction, no semantic
  codec and no canonical HACF->ECS or ECS->HACF edge.
  `research/hacf_ecs_bridge_r0` is RESEARCH_ONLY, unqualified infrastructure:
  its only observation map is a TEST_ONLY fixture, it cannot detect
  well-formed misleading evidence, and no answer it produces is authority.
* The runtime has no persistent writer at all, bounded or not.
  - The native host has no home and its containment grants no write.
  - The offline Python reference's runtime path writes nothing, and FMS is
    WARM and HOT in RAM only.
  - Every other persistent writer is an offline operator process or development
    tooling, never reachable from a runtime composer (`elpis.runtime.persistence`).
    This is a property of the code paths, proven by tests and by the traced
    composed runs; it is not a claim about host filesystem accounting.

## Research laboratories (`research/`)

* `research/ecs_dynamics` is RESEARCH_ONLY with NO_RUNTIME_AUTHORITY. It is
  not packaged, and no production module imports it (the boundary policy
  forbids the import). Its tanh probe, cubic control and associative network
  are synthetic. No result there is a claim about production ECS, a trained
  model, the DSV4 fixture or any Elpis neural component.
* Its dispositions ("supports under this frozen synthetic regime") hold only
  for the frozen specification, seed and worlds recorded in
  `research/ecs_dynamics/frozen` and `evidence/qual`. Diagnostics are
  finite-time and attractors are sampled. No Lyapunov exponent, chaos or
  bifurcation is claimed.
* Paper-derived statements rest on user-supplied PDFs identified by SHA-256
  and cited by equation in `docs/research/ECS_DYNAMICS_RESULTS.md`. The
  public cubic control is not a reconstruction of the private Structural R0
  reference family.
* `research/hecs_r0` (H-ECS R0, hierarchical ECS world models) is RESEARCH_ONLY with
  NO_RUNTIME_AUTHORITY and closed `TASK_INVALID_ON_DEV`
  (`docs/research/HECS_R0_RESULTS.md`). It stopped at DEV calibration: its
  preregistered V1 threshold lay below the SEPARATED world's achievable one-step
  error. No hierarchy, abstraction, planning, width or depth claim follows, positive
  or negative. ECS is not claimed to be a JEPA.
* `research/hecs_r1` (H-ECS R1, R0's successor with a valid task-validity law) is
  RESEARCH_ONLY with NO_RUNTIME_AUTHORITY and closed `TASK_INVALID` at QUAL
  (`docs/research/HECS_R1_RESULTS.md`): its DEV run was valid, and its single QUAL run
  failed V1C ECS adequacy in 1 of 16 instances. The hypotheses were not adjudicated,
  so no hierarchy, abstraction, planning, width or depth claim follows. The
  fast-factor loss seen at distinct upper levels comes from a fixed linear encoder,
  not from ECS learning. No hierarchy integration was authorized, and H-ECS is not
  canonical.
* `research/hecs_r2` (H-ECS R2, multi-step validity of ECS world models, then
  hierarchy) is RESEARCH_ONLY with NO_RUNTIME_AUTHORITY and closed
  `WORLD_MODEL_INVALID` at QUAL (`docs/research/HECS_R2_RESULTS.md`). Task validity
  and one-step constituent validity (S1) held in every instance, but in SEPARATED
  2 of 8 fresh worlds fell below the frozen free-running capture bound at a horizon
  the planners consume. The hierarchy hypotheses were not adjudicated: no
  hierarchy, width, temporal, representation or depth claim follows, and
  `WORLD_MODEL_INVALID` is not a hierarchy failure. MATCHED's nominal M is not
  evidence of usefulness (persistence reaches it). Finite-time tangent growth rates
  are not Lyapunov exponents, and no chaos, entropy or hyperbolicity claim is made;
  the openai/math donor supplied mathematical structure only. No integration was
  authorized, and H-ECS is not canonical.
* `research/hecs_r3` (H-ECS R3 successor lineage) is RESEARCH_ONLY with
  NO_RUNTIME_AUTHORITY (`docs/research/HECS_R3_RESULTS.md`). The C2/C2M
  constituent world-model learning law is `WORLD_MODEL_VALID` under its frozen
  regime: it is an offline learning law from observed trajectories, not a
  query-time method. The hierarchy hypothesis, adjudicated over matched
  qualified constituents, is negative (`NO_ADVANTAGE_OVER_WIDTH`). Hierarchy
  integration is NOT AUTHORIZED, no H-ECS or EDEN hierarchical runtime is
  canonical, and no runtime readiness is claimed.
* `research/ecs_memory_mf_r0` (ECS Memory MF-R0, the successor of the PR #47 model-free memory) is
  RESEARCH_ONLY with NO_RUNTIME_AUTHORITY (`docs/research/ECS_MEMORY_MF_R0.md`). It is not part of ECS_C,
  no runtime authority binds its state (a memory transition leaves the K1 digest unchanged, tested), and no
  composite retained state or LEARN authority exists for it (its canonical relocation is the next item). Its preregistered QUAL passed
  H1 (state-borne acquisition), H2 (updating), H3 (partial-cue completion, classified as algebraic completion
  by fixed encoding geometry) and H5 (consolidation of recurrent relations), and failed H4: byte-matched exact
  keyed storage retains more under interference. Codes are fixed SHA-256 derivations: no semantics, learned
  similarity or generalization to unseen entities is claimed.
* The canonical ECS memory and the composite retained state (`docs/ECS_MEMORY.md`,
  `docs/ECS_COMPOSITE_STATE.md`) are **mechanical** integration. OBSERVE and LOOKUP are exact storage of typed
  structural evidence, not truth, meaning or learning; LEARN and the FAST channel are UNQUALIFIED (refused unless a
  host admits the recurrent capability; RuntimeCore never exposes them); the MF-R0 consolidation law is not
  canonical at all, and the frozen H4 NONPASS stands. A composite identity is the runtime's authority only as
  the anchored identity of a volatile session, for that session's life; a composite snapshot round trip is an
  in-memory rollback, never a recovery from storage.
  The native host and the Python reference drive composite lineages as mechanics only (numeric LEARN and
  observe-only evidence): nothing decides what to observe, no HACF -> meaning map exists, and no codec path exists
  (`ECS_CODEC_UNQUALIFIED`). Canonical cognitive integration of the memory into the full text turn is **not**
  complete.
