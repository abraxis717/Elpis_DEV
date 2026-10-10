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
  qualified, so the canonical turn (`Runtime.run_turn`) fails closed with
  `ECS_CODEC_UNQUALIFIED`. The only codec maps in the repository are
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

## Continuity (`native/continuity`, `elpis.continuity`)

* **Byte parity, not a new format.** The Rust authority reproduces vectors
  frozen from the qualified Python implementation (PR #36) and keeps format
  v2 unchanged. Parity covers the frozen records and the frozen store session;
  it is not a proof of equivalence for every possible input sequence.
* **Python is not removed from production.** Only the continuity authority
  moved to Rust. The runtime composition is still Python and reaches the
  library through a `ctypes` adapter. A build needs a Rust toolchain.

* **Current authority, not history.** Continuity holds one fixed-size record
  (two 176-byte slots, 352 bytes in total, whatever the runtime lifetime):
  the committed K1 retained-state digest and the evolution authority
  `(revision, head, pending_assertion)`. It keeps no turn log, no receipts, no events and no
  audit trail. Nothing in it can answer what happened before the current
  authority. Per-turn diagnostics live only on the returned turn result.
* **At-most-once, not automatic recovery.** Runtime reserves the exact
  assertion before execution. Pending authority refuses further evolution;
  the caller must establish the result externally before explicit finalization.
  An uncertain final publication may reopen pending or at the next idle
  revision; neither permits the old assertion to execute again. There is no
  automatic retry, pending reset or external-side-effect reconciliation.
* **Integrity, not authentication.** The record checksum detects torn or
  corrupted slots. It does not authenticate the writer: whoever can write the
  directory can replace the register consistently. The K1 digest binds the
  identity of a K1 state, not its provenance.
* **Crash model.** The crash matrix simulates process death and torn writes
  at every named publication and initialization step, and failed writes and
  failed `fdatasync`. Durability under power loss rests on the host
  filesystem honouring `fdatasync` for an in-place, fixed-size write and on
  the rename and directory-`fsync` ordering at initialization. A torn sector
  is detected by the checksum and resolves to the other slot; stronger
  power-failure atomicity is not claimed.
* **No rollback, no synthesis.** If publication fails after a native K1
  commit, the K1 commit stands and the runtime fail-stops with the
  publication code. Restart verifies the supplied K1 state against the
  durable identity and refuses a mismatch before any mutation. No missing
  turn is replayed or synthesized; recovering from a mismatch is an explicit
  operator decision outside the runtime.
* **Single process, single owner.** A directory `flock` excludes a second
  owner on the same host. There is no cross-process transport, federation,
  replication or shared-reader semantics. One open runtime owns one K1
  lineage.
* **Not cognitive state.** Continuity holds a digest of the K1 state, never
  the state. ECS owns `W`, epoch, `H` and `a`, their snapshots and FMS
  residency. Continuity grants no authority over ECS mathematics.
* **Legacy storage is refused, not converted.** The retired receipt-history
  layout is detected and refused. Conversion is an offline step documented
  in `docs/CONTINUITY.md`.

## Evolution (`elpis.evolution`)

* The historical `elpis.rsi.*` identifiers on promotion records are persisted
  schema names. They do not claim recursive self-improvement.
* Selection, reproduction and fitness are deterministic record mechanics.
  No environment in this repository produces fitness observations. No
  adaptive or open-ended evolution claim is made.

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
  snapshot publication) is explicit, operator-governed mechanics, not autonomous
  persistence. Volatile, diskless retrieval epochs are the canonical autonomous
  retrieval path. The H-gram is a fixed-capacity, operator-provisioned
  associative store with no learning authority. There is no arbitrary semantic
  hyperedge projection, no natural-language relation extraction, no semantic
  codec and no HACF->ECS or ECS->HACF edge.
* Global autonomous unlimited persistence is prohibited. The autonomous runtime
  path may write only the two fixed continuity slots and a capacity-bounded FMS
  cold store; every other persistent writer is operator-explicit, offline or
  prohibited (`elpis.runtime.persistence`). This is a property of the code
  paths, proven by tests; it is not a claim about host filesystem accounting.

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
