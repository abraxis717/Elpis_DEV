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
  intake: the production arithmetic driver remains unimplemented. The
  principal path is not numerically equivalent to the legacy compressed-KV
  kernel, and no quality comparison between them is claimed.
* **Text output is inert.** `Runtime.run_text` records a finalized principal
  commit and an output digest in ECS. It does not ingest generated text into
  HACF. Post-sequence proposal/admission and canonical ECS-to-context projection
  remain separate integration work. No geodesic-intelligence claim follows.

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
