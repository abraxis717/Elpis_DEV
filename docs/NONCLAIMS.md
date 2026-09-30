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
  production parameter artifact is present. Compatibility with DeepSeek V4.1,
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
  separately. Admitted-context rendering is SYNTHETIC (byte to nibbles for
  the fixture tokenizer); no production tokenization is claimed. The
  principal path is not numerically equivalent to the legacy compressed-KV
  kernel, and no quality comparison between them is claimed.

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
