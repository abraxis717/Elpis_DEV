# Migration from the Elpis beta

This document records *why* the clean foundation looks the way it does. The
path-level, per-surface provenance is machine-readable in
[`migration/BETA_MIGRATION.json`](../migration/BETA_MIGRATION.json); this file
does not duplicate it.

## Donor basis

| Fact | Value |
|---|---|
| Donor | `https://github.com/abraxis717/Elpis` (read-only) |
| Planned basis commit | `8eda162712ebe291f2b88b6906366075859bc485` |
| Planned basis tree | `9e3ff61385817f94e5b0d6823fd7c471ef384c59` |
| Live `main` at migration time | `8eda162712ebe291f2b88b6906366075859bc485` (unchanged; tree verified) |
| Donor mutated | No. Accessed only through an anonymous read-only clone. |

## Census method

The census was taken from the donor tree itself, not from its READMEs:

* **Python import graph**: an AST walk over all 1,885 donor files, mapping
  every import to its owning component/runtime directory.
* **Build graph**: the root `CMakeLists.txt`, every sub-`CMakeLists.txt`, and a
  C/C++ `#include` graph of `native/semantic-spine`.
* **Test execution**: the donor native tree was built and run (105/105 native
  tests pass). The Python suites were run for ECS (547 pass), inference +
  Runtime R3 + R3SOT + StateOfThought (375 pass, 2 order-dependent isolation
  failures, both pass alone), and each Grid81 component.
* **Admission metadata**: every `COMPONENT_MANIFEST.json` / `RUNTIME_MANIFEST.json`.

## Census findings

### Live consumers vs. islands

Only a handful of beta components are imported by anything other than their
own tests:

| Donor surface | Non-test Python consumers |
|---|---|
| `ECS/runtime/elpis_ecs` | ECSContextProjector, CadenceECSProbe, Runtime R4 |
| `src/elpis/inference` | Runtime R3, R3SOT, StateOfThought, R3SOTObservationBridge |
| Runtime R3 | StateOfThought, R3SOT |
| Runtime R1 | Runtime R3 (`r1_adapter`) |
| Grid81 capability consumption → application → promotion planner → authority → candidate → publisher | each consumed by the next stage |
| TRMFractalSpine | P0, Runtime R0, TRM drivers, `elpis_reference` |
| DarwinianMatrix | Runtime R0, Runtime R4, `elpis_reference` (all TRM-driven episodes) |

**Standalone islands** (no consumer beyond their own tests): CNumPyCortex,
CadenceECSProbe, FuryanLocusOracle, R3SOTObservationBridge, Runtime R2, Runtime
R4, both platform drivers, and, notably, `Grid81StructuralSemantics`,
`Grid81TypedProjectionCompiler`, `Grid81StructuralGroupProjectionCompiler` and
`Grid81DeterministicCapabilityAuthorityEvaluator`. The front half of the Grid81
chain was connected only through sealed JSON report artifacts under an external
`reports/` tree, not through code.

### Default package, default native build, runtime admission

* **Default Python package** (`elpisai`) shipped the TRM reference runtime,
  TRMFractalSpine, P0, CNumPyCortex and Runtime R0/R1. It did **not** ship
  the Grid81 writer successors (promotion authority, candidate constructor,
  publisher), EvolutionPathGate, StateOfThought or Runtime R2/R3/R3SOT/R4.
* **Default native build** compiled HACF, the whole semantic spine *including*
  its TRM adapter/evaluation/alignment/refiner/refinement-integration layers and
  their tests, and the regex ingress stack. The HACF Python bridge and the
  inference file-asset provider were built only by separate CI jobs.
* **Runtime admission**: exactly one surface, `runtime/R3SOT`, was
  runtime-admitted. Every other component and runtime declared
  `runtime_admission = false`.

### Model dependencies

* **TRM / FPRM / Sudoku**: `src/elpis_reference` (and its vendored Torch
  model), TRMFractalSpine, P0, Runtime R0/R2, DarwinianMatrix `trm/`,
  `controller/` and episode ledger, the TRM checkpoint drivers, the FMS
  checkpoint bridge, and five native semantic-spine layers (P8, P10, P10R, P11,
  P12, plus the P13 orchestration stub).
* **Torch** is imported by the TRM path and by the DarwinianMatrix lattice
  ecology (`ecology/`, `climate/`, `geometry.py`, `runtime.py`, `evaluation/`,
  `ledger/`). DarwinianMatrix `life/` is pure Python.
* **DSV4 / DeepSeek-derived mechanics** live only in `src/elpis/inference`
  (engram addressing, row codec, compact target, experts, global context) and
  Runtime R3 (DeepSpec-derived speculative verifier). None of it needs real
  weights. Qualification uses deterministic synthetic fixtures.

### Native dependency facts

* Every TRM layer of the semantic spine is a **leaf**: no non-TRM source
  includes a TRM, refiner or refinement-integration header. The structural core
  (graph, embedding, context, retrieval, evidence, view, topology, Grid81
  packets) compiles without them.
* OpenSSL was required **only** by the TRM alignment layer.
* The regex/HACF query ingress links a narrow slice of the semantic spine
  (identity, type registry, hypergraph builder, query overlay, snapshots) plus
  the HACF corpus and context graph.
* The inference file-asset provider is a thin bridge over HACF FMS ABI v2 and
  the POSIX PAL.

### Mechanically broken or hollow donor artifacts

These are distinct from architectural rejections:

* `native/elpis-header` contracts import a nonexistent `Artifacts` package;
  five of its modules are zero-byte placeholders.
* `elpis_spine_execute` (semantic spine P13) returns hard-coded numbers
  (36 → 52 filled cells, 16 backend calls) instead of executing.
* More than 150 Grid81 component tests, the Grid81 canonical-reader adversarial
  suite and several phase verifiers read `$ELPIS_CANON_ROOT/.../reports/G*`
  evidence that is not in the donor repository. They fail in a donor checkout
  and were never run by donor CI.
* `elpis_header/observer/grid81_materialized_reducer.py` imports a
  nonexistent `Grid81.materialized_state_reader` module.
* Several Grid81 boundary tests passed vacuously: they checked that
  forbidden files do not exist under a nonexistent external path.
* The promotion planner derived phase disposition from human-edited
  markdown reports. Its own test forbidding this was red at the basis
  commit. The migration fixed it (fail closed) instead of carrying it over.
* The Grid81 runtime reducer put a caller-supplied project root on
  `sys.path` and imported the canonical reader from it. The migrated
  reduction calls the in-package reader instead.
* Structural-group and adjudication joins hard-coded the historical corpus
  size (8,192 rows and 40,960 proposals). Callers now pin the count
  explicitly.

### Duplicated mechanisms

* Three canonical-JSON digest conventions (ECS `canonical`, cross-component
  `canonical_identity`, per-component `canonical.py` files). Persisted digests
  depend on each, so each is preserved where its identities are persisted.
* Four "one-use authority" implementations: P0 lineage receipts, the Grid81
  capability lifecycle, the in-memory `ApplicationLedger`, and the durable
  SQLite ledgers. P0 lineage receipts are dropped. The capability lifecycle,
  the in-memory ledger and both durable ledgers survive: v1 is the
  publication ledger and v2 the artifact-bound application ledger. Their
  persisted identities are distinct.
* Three D4 tables (structural semantics, typed projection, structural
  groups). Persisted orbit identities are bound to each, so all three survive,
  and a test proves they are the same group.
* Two HACF Python bridges: the R1 retrieval wrapper and the TRM checkpoint
  bridge. The retrieval wrapper survives.

## Result

The migration landed in ten tranches, one commit each, plus two corrective
native commits. Per-path provenance is in `migration/BETA_MIGRATION.json`:
204 records across all eight dispositions.

| Tranche | Landed as |
|---|---|
| 1 Foundation | `ELPIS_SYSTEM.json` (sole authority), boundary suite, donor census |
| 2 ECS | `elpis.ecs`: kernel, durable history, replay, projections, Structural R0 |
| 3 Substrate | `elpis.substrate` + `native/substrate`: FMS residency, POSIX PAL, descriptor capabilities, verified file assets |
| 4 Structure | `elpis.structure` + `native/structure`: HACF structural memory, semantic core, retrieval stage |
| 5 Grid81 | `elpis.structure.grid81` and the `elpis.pipeline` canonical writer stages |
| 6 Ingress | `native/pipeline/ingress` + `elpis.pipeline.ingress`: bounded Regex → HACF query ingress |
| 7 Evolution | `elpis.evolution`: heredity, self-verified truncation selection, ECS-bound path gate, gated promotion |
| 8 Inference | `elpis.inference`: driver-neutral contracts, the DSV4 driver, decode transaction, speculative verification, steering |
| 9 Runtime | `elpis.runtime`: one composition over one ECS receipt history; `tests/integration` |
| 10 Hardening | sanitizer, no-network and donor-parity CI; this document, `docs/NONCLAIMS.md`, README |

### Retired

The retired surfaces were not moved elsewhere. Each has a record explaining
why it was retired.

* TRM, FPRM and Sudoku: the reference runtime, TRMFractalSpine, P0, the
  learned drivers and the TRM layers of the semantic spine.
* CNumPyCortex.
* Cadence and CadenceECSProbe.
* Furyan as a runtime.
* R3SOTObservationBridge.
* The runtime generations R0–R4 and R3SOT. Their surviving mechanisms were
  collapsed into subsystems: the R1 retrieval stage into structure, and the
  R3 transaction and R3SOT steering into inference.
* The QUBO selection branch.
* The beta release, sealing, PyPI and registry machinery.

No release, tag or publication authority was migrated.

### Fixes made during migration

These are distinct from renames. Each is recorded with
`semantic_code_changed: true` or described in its record:

* Grid81 joins no longer hard-code the historical corpus size.
* The promotion planner no longer reads human-edited markdown dispositions.
* The Grid81 runtime reducer no longer imports from a caller-chosen `sys.path`.
* Workspace digests and copies in evolution promotion refuse symlinks and
  special files.
* The evolution path gate requires a real ECS `ContextProjection` and a typed
  attempt result.
* The retrieval-bundle adapter fails closed with one typed error.
* Native vector outputs are defined on every path.
* Multi-MB test records live on the heap.
* Test assertions stay live in Release builds.
* A PCRE2 match-data leak on exception paths in the streaming regex ingress,
  found by LeakSanitizer, is fixed with an owning handle.

### Parity with the donor

`tests/boundary/test_donor_parity.py` runs in CI against a read-only checkout
of the donor at the basis commit. It checks two things:

* **Byte identity.** The synthetic inference address artifacts, the
  historical Grid81 canonical generation, `LICENSE` and the upstream license
  texts must be byte-identical to the donor.
* **Persisted identities.** Evolution assertion and harness-manifest
  digests, the content-map digest, canonical identity v1, and ECS state roots
  and event digests for the same history must be equal between donor and
  target.

Two further checks were run once at migration time:

* the query-ingress identities, which are pinned in the native suites;
* an 18,000-case differential run of the streaming lexer.
