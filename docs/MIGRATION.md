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

### Duplicated mechanisms

* Three canonical-JSON digest conventions (ECS `canonical`, cross-component
  `canonical_identity`, per-component `canonical.py` files). Persisted digests
  depend on each, so each is preserved where its identities are persisted.
* Four "one-use authority" implementations: P0 lineage receipts, Grid81
  capability lifecycle, the in-memory `ApplicationLedger`, and the durable
  SQLite ledger. The durable ledger and capability lifecycle survive.
* Two HACF Python bridges: the R1 retrieval wrapper and the TRM checkpoint
  bridge. The retrieval wrapper survives.
