# Elpis (development foundation)

Elpis is a research system organised around one idea, *geodesic intelligence*:
persistent structured state, memory topology, bounded transformations,
controlled evolution and inference should interact so that computation
follows structure the system has already accumulated. This is a research
direction, not a capability claim; see [`docs/NONCLAIMS.md`](docs/NONCLAIMS.md).

The division of responsibility is constitutional
([`docs/ELPIS_MISSION.md`](docs/ELPIS_MISSION.md)): **DSV4 communicates. ECS /
EDEN computes, learns and persists. FMS materializes. HACF / structure
organizes persistent structural memory. Continuity binds minimal durable
runtime lineage/authority.** DSV4 is the token boundary, not the brain.

This repository is the clean development authority. It was built by a
selective, audited migration from the Elpis beta repository. The per-path
record is [`migration/BETA_MIGRATION.json`](migration/BETA_MIGRATION.json), and
the reasoning is in [`docs/MIGRATION.md`](docs/MIGRATION.md).

The single machine-readable system authority is
[`ELPIS_SYSTEM.json`](ELPIS_SYSTEM.json). The design is described in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Merges follow
[`docs/CI_POLICY.md`](docs/CI_POLICY.md): no merge while any workflow run for
the head commit is red.

## Layout

| Path | Subsystem |
|---|---|
| `src/elpis/substrate`, `native/substrate` | resource authority, residency and materialization (FMS, PAL, file assets, execution port) |
| `src/elpis/ECS`, `native/ECS` | the one canonical ECS: cognitive/dynamical substrate (state `W`, epoch, `H`, `a`; `S3` observable; recurrence, learning, consolidation, readout; native K1 transactions, snapshots, FMS residency) |
| `native/continuity`, `src/elpis/continuity` | minimal durable runtime authority in Rust behind a stable C ABI (`include/elpis/continuity.h`): a fixed-size two-slot register of the committed K1 state digest and the evolution authority ([spec](docs/CONTINUITY.md)); `elpis.continuity` is a thin Python adapter |
| `src/elpis/structure`, `native/structure` | HACF structural memory, semantic core, retrieval, Grid81 representation |
| `src/elpis/pipeline`, `native/pipeline` | bounded Regex → HACF ingress; the canonical Grid81 writer path |
| `src/elpis/evolution` | deterministic heredity, selection, gated promotion |
| `src/elpis/inference` | the DSV4 communication codec (tokenizer, rendering); retained noncanonical model-execution mechanics for historical replay |
| `src/elpis/runtime` | the one runtime composition: codec -> ECS -> codec (fails closed until the ECS codec is qualified), bound to one continuity register |
| `research/ecs_dynamics` | RESEARCH_ONLY synthetic dynamics laboratory; not packaged, never imported by `src` ([results](docs/research/ECS_DYNAMICS_RESULTS.md)) |
| `research/dsv41_tower` | RESEARCH_ONLY DSV4.1 transformer/MoE tower, native backend, provider stream, clock and materializer; donor/oracle qualification, not Elpis cognition |

## Build and test

The native build needs CMake, a C11/C++17 compiler, pkg-config, SQLite 3,
PCRE2 and a Rust toolchain (cargo and rustc >= 1.89, for `native/continuity`; no
crates are downloaded). On Manjaro:
`sudo pacman -S --needed base-devel cmake pkgconf sqlite pcre2 rust`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

python -m pip install ".[test]"
ELPIS_NATIVE_BUILD=$PWD/build ELPIS_REQUIRE_NATIVE=1 python -m pytest
```

Continuity alone, bounded and with live output (Rust, C ABI, Python adapter and
the runtime paths that publish to it):

```bash
native/continuity/qualify.sh build
```

Base installation has no mandatory Python runtime dependencies and never
downloads model weights. NumPy is installed only by the `inference` extra (and
the `test` extra, because the full suite exercises inference). Importing
`elpis` or `elpis.runtime` needs neither Torch nor NumPy. The distribution is
`elpis-dev`, marked `Private :: Do Not Upload`. No release, tag or PyPI
publication exists.
