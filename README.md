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
Amendment 3 (the volatile-native runtime) governs "persists" and "durable":
every mutable operational state lives in volatile memory only, persistent
storage holds immutable resources the runtime reads, and no runtime state is
written back (VOLATILE MIGRATION: COMPLETE;
[`docs/VOLATILE_RUNTIME_CENSUS.md`](docs/VOLATILE_RUNTIME_CENSUS.md)).

This repository is the clean development authority. It was built by a
selective, audited migration from the Elpis beta repository. The per-path
record is [`migration/BETA_MIGRATION.json`](migration/BETA_MIGRATION.json), and
the reasoning is in [`docs/MIGRATION.md`](docs/MIGRATION.md).

The single machine-readable system authority is
[`ELPIS_SYSTEM.json`](ELPIS_SYSTEM.json). The design is described in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Merges follow
[`docs/CI_POLICY.md`](docs/CI_POLICY.md): no merge while any workflow run for
the head commit is red.

**Running it.** The production entry point is the native operator host `elpis`
([`docs/NATIVE_HOST.md`](docs/NATIVE_HOST.md)): RuntimeCore ABI v8 in a
volatile ECS_C session, with no interpreter in the process. It is RAM-only:
- it reads its pinned libraries read-only into sealed memory;
- it has no deployment home;
- it contains itself so that it holds no write permission to any file.

[`docs/OPERATING_ELPIS.md`](docs/OPERATING_ELPIS.md) installs and operates it.
It covers:
- numeric QUERY and LEARN of the native ECS composite;
- typed structural evidence (no K1 learning) and exact LOOKUP;
- the operator's source events.

A session ends with `close` or the process, and a restart is a new session at
the lineage's initial state: nothing is published, checkpointed or resumed.
There is no text interface (`ECS_CODEC_UNQUALIFIED`).

`python -B -m elpis.runtime.cli` is the offline reference interface over the
same RuntimeCore library. It serves differential tests and research, and is not
the production path.

## Layout

| Path | Subsystem |
|---|---|
| `src/elpis/substrate`, `native/substrate` | resource authority, residency and materialization (FMS, PAL, file assets, execution port) |
| `native/ECS_G`, `src/elpis/ECS` (Python bindings) | ECS_G, the geometric ECS: cognitive/dynamical substrate (state `W`, epoch, `H`, `a`; `S3` observable; recurrence, learning, consolidation, readout; native K1 transactions, snapshots, FMS residency) |
| `native/ECS_C` | ECS_C, continuity and memory: the bounded typed relational memory (`docs/ECS_MEMORY.md`) and the volatile session authority (`docs/ECS_C_SESSION.md`; held by RuntimeCore ABI v8, one session per open, on the native host and the offline reference). The durable ECS_C authority ([history](docs/ECS_C_AUTHORITY.md)) and standalone continuity before it ([record](docs/CONTINUITY.md)) are retired |
| `native/runtime/composite` | the composite retained state (K1 + memory, `ELPISEC1`), RuntimeCore's composition of the two |
| `src/elpis/structure`, `native/structure` | HACF structural memory, semantic core, retrieval, Grid81 representation |
| `src/elpis/pipeline`, `native/pipeline` | bounded Regex → HACF ingress; the canonical Grid81 writer path |
| `src/elpis/evolution` | deterministic heredity, selection, gated promotion |
| `src/elpis/inference` | the DSV4 communication codec (tokenizer, rendering); retained noncanonical model-execution mechanics for historical replay |
| `native/runtime` | RuntimeCore, the runtime's systems authority in Rust behind a stable C ABI (`include/elpis/runtime.h`, v8): lifecycle, fail-stop, the volatile ECS_C session (with native typed-evidence admission), K1 lineage binding, the managed turn's native K1 transaction and the evolution reservation, all in memory ([spec](docs/RUNTIME_CORE.md)) |
| `native/runtime/host` | the native operator host `elpis`, the production entry point: read-only pinned sealed admission, mandatory self-containment, RuntimeCore in a volatile session ([spec](docs/NATIVE_HOST.md)) |
| `src/elpis/runtime` | the Python facade over RuntimeCore (codec -> ECS_G/ECS_C composite -> codec; fails closed until the ECS codec is qualified), its process gate, the persistent-writer census (`persistence`) and the offline reference interface (`cli`, `deployment`; `acceptance` and `release` are development tooling); not the production path |
| `research/ecs_dynamics` | RESEARCH_ONLY synthetic dynamics laboratory; not packaged, never imported by `src` ([results](docs/research/ECS_DYNAMICS_RESULTS.md)) |
| `research/dsv41_tower` | RESEARCH_ONLY DSV4.1 transformer/MoE tower, native backend, provider stream, clock and materializer; donor/oracle qualification, not Elpis cognition |

## Build and test

The native build needs CMake, a C11/C++17 compiler, pkg-config, SQLite 3,
PCRE2 and a Rust toolchain (cargo and rustc >= 1.89, for
`native/ECS_C/session` and `native/runtime`; no crates are downloaded). On Manjaro:
`sudo pacman -S --needed base-devel cmake pkgconf sqlite pcre2 rust`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

python -m pip install ".[test]"
ELPIS_NATIVE_BUILD=$PWD/build ELPIS_REQUIRE_NATIVE=1 python -m pytest   # the FAST lane
```

The developer-fast command, bounded and with live output (warm build check, every
ctest test and the FAST pytest lane; 46 s warm on a 4-core VM):

```bash
tests/qualify.sh fast build
```

A bare `pytest` runs the FAST lane only. The other lanes are explicit:
`tests/qualify.sh stress|scientific|historical|all build` (or `pytest --lane ...`).
Ownership of every test by a lane and of every lane by a CI job:
[docs/CI_POLICY.md](docs/CI_POLICY.md#lanes).

The volatile session and RuntimeCore alone (the Rust suites, the C ABIs over
real native K1, and the native host):

```bash
ctest --test-dir build -L ECS_C --output-on-failure
ctest --test-dir build -L runtime --output-on-failure
```

Base installation has no mandatory Python runtime dependencies and never
downloads model weights. NumPy is installed only by the `inference` extra (and
the `test` extra, because the full suite exercises inference). Importing
`elpis` or `elpis.runtime` needs neither Torch nor NumPy. The distribution is
`elpis-dev`, marked `Private :: Do Not Upload`. No release, tag or PyPI
publication exists.
