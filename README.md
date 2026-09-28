# Elpis (development foundation)

Elpis is a research system organised around one idea, *geodesic intelligence*:
persistent structured state, memory topology, bounded transformations,
controlled evolution and inference should interact so that computation
follows structure the system has already accumulated. This is a research
direction, not a capability claim; see [`docs/NONCLAIMS.md`](docs/NONCLAIMS.md).

This repository is the clean development authority. It was built by a
selective, audited migration from the Elpis beta repository. The per-path
record is [`migration/BETA_MIGRATION.json`](migration/BETA_MIGRATION.json), and
the reasoning is in [`docs/MIGRATION.md`](docs/MIGRATION.md).

The single machine-readable system authority is
[`ELPIS_SYSTEM.json`](ELPIS_SYSTEM.json). The design is described in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Layout

| Path | Subsystem |
|---|---|
| `src/elpis/substrate`, `native/substrate` | memory and residency below every model (FMS, PAL, file assets) |
| `src/elpis/ecs` | identity, durable history, replay, projections |
| `src/elpis/structure`, `native/structure` | HACF structural memory, semantic core, retrieval, Grid81 representation |
| `src/elpis/pipeline`, `native/pipeline` | bounded Regex → HACF ingress; the canonical Grid81 writer path |
| `src/elpis/evolution` | deterministic heredity, selection, gated promotion |
| `src/elpis/inference` | inference behind contracts; the DSV4 driver on synthetic fixtures |
| `src/elpis/runtime` | the one runtime composition over one ECS receipt history |

## Build and test

The native build needs CMake, a C11/C++17 compiler, pkg-config, SQLite 3 and
PCRE2. On Manjaro: `sudo pacman -S --needed base-devel cmake pkgconf sqlite pcre2`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

python -m pip install ".[test]"
ELPIS_NATIVE_BUILD=$PWD/build ELPIS_REQUIRE_NATIVE=1 python -m pytest
```

Base installation depends only on NumPy. It never downloads model weights,
and importing `elpis` needs neither Torch nor NumPy. The distribution is
`elpis-dev`, marked `Private :: Do Not Upload`. No release, tag or PyPI
publication exists.
