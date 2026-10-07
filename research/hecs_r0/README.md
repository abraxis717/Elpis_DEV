# H-ECS R0 laboratory (`research/hecs_r0`)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `PREREGISTERED`

A hierarchy of canonical ECS states (one native K1 state per level, through the existing C ABI) at distinct
temporal scales, with and without distinct learned representations between levels, compared with width
scaling of one ECS. Question, design, gates and dispositions: [`PREREGISTRATION.md`](PREREGISTRATION.md).
Results: [`docs/research/HECS_R0_RESULTS.md`](../../docs/research/HECS_R0_RESULTS.md).

**Status: closed `TASK_INVALID_ON_DEV`** at DEV calibration (the preregistered V1 threshold lies below the
SEPARATED world's achievable one-step error). No DEV run or QUAL run exists; R0 is not rescued.

Not on the canonical path: nothing in `src/` imports or calls it, and it has no ECS<->DSV codec, no HACF
edge and no runtime authority.

## Layout

| Path | Content |
|---|---|
| `rust/` | the `hecs_r0` crate (standard library only): hierarchy spec and bounded controller, encoders, K1 binding, world, planner, measurements, gates |
| `specs/hecs-r0.v1.spec.json` | the frozen specification (`hecs spec` must reproduce it byte for byte) |
| `evidence/dev/` | DEV calibration (the study stopped there) |

## Test lanes

| Lane | Command | Bound |
|---|---|---|
| FAST MECHANICS / NATIVE-RUST | `ctest -L hecs` (runs `cargo test` against the built K1 library) | seconds |
| H-ECS SCIENCE (deliberate only) | `hecs calibrate`, `hecs run --phase dev`, `hecs run --phase qual` (below) | minutes |
| Evidence guards | `pytest tests/research/hecs_r0` | seconds |

The science commands never run from `ctest` or `pytest`.

```bash
K1=build/native/ECS/libelpis_ecsg_k1.so
cd research/hecs_r0/rust
cargo run --release --offline -- calibrate --k1 "$PWD/../../../$K1" --out ../evidence/dev/hecs-r0.v1.calibration.json
cargo run --release --offline -- run --phase dev  --steps N --k1 "$PWD/../../../$K1" --out ../evidence/dev/hecs-r0.v1.dev.json
cargo run --release --offline -- run --phase qual --steps N --k1 "$PWD/../../../$K1" --out ../evidence/qual/hecs-r0.v1.qual.json
```

Progress is printed to stderr per world, seed and condition.
