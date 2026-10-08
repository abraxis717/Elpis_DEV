# H-ECS R1 laboratory (`research/hecs_r1`)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `PREREGISTERED`

The successor of H-ECS R0 (closed `TASK_INVALID_ON_DEV`, never edited). The same question and mechanics, with a
valid task-validity law: an analytic floor, a held-out best-linear reference and ECS adequacy relative to that
reference, on fresh SHA-256-derived seeds. Question, design, gates, stop laws and the integration law:
[`PREREGISTRATION.md`](PREREGISTRATION.md). Source equivalence with R0: [`SOURCE_EQUIVALENCE.json`](SOURCE_EQUIVALENCE.json).

**Status: closed `TASK_INVALID` at QUAL** ([`docs/research/HECS_R1_RESULTS.md`](../../docs/research/HECS_R1_RESULTS.md)).
DEV calibrated at 6000 K1 steps and was valid; the one QUAL run failed V1C (ECS adequacy) in 1 of 16 instances.
The hypotheses were not adjudicated; no integration is authorized.

Not on the canonical path: nothing in `src/` imports or calls it, and it has no ECS<->DSV codec, no HACF edge
and no runtime authority.

## Layout

| Path | Content |
|---|---|
| `rust/` | the `hecs_r1` crate (standard library only), R0's laboratory plus `validity.rs` |
| `specs/hecs-r1.v1.spec.json` | the frozen specification (`hecs1 spec` must reproduce it byte for byte) |
| `evidence/design/` | DESIGN-seed validity diagnostics (no hypothesis quantity) that informed the margins |
| `evidence/dev/`, `evidence/qual/` | DEV calibration and run; QUAL run (write-once) |

## Test lanes

| Lane | Command | Bound |
|---|---|---|
| FAST MECHANICS / NATIVE-RUST | `ctest -L hecs` (`cargo test` against the built K1 library) | seconds |
| H-ECS SCIENCE (deliberate only) | `hecs1 design`, `hecs1 calibrate`, `hecs1 run --phase dev|qual` | minutes |
| Evidence guards (SCIENTIFIC lane) | `pytest tests/research/hecs_r1` | seconds |

The science commands never run from `ctest` or `pytest`.

```bash
K1=$PWD/build/native/ECS/libelpis_ecsg_k1.so
cd research/hecs_r1/rust
cargo run --release --offline -- calibrate --k1 "$K1" --out ../evidence/dev/hecs-r1.v1.calibration.json
cargo run --release --offline -- run --phase dev  --steps N --k1 "$K1" --out ../evidence/dev/hecs-r1.v1.dev.json
cargo run --release --offline -- run --phase qual --steps N --k1 "$K1" --out ../evidence/qual/hecs-r1.v1.qual.json
```
