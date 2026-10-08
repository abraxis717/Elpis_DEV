# H-ECS R2 laboratory (`research/hecs_r2`)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `PREREGISTERED`

A fresh experiment after H-ECS R0 (closed `TASK_INVALID_ON_DEV`) and R1 (closed `TASK_INVALID`), both never
edited. The primary question is whether the ECS/K1 world model, trained one step ahead, gives a multi-step
predictive object stable enough at the horizons the H-ECS planners consume. Hierarchy (width vs temporal-shared
vs distinct-representation levels) is adjudicated only if it does. Question, design, gates, stop laws and the
integration law: [`PREREGISTRATION.md`](PREREGISTRATION.md). Source equivalence with R1:
[`SOURCE_EQUIVALENCE.json`](SOURCE_EQUIVALENCE.json).

**Status: preregistered.** The DESIGN record is committed with the frozen specification; DEV and QUAL follow the
frozen law.

Not on the canonical path: nothing in `src/` imports or calls it, and it has no ECS<->DSV codec, no HACF edge
and no runtime authority.

## Layout

| Path | Content |
|---|---|
| `rust/` | the `hecs_r2` crate (standard library only): R1's laboratory plus `multistep.rs` |
| `specs/hecs-r2.v1.spec.json` | the frozen specification (`hecs2 spec` must reproduce it byte for byte) |
| `evidence/<phase>/RECORD.json` | the compact attestation of each phase (DESIGN, DEV, QUAL; write-once, under 64 KiB) |

## Test lanes

| Lane | Command | Bound |
|---|---|---|
| FAST MECHANICS / NATIVE-RUST | `ctest -L hecs` (`cargo test` against the built K1 library) | seconds |
| H-ECS SCIENCE (deliberate only) | `hecs2 design`, `hecs2 calibrate`, `hecs2 run --phase dev|qual` | tens of minutes |
| Evidence guards (SCIENTIFIC lane) | `pytest tests/research/hecs_r2` | seconds |

The science commands never run from `ctest` or `pytest`. The mechanics tests recompute every ECS-free
references block from its recorded seed and require its SHA-256 to match the phase record.

## Evidence policy

**Science-run output is raw telemetry and must be written outside the repository. It is not committed
evidence.** The committed evidence is one compact record per phase, `evidence/<phase>/RECORD.json`: the SHA-256
and byte count of the raw output, the execution source (with the SHA-256 of every `hecs2` source file) and engine
identities, the seeds, the SHA-256 of every ECS-free references block, and the decisive numbers from which
`tests/research/hecs_r2` recomputes the phase's decision under the frozen law. Those guards admit no other file
under `evidence/`, and `.gitignore` ignores the raw output names.

```bash
K1=$PWD/build/native/ECS/libelpis_ecsg_k1.so
: "${OUT:?set OUT to a directory outside the repository}"
cd research/hecs_r2/rust
cargo run --release --offline -- design    --k1 "$K1" --out "$OUT/hecs-r2.v1.design.json"
cargo run --release --offline -- calibrate --k1 "$K1" --out "$OUT/hecs-r2.v1.calibration.json"
cargo run --release --offline -- run --phase dev  --steps N --k1 "$K1" --out "$OUT/hecs-r2.v1.dev.json"
cargo run --release --offline -- run --phase qual --steps N --k1 "$K1" --out "$OUT/hecs-r2.v1.qual.json"
```
