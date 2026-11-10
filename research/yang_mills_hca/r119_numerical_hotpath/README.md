# R119 — numerical hot-path qualification of the sparse AWR

**Independent Elpis research. RESEARCH_ONLY. Not production-admitted.**
`R93_GATE=ENERGY_NORMALIZED_DENOMINATOR_HOLD` is unchanged; R118-E remains descriptive.
No AWR capacity was expanded and the learned HCA (six frozen R77/R78 checkpoints) is
unmodified. R118 (`../r118_scalable_awr/`) is frozen; its closure record is
`../r118_scalable_awr/evidence/CLOSURE.json`.

R118-D qualified the hot path for its registered workload but left one exception open:
the exact numerical path exceeded the 20 ms p95 target at 1,024 active HCAs in the
input-free decayed-state regime. R119 locates the floating-point operations responsible,
proposes an exact repair, and qualifies three numerical modes under a protocol registered
before measurement (`PROTOCOL_R119.json`, targets inherited unchanged: 2 / 5 / 20 ms).

Results and dispositions: [RESULTS.md](RESULTS.md). Mechanism and correctness arguments:
[ARCHITECTURE.md](ARCHITECTURE.md).

## Components

| File | Role |
|---|---|
| `forensics.py` | instrumented pure-Python replica of the exact kernel (bit-identical to native), per-site census of subnormal operations along the failure trajectory |
| `microbench.py`, `csrc/fp_microbench.c` | per-operation cost of subnormal operands and results on this CPU, default vs FTZ/DAZ |
| `csrc/awr_kernel_r119.c` | R118 kernel arithmetic plus four guarded exact fast paths |
| `guards.py` | exact rational computation of the gate-guard threshold θ |
| `engine_r119.py` | modes `EXACT_R118`, `EXACT_GUARDED`, `FTZ_DAZ` on the frozen R118 engine |
| `differential.py` | kernel-level bit-identity harness (R118 vs R119 guards off / on) |
| `qualify_r119.py` | executes the registered protocol and applies its decision rules |
| `native_r119.py` | builds C sources at runtime in volatile scratch (no binary is committed) |
| `tests/` | guard soundness, differential bit identity, engine-mode equivalence |

## Running (from the repository root)

```bash
python -B -m unittest discover -s research/yang_mills_hca/r119_numerical_hotpath/tests -t .
```

```bash
python -B -m research.yang_mills_hca.r119_numerical_hotpath.qualify_r119 --all --out /tmp/r119.json
```

```bash
python -B -m research.yang_mills_hca.r119_numerical_hotpath.microbench /tmp/r119_microbench.json
```

The numerical part of the qualification needs torch (the frozen oracle); latency parts
never import it. All scratch is in `/dev/shm`; nothing is written under the repository.

## Attribution

Upstream mathematical context: [creelie/4Y-M](https://github.com/creelie/4Y-M) by Deep
Bhattacharjee, Ushashi Bhattacharya, James Montgomery Lockwood and Tristan Romanov
(v4.21.0, MIT; see [../PROVENANCE.md](../PROVENANCE.md)). No upstream code is used; the
upstream authors did not develop, endorse or qualify any of this work.
