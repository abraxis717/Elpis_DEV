# ECS native K1 runtime: performance

`PERFORMANCE_ONLY` · `NO_SCIENTIFIC_CLAIM`. Contract: [`docs/ECS_K1_RUNTIME.md`](../ECS_K1_RUNTIME.md).

No latency or throughput threshold was registered before these measurements, so every number here is descriptive.
The gates are structural and are enforced by tests on every CI configuration:

| gate | evidence |
|---|---|
| no allocation after create/reserve on any warm operation | link-time allocator interposition on every allocator (`test_ecsg_k1_alloc`, `test_ecsg_k1_fms_alloc`); the K1 heap counter is unchanged across every measured warm loop (`test_ecsg_k1_performance`) |
| one native call per operation, whatever K; none proportional to K | `tests/ECS/test_k1_runtime.py::test_one_native_crossing_per_operation_and_none_proportional_to_k` (K = 1, 10, 1000: exactly one `learn` crossing); `research/ecs_k1_native/overhead.py` reports one crossing per call |
| no transient executor or state per warm query | the warm path pins and binds the persistent workspace. Its heap counter does not move, and the adapter calls no executor API (`tests/boundary/test_k1_runtime.py`). |
| sanitizers | gcc ASan+UBSan and gcc TSan run the whole ECS native suite, including every K1 test |

## Measured on

- **Host:** Intel Xeon @ 2.80 GHz, 4 vCPUs, a shared VM.
- **Build:** gcc 13.3.0, `Release` (`-O3 -DNDEBUG`), K1 `-ffp-contract=off`, hot kernels
  `target_clones("avx2","default")`.
- **Regime:** `dim = 6`, `width = 36`, F = 83, 64 rows.
- **Sizes:** image 30,344 B, envelope 30,376 B, workspace 95,040 B.
- **Sampling:** 200 samples (50 for K = 100) after construction.

The sample counts are `ctest -R test_ecsg_k1_performance` defaults. `ELPIS_K1_PERF_RUNS` and `ELPIS_K1_PERF_FULL=1`
change them.

H is non-zero in every learning measurement, so every step applies the K1 correction.

### Native (C, no Python)

| operation | p50 | p95 | p99 | throughput |
|---|---|---|---|---|
| query, 64 rows | 5.1 µs | 6.1 µs | 10.0 µs | 186k ops/s |
| learn K = 1 | 23.4 µs | 32.2 µs | 58.9 µs | 42k steps/s |
| learn K = 10 | 239 µs | 337 µs | 530 µs | 42k steps/s |
| learn K = 100 | 1.57 ms | 3.29 ms | 3.65 ms | 51k steps/s |
| consolidate, 64 rows | 85 µs | 109 µs | 147 µs | 11.2k ops/s |
| transaction begin + learn K=10 + consolidate + commit | 247 µs | 376 µs | 426 µs | 3.8k txn/s |
| snapshot (encode + SHA-256) | 132 µs | 223 µs | 266 µs | 6.7k ops/s |
| restore (validate + allocate + decode) | 134 µs | 195 µs | 228 µs | 7.0k ops/s |
| FMS warm query | 3.5 µs | 3.6 µs | 7.4 µs | 275k ops/s |
| FMS warm learn K = 10 | 149 µs | 281 µs | 312 µs | 55k steps/s |
| FMS warm consolidate | 86 µs | 135 µs | 164 µs | 10.6k ops/s |
| FMS warm transaction cycle | 244 µs | 289 µs | 469 µs | 3.9k txn/s |
| FMS warm snapshot | 128 µs | 171 µs | 189 µs | 7.4k ops/s |
| FMS COLD→WARM + query | 139 µs | 168 µs | 199 µs | 6.9k ops/s |
| FMS WARM→COLD, cold replica valid | 0.37 µs | 0.68 µs | 1.6 µs | — |
| FMS WARM→COLD, dirty (write-back) | 1.34 ms | 2.17 ms | 3.47 ms | — |

**Reading the numbers:**

- A corrected K1 step costs about 20–24 µs at this shape.
- The qualified regime of 4000 steps per experience is therefore about 0.1 s per experience in one native call
  (`ELPIS_K1_PERF_FULL=1` measures it).
- The FMS warm path adds no measurable cost over standalone: a pin and an unpin.
- COLD→WARM is dominated by the cold read plus digest. A demotion with a valid replica only drops the RAM copy; a
  dirty demotion writes the replica back.

### Python control-plane overhead (split timing)

`research/ecs_k1_native/overhead.py` times each public call in total and, inside it, the one native call it makes.
Overhead is total minus native, at p50.

| call | total | native | overhead p50 | overhead p95 | native calls |
|---|---|---|---|---|---|
| `K1State.query` | 6.7 µs | 4.0 µs | 2.7 µs | 5.5 µs | 1 |
| `K1State.query_into` | 6.6 µs | 4.0 µs | 2.6 µs | 5.0 µs | 1 |
| `K1State.learn` K = 1 | 20.8 µs | 16.9 µs | 3.9 µs | 9.1 µs | 1 |
| `K1State.learn` K = 10 | 156 µs | 150 µs | 5.3 µs | 11.6 µs | 1 |
| `K1State.learn` K = 100 | 1.52 ms | 1.51 ms | 12.3 µs | 27.1 µs | 1 |
| `K1State.consolidate` | 91.7 µs | 87.9 µs | 3.5 µs | 8.7 µs | 1 |
| `K1FMSRuntime.query` | 7.5 µs | 4.7 µs | 2.9 µs | 4.6 µs | 1 |
| `K1FMSRuntime.learn` K = 10 | 153 µs | 148 µs | 4.7 µs | 12.2 µs | 1 |

The overhead grows slightly with K for the same reason Runtime R1 recorded: cache, predictor and clock state after
a long native call. It is not work. The call pattern is identical for every K, with one crossing and no Python loop.
