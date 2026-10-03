# ECS Runtime R1 benchmark laboratory

PERFORMANCE_ONLY. NO_SCIENTIFIC_CLAIM. RESEARCH_ONLY, NO_RUNTIME_AUTHORITY.

Measures how ECS_G executes, under the contract in
[`docs/ECS_RUNTIME_R1.md`](../../docs/ECS_RUNTIME_R1.md). Nothing measured here
is evidence about cognition, and no width in the scaling set is "supported
cognition".

* `workloads.py`: the fixed workloads, data regime, sampling policy and the
  pre-registered gates and thresholds (committed before any measurement).
* `native/ecsg_runtime_bench.c`: direct-native driver (no Python) for the ECS_G
  ABI; one timed sample per operation after an explicit warmup.
* `bench.py`: the harness. It runs the native driver and the Python binding on
  every workload and writes one evidence file that binds the git head and
  tree, the SHA-256 of every ECS_G source, header, binding and build file, the
  CMake configuration, the compiler and the exact shared-library SHA-256.
* `evidence/`: write-once evidence files (`open(..., "x")`).

```sh
cmake -S . -B build/r1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc
cmake --build build/r1 -j
PYTHONPATH=src python -m research.ecs_runtime_r1.bench --build build/r1 --phase baseline \
    --out research/ecs_runtime_r1/evidence/<name>.json
```

Each result reports the first (cold) call separately from steady state: min,
p50, p95, p99, max, mean and throughput (`1e9 / p50`). Timing on a shared or
virtualized host is noisy; tails are reported, not hidden.
