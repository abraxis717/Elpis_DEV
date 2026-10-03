# ECS Runtime R1: measured results

PERFORMANCE_ONLY. NO_SCIENTIFIC_CLAIM. This report admits what the evidence in
`research/ecs_runtime_r1/evidence/` measured about how ECS_G executes. It says
nothing about cognition. Contract and gates:
[`docs/ECS_RUNTIME_R1.md`](../ECS_RUNTIME_R1.md) (gates registered at R1A
`3568a77`, before any measurement).

## Identity of what was measured

| | |
|---|---|
| measured head | `3afdae34` (R1G tooling), clean tree, tree `192386b8` |
| baseline head | `3568a77f` (R1A), clean tree, pre-nativeization binding |
| library | `libelpis_ecsg_math.so` SHA-256 `2374876e3a837761c87bdbc39389a2c5b2e007fcbaa81dd4b39e05deb1e5dc09` (baseline: `11fe7a56...`) |
| bench driver | SHA-256 `69320523...` |
| compiler, profile | gcc 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1), `Release`, `-O3 -DNDEBUG`; ECS_G `-ffp-contract=off`; hot kernels `target_clones("avx2","default")` |
| host | Intel Xeon @ 2.80 GHz, 4 vCPUs, AVX2/AVX-512/FMA available, Linux 6.18 (shared VM), glibc 2.39 |
| Python | CPython 3.11.15, ECS_G binding standard-library only |
| source digests | every ECS_G source, header, binding and build file, per evidence file (`binding.sources`); `tests/research/ecs_runtime_r1/test_evidence.py` fails if the measured code changes |

Sampling: per measurement an explicit warmup, then 20-2000 individually timed
samples within a 1.5 s budget; the first (cold) call is reported separately.
Native-direct numbers come from a C driver with no Python; Python numbers
time the public API in process.

## Gate verdicts

* `material_speedup`: PASS. New `CognitiveCore.learn` p50 is 3.83x (K=10),
  11.6x (K=100), 15.9x (K=1000) and 16.7x (K=4000) faster than the
  pre-nativeization baseline (threshold 2x for K >= 10).
* `python_overhead_k_independent`: FAIL. As registered (Python p50 minus
  native-direct p50, measured in separate processes) the audited
  `CognitiveCore.learn` shows 1.30 ms at K=4000 against a limit of 0.73 ms
  (2 x 316 us + 100 us); the other three Python modes pass. The estimator is
  not monotonic in K (316 us at K=10, 134 us at K=1000) and is negative for
  every non-receipt mode at K=1000 (-210 to -331 us): at 7-27 ms per call,
  drift between separately sampled processes exceeds the quantity measured.
  The registered verdict stands as FAIL.
* `python_overhead_split_timing`: PASS (supplementary, not a replacement).
  Measured inside each call as total time minus the time spent in the native
  learn call, every Python mode meets the registered rule: buffers 4.3 us at
  K=10 and 80 us at K=4000 (limit 109 us), lists without receipt 21 / 101 us
  (limit 142 us), `Executor.learn` 4.4 / 69 us (limit 109 us), audited receipt
  299 / 507 us (limit 698 us). The growth with K is not work: the identical
  K=1 call spends 2.8 us of Python before the native call back to back,
  10.3 us after a 1 ms idle gap and 73 us after a 27 ms idle gap (cold
  caches, predictors and clocks). Structurally the control plane is
  K-independent: one native learn call for every K
  (`test_learn_is_one_native_call_whatever_k`, identical call pattern for
  K=1 and K=4000) and no step loop in Python (`tests/boundary/test_runtime_r1.py`).
* `single_ffi`: PASS (tests). `no_hot_allocation`: PASS (allocation
  interposition on every allocator; zero allocations in query, learn,
  schedules, refusals and every transaction path). `scalar_parity`: PASS
  (bitwise; see Correctness).
* `sanitizers`: PASS. gcc ASan+UBSan and gcc TSan: all 7 ECS_G native tests;
  226 Python executor and differential tests under ASan.
* `jitter_investigation`: done; 11 short operations exceed p99/p50 = 3 (see
  Tail latency).

## R0 learning (d=6, N=36, R=64): p50 / p95 / p99

Microseconds unless marked ms. "Baseline" is the pre-nativeization code at
`3568a77`; the scalar reference C is unchanged, so its rows agree.

| K | direct scalar C (reference) | old Python binding (`CognitiveCore.learn`) | new executor, direct C | new `CognitiveCore.learn` (receipt) | new, `receipt=False` | new, buffers |
|---|---|---|---|---|---|---|
| 1 | 24.8 / 45.0 / 75.4 | 407.1 / 537.0 / 793.3 | 6.7 / 12.3 / 30.9 | 310.1 / 393.1 / 502.8 | 28.3 / 45.0 / 66.8 | 11.9 / 16.6 / 35.4 |
| 10 | 252.2 / 410.8 / 456.2 | 1,457.1 / 1,823.4 / 2,797.5 | 64.5 / 82.9 / 95.0 | 380.0 / 588.7 / 813.3 | 86.1 / 114.9 / 125.7 | 68.9 / 96.2 / 100.9 |
| 100 | 2,577.2 / 3,861.3 / 4,797.3 | 11.65 ms / 20.14 ms / 20.82 ms | 664.6 / 1,159.2 / 1,273.4 | 1,001.6 / 1,703.0 / 2,676.5 | 695.2 / 1,080.9 / 1,408.1 | 678.5 / 781.6 / 1,014.9 |
| 1000 | 26.25 / 39.70 / 48.06 ms | 113.72 / 116.78 / 117.17 ms | 7.01 / 11.70 / 13.23 ms | 7.15 / 10.48 / 11.44 ms | 6.80 / 8.10 / 9.29 ms | 6.79 / 12.48 / 12.88 ms |
| 4000 | 103.15 / 115.70 / 129.45 ms | 471.38 / 533.43 / 534.40 ms | 26.99 / 38.42 / 47.03 ms | 28.29 / 37.39 / 53.29 ms | 26.98 / 37.50 / 59.65 ms | 26.96 / 42.40 / 49.86 ms |

Speedups at p50 (baseline `CognitiveCore.learn` / new): receipt 1.31x, 3.83x,
11.6x, 15.9x, 16.7x for K = 1, 10, 100, 1000, 4000; `receipt=False` 14.4x to
17.5x; buffers 17.2x to 34.2x. Native executor vs reference loop order: 3.7x to
3.9x. Throughput: about 148,000 G1 steps/s through the executor (native or
Python, K=4000) against 8,500 steps/s through the old binding. A native
transaction (`exec_txn`: begin, learn, commit) costs the same as a direct
learn (6.7 us at K=1). The audited receipt costs about 0.3 ms per call
(0.5 ms right after a long native call, on cold caches) and does not grow
with the work: a JSON experience digest and two snapshot identities.

## R0 query: p50 / p95 / p99 (us)

| rows | direct scalar C | old binding (`query`, lists) | new executor, direct C | new `query_into` (buffers) | new `query` (lists) |
|---|---|---|---|---|---|
| 64 | 9.4 / 9.5 / 28.3 | 87.6 / 182.8 / 226.7 | 3.2 / 3.2 / 6.0 | 6.7 / 8.7 / 14.9 | 21.6 / 29.0 / 53.4 |
| 256 | 36.5 / 53.5 / 68.0 | 368.5 / 468.3 / 626.0 | 12.6 / 20.8 / 39.1 | 16.3 / 22.3 / 46.0 | 73.6 / 152.4 / 192.9 |

Query throughput (R=256): 20.3 M rows/s native, 15.7 M rows/s through
`query_into`, 0.69 M rows/s through the old binding. A query is one native
call, read-only (no snapshot, hash, JSON or receipt).

## Cold versus warm

First call / warm p50 (us): executor create+destroy 17.7 / 0.40 (reference
state 3.4 / 0.26); executor forward 11.8 / 3.2; executor learn K=1 22.0 / 6.7;
`query_into` 26.6 / 6.7; `query` 58.4 / 21.6; `CognitiveCore.learn` K=1 with
receipt 669 / 310 (baseline 616 / 407); snapshot 47.3 / 1.4; restore 20.0 /
6.7; Python transaction begin+learn+commit 57.5 / 15.5 (baseline Python
fork+adopt 85.4 / 32.4). Cold costs are first-touch page faults and cold
caches; none recurs.

## Scaling (PERFORMANCE_ONLY, NO_SCIENTIFIC_CLAIM; not supported cognition)

p50, d=6; learn workloads K=10.

| workload | ref C | old binding | executor C | new Python (buffers) | new `CognitiveCore` | native speedup | `CognitiveCore` speedup |
|---|---|---|---|---|---|---|---|
| query N=72 R=64 | 18.4 us | 98.3 us | 5.7 us | 9.2 us | 24.2 us | 3.21x | 4.07x |
| learn N=72 R=64 | 515.6 us | 1.70 ms | 115.7 us | 124.4 us | 451.6 us | 4.46x | 3.77x |
| query N=72 R=256 | 74.4 us | 404.9 us | 22.8 us | 26.3 us | 84.8 us | 3.27x | 4.78x |
| learn N=72 R=256 | 2.18 ms | 6.68 ms | 461.9 us | 480.3 us | 1.93 ms | 4.72x | 3.46x |
| query N=288 R=64 | 73.7 us | 157.1 us | 21.5 us | 25.1 us | 40.6 us | 3.43x | 3.87x |
| learn N=288 R=64 | 2.02 ms | 3.40 ms | 458.5 us | 466.6 us | 843.3 us | 4.40x | 4.03x |
| query N=288 R=256 | 298.3 us | 632.5 us | 85.6 us | 90.1 us | 153.7 us | 3.49x | 4.12x |
| learn N=288 R=256 | 8.96 ms | 14.00 ms | 1.78 ms | 1.81 ms | 3.20 ms | 5.03x | 4.37x |
| query N=1152 R=64 | 309.2 us | 410.3 us | 95.7 us | 99.0 us | 114.6 us | 3.23x | 3.58x |
| learn N=1152 R=64 | 8.79 ms | 11.23 ms | 2.03 ms | 2.07 ms | 3.04 ms | 4.32x | 3.70x |
| query N=1152 R=256 | 1.22 ms | 1.59 ms | 395.3 us | 402.6 us | 473.2 us | 3.08x | 3.37x |
| learn N=1152 R=256 | 40.36 ms | 48.33 ms | 8.35 ms | 8.60 ms | 10.28 ms | 4.84x | 4.70x |
| query N=4608 R=64 | 1.26 ms | 1.38 ms | 457.9 us | 450.1 us | 469.6 us | 2.75x | 2.95x |
| learn N=4608 R=64 | 38.20 ms | 44.26 ms | 10.05 ms | 9.91 ms | 12.36 ms | 3.80x | 3.58x |
| query N=4608 R=256 | 5.07 ms | 5.53 ms | 1.78 ms | 1.79 ms | 1.88 ms | 2.86x | 2.95x |
| learn N=4608 R=256 | 224.28 ms | 316.83 ms | 39.76 ms | 40.59 ms | 50.84 ms | 5.64x | 6.23x |

At large shapes the Python share vanishes and the native kernel is the
whole cost; at R0 shapes the old binding's per-step Python dominated.

## Memory

Executor workspace (one arena, allocated at creation; layout and ownership
in [`docs/ECS_RUNTIME_R1.md`](../ECS_RUNTIME_R1.md#memory-ownership-the-fms-boundary)):
31,744 bytes for R0 with capacity 64 and 99,328 bytes at the Python default
capacity 256; 3.5 MB at N=4608 with capacity 64 and 10.6 MB at capacity
256. It grows only by an explicit `reserve`. The old binding allocated per
Python step a scratch buffer of (RN + R + dN) doubles (20,672 bytes at R0,
9.7 MB at N=4608, R=256) and, per learn, a forked native state; the executor
allocates nothing after creation. Learned state is `8 dN` bytes plus the
epoch (1,728 bytes at R0).

## Native profile (callgrind, R0, K=100)

Before (R1C/R1D executor, reference loop order): 314,448 instructions per
step, dominated by the scalar z loop over `a`, the gradient loop that
recomputed `phi'(z)` once per input dimension and walked z with stride N,
and loop control. After (R1F): 69,348 instructions per step (4.53x fewer),
`g1_step.avx2` 89% and `row_sums.avx2` 6% of instructions. The rewrite keeps
every per-element operation and order (z over a ascending, sequential row
sums, gradient over r ascending; 8 row sums run as independent chains),
compiles without FMA (`-ffp-contract=off`, checked in both gcc and clang
builds) and selects AVX2 or baseline code at load time. Evidence:
`evidence/profile.json`.

## Tail latency

Operations above p99/p50 = 3 in `final.json`: all are shorter than 100 us,
and they include the unchanged scalar reference itself (`ref_forward` 3.01,
`ref_learn` K=1 3.04); the largest is executor learn K=1 (6.7 us p50, 30.9 us
p99, 4.64). Investigation (`evidence/jitter.json`):

* a calibration loop of matched duration with no memory traffic stays at
  p99/p50 1.0-1.6, while memory-touching operations of the same length reach
  2-3.7: the tail comes from the shared host (cache and memory interference,
  interrupts, vCPU scheduling), not from executor control flow, which does
  no data-dependent work beyond its refusal checks and allocates nothing;
* pinning to one CPU halves the K=1 executor tail (3.65 to 1.87): part of
  the tail is migration;
* long operations (K=4000, 27 ms) have tails of 1.1-1.9, like their
  calibration;
* disabling Python's cyclic GC does not reduce Python tails (3.37 vs 6.80,
  2.14 vs 2.89): GC is not a source; the measured buffer path creates no
  per-element Python objects.

## Correctness

* Bitwise parity: the executor's forward, K-step learn, schedules and
  transactions equal the scalar reference byte for byte over 240 C
  shape/rate combinations (dispatched AVX2 and baseline-ISA builds), 160 seeded random Python cases
  with refusals where the reference refuses (10 of 160), and 40
  schedule/transaction cases; snapshots are byte-identical.
* Cognitive R0: all 8 QUAL worlds reproduce their recorded measurements
  exactly on the executor core (was 2 of 8 checked before R1).
* Failure atomicity after at least one step, epoch +K, generation +1, read-only
  query, staleness, bounds (overflow, capacity, step count, rate, non-finite
  input, epoch headroom, destroyed handles, BUSY) are tested natively and
  through Python.

## Limits of these results

* One shared virtual machine, one compiler, one ISA family. Absolute numbers
  do not transfer; ratios against the reference measured in the same run are
  the meaningful figures. The AVX2 clone exists for x86-64 only; elsewhere the
  baseline code runs (same results).
* The registered overhead estimator failed as registered; its failure is
  explained above and not reinterpreted as a pass.
* SINGLE_WRITER only: overlapping calls are refused, never serialized.
* FMS does not yet own mutable ECS state; retention under sequential
  learning, the ECS<->DSV semantic codec and HACF->ECS remain missing
  (`docs/COGNITION_R0.md`). Nothing here is a cognition claim.
