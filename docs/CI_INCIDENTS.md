# CI incidents

Authority: `docs/CI_POLICY.md`. Each entry classifies one red line, records the evidence and the repair, and states
what the evidence does not establish. Entries are append-only.

## CI-HECS-ASAN-1 — H-ECS Rust harness crashes under ASan

**Status:** repair applied (serialize the harness under ASan); root cause classified, the exact sanitizer-runtime race
**not identified** (reproducibility limitation below).

### Observations

| run | head | job | test | failure |
|---|---|---|---|---|
| 204 | `8561838` | Native ASan+UBSan | `research.hecs_r1_mechanics` | `AddressSanitizer: CHECK failed: sanitizer_thread_registry.cpp:316`, garbage thread status `0xa33e9500` |
| 205 | next head | Native ASan+UBSan | same H-ECS R1 sources | green |
| 215 (38451028810) | `91516e5` (main, post-merge) | Native ASan+UBSan | `research.hecs_r0_mechanics` | SIGSEGV, no sanitizer report, after three tests passed |

Every other job of run 215 that had finished was green, including all `research.ecs_mem*` tests under ASan. Run 201
(green) and run 215 used the same runner image (ubuntu-24.04 `20261004.327.1`); the hosts differed (Intel Xeon 8370C
vs AMD EPYC 7763) and run 215 was markedly slower (hecs_r2 89.8 s vs 50.0 s). The three H-ECS crates' CMake files
are identical up to names; none of their sources changed between green and red runs.

### Mechanism (what is evidenced)

* The H-ECS crates run an **uninstrumented** Rust libtest process with `libasan` preloaded by
  `native/cmake/sanitizer-runner.sh`; only the dlopened K1 library is instrumented.
* Local reproduction (ASan, gcc, Debug; same runner): 1 failure in 300 runs of the default parallel harness, an ASan
  `CHECK failed: ((i)) < ((size_))` in `ThreadRegistry::StartThread` with a garbage tid (`0x9a119e18`), raised in
  `asan_thread_start` — the moment a new test thread starts. Both CI signatures are in the same place: the sanitizer
  runtime's own thread registry, at thread creation, with corrupt bookkeeping values.
* No report in any run concerns K1 memory (no use-after-free, overflow or leak report against K1 or MF-R0 code).
* Refuted: a Rust/ASan alternate-signal-stack conflict. With `libasan` preloaded, ASan owns the SIGSEGV handler and
  installs its own alternate stacks (191 232 bytes); the Rust runtime installs none.
* libtest with `--test-threads=1` still runs each test on its own thread, but strictly one at a time (measured:
  at most one live test thread), so no test thread is created while another is alive.

### Classification

NONDETERMINISTIC / ENVIRONMENT-SENSITIVE (`docs/CI_POLICY.md` rule 3): a crash in the sanitizer runtime's thread
bookkeeping of an uninstrumented process with a preloaded ASan runtime, at overlapping test-thread lifetimes. It is
**not** classified as a K1, H-ECS or MF-R0 implementation defect, and **not** dismissed as harmless: it is a real
failure of a required job.

### Reproducibility limitation

After the single local reproduction, 0 failures were observed in 8 000 further runs (default parallel,
`--test-threads=1`, without K1, `use_sigaltstack=0`), 0 in 2 400 oversubscribed runs, and 0 in purpose-built C
reproducers (thread waves, a libtest-like overlapping pool with and without `dlopen`, late re-arming pthread-key
destructors). The local rate is below 1e-4 per run while the CI rate was roughly one red job in twenty runs. The
exact race inside `libasan` is therefore not identified, and the absence of failures with `--test-threads=1` is
**not** statistical proof that the repair removes it: the local baseline is too rare to discriminate.

### Repair

`research/hecs_r{0,1,2}/CMakeLists.txt`: under `ELPIS_ENABLE_ASAN` only, the cargo test command passes
`-- --test-threads=1`. This removes the one condition every observed crash shared (a test thread created while
others are alive). No test is excluded, no assertion changes, sanitizers stay enabled with leak detection and
abort-on-error, and the TSan configuration keeps the parallel harness so concurrent K1 use is still exercised where
races are detected. `tests/boundary/test_sanitizer_runners.py` keeps all three properties.

Not done, by policy: disabling a sanitizer, excluding an H-ECS test, re-running CI until green, editing frozen
evidence.

### Closure condition

The repair is accepted only with the required jobs green on its exact head commit (`docs/CI_POLICY.md` rules 1, 2).
A recurrence with `--test-threads=1` reopens this incident and refutes the overlapping-lifetime explanation.

## CI-HOTRES-1 — a released FMS-resident K1 state assumed to be the next eviction victim

**Status:** repair applied (test contract); classified TEST-CONTRACT DEFECT. The FMS residency implementation is
unchanged.

### Observations

| run | head | job | test | failure |
|---|---|---|---|---|
| PR #50 (38674420705) | `68e8bc3` | Python 3.11 | `tests/integration/test_hot_residency.py` | `assert before["tier"] == 2` (the released state was still WARM after four rounds of pressure), then `BUSY: K1 FMS runtime destroy` from the cleanup |
| post-merge main (38674556226) | `21562cd` | Python without network | same | same |

The same tree was green in the other Python jobs of both runs (Python 3.12, and Python 3.11 / 3.12 on main), and the
test passed 10/10 locally: one tree, different verdicts (`docs/CI_POLICY.md` rule 3).

### Mechanism (what is evidenced)

* FMS chooses a demotion victim among *demotable* objects only (`fms_core.c`, `demotable`: never a pinned one) by the
  largest `size / evict_cost_ns`, and `evict_cost_ns` weights the promotion cost by `p = tau / (tau + age)`: `age` is
  the time since the last access and `tau` the mean inter-access interval, both on the PAL's monotonic wall clock.
* `fms_release` sets `last_access_ns` (`fms_core.c`, `fms_release`), so releasing a state is an access: it makes the
  state *more* recent, never the next victim.
* The old test released a frequently accessed state and then expected one of up to four newcomers to demote it while
  another unpinned state was also WARM. Which of the two FMS picked depended on access counts and on milliseconds of
  wall-clock time. A diagnostic outside the repository (same FMS configuration, two unpinned WARM K1 states, varying
  only the access order, the access count and a 0 / 5 / 50 ms wait) demoted either state depending on those alone.
* The `BUSY` at destroy was a second test defect: the cleanup closed only the pressure states, so after the primary
  failure the main state was still registered and its report hid the first one.
* `native/runtime/tests/test_runtime_hot_io.c` had the same latent loop (it passed on CI timing).

### Classification

TEST-CONTRACT DEFECT: the test asked for a victim order FMS never promised. Not an FMS, K1 FMS adapter or RuntimeCore
defect; no source of either changed.

### Repair

* Every COLD precondition is made exact through FMS's one hard law, a pinned object is never demoted: every other WARM
  state is held (the adapter's residency hold) so that exactly one state is eligible, and only then a newcomer arrives.
  The Python binding gained the adapter's existing `resident_hold` / `resident_release` and its read-only `stats`
  (`K1FMSRuntime.metrics()`: FMS promotions, demotions, cold reads and writes), all ABI-v1 entry points already used
  by RuntimeCore.
* The invariants are separate tests (`test_hot_residency.py`): A (a bound state is never demoted and HOT work on it
  promotes nothing and reads no cold replica; with every other WARM state held the newcomer is refused rather than the
  bound state demoted), B (QUERY, LEARN and an identity-only durable request on a demoted state are refused
  `RUNTIME_NOT_RESIDENT` with FMS counters, the adapter's state record, holds, continuity and the durability lane all
  unchanged, and no fail-stop), C (only the explicit materialization reads the replica: exactly one promotion and one
  cold read; the released live lineage resumes at its generation). The fixture closes every state it registered, so a
  pass also proves no hold leaked and a failure keeps its own report.
* The native test establishes its preconditions the same way, adds real pressure while bound and the refused
  identity-only request with byte-exact continuity and lane status.
* Mutation sensitivity (rebuilt mutants, outside the repository): RuntimeCore binding without the hold, the adapter
  counting a hold without pinning it, a hold that promotes synchronously, NOT_RESIDENT ignored by RuntimeCore, a hold
  leaked by release, a refusal that fail-stops, a refused durable request that publishes the released identity, and a
  Python facade that materializes and retries: all eight killed by the corrected suite (the last but one only by B and
  the native test; the old test let it survive).

Not done, by policy: an `any(... == COLD)` relaxation, a mocked eviction, a retry loop, a sleep.

### Closure condition

The repair is accepted only with the required jobs green on its exact head commit (`docs/CI_POLICY.md` rules 1, 2).
