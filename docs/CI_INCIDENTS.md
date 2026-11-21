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

## CI-RUNNER-IMAGE-1 — `ubuntu-latest` became Ubuntu 26.04 under the native jobs

**Status:** repair applied: the jobs that build or run native code are pinned to `ubuntu-24.04`, the image their
toolchain was qualified on. Classified CI-ENVIRONMENT DEFECT. Whether the native code is correct under GCC 15.2 and
clang 21.1 is **not established** (below).

### Observations

Each job log records its image (`Runner Image`). `src/` and `native/` are byte-identical in every head below; the
workflow differs between them only in the Elpis Torch research job.

| run | head | image (in every job log checked) | verdict |
|---|---|---|---|
| 474 (44346508710), 16:22 UTC 18 Nov | `93b0db6` | ubuntu-24.04 `20261111.127.1`, `20261111.370.1` | 16 of 16 green |
| 475 (44348237360), 16:31 UTC | `7106981` (adds one JSON record) | mixed | 12 green, 4 red. Native (gcc, Release) and Native (clang, Release) green on ubuntu-24.04; Python 3.11 and Native (clang, Debug) red on ubuntu-26.04 `20261111.196.1` |
| 478 (44358066804) | `bfe4a12` (`main`, after #76) | ubuntu-26.04 `20261111.196.1` | 10 red |
| 480 (44383238228), 481 (44383589088) | `9338e9d` (R8, #78) | ubuntu-26.04 `20261111.196.1` | the same 10 red |

The 10 red jobs are Native (gcc, Release), Native (clang, Release), Native (clang, Debug) and the seven jobs whose first
step is the Release native build: Boundary, Python 3.11, Python 3.12, Python stress and historical, Python without
network, Installed system acceptance and Scientific authority. `NATIVE_DEPS` installs the image's distribution
toolchain:

| | ubuntu-24.04 (24.04.5) | ubuntu-26.04 (26.04.1) |
|---|---|---|
| GCC | 13.3.0 | 15.2.0 |
| clang | 18.1.3 | 21.1.8 (package `1:21.1.6-71`) |
| CMake | 3.28.3 | 4.2.3 |
| Landlock ABI reported by the native host | not printed (its smoke test passes) | 8 |

### Mechanism (what is evidenced)

* **GCC 15.2, Release.** `ELPIS_WARNINGS_AS_ERRORS` is ON by default (`CMakeLists.txt`).
  * GCC 15.2 reports four `-Wmaybe-uninitialized` errors in `native/pipeline/ingress/src/regex_hacf_query_ingress.cpp`. They sit inside libstdc++'s `<variant>` move, for the `std::nullptr_t` and `bool` alternatives of `detail::J`, inlined from the two moves of a `detail::J::O` into a `detail::J` in `build_context_proposal` (lines 269 and 284).
  * The target `elpis_query_ingress` fails, and so does every job whose first step is the Release build.
  * The Debug builds (Native gcc Debug, ASan+UBSan, TSan) compile the file without the diagnostic.

  Run 481, Native (gcc, Release):

  ```
  -- The CXX compiler identification is GNU 15.2.0
  /usr/include/c++/15/variant:229:11: error: '*(std::nullptr_t*)((char*)&<unnamed> + offsetof(elpis::regex_hacf_query_ingress::detail::J, [...]::_M_u))' may be used uninitialized [-Werror=maybe-uninitialized]
  [...]/native/pipeline/ingress/src/regex_hacf_query_ingress.cpp:284:55: note: '<anonymous>' declared here
    284 |             hit_rows.push_back(detail::J(std::move(ho)));
  cc1plus: all warnings being treated as errors
  gmake[2]: *** [native/pipeline/CMakeFiles/elpis_query_ingress.dir/build.make:79: native/pipeline/CMakeFiles/elpis_query_ingress.dir/ingress/src/regex_hacf_query_ingress.cpp.o] Error 1
  ```

* **clang 21.1, Release and Debug.**
  * The build succeeds, and 185 of 186 ctest tests pass. `runtime.native_host_smoke` fails.
  * The host establishes its containment (Landlock ABI 8, no write grant). Every ECS operation then fails with `ECS_STATE`, and the host exits 3.

  Run 481, Native (clang, Release):

  ```
  -- The CXX compiler identification is Clang 21.1.8
  152/186 Test #152: runtime.native_host_smoke ........................***Failed    0.07 sec
  CMake Error at [...]/native/runtime/host/tests/host_smoke.cmake:42 (message):
    [...]/build/native/runtime/elpis;--native-build;[...]/build;--catalog-sha256;def766f8[...];--seed;7;session;--script;-:
    exit 3 (expected 0)
    {"op":"status", [...] "containment":{"restricted":true,"landlock_abi":8,"write_grants":0,"core_limit":0,"dumpable":false,"fsize_limit":0,"swap_areas":1,"memory_locked":true}, [...]}
    {"op":"source_found","ok":false,"code":"ECS_STATE","detail":"the lineage's initial composite could not be built"}
    {"op":"learn","ok":false,"code":"ECS_STATE","detail":"the lineage's initial composite could not be built"}
  99% tests passed, 1 tests failed out of 186
  ```

  * On the same image, the GCC 15.2 Debug build passes the same test (run 481, Native (gcc, Debug): `Test #152: runtime.native_host_smoke ... Passed`, `100% tests passed out of 186`), and so do the GCC sanitizer builds.
  * On ubuntu-24.04 the clang 18.1 builds pass it (run 474; run 475, Native (clang, Release)).
* In run 475 the verdict follows the image job by job, on one commit.

### Classification

CI-ENVIRONMENT DEFECT (`docs/CI_POLICY.md`, "Repairing a red line"):

* The jobs' runner image changed under an unchanged native tree, and with it their compilers, CMake and kernel.
* It is the case rule 3 names: one tree with different verdicts, environment-sensitive. The deciding environment is identified. Every job checked on ubuntu-24.04 passed, and every failing job checked ran on ubuntu-26.04.

### What the evidence does not establish

* **The GCC 15.2 diagnostic.** Whether it is a false positive of the `std::variant` move (the moved alternative is a `detail::J::O`, not the flagged `std::nullptr_t` or `bool`), or a real read of uninitialized storage. That question is work in `native/` and was not taken up here.
* **The clang 21.1 host-smoke failure.** Whether it comes from the compiler, from the Landlock ABI 8 kernel, or from both. The GCC Debug build passing on the same kernel shows that ABI 8 alone does not fail it. A local check could not separate the two: the kernel available to it reports Landlock ABI 7.
* **ubuntu-26.04 is therefore not qualified.** The native code is not qualified on ubuntu-26.04 (GCC 15.2, clang 21.1, Landlock ABI 8). The repair restores the qualified environment; it does not qualify the new one.

### Repair

`.github/workflows/ci.yml`: every job that compiles or executes the repository's native code runs on `ubuntu-24.04`:

* Boundary;
* Python 3.11 and 3.12;
* Python stress and historical;
* Python without network;
* Installed system acceptance;
* Scientific authority;
* Continuity and RuntimeCore (Rust);
* the four Native jobs;
* Native ASan+UBSan and Native TSan;
* Elpis Torch research.

They are pinned together, so that one run never qualifies the native code under two toolchains. Four of these jobs were green on ubuntu-26.04 and would otherwise have stayed there, building with GCC 15: Native (gcc, Debug), the two sanitizer jobs, and the research job, which builds the ECS_G kernel. Donor parity runs no native code and keeps `ubuntu-latest`. `NATIVE_DEPS` is unchanged; on ubuntu-24.04 it installs the toolchain above.

Not done, by policy:

* No source, test, build flag or warning setting changed; `ELPIS_WARNINGS_AS_ERRORS` stays ON.
* No test was excluded or skipped.
* No containment check was relaxed.
* No job was re-run until green.

### Closure condition

* **Acceptance.** The repair is accepted only with every job green on its exact head commit, in both the push and the pull-request run (`docs/CI_POLICY.md` rules 1, 2). The logs must show `Image: ubuntu-24.04` and the toolchain above.
* **Duration.** The pin lasts only as long as GitHub provides ubuntu-24.04.
* **Qualifying ubuntu-26.04.** This needs the two open questions above answered in `native/`, under the owner's authorization.
* **Reopening.** A pinned job failing for a toolchain or image reason reopens this incident.
