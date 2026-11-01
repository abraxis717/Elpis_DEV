# VOLATILE_RUNTIME_V1: zero persistent writes on the composed runtime

Classification:
- QUALIFICATION evidence for closure condition 11 of the volatile-native migration (`docs/ELPIS_MISSION.md`,
  Amendment 3; `src/elpis/runtime/persistence.py`, `CLOSURE`).
- It extends `research/volatile_runtime_v0`, whose recorded evidence (`ZERO_WRITE_PASS` at `e61d5dd`) covered the
  volatile ECS_C session's own test paths only. That evidence is unchanged. This harness imports V0's tracer,
  detector and confinement from `research/volatile_runtime_v0/zero_write.py` and does not modify them.

## Protocol (declared before the recorded run)

The harness `zero_write_composed.py` prepares every input in a work tree before any traced run:
- the session scripts;
- an immutable COLD asset (mode 0444);
- a retained H-gram file (mode 0400);
- library copies with one missing and one drifted;
- a fake `/proc/swaps`;
- the write-probe targets.

After that the traced processes write nothing at all.

| Path | What it exercises |
|---|---|
| `host_session` | The native host, the production entry point: RuntimeCore ABI v8 over the volatile ECS_C session, ECS_G and ECS_C composed (the ECS composite); sources and operator events; typed HACF evidence and LOOKUP; managed numeric LEARN and QUERY; in-memory refusals (fuel, a negative rate, two invalid typed relationships, a dormant source's evidence, a stale source coordinate); the retired commands; `close` and a fresh session in the same process |
| `host_long` | A randomized 3,000-operation session (seed 11); `host_long_short` is the same law for 300 operations, for boundedness |
| `host_restarts` | 25 plain and 3 traced restarts: every process starts at the lineage's one initial identity, and a closed session is discarded |
| `host_differential`, `python_cli` | The same differential script through the host and through the Python reference interface, compared answer by answer (identities, generations, epochs, query digests, LOOKUP answers, refusal codes) |
| `python_fms_hacf` | The Python reference in one process (`python_fms_hacf.py`): the process gate; an FMS-resident K1 lineage under RuntimeCore on a RAM-only FMS context at its bound (a newcomer refused `CAPACITY`, never spilled); 12 managed numeric LEARN/QUERY; a fuel refusal; two sessions in a row (the second starts at genesis); three volatile HACF retrieval epochs with query ingress whose state root is never created; the retired operations absent; the retired `cold_root` parameter refused |
| `fms_cold` | `test_fms_volatile_cold --asset`: frozen COLD → WARM through the read-only file service, WARM → HOT, a volatile HOT mutation, dematerialization, reload by SHA-256, repetition, concurrent references, budget exhaustion without spill, a failed materialization, an unavailable resource, HOT exhaustion, every legacy COLD writer refused (the writable descriptor it refuses is the harness's) |
| `hgram_read` | A retained H-gram, a read-only static resource: probe and three association reads |
| `runtime_core.*` | RuntimeCore's C suites over real native K1 and the composite: session, composite, ABI, lifecycle, hot I/O |
| `ecs_c_session.*` | The session's C ABI and the relational view over it (V0's P1 and P2, rerun) |

Every path runs in three modes:
- **plain**: exit status, wall time, and peak RSS (V0's `peak_rss` launcher).
- **observed**: under `strace -f -y` with every write permitted. The C suites and the Python reference run as root.
  The host and the H-gram reader run as uid 65534, because they refuse a privileged process or a foreign owner.
- **confined**: V0's confinement under `strace -f -y`:
  - a private mount namespace with every mount but `/proc` read-only;
  - uid 65534 with no capabilities;
  - `elpis_autonomous_run` with no grant;
  - `RLIMIT_CORE` 0.

The traced system calls are V0's. A write-class event on storage is counted whether it was attempted or succeeded.

**Pass criteria, all required (`ZERO_WRITE_PASS`):**
1. Every path exits 0 in every mode, with **zero** write-class events in the observed and confined modes.
2. The composed session answers as expected: LOOKUP finds in the session and not after `close`; a fresh session
   starts at the initial identity; containment is established; the host reports persistence `NONE`.
3. The Python reference answers as the host does on the differential script.
4. Every restart starts at the one initial identity. Beyond the extra bytes of its script, which the host holds in
   memory, the long session's peak RSS grows by at most 512 KiB.
5. The H-gram reads return exactly the records the harness made.
6. Every adversarial run ends as predicted with zero write events.
7. The negative controls behave as predicted.
8. No binary the runtime loads imports a write-class C library function.
9. Every input's SHA-256 is unchanged after all runs, and the work tree is exactly as made.

Reproduce, as root on Linux x86-64 with strace, unshare, setpriv, prlimit, a C compiler and Landlock ABI ≥ 5:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
    python3 -I -B research/volatile_runtime_v1/zero_write_composed.py --repo . --build build \
        --out research/volatile_runtime_v1/results

## Results (recorded run at `66a11a4`, clean tree; `results/`)

**Verdict: `ZERO_WRITE_PASS`.** Every pass criterion holds. Host: Linux 6.18 x86-64, ext4, Landlock ABI 7, no swap,
`core_pattern` `core`.

An earlier run at `0d6d16f` was also `ZERO_WRITE_PASS`, with the same traced calls and zero write events. This run
replaces it because a fix to FMS (`66a11a4`: an object another holder has pinned or leased never moves) landed after
it. `66a11a4`'s `native/` and `src/` are the code that ships, except the census flag for this condition.

| Path | plain: exit, peak RSS, s | observed: exit, **write events** (traced calls) | confined: exit, **write events** (traced calls) |
|---|---|---|---|
| host_session | 0, 3,792 KiB, 0.028 | 0, **0** (140) | 0, **0** (151) |
| host_long (3,000 ops) | 0, 4,620 KiB, 2.655 | 0, **0** (3,114) | 0, **0** (3,125) |
| host_long_short (300 ops) | 0, 3,920 KiB, 0.272 | - | - |
| host_differential | 0, 3,536 KiB, 0.026 | - | - |
| python_cli | 0, 27,108 KiB, 0.202 | 0, **0** (361) | 0, **0** (396) |
| python_fms_hacf | 0, 28,212 KiB, 0.192 | 0, **0** (329) | 0, **0** (366) |
| fms_cold | 0, 2,464 KiB, 0.092 | 0, **0** (39) | 0, **0** (76) |
| hgram_read | 0, 2,256 KiB, 0.007 | 0, **0** (45) | 0, **0** (56) |
| runtime_core.session | 0, 2,732 KiB, 0.009 | 0, **0** (68) | 0, **0** (105) |
| runtime_core.composite | 0, 3,284 KiB, 0.022 | 0, **0** (73) | 0, **0** (110) |
| runtime_core.abi | 0, 2,468 KiB, 0.007 | 0, **0** (41) | 0, **0** (78) |
| runtime_core.lifecycle | 0, 2,600 KiB, 0.022 | 0, **0** (47) | 0, **0** (84) |
| runtime_core.hot_io | 0, 2,868 KiB, 0.018 | 0, **0** (48) | 0, **0** (85) |
| ecs_c_session.abi | 0, 1,896 KiB, 0.535 | 0, **0** (20) | 0, **0** (57) |
| ecs_c_session.relational | 0, 1,924 KiB, 0.005 | 0, **0** (36) | 0, **0** (73) |

The traced calls were:
- read-only `openat` of libraries, scripts, assets and system files;
- private or anonymous `mmap`s;
- `write`s to pipes, and to memfds while sealing admitted libraries.

**Restarts:** all 28 processes exited 0. Every one started at the one initial identity, and every closed session was
discarded. The 3 traced restarts recorded 0 write events.

**Boundedness:**
- The composite envelope is fixed at 137,960 bytes; the session capacity (16) and the memory's slot capacity (64)
  are fixed at creation.
- From 300 to 3,000 operations, peak RSS went from 3,920 to 4,620 KiB. The script grew by 1,050 KiB, so the state
  delta is −350 KiB.

**Adversarial runs** (all with **0** write events):

| Run | Mode | Exit | Outcome |
|---|---|---|---|
| standard output on storage | uid 65534 | 4 | refused, with **0 bytes** written there and the reason on standard error |
| an inherited writable descriptor | uid 65534 | 0 | closed by the host's containment; the session ran |
| swap active (the testing host's `/proc/swaps` override) | uid 65534 | 0 | `swap_areas` 1, `memory_locked` true |
| core limit unlimited:unlimited | raised as root, then uid 65534 | 0 | `core_limit` 0, `dumpable` false |
| 6 MiB address space | uid 65534 | 4 | refused at admission (`INSTALL_PIN`): no signal |
| 10 and 14 MiB address space | uid 65534 | 0, 0 | the composed session completed within the limit |
| a privileged start | root | 4 | `CONTAINMENT_REFUSED` |
| a missing library / a drifted library | uid 65534 | 4 / 4 | `INSTALL_PIN` |
| `init`, `publish`, `reconcile`, `--home` | uid 65534 | 2 each | `OPERATION_RETIRED` |

**Negative controls** (separate from the legitimate runs; all as predicted):
- **root, unconfined:** creation, write, rename, `sync` and removal all succeed, and all 5 are detected.
- **uid 65534 alone:** the write into its own directory succeeds and is detected.
- **Landlock alone:** the write is refused, and the attempt is detected.
- **Full confinement:** the write is refused `EROFS`, and the attempt is detected.
- **The testing host's deliberate write probes under confinement:** every attempt is refused and all 6 are detected
  as attempts:
  - create, write and extend: `EROFS`;
  - remove, rename and mkdir: `EPERM`, from the host's seccomp filter.

  The probe directory is unchanged.

**Static:**
- The production host and the 13 libraries the runtime loads import no write-class C library function. Those are the
  six admitted libraries, the session, the FMS adapters and file-asset library, the retrieval and ingress bridges and
  the execution budget.
- The 45 inputs' SHA-256 are unchanged after all runs: binaries, libraries, the COLD asset, the H-gram, the library
  copies and the scripts.

## What this does not show

- **Host services outside Elpis's process-control boundary:**
  - swap configuration, crash collectors, journald, backups;
  - filesystem metadata such as `atime`: the test mounts are read-only in the confined mode; elsewhere a
    filesystem's own access-time policy is the host's.

  Process controls cannot stop them, and the census, section 6, lists what the host must configure. Nothing is
  claimed about how much the operating system writes.
- **Inputs beyond the exercised ones.** Tracing shows what these runs did, not every possible input. Three things
  bound the rest:
  - the static properties: the import gate, the census's code bindings, and the absence of every retired entry
    point;
  - the native host's containment: Landlock with no write right, seccomp, and no writable descriptor;
  - the Python process gate.
- **Power loss or kernel crashes.** There is nothing to recover by design: a restart is a new session.
