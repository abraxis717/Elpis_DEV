# VOLATILE_RUNTIME_V0: zero filesystem writes on the volatile ECS_C session's tested paths

Classification:
- DEV evidence for phases V1 and V2 of the volatile-native migration (`docs/ELPIS_MISSION.md`, Amendment 3;
  `docs/VOLATILE_RUNTIME_CENSUS.md`).
- **VOLATILE MIGRATION: INCOMPLETE.** This evidence covers the session's own test paths only. It is not evidence for
  any composed runtime path. RuntimeCore ABI v7, the native host and the Python reference interface still write
  persistent state (census, sections 3 and 4).

## Protocol (declared before the recorded run)

The harness `zero_write.py` runs built binaries:

| Path | Binary | What it exercises |
|---|---|---|
| P1 | `test_ecs_c_session_abi` | the session C ABI end to end: lifecycle, every refusal class (atomic), the retired kinds, foreign coordinates, capacity, receipts, a 100,000-event run and 1,000 create/destroy cycles with heap measured |
| P2 | `test_ecs_c_relational_session` | the relational view over the session and the real ECS_C memory |
| P3 | the session crate's release test binary (`ECS_C.cargo_test_session`, `--test-threads=1`) | 12 tests, among them the 36,000-event differential against the law, allocation-free transitions and a 100,000-event run |
| P4 | `nomem_probe` (`nomem_probe.c`, built by the harness) | memory exhaustion under `RLIMIT_AS`: `NOMEM` with nothing created, survivors still transition, reclaimed memory is reusable |

Each path runs in three modes:
- **plain**: exit status, wall time, and the command's peak RSS. `peak_rss.c`, a small launcher, measures it
  with `wait4`, because `ru_maxrss` also counts the image a process was forked from.
- **observed**: as root, where any write would be permitted, under `strace -f -y`. The claim is that no write is
  even attempted.
- **confined**: all of the following, under `strace -f -y`:
  - a private mount namespace with every mount except `/proc` remounted read-only; the topmost mount of every mount
    point is verified, and the harness fails closed otherwise;
  - uid/gid 65534 with no groups and no capabilities;
  - `elpis_autonomous_run` with no grant: Landlock denies write, create, remove, rename, truncate and device ioctl;
    seccomp denies metadata syscalls; descriptors ≥ 3 are closed;
  - `RLIMIT_CORE` 0 and an empty environment.

The tracer's log goes into a pipe, so no file is created even for the trace.

The traced system calls cover every way to write, create, delete or sync storage, including writable mappings:
- opens and `creat`;
- the `write` / `pwrite` family;
- `rename*`, `unlink*`, `mkdir*`, `rmdir`, `truncate`/`ftruncate`, `fallocate`;
- `fsync`, `fdatasync`, `sync`, `syncfs`, `sync_file_range`, `msync`;
- `link*`, `symlink*`, `mknod*`;
- `chmod*`, `chown*`, `utimensat`, xattrs;
- `mmap`, `copy_file_range`, `sendfile`, `splice`;
- `io_uring_setup` and the handle opens.

A **write-class event on storage** is any of these. Opens and writes count unless they target a pipe, socket,
terminal, `/dev/null`, an anonymous inode or a memfd, and `mmap` counts when it is shared, writable and file-backed.
Attempts count as well as successes.

**Pass criteria, all required (`ZERO_WRITE_PASS`):**
1. Every path exits 0 in all three modes.
2. Every observed and confined run has **zero** write-class events on storage.
3. The negative controls behave as predicted, so the detector is not blind and each containment layer is effective
   on its own:
   - **root unconfined:** creation, rename, `sync` and removal all succeed and are all detected;
   - **uid 65534 alone**, into a directory it owns: the write succeeds and is detected. The uid layer alone is not
     containment;
   - **Landlock alone:** the same write is refused `EACCES` and detected;
   - **full confinement:** the same write is refused `EROFS` and detected.
4. The session library imports nothing beyond `abort`, `free`, `getrandom`, `malloc`, `memcpy`, `memmove`,
   `memset`, `memcmp`, `bcmp` and `posix_memalign`, plus the weak C runtime symbols.

The results carry the source commit, environment, hashes of every binary and source, footprints, traced-call counts
and every write event. They contain no host-local path: the repository and build tree appear as `<repo>` and
`<build>`.

Reproduce, as root on Linux x86-64 with strace, unshare, setpriv, a C compiler and Landlock ABI ≥ 5:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
    ctest --test-dir build -R '^ECS_C\.cargo_test_session$'     # builds the P3 test binary
    python3 -I research/volatile_runtime_v0/zero_write.py --repo . --build build --out research/volatile_runtime_v0/results

## What this does not show

- Nothing about RuntimeCore, the native host, the Python runtime, FMS, HACF or any composed path. They still write
  (census, sections 3 and 4).
- Nothing about host services outside the process, such as swap, crash collectors, journald or backups. Process
  controls cannot stop them (census, section 6).
- Nothing beyond the exercised inputs. Tracing shows what these runs did, not what every possible input would do.
  The static properties (`no_std`, the import set, an ABI without paths) are what bound the latter.

## Results (recorded run at `e61d5dd`, clean tree; `results/`)

**Verdict: `ZERO_WRITE_PASS`.** Every pass criterion holds.

| Path | plain: exit, peak RSS, s | observed: exit, write events (traced calls) | confined: exit, write events (traced calls) |
|---|---|---|---|
| P1 session C ABI | 0, 1,812 KiB, 0.66 | 0, **0** (20) | 0, **0** (56) |
| P2 relational over the session | 0, 1,904 KiB, 0.005 | 0, **0** (36) | 0, **0** (72) |
| P3 cargo tests (12 passed) | 0, 3,588 KiB, 1.20 | 0, **0** (75) | 0, **0** (111) |
| P4 memory exhaustion | 0, 18,500 KiB, 0.19 | 0, **0** (101) | 0, **0** (137) |

What the traced calls were:
- read-only `openat` of libraries and system files, and `/proc/self/statm` in P4;
- private or anonymous `mmap`s;
- `write`s to the stdout pipe.

The confined runs trace more calls because `setpriv` and `elpis_autonomous_run` run under the tracer too.

**P4** (`nomem_probe`) ran with `RLIMIT_AS` set to its mapping plus 48 MiB:
- 26 sessions of capacity 4,096 (1,901,968 bytes each) were created;
- the 27th was refused `ECS_C_NOMEM` with nothing created and no abort;
- all 26 then transitioned, with a constant footprint;
- after 13 were destroyed, creation succeeded again.

Memory pressure caused no write-class call on any file in either mode.

**Negative controls**, all as predicted:

| Control | Exit | File created | Detected |
|---|---|---|---|
| root, unconfined | 0 | yes (then renamed and removed) | the create `openat`, `write`, `renameat2`, `sync` and `unlinkat` |
| uid 65534 alone, in a directory it owns | 0 | **yes** | the create `openat`: the uid layer alone is not containment |
| uid 65534 + Landlock (`elpis_autonomous_run`) | 1 | no | the create `openat` = −1 `EACCES`, and `utimensat` = −1 `EPERM` (seccomp) |
| full confinement (+ read-only mounts) | 1 | no | the create `openat` = −1 `EROFS`, and `utimensat` = −1 `EPERM` |

**Static.** `libelpis_ecs_c_session.so` (SHA-256 `4d864fc6…`) imports the weak C runtime symbols plus `abort`,
`free`, `getrandom`, `malloc`, `memcpy`, `memmove` and `memset`, and nothing else. Its footprint is exactly
2,352 / 8,848 / 31,120 / 120,208 / 476,560 / 1,901,968 bytes at capacity 2 / 16 / 64 / 256 / 1,024 / 4,096, and 0
(refused) at 1 and 4,097.

**Bounds from the tests themselves.** These figures come from the test output, not the harness:
- 0 heap allocations over 20,000 transitions (P3);
- creation makes 3 allocations;
- heap in use is constant over a 100,000-event run (12,544 → 12,544 bytes) and over 1,000 sessions
  (17,584 → 17,584 bytes, after one warm-up cycle for glibc's tcache) (P1).

**Environment** (`results/ENVIRONMENT.txt`): Linux 6.18 x86-64, Landlock ABI 7, strace 6.8, GCC 13.3, rustc 1.97.
Swap is 0. `core_pattern` is `core` (`RLIMIT_CORE` 0 in the confined runs). `/` and `/tmp` are on persistent ext4.
`results/MANIFEST.json` binds the SHA-256 of every source and result.

An earlier run at `6161ae8` passed identically on writes. Its plain-mode RSS figures were invalid (they counted the
Python harness's image; see `e61d5dd`), so its results were not kept.
