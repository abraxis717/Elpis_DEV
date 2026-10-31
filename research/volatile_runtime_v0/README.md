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
- **plain**: exit status, wall time, and the process's own peak RSS (`wait4`).
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
