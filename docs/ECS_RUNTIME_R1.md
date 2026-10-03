# ECS Runtime R1: native hot-path contract

Authority for making the ECS execute like a model runtime, under
[`ELPIS_MISSION.md`](ELPIS_MISSION.md) (DSV4 communicates; ECS computes, learns
and persists) and [`COGNITION_R0.md`](COGNITION_R0.md) (ECS is the model). R1 is
systems engineering. It changes how the qualified ECS_G computation executes,
never what it computes.

## The invariant

    PYTHON MAY CONTROL THE ECS.
    PYTHON MUST NOT EXECUTE THE ECS HOT PATH.

Once admitted to the ECS runtime, no work whose cost grows with width `N`,
rows `R`, steps `K`, transitions, query batch size or materialized state runs
in Python. Python keeps configuration, explicit orchestration, lifecycle,
receipts, logging, harnesses, authority checks and persistence policy.

Forbidden in Python on the hot path: per-entity, per-row or per-step loops;
repeated buffer construction or state copies inside a transition; snapshot
hashing, JSON or digests inside a numerical operation; Python<->C crossings
proportional to `K`; per-element conversions in the fast path; cognition
through Python callbacks.

Forbidden inside the native critical path (query and state transition): JSON,
Python objects, SHA-256, filesystem I/O, ECS_C history writes, HACF traversal,
logging, model calls, callbacks, tokenization, snapshot serialization, and
heap allocation after executor creation.

## What stays frozen

The qualified mathematics: `phi(z) = 0.5z + 0.5z^2 + 0.5z^3`,
`f_W(x) = sum_i phi(x . w_i)`, the G1 step
`W' = W - eta (2/R) X^T [e * phi'(XW)]` with `e = f_W(X) - y`, microscopic `W` as
authority, `S3` as a diagnostic, the snapshot format, epoch semantics and
failure atomicity. `native/ECS_G/src/ecsg_math.c` and `ecsg_state.c` remain the
scalar reference and correctness authority (digest-pinned by the mission
gate). An executor that keeps the reference's per-element floating-point
order must be bitwise equal to it. Any backend that changes accumulation order
needs its own predefined tolerance contract and differential qualification
before the runtime may select it. Cognitive R0 v1 evidence is never rewritten.

## Required runtime shape

A native executor under `native/ECS_G` owns, for one ECS_G state:

* authoritative `W`, staging/candidate `W`, reusable scratch, admitted
  capacity, epoch and a commit generation;
* **QUERY**: one call, `W + X -> out`, read-only, no allocation, no snapshot,
  hash, JSON or receipt;
* **LEARN**: one call for `K` steps. `X`/`y` are admitted once and the whole
  `K` loop runs natively on staging. Any failure leaves authoritative `W` and
  epoch unchanged. Success is one commit: `W -> W'`, epoch `+K`;
* **native candidate commit**: source generation capture, candidate state,
  stale detection, commit and abort, with commit by buffer exchange rather
  than a copy of `W`.

Concurrency model: **SINGLE_WRITER**. Calls on one executor must be serialized
by the caller. The executor detects concurrent entry and refuses it (`BUSY`).
It claims no multi-writer atomicity and no concurrent readers.

Python is a thin control plane: validate and admit top-level arguments, one
native call, package the result. Receipts and digests are produced outside
the numerical operation, never feed the commit, and their cost does not grow
with `K`; a caller that wants no receipt pays for none. A runtime turn with
several ordered drives is one native transaction: begin, one native learn
over all drives, readout, commit or abort. A contiguous-buffer path (`array('d')`, `memoryview`)
must exist; list/tuple input is a convenience and does not set the
performance ceiling. ECS_G stays standard-library only. A future native codec
calls the executor ABI directly, without Python.

## Measurement protocol

Benchmarks are performance evidence, not scientific evidence, and never
cognition claims. The harness (`research/ecs_runtime_r1`) reports min, p50,
p95, p99 and throughput after an explicit warmup, separates cold-start from
steady state, and records the compiler, build type, flags, CPU, Python, the
exact source digests and the exact shared-library SHA-256.

Workloads are fixed in `research/ecs_runtime_r1/workloads.py` before any
measurement:

* **R0 authoritative**: `d=6`, `N=36`, `R=64`; query, one step, `K` in
  {10, 100, 1000, 4000};
* **scaling** (`PERFORMANCE_ONLY`, `NO_SCIENTIFIC_CLAIM`): `d=6`,
  `N` in {72, 288, 1152, 4608}, `R` in {64, 256}; query and `K=10`. These widths
  are not "supported cognition".

Gates are pre-registered in the same file: one learn FFI call per
`CognitiveCore.learn(K)` with call counts independent of `K`; no heap
allocation in executor query/learn/transaction paths after creation; Python
wrapper overhead independent of `K`; a material speedup over the
pre-nativeization `CognitiveCore.learn`; bitwise parity with the scalar
reference (and Cognitive R0 reproduction on all 8 QUAL worlds); ASan/UBSan
clean. Tail latency (p99 against p50) is investigated and its sources are
reported.

## Commit order

R1A contract, workloads and gates (this commit, no results) -> R1B baseline
evidence -> R1C/R1D native executor and native commit -> R1E Python control
plane -> R1F profile-guided optimization -> R1G differential, sanitizer and
latency qualification -> R1H admit only the measured claims.
