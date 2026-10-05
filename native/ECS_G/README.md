# ECS_G — geometric dynamical substrate primitive

`ECS_G` is a qualified native geometric/dynamical substrate primitive for
Elpis's cognitive ECS/EDEN system (`docs/ELPIS_MISSION.md`: DSV4
communicates, ECS computes and persists).

It is intentionally small and independent. It does not contain inference,
codec, history, retrieval, planning, evolution, or legacy Elpis component code.

What is qualified, and what is not:

- `W` is the authoritative microscopic state of this kernel.
- `S3` is a derived coarse observable of `W`, not a second state.
- The mathematics and state mechanics below are qualified in the frozen
  `d=6`, `N in {36,48,72}` regime. Those experiments do not establish that
  this small kernel alone is the entire eventual Elpis cognitive substrate.

## Authoritative state

The authoritative microscopic state is `W in R^(d x N)`.

For the qualified scientific regime, `d=6` and tested widths are
`N in {36,48,72}`.

The cubic primitive is:

    phi(z) = 0.5 z + 0.5 z^2 + 0.5 z^3

and:

    f_W(x) = sum_i phi(x^T w_i)

The derived raw-sum macrostate is `S3 = (mu, M, T3)`:

    mu = sum_i w_i
    M  = sum_i w_i w_i^T
    T3 = sum_i w_i^(tensor 3)

For `d=6`, packed `S3` has dimension `6+21+56=83`.

`W` remains authoritative. `S3` is a projection of that state.

## Recurrence

G1 implements:

    e       = f_W(X) - y
    grad W  = (2/R) X^T [e * phi'(XW)]
    phi'(z) = 0.5 + z + 1.5 z^2
    W'      = W - eta grad W

The frozen Branch36 reference learning rate is `0.002`, with no width
normalization. The API keeps the rate explicit.

A step is atomic: candidate `W'` is fully computed and finite-validated before
the live state and epoch change.

## Snapshot

G2 adds deterministic portable snapshot/restore for exactly:

- format version;
- `d`;
- `N`;
- epoch;
- microscopic `W`.

The snapshot API performs no file I/O and embeds no timestamps, paths, hashes,
or external subsystem state. Storage and provenance belong to the caller.

## Cognitive R0 (qualified under its frozen synthetic regime)

ECS_G supports stateful learned input-response computation under the
qualified Cognitive R0 regime (`docs/COGNITION_R0.md`,
`docs/research/COGNITION_R0_RESULTS.md`): one state learns a bounded synthetic
input -> response relationship into `W` through the G1 recurrence, and answers
later queries with the native forward map of that `W`. The learned behaviour
follows `W` (reset, transplant, zero), survives snapshot and a clean process
bitwise, changes with further experience and is deterministic. Two
microstates with identical `S3` diverge after one identical learning step, so
`W`, not `S3`, is authoritative.

Not established: language or meaning; retention under sequential learning
(the QUAL found interference in 8 of 8 worlds, catastrophic in 6 of 8);
anything about the cubic kernel, `d=6`/`N=36`, `S3` or gradient descent beyond
that regime.

The Python surface is `elpis.ECS_G.cognition.CognitiveCore` (`query`,
`learn`, `snapshot`/`restore`) over the native executor below; `WorldState`
is the scalar reference binding. The reference kernel is unchanged.

## Runtime R1 executor (`ecsg_executor.h`)

The runtime form of an ECS_G state (`docs/ECS_RUNTIME_R1.md`). One executor
owns the authoritative `W`, staging and transaction `W` buffers, scratch and
an admitted-experience capacity in one arena allocated at creation; no
operation allocates afterwards except an explicit `reserve`.

- `forward`: one call, read-only.
- `learn` / `learn_schedule`: `K` G1 steps (or ordered drives) in one call;
  X/y admitted once; any refusal leaves `W`, epoch and generation unchanged;
  success commits by pointer exchange (epoch `+K`).
- `txn_*`: a native candidate, committed by the same exchange or refused
  `STALE` when another commit replaced its source (generation check).
- SINGLE_WRITER: overlapping entry is refused with `BUSY`.

Its kernels keep the reference's per-element floating-point order and are
bitwise equal to `ecsg_math.c`/`ecsg_state.c` (C tests in dispatched and
baseline-ISA builds, a randomized Python differential, all 8 Cognitive R0
QUAL worlds reproduced). Measured (PERFORMANCE_ONLY,
`docs/performance/ECS_RUNTIME_R1.md`): an R0 step (d=6, N=36, R=64) in
6.7 us native vs 24.8 us for the reference loop order; `CognitiveCore.learn`
3.8x to 16.7x faster than before R1 for K = 10..4000. Its sources are pinned
with the kernel by the mission gate.

## Boundary

This directory owns only ECS_G.

It does not import or call ECS_C, HACF, inference, the DSV4 codec, runtime,
evolution, FMS, TRM, DarwinianMatrix, AnchorSpine, CNumPyCortex, or retired
Elpis systems.

The constitutional direction is:

    DSV4 encoded communication
            |
            v
        ECS / EDEN
       active dynamics        (ECS_G: W_t -> W_t+1)
            |
            v
        ECS readout           (S3(W), a coarse observable)
            |
            v
       DSV4 decode

The runtime composition (`src/elpis/runtime/cognition.py`) orchestrates it
over native K1 (`docs/ECS_K1_RUNTIME.md`). A stimulus enters only as a
native-ready ordered experience schedule; one native K1 call learns and
consolidates every experience on a transaction candidate of `(W, epoch, H, a)`,
and `S3` of the candidate `W` is what ECS exposes to the decode boundary. The semantic maps between DSV4 token space
and those drives and readouts are not defined or qualified anywhere in this
repository, so the canonical turn fails closed without them.

ECS_G is never a conditioning input to a DSV model and is never driven by a
DSV model's output statistics. That sidecar topology (commits `f4e1f05`,
`75313fb`) was removed from the canonical path.

The Python binding (`src/elpis/ECS_G/native.py`) imports only the standard
library and takes an already loaded library handle.

## Mutable FMS R0 residency

The separate `elpis_ecsg_fms` library stores the existing portable ECS_G
snapshot bytes as generic FMS objects. It uses only the public Runtime R1 executor
ABI; `ecsg_executor.c`, its Python binding and the canonical cognitive turn are not
modified. An open transaction retains one executor and one FMS WRITE lease; direct
operations restore a transient executor. See `docs/ECS_MUTABLE_FMS_R0.md`.
