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

The runtime composition (`src/elpis/runtime/cognition.py`) orchestrates it.
A stimulus enters ECS_G only in the kernel's qualified input form, ordered
drives `(X, y)` applied as atomic gradient steps, and `S3(W)` is what ECS
exposes to the decode boundary. The semantic maps between DSV4 token space
and those drives and readouts are not defined or qualified anywhere in this
repository, so the canonical turn fails closed without them.

ECS_G is never a conditioning input to a DSV model and is never driven by a
DSV model's output statistics. That sidecar topology (commits `f4e1f05`,
`75313fb`) was removed from the canonical path.

The Python binding (`src/elpis/ECS_G/native.py`) imports only the standard
library and takes an already loaded library handle.
