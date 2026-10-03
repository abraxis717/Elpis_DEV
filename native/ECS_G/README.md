# ECS_G — geometric world model

`ECS_G` is Elpis_DEV's native geometric world-model subsystem.

It is intentionally small and independent. It does not contain inference,
history, retrieval, planning, evolution, or legacy Elpis component code.

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

It does not import or call ECS_C, HACF, inference, runtime, evolution, FMS,
TRM, DarwinianMatrix, AnchorSpine, CNumPyCortex, or retired Elpis systems.

The eventual integration boundary is:

    completed inference
            |
            v
         ECS_G
       W_t -> W_t+1
            |
            v
      future inference

The mapping between current DSV4.1 inference outputs and ECS_G recurrence
inputs is deliberately not defined here.
