# Elpis Torch R9 — Gate A: projection and transport falsification

**Disposition: HOLD.**

The repository contains no executable seam between CGC/K1 state and a geometric transport representation (section 1).
This document freezes the protocol a qualifying experiment would have to follow (section 2). Nothing in it was run,
and nothing was simulated in place of the missing seam.

## 0. The hypothesis, exactly

> Independently learned CGC/K1 representations can be projected into a shared relation space and transported across
> geometric frames without destroying task-relevant retained information.

Equivariance is necessary for some geometric mechanisms but not sufficient: a transport that commutes with frame
changes can still destroy what the task needs.

This gate concerns the **native K1** retention state and its R3 task. It is not about the PyTorch `CGCTransform`. That
operator holds no state beyond its weights; nothing per node exists to project or transport. The two must not be
conflated (`CLAIM_AUDIT_R9.md`, L1–L3).

## 1. Audit: is there an executable seam?

**No.** The evidence:

**1. The K1 state.**
- The native K1 retention state `(W, epoch, H, a)` is defined, digested and transplantable. See:
  - `docs/research/ECS_RETENTION_R3_RESULTS.md`;
  - `docs/research/ECS_K1_NATIVE_RESULTS.md`;
  - `research/ecs_retention_r3`, `research/ecs_k1_native`.
- The extra retention state is 28,552 bytes.
- No code exports it into any relation space, and no code re-imports a transformed state into a K1 node.

**2. The transport representation** (`research/yang_mills_hca/r118_scalable_awr/gauge.py`):
- Relation rows of 8 floats: two 4-vectors.
- An SO(2)×SO(2) relation gauge acting as block-diagonal 2×2 rotations.
- Frames per incidence (`g_r U[r,p] h_p^T`).
- The operator is the frozen 88-parameter R77/R78 HCA.

**3. The HCA line keeps CGC out.**
- Its own architecture lists "sparse gather and frame transport … end-to-end gauge covariance" among its
  **unqualified engineering seams** (`ARCHITECTURE.md`).
- Every frozen protocol (R71–R78) records `CGC: SHADOW_ONLY`.
- The line carries `R93_GATE=ENERGY_NORMALIZED_DENOMINATOR_HOLD` and reports poor teacher-forced local transition
  fidelity (`README.md`).

**4. The connecting pieces do not exist.** No projection φ from K1 state to relation rows, no inverse or readout, no
shared relation-space contract and no task that consumes transported K1 information exists anywhere in the
repository.

Building those pieces would be inventing the integration seam this gate is meant to test. R9 therefore stops at HOLD.

## 2. Frozen protocol (for a future, separately authorized experiment)

These definitions and thresholds are frozen by this commit, before any Gate A evaluation exists. A later experiment
may add detail but may not loosen them.

### Preconditions (all required; each needs the owner's authorization)

- **P-1 Seam specification.** It must state:
  - which K1 components are exported (W, H, a; epoch as metadata only), with their layout;
  - how a transported state is re-imported into a K1 node for evaluation.
  It must be reviewed and committed before any confirmatory run.
- **P-2 Transport contract.** The relation-row dimension, frames and gauge group come from the HCA line's own code. They
  are imported, not re-implemented, and that code's qualification status is stated.
- **P-3 Authority.**
  - Native K1 authority and historical checkpoints are unchanged.
  - The work stays research-only.
  - No production path changes.

### A1 — Frozen projection φ

- **Form.** φ maps a K1 state (flattened, float64) to `R` relation rows of 8 floats, with `R` registered in advance.
  It is linear. It is fixed before evaluation, from DEV-world K1 states only — a seeded orthonormal projection, or the
  leading principal basis of DEV states; the choice is registered.
- **One φ.** The same φ serves every node compared. It is never fitted jointly with the transport, and never on the
  evaluation (QUAL) worlds.
- **Reference path.** An unprojected path (direct K1) is preserved as the reference.
- **Report:**
  - input dimension and output dimension `8R`;
  - the rank;
  - the DEV-state variance retained;
  - the information discarded, as `1 −` the variance retained, together with the null space's dimension.

### A2 — Task-relevant approximate collisions

- **Distance.** `δ(x, y) = ‖φ(x) − φ(y)‖ / m`, where `m` is the median pairwise distance of φ over DEV states.
- **Collision.** A pair of distinct evaluation states collides when `δ < τ`, with **τ = 0.01**.
- **Report:**
  - the collision incidence;
  - the share of colliding pairs whose states require **different correct R3 outputs**, meaning a different R3
    decision record or answer — the task-relevant collisions.
- Exact floating-point equality is recorded as a diagnostic only.

### A3 — Null controls

| Transport | Definition |
|---|---|
| Identity | φ then φ⁺, no transport; this measures the projection loss alone |
| Random gauge-consistent | SO(2)×SO(2) relation gauges drawn per relation, matched in norm and rank to the proposed transport |
| Learned | fitted on training frames, evaluated on held-out, unseen frames |
| Proposed | the HCA frame transport |

- **Equal budgets.** All four receive the same information and the same compute.
- **Covariance is not a result.** Passing a covariance identity alone is not a positive outcome.

### A4 — Frames

- **Valid SO(2)×SO(2) changes.** Equivariance error `‖T(g·x) − g·T(x)‖ / ‖T(x)‖`. The threshold is **1e-9**
  (float64).
- **Robustness (a separate metric).** Badly conditioned, near-singular, non-orthogonal transforms at condition numbers
  1e2, 1e4 and 1e6, applied only where the mathematics defines a meaningful input. Reported as error amplification.
- **Never** classified as gauge changes.

### A5 — Multistep composition

- **Chain lengths.** K = 1, 8 and 32, each run as a **complete** composed sequence; single steps are never
  extrapolated to K.
- **Report:** accumulated state error and R3 task degradation against K.

### A6 — Retention after transport

**Task.** The R3 aliasing-conflict task. The metric is earlier-experience NMSE at D, and whether the sequence is held,
in each of these cases:
- before transport;
- after φ, the transport `T^K` and re-import;
- untransported K1;
- each null.

**PASS** requires both of the following, on the 32 frozen R3 QUAL worlds:
- **Non-inferiority to untransported K1:**
  - the median NMSE is ≤ max(2 × the untransported median, untransported + 0.002);
  - the sequence is held in ≥ 30/32 worlds.
- **Superiority over the identity and random-gauge nulls:** the proposed transport's median NMSE is lower than each
  null's by more than 0.002.

A nonzero readout alone is never a pass.

### A7 — Stranger nodes

- **Required arm.** At least one arm uses K1 nodes trained **independently**: different worlds and seeds, with no
  shared φ or transport contract during learning. They are connected only after training.
- **Comparison.** Jointly adapted nodes are compared against them.
- **Interpretation.** If stranger nodes fail where jointly trained nodes succeed, that is reported as
  **co-adaptation**, not interoperable transport.

### Dispositions

| Disposition | Condition |
|---|---|
| PASS | A1–A7 executed; A6 and A7 (stranger) pass; A4 equivariance within threshold |
| PARTIAL | A6 passes for jointly trained nodes only, or for K = 1 but not K = 32 |
| FAIL | A6 non-inferiority fails for the proposed transport, or the transport does not beat both nulls |
| NONQUAL | the preconditions are met, but the experiment cannot complete as registered |
| HOLD | a precondition (P-1, P-2) is absent |
| NOT_RUN | not attempted |

### Limits that hold whatever the outcome

- **No promotion.** No Yang–Mills/HCA component is promoted into the production path.
- **No change to authority.** Native K1 authority and the historical frozen checkpoints are unchanged.
- **No generalization.** Success on this toy tensor transport would not imply that learned Elpis language
  representations can be transported.

## 3. R9 disposition

**HOLD.** Preconditions P-1 and P-2 do not exist in the repository, and constructing them is outside R9's
authorization. A1–A7: NOT_RUN.
