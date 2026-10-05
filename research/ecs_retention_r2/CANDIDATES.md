# ECS_G Retention R2: task, candidates and controls (RET2A)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE` · `PREREGISTERED`

Written at RET2A, before any R2 laboratory code, DEV run or QUAL run. The
numbers, thresholds, DEV rules and QUAL pass rule are in
`specs/ecsg-retention-r2.v1.spec.json`. Both files are write-once: a change is a
new experiment version. Retention R1 v1 (`research/ecs_retention_r1`) is closed
and untouched. R2 is a new lineage with its own name, seed, worlds and evidence.

## 1. Starting point

* **Retention R0** (frozen `2b4e0618`, QUAL `f8e63116`): plain sequential G1
  forgets catastrophically. The selected consolidation, C1 at `lambda = 4`,
  retained A in 18 of 24 worlds but learned B in only 10 of 24
  (`PARTIAL_REDUCTION`). The joint ceiling was feasible in only 8 of 24 worlds,
  so capacity decided most of the outcome.
* **Retention R1 v1** (DEV record `76815470`): `TASK_INVALID_ON_DEV`. Its arm S
  was jointly realizable, but at every scale plain G1 kept almost everything:
  - it held the whole sequence in 5/8, 8/8 and 8/8 DEV worlds;
  - its median earlier-experience nmse at D was 0.03 to 0.04;
  - later experiences were insufficiently novel (minimum `O_prev` nmse 0.08 to
    0.12).

  No candidate was evaluated. R1 says nothing about K1, K2, K3 or C1R.

R2 keeps the R1 mechanisms and analysis unchanged (R1 `CANDIDATES.md`
sections 2, 5 to 12). It replaces only the primary task. The task must be
jointly realizable, conflict-bearing and novel by construction, not by scale.

## 2. Why R1's task could not forget

Write `s = S3(W)`, so that `f_W(x) = 1/2 phi(x) . s`. Let experience t's
training features be `Phi_t`, and let the jointly realizable truth be `s*`.

**Lemma (where consistent sequential learning can forget).** Suppose B's
targets are consistent with `s*` and `W_A` fits A. Then the function on A's
inputs can change during B's learning only through directions that B's
learning moves and B's own loss does not hold:

- anything B observes is held at `s*` by B's loss;
- anything B's update never moves is untouched.

A's function is therefore damaged only where three things meet:

- B's update moves it;
- A observes it;
- B's data do not constrain it.

R1's arm S made that intersection small, for two reasons:

- **Disjoint rows.** Inputs vanished off the active coordinates, so B's G1
  update moved only B's rows of `W`.
- **Compatible shared structure.** The shared coordinates carried the same
  teacher, so B's loss pinned them to the values A needed.

Only a few mixed monomials remained exposed, and the damage stayed inside the
class thresholds.

A design that rotates the coordinates but keeps a shared core with private
axes fails the same way. B's update never touches A's private axis, and B's
loss holds the shared core. Novelty also fails for generic random planes: a
minimum-norm fit to earlier planes predicts most of a later plane, because a
generic teacher's energy is visible on every plane.

**Consequence.** Conflict needs an *aliased* direction:

- A observes it;
- B's update moves it;
- B's data cannot tell it apart from a direction A cannot see.

## 3. The primary task family: aliased hidden ridges

Per world, all streams are frozen in the specification (`spec.task`):

* **Basis.** A uniformly random orthonormal basis `q1, ..., q6` of R^6 (QR of a
  Gaussian matrix; the sign convention is fixed in the specification).
* **Witness teacher.** `T* = [r_A, g q4, g q5, g q6]`.
  - `r_A` is a uniformly random unit vector in `S = span(q1, q2, q3)`.
  - `g` is the hidden-ridge gain.
  - `g*(x) = sum_j phi(x . t*_j)`.
* **Planes.** Experience A observes `S` itself. B, C and D each replace one axis
  of `S` by an aliased axis at angle `theta = 45 deg`:

      P_A = span(q1, q2, q3)
      P_B = span(b, q2, q3),   b = cos(theta) q1 + sin(theta) q4
      P_C = span(q1, c, q3),   c = cos(theta) q2 + sin(theta) q5
      P_D = span(q1, q2, e),   e = cos(theta) q3 + sin(theta) q6

* **Inputs and rows.** `x = M_t z` with `z ~ N(0, I_3)` in the plane's
  orthonormal frame `M_t`. Targets are `g*(x)`. There are 64 experience rows and
  256 held-out rows per experience; `W0` and every other regime constant are as
  in R0/R1.

### Analytic properties (checked mechanically at RET2B)

1. **Joint realizability.** `W* = [T* | 0]` (`6 x 36`; `phi(0) = 0`) answers
   every experience exactly. Retention cannot be confounded with capacity.
2. **Aliasing.**
   - On `P_B`, with coordinates `(alpha, beta, gamma)` along `(b, q2, q3)`, a
     ridge direction `w` enters only through
     `(cos(theta) w_1 + sin(theta) w_4, w_2, w_3)`. B's data cannot distinguish
     a ridge along `q4` from one along `q1`.
   - A, which observes `q1` but not `q4`, can.
   - C (aliasing `q2` with `q5`) and D (aliasing `q3` with `q6`) are the same
     construction.
3. **Structural novelty.** The hidden ridge `g q(3+k)` vanishes identically on
   every earlier plane. It is perpendicular to `S` and to the earlier aliased
   axes, which lie in `span(S, q4, ..., q(3+k-1))`. Earlier experiences
   therefore carry no information about it. On its own plane it contributes
   `phi(g sin(theta) alpha)`, a large share of the variance for `g >= 1.5`.
4. **Structural conflict.** G1's update on experience B is
   `sum_r e_r phi'(x_r . w_i) x_r` with `x_r` in `P_B`. Every entity moves
   within `P_B`, so any growth that represents the hidden ridge happens along
   `b`, which is half `q1`. On A's plane this is a spurious ridge along `q1`
   (A cannot cancel it, since A's own data were already fitted). This forgetting
   follows from the update geometry, not from the metric. B's loss cannot
   correct it, because on `P_B` the `q1` and `q4` components are the same
   function.
5. **Identifiability by the union, and replay.** A's data pin the function on
   `S`; each later plane then identifies its hidden ridge. The joint
   least-squares ceiling over all four experiences is exact. Rehearsal (M1)
   sees A's rows while learning B, so the joint gradient steers the new ridge
   from `q1` to `q4`.

### The only task parameter

The hidden-ridge gain `g`, with the ordered grid `(2.0, 2.5, 1.5)`, is chosen on
DEV by controls only. The aliasing angle, the input scale, the basis
construction and every regime constant are fixed.

### Design analysis disclosed (controls only; not evidence)

The family was chosen by mathematical analysis plus control-only runs on
`design-*` worlds. These were drawn from a scratch generator unrelated to this
specification's seed, name and streams, and none of them is a DEV or QUAL world.
Only M0, M1, O and WSTAR were run. No candidate, reference or consolidation
mechanism was run at any point of the design. Summary over 8 design worlds per
setting, `theta = 45 deg`:

| hidden gain g | M0 held the sequence | M0 learned every stage | M0 median earlier-experience nmse at D | min `O_prev` nmse | M1 held | joint ceiling |
|---|---|---|---|---|---|---|
| 1.5 | 1/8 | 8/8 | 0.43 | 0.30 | 8/8 | exact |
| 2.0 | 0/8 | 8/8 | 0.70 | 0.69 | 8/8 | exact |
| 2.5 | 0/8 | 7/8 | 1.56 | 0.79 | 8/8 | exact |

Rejected families, same analysis:

| family | why rejected |
|---|---|
| R1's coordinate subspaces | M0 kept everything |
| Haar-random planes with one teacher | median 0.27; novelty down to 0.02 |
| shared core with private axes | M0 median 0.004 to 0.015 |
| aliasing at 30 or 60 degrees | weaker conflict, or M0 learning failures |

These numbers motivated the grid order. They decide nothing: DEV on fresh worlds
does.

## 4. Mechanisms (unchanged from R1 v1; experimental isolation)

The definitions, constants and state of every mechanism are those of R1 v1
(R1 `CANDIDATES.md` sections 6 and 7; R1 spec `mechanisms`). R2 re-implements
them exactly.

| mechanism | role | definition (as R1) | extra persistent state |
|---|---|---|---|
| M0 | negative baseline (control) | plain sequential G1, `W_t = G1^K(W_{t-1}; t)` through the canonical core | none |
| M1 | `EXTERNAL_MEMORY_CONTROL`, never eligible | cumulative rehearsal: G1^K on every experience so far, one batch | stored `(X, y)`, 3,584 B per experience, growing |
| O | `EXPERIMENT_ONLY` | minimum-norm least-squares ceilings: per stage on the experiences seen so far; `O_prev` on those before `t` | none |
| WSTAR | `EXPERIMENT_ONLY` | the witness `[T* | 0]` | none |
| C1R | `REFERENCE`, never selected | R0 C1 unchanged, `lambda = 4` | `H, b`: 28,552 B |
| K1 | candidate | exact functional consolidation: `H += Sigma_t`, `a = S3(W_t)`; step `G1(W) - eta J^T u`, `u = 1/2 H (S3(W) - a)`; `lambda = 1` | `H, a`: 28,552 B |
| K2 | candidate | protected functional subspace: `H += Sigma_t`; `U` the eigenvectors of `H` above `1e-9 x` the largest; step `G1(W) - C^T (C C^T)^+ C Delta` with `C = U^T J(W)` and Gram pseudo-inverse cutoff `1e-12` | `H`: 27,888 B |
| K3 | candidate | K1, plus after each consolidation the registered fibre reconditioning minimizing `Omega = |P K (I-P)|^2 / |K|^2` over `S3 = const`: 200 iterations, relative step `1e-2` halving on rejection, stop after 20 rejections, minimum-norm Gauss-Newton retraction (10 iterations, `1e-13`) | `H, a`: 28,552 B |

No consolidation strength is tuned. No mechanism is redesigned.

Every candidate learning step is the canonical native G1 step of the current
`W`, followed by the correction at the pre-step `W`. With the mechanism removed,
the engine is bitwise the canonical core. Learning an experience and
consolidating it form one atomic transition. Consolidation receives `W` and the
current experience's inputs only. Every query is the canonical native forward
map of `W` alone.

State accounting, native-runtime requirements and FMS implications are those of
R1 `CANDIDATES.md` sections 10 to 12. They depend only on the mechanisms and on
`d = 6`, `N = 36`, which are unchanged.

## 5. Ablations and microstate check, adapted to this task

* `state_removed`, `extended_state_transplant_at_B` and
  `consolidation_reset_at_B` are gating, exactly as R1 defines them.
* `mismatched_statistics` (reported): every consolidation uses inputs from the
  never-observed plane `span(q4, q5, q6)` (own stream) instead of the
  experience's.
* `uninformed_statistics`, `per_experience_anchors` (K1) and `fibre_removed`
  (K3) are reported, as in R1.
* **PTE microstate check, made nondegenerate for this task.** Take the R0 PTE
  values (`0, 4, 7, 11` and `1, 2, 9, 10`, times `0.125`) and place them along
  B's aliased axis `b` instead of coordinate 0. Base entities are
  `W0[:, :32]`; tail entities are `0.125 v b` for each PTE value `v`.
  - Power sums along one direction are equal, so the two microstates have
    identical `S3`.
  - B's inputs have unit variance along `b`, so a B step reads exactly the
    coordinate where the microstates differ. R1's arm-S degeneracy cannot recur.

## 6. Validity (DEV task rule; controls only)

All six conditions must hold on every DEV world at a grid value before any
candidate runs:

- **V1 realizability:** WSTAR within `1e-20` nmse on every experience.
- **V2 real forgetting.**
  - M0 holds the sequence in at most 25% of DEV worlds.
  - The median over DEV worlds of M0's mean earlier-experience nmse at D is at
    least 0.5. This is an absolute condition, at the class threshold.
  - The median A end ratio is at least 10. A ratio alone never establishes
    forgetting.
- **V3 novelty:** `O_prev` nmse at least 0.5 at every stage B, C, D in every DEV
  world (R1's floor was 0.25).
- **V4 learnability:** M0 learns every stage in every DEV world.
- **V5 joint solvability:** M1 holds the whole sequence in every DEV world.
- **V6 capacity:** the joint least-squares ceiling over the experiences seen so
  far has nmse at most `1e-6` on every seen experience, at every stage, in
  every DEV world.

The first grid value satisfying V1-V6 is chosen. If none does, R2 stops as
`TASK_INVALID_ON_DEV`: no candidate, no freeze, no QUAL.

## 7. Predictions (pre-registered; not gates)

* M1 holds the sequence in most QUAL worlds (validity requires at least 75%).
* K1 matches or exceeds M1. Its objective equals joint training with no stored
  rows, so it should resolve the aliasing the way replay does.
* K2 retains by first-order protection of A's function. Its risks:
  - acquisition: its constraint must leave room to grow the hidden ridge along
    the fibre and the unprotected directions;
  - second-order drift.
* K3 against K1: null hypothesis, no difference.
* C1R (`lambda = 4`) favours earlier experiences; acquisition is its risk.
* `state_removed` reproduces M0 bitwise.

## 8. What R2 can and cannot establish

An `OUTCOME_A` would support retention by declared ECS state on a jointly
realizable, aliasing-conflict synthetic sequence, at `d = 6`, `N = 36`, with the
qualified cubic kernel and G1 as the base step. It would make the selected
candidate eligible for a canonical native milestone, not canonical. Nothing in
R2 bears on capacity-limited regimes, larger `d`, other kernels, language or
meaning.
