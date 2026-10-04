# ECS_G Retention R1: candidate mechanisms (RET1A)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE` · `PREREGISTERED`

Written at RET1A, before any Retention R1 laboratory code, candidate
implementation, DEV run or QUAL run. The numbers, thresholds, DEV selection
rule and QUAL pass rule are in `specs/ecsg-retention-r1.v1.spec.json`. This file
says why the candidate set is what it is, what each mechanism is, what state it
needs, what its eventual native and FMS form would require, and what would
falsify it. Both files are write-once: a change is a new experiment version.

## 1. Starting point (Retention R0 v1, as recorded)

Frozen authority `2b4e0618...`, QUAL `f8e63116...`
(`docs/research/ECS_RETENTION_R0_RESULTS.md`), 24 QUAL worlds, task A -> B:

* plain sequential G1 (M0) retained A in 0/24 worlds, catastrophically in 18/24;
* the selected candidate C1 (coarse S3-space functional consolidation,
  `lambda` 4) retained A in 18/24 but learned B in 10/24 (both in 7/24); gates B
  and C failed: `PARTIAL_REDUCTION` (OUTCOME_C);
* rehearsal of stored experience (M1, control) held both in 9/24;
* the unconstrained joint least-squares ceiling met both criteria in only 8/24.

R1 does not reopen R0. Its evidence is untouched and its disposition stands. R1
is a new experiment with its own name, seed and worlds.

## 2. What the R0 failure is, in the kernel's own mathematics

Notation as in R0: packed cubic features `phi(x)` in `R^F` (`F = 83` for
`d = 6`), `s = S3(W)`, `J(W) = dS3/dW` (`F x dN` = 83 x 216 at R0).

**F1. Response.** `f_W(x) = 1/2 phi(x) . s` exactly.

**F2. Every experience's loss is an exact quadratic in `s`.** For experience `t`
with `R` rows,

    L_t(s) = mean_r (1/2 phi(x_r) . s - y_r)^2 = 1/4 (s - s^_t)^T Sigma_t (s - s^_t) + r_t

with `Sigma_t = mean_r phi(x_r) phi(x_r)^T` (inputs only), `s^_t` any
least-squares solution and `r_t` the irreducible residual. The G1 step is the
gradient step on `L_t` through `J`: `grad_W L_t = J(W)^T grad_s L_t`.

**F3. Fixed-size state already carries all past losses.** By F2 the sum of past
losses is an exact quadratic held by an `F x F` curvature and an anchor,
independent of the number of experiences and of `N`. R0's C1 state `(H, b)` is
that structure with the target-free anchor `s_t = S3(W_t)` in place of
`s^_t`. With `lambda = 1` and `W_A` at A's optimum, C1's objective while
learning B equals `L_A + L_B` up to a constant: the same minimizers as the
rehearsal control M1 (one 128-row batch, loss `(L_A + L_B) / 2`), with no stored
row and no target. So C1 was not missing information about A. What decided R0:

* (a) **capacity**: in 16 of 24 QUAL worlds even the unconstrained joint
  least-squares fit of this function class to both experiences failed the
  criteria. No mechanism that learns this class from that data could be
  expected to pass there, for any `N` (F4);
* (b) **weighting**: DEV chose `lambda = 4`, which weights A four times B at the
  compromise, trading B for A;
* (c) **anchoring**: cumulative C1 anchors each `Sigma_t` at the state reached
  after `t`, which after the first experience is already a compromise, so the
  accumulated quadratic leans toward earlier experience (consistent with R0's
  sequence: cumulative C1 kept the first experience and blocked the later
  ones). The exact target-free law follows from F2: if `s_t` minimizes
  `Q_{t-1} + L_t`, then `Q_{t-1} + L_t = 1/4 (s - s_t)^T (H_{t-1} + Sigma_t) (s - s_t) + const`
  exactly. The accumulated quadratic is re-anchored at the current state;
* (d) **dynamics**: G1 moves `s` through the S3 tangent kernel
  `K(W) = J(W) J(W)^T`, `ds = -eta K(W) grad_s(loss)`. Under a fixed step
  budget, the stiffness of the consolidation term and the coupling of `K`
  between protected and unprotected function directions decide how much of the
  joint optimum is reached.

**F4. Capacity does not grow with `N`.** `S3` is a sum over entities, so for
every `N` the response class is the at most 83-dimensional space of cubic
polynomials vanishing at 0 (with `M = sum w w^T` positive semidefinite). More
entities, recruited entities or subpopulations add no representable function.

**F5. `W` is authoritative for dynamics, `S3` for function.** At generic `W`
with `N = 36` the fibre `ker J(W)` has dimension `216 - 83 = 133`: microscopic
moves that change no response now but change `J`, hence `K(W)`, hence every
later learning step (Cognitive R0 gate G; the R0 PTE check). Two consequences:

* protecting only `S3` leaves the fibre free; the current function is kept,
  but where the microstate drifts within its fibre, and so what the next
  experience can learn, is left undetermined;
* protecting `W` per weight (R0 C2) also pins fibre directions that cannot
  change any response, wasting plasticity, while the entity sum keeps
  function-changing combinations of weights coupled. R0 measured it: C2 held
  both in 3/24 and its cumulative penalty diverged in 33 learns.

For a function direction `u` in the linear coefficients alone,
`u^T K(W) u = N |u_1|^2` for every `W`: there, no microstate removes plasticity.

**F6. No locality.** `phi` is a global polynomial, so any retaining mechanism
needs memory of where earlier experience lived (R0, unchanged). Every R1
candidate takes that memory from experienced inputs (`Sigma_t`) and from its
own state, never from targets.

## 3. Consequences for the design

1. **Separate capacity from stability-plasticity.** A retention mechanism
   cannot exceed the capacity of the function class it protects (F4). R1
   preregisters two arms:
   * **Arm S (primary, gating).** A jointly realizable sequence: one ECS_G
     state, the witness `W* = [T* | 0]`, answers every experience exactly, so
     a failure cannot be blamed on capacity. The controls must still certify
     a real test: plain G1 must forget, rehearsal must succeed, and every
     experience must carry information earlier ones do not determine.
   * **Arm R (secondary, descriptive).** The R0 v1 independent-teacher regime,
     with the representability ceiling stratifying worlds. It shows how each
     mechanism trades old against new experience where capacity binds. It
     never decides or upgrades the disposition.
2. **A sequence, not a pair.** The primary requirement is A -> B -> C -> D with
   every stage boundary held. Stage B is R0's pairwise test. Stages C and D
   consolidate already consolidated state, with accumulated curvature, anchor
   drift and fixed-size state, which is where R0's candidates broke.
   Pairwise success would not show persistently learnable state.
3. **No tuned consolidation strength.** F2/F3 fix the relative weight of past
   and new experience: each experience's mean loss counts once, as in joint
   training (`lambda = 1`). No candidate has a DEV-tuned strength. DEV chooses
   only the arm-S input scale (by controls alone) and one candidate among three.
4. **Exploit the S3/W distinction explicitly.** K1 protects function with an
   exact soft restoring force. K2 protects function with a hard first-order
   constraint realized through the current microscopic Jacobian, leaving the
   fibre and unprotected function directions free. K3 keeps function and
   consolidation state identical to K1 and changes only the microstate within
   its fibre. It asks whether microscopic state, not function, decides
   retention dynamics.

## 4. Arm S: the jointly realizable overlapping-subspace sequence

Per world (frozen RNG streams, `spec.arms.S`):

* `W0`: the R0 initialization (`6 x 36`, entries `N(0, 0.18^2)`);
* one world teacher `T*` (`6 x 4`, entries `N(0, 0.5^2)`),
  `g*(x) = sum_j phi(x . t*_j)`;
* experience `t` has active coordinates `S_A = {0,1,2}`, `S_B = {2,3,4}`,
  `S_C = {0,4,5}`, `S_D = {1,3,5}`; inputs `x_a = sigma z_a` on `S_t` (`z`
  standard normal) and `x_a = 0` elsewhere; targets `g*(x)`; 64 experience rows
  and 256 held-out rows per experience.

Properties (analytic; the witness is also a mechanics check):

* every pair of experiences shares exactly one coordinate, and every coordinate
  belongs to exactly two experiences;
* experience `t` sees only the 19 monomials within `S_t`, and its G1 gradient
  moves only rows `S_t` of `W`. Interference therefore runs exactly through the
  shared rows. Learning B moves row 2, which changes A's monomials that mix
  `x_2` with `x_0, x_1`, and B does not constrain those. By D every row A uses
  has been moved by a later experience;
* distinct monomials seen after A, B, C, D: 19, 35, 48, 58 of 83. Every later
  experience has 10 to 16 monomials that no earlier experience determines;
* the witness `W* = [T* | 0]` (`phi(0) = 0`) answers every experience exactly: the
  sequence is jointly realizable by a single ECS_G state of the qualified
  class.

`sigma` is chosen on DEV from `(1.25, 1.0, 0.75)`. It is the first value, in
that order (most nonlinear first), at which the controls certify a valid test
(`spec.dev_rules.task`). No candidate runs on DEV before it is chosen. If no
value qualifies, R1 stops at DEV as `TASK_INVALID_ON_DEV` and a revised
design is a new version.

## 5. State classes

`COGNITIVE_STATE` (authoritative, mutable, persistent; shapes responses or
learning), `DERIVED` (a function of cognitive state, recomputable),
`SCRATCH` (per-step or per-call workspace), `CONTROL_STATE` (fixed program
parameters), `EXPERIMENT_ONLY` (exists only to construct or evaluate the
test), `EXTERNAL_MEMORY_CONTROL` (stored experience; never eligible).

For every mechanism, every query is the canonical native forward map of its
current `W` alone (`Executor.forward`). One executor created from the
evaluated `W` answers the held-out inputs of every experience; no task label,
example, target or consolidation state reaches a query.

Every learning step of every candidate is the canonical native G1 step of the
current `W` on the current experience. The laboratory creates an executor from
`W` and performs one learn step, then applies the candidate's correction
evaluated at the pre-step `W`. With the correction removed the procedure is
bitwise the canonical core (mechanics check). Learning one experience and
consolidating it form one atomic transition. If any step is refused or any
value is non-finite, `W` and the consolidation state stay as they were before
the experience (canonical refusal semantics), and the refusal is recorded.

## 6. Controls and references (never selected)

**M0: plain sequential G1 (negative baseline).** `W_t = G1^K(W_{t-1}; t)`
through `CognitiveCore.learn`, `K = 4000`, rate 0.002. State: `W` and epoch.
Arm S validity requires it to forget.

**M1: cumulative rehearsal (`EXTERNAL_MEMORY_CONTROL`).** At stage `t`,
`G1^K(W_{t-1}; all experiences so far)` in one batch through the canonical
executor. It stores every experience, with targets: 3,584 bytes per experience,
growing. It is the ceiling for what explicit replay achieves within the budget.
Arm S validity requires it to succeed. **NONCANONICAL. NEVER ELIGIBLE.**

**O: representability ceilings (analysis).** The unconstrained minimum-norm
least-squares `c` with `f = 1/2 phi . c` fitted on the training rows of the
experiences seen so far (stage ceilings), and on the experiences before `t` only
(`O_prev`, for novelty). `EXPERIMENT_ONLY`; it ignores realizability by 36
entities.

**WSTAR: realizability witness (analysis, arm S).** `W* = [T* | 0]`. It must
answer every experience with nmse at most `1e-20` (mechanics).

**C1R: the R0 selected mechanism, unchanged (`REFERENCE`).** R0's C1 exactly:
`lambda = 4`, `H += Sigma_t`, `b += Sigma_t S3(W_t)`, correction
`J^T u`, `u = (lambda / 2)(H S3(W) - b)`. It is run so that R1 can say whether
the old mechanism already suffices where capacity does not bind. It is never
selected and never tuned.

## 7. Candidates

All three use only the current experience's inputs at consolidation. Learning
reads the current experience `(X_t, y_t)` exactly as G1 does. No candidate stores
rows, targets, labels or task identity.

### K1: exact functional consolidation (S3 Laplace, re-anchored)

* **Family:** slow consolidated state in the kernel's own coordinates; the
  exact form of functional regularization / online Laplace for this kernel.
* **Consolidation** (after experience `t` is learned, inside the same
  transition):

      H <- H + Sigma_t            (Sigma_t from X_t only)
      a <- S3(W_t)                (the whole accumulated quadratic is re-anchored)

* **Learning step** (experience `t`, every step, correction at the pre-step `W`):

      W <- G1(W; X_t, y_t) - eta J(W)^T u,    u = 1/2 H (S3(W) - a)

  i.e. gradient descent on `L_t(s) + Q(s)`, `Q(s) = 1/4 (s - a)^T H (s - a)`,
  `lambda = 1` fixed by F2/F3. With `H = 0` it is the canonical core.
* **Why it could resolve the tradeoff:** by F2/F3 its objective is the
  joint-training objective, up to the anchor error of target-free
  consolidation. It has no free strength to trade old against new (R0 cause
  b), and its re-anchoring is the exact accumulation law (R0 cause c). In a
  jointly realizable sequence the joint optimum holds every experience, and K1
  reaches it without stored experience.
* **Relation to C1:** for the pair A -> B, K1 equals C1 at `lambda = 1`. From
  stage C on, it differs in the anchoring law. It is not a retuning of C1: it
  has no hyperparameter, and the reference C1R runs unchanged beside it.
* **State:** authoritative `W` (`dN`), `H` (packed symmetric `F(F+1)/2`),
  `a` (`F`), epoch. Derived/scratch: `S3(W)`, `u`, `U_2` (`d x d`), `U_3`
  (packed `T`). Control: rate, budget, `lambda = 1`, consolidation once per
  learned experience. Experiment-only: teachers, targets, held-out rows.
  External memory: none. Extra persistent bytes at `d = 6`: 28,552 (16.5x `W`).
* **Scaling:** `H, a` are `O(F^2) = O(d^6)`, independent of `N`, of the number of
  experiences and of entities. Steep in `d`: 0.8 MB at `d = 12`, 171 MB at
  `d = 32`. A larger-`d` substrate would need structured curvature, which is a
  new experiment.
* **Query-time dependencies:** `W` only. **Learning-time:** `W, H, a`, the
  current experience. **Persistence:** `(W, epoch, H, a)` atomically. Restoring
  `W` without `(H, a)` is a consolidation reset, which changes future
  behaviour (gate G iv), so it must never be silent.
* **Native runtime:** per-step extra work of about 42,500 operations, against
  G1's 82,944 at `R = 64`: one global `S3` reduction, one `F x F` symmetric
  mat-vec and one local contraction per entity driven by the broadcast `u`. It
  is fusable into the executor's K-step loop. Consolidation is
  `O(R F^2 + N d^3)` at the end of a learn transaction.
* **Falsified if:** at QUAL the arm-S gates fail; or removing `(H, a)` does not
  remove retention; or resetting `(H, a)` after stage B leaves A and B held
  (then `W`, not the declared state, carries retention).

### K2: protected functional subspace (hard first-order constraint, current Jacobian)

* **Family:** low-rank/protected functional subspace; constrained update
  directions (gradient compatibility in its exact form for this kernel);
  selective plasticity expressed in function coordinates rather than per weight.
* **Consolidation:** `H <- H + Sigma_t`. The protected basis `U` is the
  eigenvectors of `H` with eigenvalue above `1e-9 x` its largest. `U` is
  derived and recomputed at each consolidation.
* **Learning step:** with the canonical increment `Delta = G1(W) - W` and the
  constraint `C(W) = U^T J(W)` (`k x dN`),

      W <- W + Delta - C^T (C C^T)^+ C Delta

  so the step changes no protected function direction to first order: the
  response on every past experience's inputs is unchanged to first order. The
  step uses the current `J(W)`, so it moves freely in the fibre and in
  unprotected function directions, and can move rows a past experience uses
  when other entities compensate. `(C C^T)^+` takes a relative eigenvalue cutoff
  of `1e-12`. With `k = 0` it is the canonical core.
* **Why it could resolve the tradeoff:** it is the hard limit of
  stability. Past function is not traded at all, and plasticity is whatever
  the fibre and the free function directions leave. In arm S the free
  directions contain each new experience's new monomials (section 4), so
  acquisition is possible without touching past function. It differs from K1
  in having no anchor, no restoring force and no compromise, and from C2 in
  protecting function through the current microscopic Jacobian rather than
  weights.
* **State:** authoritative `W`, `H` (packed), epoch. Derived: `U` (`F x k`,
  `k <= F`), `J`, `C`, `C C^T` and its factor. Control: rate, budget, the
  eigenvalue cutoffs. External memory: none. Extra persistent bytes: 27,888.
  The derived cache is at most 55,112 bytes and reconstructible.
* **Scaling:** as K1 for persistent state. Per-step work
  `O(N d F k + N d k^2 + k^3)` grows with the protected dimension `k <= F`, not
  with the number of experiences.
* **Query-time:** `W` only. **Learning-time:** `W, H` (via `U`), the current
  experience. **Persistence:** `(W, epoch, H)` atomically.
* **Native runtime:** per-step Jacobian assembly, a `k x k` Gram matrix
  (a global reduction over all entities), a deterministic Cholesky/eigen
  factorization and two projections: about 1.4 M operations per step at
  `k = 48` (17x G1), at most about 3.2 M at `k = F` (39x). A deterministic
  symmetric eigensolver runs at consolidation (cold path). Second-order drift
  is not corrected; it is part of the hypothesis.
* **Falsified if:** arm-S gates fail, in particular acquisition (the
  constraint leaves too little microscopic room) or retention through
  accumulated second-order drift; or the causal ablations fail as for K1.

### K3: fibre reconditioning over K1 (function-preserving microstate reorganization)

* **Family:** state-dependent plasticity through the microstate, the direct
  use of F5. Retention is attempted through which microstate realizes the
  function, not through what the function is.
* **Definition:** K1 in every respect, plus one reconditioning after each
  consolidation (cold path, inside the same transition). Over the fibre
  `{W' : S3(W') = S3(W_t)}`, minimize the protected/unprotected coupling of the
  S3 tangent kernel

      Omega(W) = |P K(W) (I - P)|_F^2 / |K(W)|_F^2,    K = J J^T,  P = U U^T  (U as in K2)

  i.e. how strongly a learning signal in unprotected function directions leaks
  into protected ones under the unmodified G1 law. The procedure is fixed:
  at most 200 iterations. Each iteration takes a Riemannian gradient step of
  relative size `alpha |W|_F` along the fibre-projected gradient (`alpha` starts
  at `1e-2` and halves on each rejected iteration). A minimum-norm Gauss-Newton
  retraction to `S3 = S3(W_t)` follows (at most 10 iterations, relative error at
  most `1e-13`). An iteration is accepted only if the retraction converges and
  `Omega` decreases. The procedure stops after 20 consecutive rejections.
  Function is preserved to `1e-10` relative on every experience's inputs
  (mechanics). The anchor `a` equals `S3` of the reconditioned `W`.
* **Why it could resolve the tradeoff:** under K1 the residual tradeoff is
  dynamical (F3 d). The new experience's gradient, filtered through `K(W)`,
  pushes protected function away and the restoring force pushes it back, at
  the cost of budget and stiffness. That coupling depends on the microstate,
  and the microstate can be changed along a 133-dimensional fibre without
  changing any response. K3 against K1 is a clean causal contrast: identical
  function and identical `(H, a)` at every consolidation, different `W`. Null
  hypothesis: no difference.
* **State:** as K1 (28,552 extra persistent bytes). The reconditioned `W`
  replaces `W` (cognitive). Derived/scratch: `P`, `K`, `Omega` and its gradient,
  `J`, the retraction solves. Control: the fixed procedure constants.
* **Scaling:** persistent state as K1. Cold path per consolidation at most
  about `200 x (N d F^2 + 4 F^3 + 10 x (N d F^2 + F^3))`, around `1e9` to
  `4e9` operations at `d = 6`, independent of the number of experiences.
* **Query-time:** `W` only. **Learning-time:** as K1, plus the reconditioning
  at consolidation. **Persistence:** as K1.
* **Native runtime:** hot path identical to K1. Consolidation needs a native,
  bounded, deterministic fibre optimizer (Jacobian, `F x F` solves, analytic
  `Omega` gradient). Reconditioning rewrites `W` without a G1 step, so the
  executor would need an epoch/generation rule for a function-preserving
  microstate commit (an open native question, recorded).
* **Falsified if:** as K1. Separately, if K3 and K1 do not differ, the
  microstate-only hypothesis is not supported.

## 8. Ablations (pre-registered; run for the DEV-selected candidate at QUAL)

* `state_removed` (gating, G ii): the same engine with the mechanism removed
  from the start (`H = 0`, no projection, no reconditioning). By engine
  equivalence it is bitwise M0.
* `consolidation_reset_at_B` (gating, G iv): after stage B, keep `W` and empty
  the consolidation state, then continue C and D with the candidate engine.
* `extended_state_transplant_at_B` (gating, G iii): after stage B, serialize the
  complete declared state (`W` snapshot bytes plus consolidation state and
  control constants), restore it into a fresh engine (every world in process;
  `qual-0000` also in a clean process) and continue. It must be bitwise equal to
  the uninterrupted run, which proves nothing undeclared carries retention.
* `mismatched_statistics` (reported): every consolidation uses inputs from the
  complementary coordinate triple (`{3,4,5}` for A, `{0,1,5}` for B,
  `{1,2,3}` for C, `{0,2,4}` for D; own stream) instead of the experience's.
* `uninformed_statistics` (reported): K1/K3 use `(tr Sigma_t / F) I` in place of
  `Sigma_t`; K2 uses a uniformly random protected subspace of the same dimension
  (own stream).
* family-specific (reported): K1 `per_experience_anchors` (C1's anchoring law at
  `lambda = 1`); K3 `fibre_removed` (the K1 run: the microstate-only contrast).
* `pte_microstate` (mechanics): two microstates with identical `S3` (the R0
  PTE construction), each with its own consolidation from A's inputs, diverge
  after one identical candidate step on B: `W` stays authoritative.

## 9. Families assessed and excluded before DEV

| family | decision | reason (from the mathematics and R0, before any R1 run) |
|---|---|---|
| selective / state-dependent plasticity, per weight or per entity | excluded as weight-level; included in function coordinates (K2) and microstate form (K3) | R0 tested its best-informed local form (C2: 3/24 both, divergent cumulatively). By F5, importance is not a property of single weights or entities: function lives in entity-coupled `S3` coordinates with a 133-dimensional fibre. Weight gating both pins fibre moves and leaves function-changing combinations open |
| slow / fast learned state | included in its exact form (K1) | for this kernel the exact slow state is the sufficient statistic `(H, a)` with `W` as the fast state (F2/F3). Splitting `W` into slow and fast entity populations changes neither the function class (`S3` is additive) nor the information about the past. Without past-input memory a fast population relearning B still changes A's region (F6). With it, it reduces to K1 |
| low-rank / protected functional subspace | included (K2), exact rank | at `d = 6`, `H` is only 83 x 83. A chosen rank would be a tunable grid with no mathematical reason. Low-rank curvature is recorded as the scaling question for larger `d` |
| gradient compatibility / constrained update directions | included in its exact form (K2); episodic forms excluded | A-GEM-type compatibility needs stored examples (external memory). At the anchor, the first-order gradient of past loss is zero in this kernel, so "do not increase past loss to first order" is vacuous. Compatibility is a second-order, subspace condition, which is K2 |
| modular or recruitable subpopulation capacity | excluded | F4: no capacity gain for any `N`. A recruited entity at `w = 0` initially moves only linear coefficients (`dS3/dw` at 0 is `(I, 0, 0)`). Without past-input memory it cannot retain (F6). With it, it is K1/K2 restricted to a subset, dominated by the unrestricted form |
| sparse / entity-local learning | excluded | updating a subset of entities restricts the tangent kernel to a partial sum, an uninformed plasticity restriction. The dense `d = 6` kernel offers no sparsity to exploit. Informed versions reduce to K2 |
| topology-mediated retention | excluded | the qualified state is an unordered set of entities (`S3` is permutation-invariant). No topology or HACF -> ECS edge is defined or qualified. Adding one is a new ontology not derived from the R0 failure. Deferred until such an edge exists |
| capacity expansion | excluded from R1 candidates; separated by design | more entities do not expand the class (F4). Real expansion needs a different kernel (non-polynomial `phi`, higher degree or input-gated entities): a new primitive that requires its own mathematics, native reference, bitwise executor parity and Cognitive-R0-style qualification before a retention experiment may use it. Mixing it in would confound capacity with retention. R1 removes capacity in arm S by construction and measures where it binds in arm R |
| exact target-statistic accumulation (`mean(y phi)`) | excluded | it is a sufficient statistic of the targets, i.e. a compressed answer store. R1 keeps R0's rule: consolidation reads the state and experienced inputs only |
| pseudo-rehearsal / generative replay | excluded | it needs a generator outside ECS_G or stored inputs. For this kernel, `H` is already the exact input-moment summary that replaying inputs would supply |
| attractor / settling dynamics | excluded | the qualified recurrence has no autonomous dynamics between experiences (R0). It would need a new law |

## 10. State accounting (`d = 6`, `N = 36`, `F = 83`; bytes are binary64)

| mechanism | cognitive (authoritative, mutable, persistent) | derived / scratch | control | experiment-only | external memory | extra persistent bytes | scaling |
|---|---|---|---|---|---|---|---|
| M0 | `W` (1,728 B), epoch | `S3` | rate, budget | teachers, targets, held-out rows | none | 0 | `dN` |
| M1 | `W`, epoch | `S3` | rate, budget | as M0 | all experiences `(X, y)` | 3,584 per experience (14,336 after D), external | grows with experiences |
| O | none | `c` | none | everything | n/a | n/a | n/a |
| WSTAR | none | none | none | `T*` | n/a | n/a | n/a |
| C1R | `W`, `H`, `b`, epoch | `u`, `S3` | rate, budget, `lambda = 4` | as M0 | none | 28,552 | `O(d^6)`; fixed in `N`, experiences, entities |
| K1 | `W`, `H`, `a`, epoch | `u`, `U_2`, `U_3`, `S3` | rate, budget, `lambda = 1` | as M0 | none | 28,552 | `O(d^6)`; fixed in `N`, experiences, entities |
| K2 | `W`, `H`, epoch | `U` (at most 55,112 B), `J`, `C`, Gram, factor | rate, budget, cutoffs `1e-9`, `1e-12` | as M0 | none | 27,888 | `O(d^6)`; fixed in `N`, experiences, entities |
| K3 | `W` (reconditioned), `H`, `a`, epoch | as K1, plus `P`, `K`, `Omega`, retraction solves | as K1, plus procedure constants | as M0 | none | 28,552 | `O(d^6)`; fixed in `N`, experiences, entities |

## 11. Native-runtime feasibility

`PYTHON MAY CONTROL THE ECS. PYTHON MAY NOT EXECUTE THE ECS HOT PATH.` R1
implements nothing natively. A promotion would need its own native milestone.
What each candidate would require:

| requirement | C1R | K1 | K2 | K3 |
|---|---|---|---|---|
| changes to `ecsg_executor` | yes: fused correction in the K-step loop; consolidation at the end of a learn transaction | as C1R | yes: per-step Jacobian, Gram, factorization and projection kernels; consolidation with a deterministic symmetric eigensolver | as K1, plus a bounded native fibre optimizer at consolidation |
| additional resident state | `H`, `b` (28,552 B) | `H`, `a` (28,552 B) | `H` (27,888 B), plus derived `U` | `H`, `a` (28,552 B) |
| per-step global reductions | `S3(W)` and an `F x F` mat-vec | same | `C C^T` over all entities, `C Delta` | same as K1 |
| per-entity local state | none (a broadcast `u` drives local updates) | none | none | none |
| sparse / topological traversal | none | none | none | none |
| state growth | none | none | none | none |
| additional transaction buffers | candidate `H, b`; scratch `u, U_2, U_3` | candidate `H, a`; same scratch | candidate `H`, `U`; scratch `J` (143 KB), `C`, Gram | as K1, plus optimizer scratch; reconditioned `W` in the candidate buffer |
| snapshot-format evolution | yes: `W`, epoch and a typed consolidation block | yes | yes | yes, plus a commit rule for a function-preserving microstate change (open) |
| FMS-compatible materialization | yes: fixed block, learning-time only | yes | yes | yes |
| hot-path extra operations per step (`R = 64`) | about 42,500 (0.51x G1) | about 42,500 (0.51x) | about 1.4 M at `k = 48`, at most about 3.2 M (39x) | as K1 |
| cold-path operations per consolidation | `O(R F^2)` | `O(R F^2)` | `O(R F^2 + F^3)` | at most about `4e9` |

## 12. FMS implications (no FMS code in R1)

| mechanism | must persist with `W` | reconstructible scratch | size | independently materializable | locality / granularity of a future FMS edge |
|---|---|---|---|---|---|
| K1, C1R | `(H, a)` / `(H, b)`, co-versioned with `W` and the epoch | `u`, `U_2`, `U_3`, `S3`, `J` | fixed, 28,552 B at `d = 6`, `O(d^6)` | yes: QUERY never reads it, so it can stay non-resident between learn transactions | one global dense block, committed atomically with `W` (same epoch and generation); no per-entity paging |
| K2 | `H` | `U`, `J`, `C`, Gram, factor | fixed, 27,888 B (+ derived cache at most 55,112 B) | yes, learning-time only | global dense block, atomic with `W` |
| K3 | `(H, a)` | as K1, plus optimizer scratch | fixed, 28,552 B | yes, learning-time only | as K1. Reconditioning is a `W` commit, so a materializer must treat it as a state transition |
| M1 | stored experience | none | grows per experience | n/a | external memory; never a cognitive FMS asset |

Common finding: every candidate keeps `W` as the only query-time state and adds
one fixed-size, learning-time-only block that must be co-versioned with `W`. A
future mutable FMS edge needs a materialization class of that kind: resident
during LEARN, cold otherwise, committed atomically with `W`.

## 13. Predictions (pre-registered; not gates)

* Arm S: M1 holds the sequence in most worlds (validity requires at least 75% at
  QUAL). K1 holds it in at least as many worlds as M1: it has the same
  objective, no stored rows, and an effective per-experience rate four times
  M1's at stage D.
* Arm S: K2 retains through first-order exact protection. Its risks are
  acquisition (too little microscopic room under the constraint) and
  accumulated second-order drift over three later stages.
* K3 versus K1: null hypothesis, no difference in counts. A difference would
  show that the microstate, at identical function and consolidation state,
  changes retention.
* Arm R: no mechanism exceeds the ceiling-feasible worlds. K1 tracks M1. K2
  under-acquires B where A and B disagree on shared monomials. C1R favours the
  first experience, as in R0.
* `state_removed` reproduces M0 bitwise.

## 14. What R1 can and cannot establish

If the selected candidate passes every arm-S gate, R1 supports retention by
declared ECS state, under a jointly realizable synthetic sequence, at
`d = 6`, `N = 36`, with the qualified cubic kernel and G1 as the base step.
That would make the candidate *eligible* for a canonical milestone, not
canonical. The milestone would need the native executor form, snapshot
evolution, an FMS edge for the consolidation block, and its own
qualification. Nothing in R1 bears on capacity-limited regimes beyond arm R's
description, on larger `d`, on other kernels, or on language or meaning.
