# ECS_G Retention R0: candidate mechanisms

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_LANGUAGE_CLAIM` · `SYNTHETIC` · `SEMANTICS=NONE`

Written before any candidate was implemented or run. The numeric
specification, thresholds and selection rules are in
`specs/ecsg-retention-r0.v1.spec.json`; this file defines what each mechanism
is, what state it needs and what would falsify it.

## The structure that decides the candidate set

For the qualified ECS_G kernel, with packed cubic features

    phi(x) = ( x_a ;  m2(a,b) x_a x_b  (a <= b) ;  m3(a,b,c) x_a x_b x_c  (a <= b <= c) )

(`m2` = 1 if a = b else 2; `m3` = number of distinct permutations: 1, 3 or
6) and the packed raw-sum observable `S3(W) = (mu, M, T3)` of
`ecsg_math.h`, the response is exactly

    f_W(x) = 1/2 * phi(x) . S3(W)          (83 coefficients for d = 6)

so the learned *function* depends on `W` only through `S3(W)`, while the G1
recurrence moves the 216 microscopic weights (`W` stays authoritative: two
microstates with equal `S3` answer identically but learn differently,
Cognitive R0 gate G). Two consequences:

1. **No locality.** `phi` is a global polynomial: changing any entity changes
   the response on every input region. Plain G1 on experience B fits `S3`
   to B's data and moves the function on A's region without any information
   about where A lives. Experience B carries no information about A's
   inputs, so any mechanism that retains A must carry *some* memory of A
   (its input distribution, its importance for the weights, or its
   examples).
2. **Retention is a question about 83 function coefficients**, learning is a
   question about 216 weights. A mechanism can protect A in function space
   (in `S3` coordinates) or in parameter space (per microscopic weight).
   These are the two candidate families below; both draw their memory of A
   from the same source (A's experienced inputs, never its targets) and
   differ in where and how locally the protection lives.

`G1(W; X, y)` is always the canonical native step: the laboratory creates an
executor from the current `W` and performs one learn step, then subtracts the
candidate's consolidation term evaluated at the pre-step `W`. With the term
removed the procedure is bitwise the canonical core.

## State classes

`COGNITIVE_STATE` (learned, persistent, shapes behaviour or learning),
`CONTROL_STATE` (fixed program parameters), `EXPERIMENT_ONLY` (exists only to
evaluate), `EXTERNAL_MEMORY_CONTROL` (stored experience; never eligible for
promotion). Every mechanism answers every query with the canonical native
forward map of its final `W` alone (`Executor.forward`); no mechanism consults
anything else at query time.

## M0: plain sequential G1 (frozen negative baseline)

* Definition: `W_A = G1^K(W0; A)` then `W_AB = G1^K(W_A; B)`, `K` = 4000 steps
  at rate 0.002, through the canonical core (`CognitiveCore.learn`).
* State: `W` (`COGNITIVE_STATE`, `d N` doubles); epoch (`CONTROL_STATE`);
  rate, budget (`CONTROL_STATE`). Additional: none.
* Changes G1 / topology / query / learning: no / no / no / no.
* Runtime: the Runtime R1 executor as is.
* Role: must reproduce the Cognitive R0 interference finding (gate H).

## M1: rehearsal with stored experience (external-memory control)

* Definition: after `W_A`, experience A `(X_A, y_A)` is kept and learned
  jointly with B: `W_AB = G1^K(W_A; A u B)` (one batch of 128 rows, equal
  weight per row), through the canonical executor.
* State: `W` (`COGNITIVE_STATE`); stored `(X_A, y_A)`, `n_A (d + 1)` doubles
  (3,584 bytes) (`EXTERNAL_MEMORY_CONTROL`). It stores targets.
* Changes G1 / topology / query / learning: no / no / no / data only.
* Runtime: native as is; needs historical examples (grows with the number
  of experiences).
* Role: how much retention is achievable when old experience is explicitly
  available, and whether the task admits one state for both experiences at
  all. **NONCANONICAL. NOT ELIGIBLE FOR PROMOTION BY ITSELF.** It is never
  the selected candidate and cannot satisfy the candidate gates.

## O: representability ceiling (analysis, not a mechanism)

* Definition: the unconstrained least-squares coefficient vector `c*` fitting
  `1/2 phi(x) . c` to experience A and B jointly; held-out nmse of
  `1/2 phi(x) . c*` on A and B.
* State: `EXPERIMENT_ONLY` (computed, never a state of the kernel; it
  ignores that `M` must be positive semidefinite and realizable by `N`
  entities).
* Role: the best any single state of this function class could do with both
  experiences; interpretation only.

## C1: coarse (S3-space) functional consolidation

* Family: consolidation state in the kernel's own coarse coordinates;
  constrained update directions in function space. Named analogues:
  functional regularization / distillation on stored input statistics;
  online Laplace (EWC) with the exact Gauss-Newton curvature of the cubic
  kernel's features.
* Consolidation (at the end of each experience `t` with experienced inputs
  `X_t`, `n_t` rows; targets are never read):

      Sigma_t = (1 / n_t) sum_r phi(x_r) phi(x_r)^T        (input statistics)
      s_t     = S3(W_t)                                    (the consolidated function)
      H += Sigma_t ;   b += Sigma_t s_t

* Learning a new experience `(X, y)`: the G1 step plus a consolidation term,

      W <- G1(W; X, y) - eta * grad_W P(W)        (= W - eta (grad L_G1 + grad P))
      P(W) = (lambda / 4) * sum_t (S3(W) - s_t)^T Sigma_t (S3(W) - s_t)
           = lambda * sum_t mean_{x in X_t} ( f_W(x) - f_{W_t}(x) )^2
      grad_W P = J(W)^T u,   u = (lambda / 2) (H S3(W) - b),   J = dS3/dW
      per entity: d(u . S3)/dw_i = u_1 + 2 U_2 w_i + 3 U_3(w_i, w_i, .)

  (`U_2`, `U_3` the symmetric tensors of the packed `u`).
* Additional mutable state: `H` (`F x F` symmetric, `F` = 83 for d = 6;
  3,486 unique doubles) and `b` (`F` doubles): `COGNITIVE_STATE`, 28,552
  bytes, 16.5x the size of `W` at R0. Size `O(F^2) = O(d^6)`, independent of
  `N` and of the number of experiences; no examples, no targets. With 64
  experienced rows `Sigma_t` has rank <= 64 and may determine the input set
  itself, so it is memory of A's inputs (declared, not hidden).
* Changes G1: yes (additive consolidation gradient). Topology: no. Query: no.
  Learning: yes.
* Runtime: per step `O(N d^3 + F^2)` extra (one `S3` reduction over entities,
  one `F x F` mat-vec, one local contraction per entity) against G1's
  `O(R N d)`; no historical examples; one auxiliary matrix; native-
  implementable and fusable into the executor's K-step loop (the `S3`
  projection is already native). Global reduction over `W`, then local
  entity updates driven by one broadcast vector `u`.
* Why it could preserve A: because the kernel is linear in `S3`, `P` is
  exactly the mean squared change of the function over A's experienced
  inputs, i.e. the curvature of A's data loss. With `lambda` = 1 and `W_A` at
  A's optimum it equals the A term of joint training, without keeping any
  target.
* Falsified if: at the DEV-selected `lambda` the QUAL retention gate fails;
  or B is not acquired (over-constraint); or removing the state (`lambda` = 0)
  does not remove retention (the state is then not the cause).

## C2: local per-weight consolidation

* Family: protected/slow state, state-dependent plasticity, local to each
  microscopic weight (each entity carries its own consolidation). Named
  analogue: diagonal EWC / synaptic-importance consolidation.
* Consolidation at the end of experience `t` (inputs only):

      omega_t[k] = 1/4 [ J(W_t)^T Sigma_t J(W_t) ]_kk     for every weight k of W
      Omega += omega_t ;   A_t += omega_t * W_t   (elementwise)

  i.e. the diagonal, in `W` coordinates, of C1's quadratic form at `W_t`.
* Learning: `W <- G1(W; X, y) - eta * 2 lambda (Omega * W - A_t)`
  (`P = lambda sum_t sum_k omega_t[k] (W_k - W_t[k])^2`).
* Additional mutable state: `Omega`, `A_t` (`d x N` each): `COGNITIVE_STATE`,
  3,456 bytes, 2x `W`. `O(d N)`, local per entity, independent of the number
  of experiences; no examples, no targets.
* Changes G1: yes. Topology: no. Query: no. Learning: yes.
* Runtime: per step `O(d N)` extra, elementwise; trivially fusable and
  compatible with sparse/materialized execution; consolidation itself is a
  cold-path `O(F^2 d N)` computation once per experience.
* Why it could preserve A: weights that A's responses depend on are held near
  their consolidated values in proportion to their importance.
* Why it may not: `S3` couples all entities (sums over `i`) and the map
  `W -> S3` has a 133-dimensional fibre for R0; a diagonal penalty ignores the
  couplings and also penalizes fibre directions that do not change `f`.
  The C1/C2 contrast asks whether retention in this kernel can be local per
  weight, or needs a global function-space consolidation operator.
* Falsified if: as for C1.

## Ablations (pre-registered, run for the DEV-selected candidate)

* `state_removed`: `lambda` = 0, same engine and budget (gating, gate E).
* `mismatched_statistics`: consolidation statistics built from inputs of a
  different region (task C's input distribution) instead of A's experienced
  inputs (reported).
* `uninformed_statistics`: C1 with `Sigma_t` replaced by a trace-matched
  multiple of the identity; C2 with `omega_t` replaced by its mean (uniform)
  (reported). Asks whether the input-distribution memory, not mere
  anchoring, is what retains.

## Considered and not tested

* Entity partition / recruitment without A information: because `phi` has
  no locality, plastic entities fitting B's residual change the response on
  A's region without bound; it carries no memory of A, so by point 1 above it
  cannot retain A except by accident. Recorded as an untested hypothesis.
* Hard gradient projection onto the complement of A's protected subspace: the
  hard-constraint limit of C1 (`lambda` -> infinity on A's top subspace);
  subsumed, not separately tested.
* Attractor-supported stabilization: the qualified ECS_G recurrence has no
  autonomous dynamics between experiences (the state moves only when it
  learns), so there is no attractor mechanism to test without changing the
  model.
* Pseudo-rehearsal from generated inputs: needs a generator outside ECS_G;
  out of scope.

## State authority summary

| mechanism | `COGNITIVE_STATE` | `CONTROL_STATE` | `EXTERNAL_MEMORY_CONTROL` | extra bytes (R0) |
|---|---|---|---|---|
| M0 | `W` | epoch, rate, budget | none | 0 |
| M1 | `W` | epoch, rate, budget | `(X_A, y_A)` | 3,584 (external) |
| C1 | `W`, `H`, `b` | epoch, rate, budget, `lambda` | none | 28,552 |
| C2 | `W`, `Omega`, `A_t` | epoch, rate, budget, `lambda` | none | 3,456 |
| O | none (analysis) | | | |

`W` alone answers every query for every mechanism. For C1 and C2 the
additional state shapes only learning; it is persistent learned state and
would have to persist with `W` in any snapshot if a mechanism were ever
promoted.
