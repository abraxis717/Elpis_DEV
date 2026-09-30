# ECS neural dynamics: laboratory results (v1)

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_PRODUCTION_NEURAL_CLAIM`

Branch `research/ecs-neural-dynamics-r1`, based on `main` at
`6cd647c520f0de233f380ebba44546f95afa2186`. The laboratory is in
[`research/ecs_dynamics`](../../research/ecs_dynamics/README.md).

Every result below comes from a synthetic system:
* the tanh probe;
* the exact cubic control;
* a small re-implementation of one published associative network;
* the public ECS kernel run in temporary storage.

No result is a statement about production ECS, a trained model, or any Elpis
neural component. Public ECS has no recurrent, real-valued dynamics; see
[§8](#8-what-this-means-for-ecs).

## 1. Paper access

The papers could not be reached from this environment: the egress policy
refused `arxiv.org` and its mirrors with a 403. The requester then supplied
the PDFs, and every paper-derived statement here rests on those files. The
PDFs are not committed.

| id | status | SHA-256 of the supplied PDF | title |
|---|---|---|---|
| 2609.19288 | `AVAILABLE_USER_SUPPLIED_PDF` | `595988809f49614c81ae065860ec3fc64a8ae02a027b36b09b1016c901daff10` | Vaidya, *Learning-Induced Dynamical Transition in Recurrent Neural Networks* |
| 2609.29834 | `AVAILABLE_USER_SUPPLIED_PDF` | `34edc64b4811662a26323b5578ebb4f61d8ba31fbd32830a659bfd6bd985f2e4` | Wen & Lu, *Designing collective behaviour within a fixed coarse-grained description* |
| 2609.19424 | `AVAILABLE_USER_SUPPLIED_PDF` | `422ea8452ad9b8f049c93a41b170207da58aaee7ca1f74da07c9b0b6b7682a5e` | Lozano-Duran, *Ladder of Information Limits on Prediction for Reduced-Order Models* |
| 2609.07341 | `AVAILABLE_USER_SUPPLIED_PDF` | `0c3afa02e8cd3d4dc7abf1513de63da4abd481d2fd2fcde1ef0c662e518051b7` | Aguilera & De Martino, *Storing Infinite Dynamical Attractors in Nonreciprocal Associative Neural Networks* |

`PAPER_DERIVED_CLAIMS=CITED_PER_RESULT`. Each result's `sources` field and
[§6](#6-paper-dispositions) cite the section or equation used. Anything
marked `LAB:` is this laboratory's construction, not the paper's.

## 2. Definitions

* **Coarse map** `C`: an explicit function of a microscopic object (a state or
  a rule). Examples: raw moments `S1=(m1)`, `S2=(m1,m2)`, `S3=(m1,m2,m3)`;
  `projection[0,1]`; `mean+second_moment`; the ECS topology analysis without
  its `topology_digest`.
* **Target classes**: `INSTANTANEOUS_OUTPUT`, `ONE_STEP_TRANSITION`,
  `K_STEP_TRAJECTORY`, `EVENT`, `ATTRACTOR_CLASS`, `LONG_RUN_STATISTIC`,
  `INTERVENTION_RESPONSE`.
* **Sufficiency finding**: always "`C` was tested for sufficiency with respect
  to target `T` under regime `R`". The outcome is `COUNTEREXAMPLE_FOUND`,
  `NO_COUNTEREXAMPLE_UNDER_REGIME` (not a proof) or `NOT_TESTED`. No record
  says "`C` is sufficient".
* **Fibre** (Wen & Lu, main text): `{z' : C(z') = C(z)}`. At a regular point
  its tangent space is `ker DC`.
  * Response capacity `Gamma = rank(DPhi restricted to ker DC)`, their Eq. (2).
  * Prediction loss: `sigma_max(DPhi N)` for an orthonormal basis `N` of
    `ker DC`. This is the largest first-order target change per unit
    Euclidean budget inside the fibre. It is a Euclidean-budget adaptation
    of their Eq. (3), which uses an L-infinity kernel budget.
* **Coarse-preserving intervention**: a step in `ker DC`.
  * It is exact for a linear `C`.
  * For a nonlinear `C` it is followed by Gauss-Newton retraction.
  * The preservation residual `||C(z') - C(z)|| / max(1, ||C(z)||)` must be at
    most the frozen tolerance; otherwise the intervention is rejected and
    counted.
* **Phase diagnostics**:
  * finite-time tangent growth rate (not a Lyapunov exponent);
  * recovery class `DECAY` / `PERSIST` / `GROW` (separation ratio at most
    1e-2, or at least 1e2);
  * tail classification `FIXED_POINT`, `PERIODIC` (repetition within
    tolerance across at least 3 cycles), `UNBOUNDED`, or
    `UNRESOLVED_NONPERIODIC`. The last label covers slow transients,
    quasi-periodic motion and long periods. Nothing is labelled chaotic.
* **Two-time correlation**: `C(t,s) = (1/N) sum_i phi(x_i(t)) phi(x_i(s))`
  (Vaidya, Eq. 2.5). Here `phi` is the identity, because the state is
  already the rate.
* **tanh probe**: `x_{t+1} = tanh(g J x_t + b + B u_t)` with
  `J = (S + gamma A)/sqrt(1+gamma^2)`, where `S = (G1+G1^T)/sqrt(2d)` and
  `A = (G2-G2^T)/sqrt(2d)`.
  * Off-diagonal entries have variance `1/d`.
  * `corr(J_ij, J_ji) = (1-gamma^2)/(1+gamma^2)`, checked in the tests.
  * `gamma = 1` gives i.i.d. `N(0, 1/d)`.
* **Cubic control**: `phi(z) = 0.5z + 0.5z^2 + 0.5z^3`,
  `f_W(x) = sum_i phi(x . w_i) = 0.5 (x.m1 + x^T m2 x + m3[x,x,x])`. The
  rule update is `F(W) = W + eta W^3`, elementwise.

## 3. Commands

```
S=<venv with numpy>; export PYTHONPATH=src
$S/bin/python -m research.ecs_dynamics.run dev                      # DEV worlds only
$S/bin/python -m research.ecs_dynamics.run freeze --commit 7cd8158d4123220e376e99e852e6ac0c7987234b
$S/bin/python -m research.ecs_dynamics.run qual                     # once
$S/bin/python -m research.ecs_dynamics.run status
$S/bin/python -m pytest tests/research tests/boundary tests/ecs
```

Numerical profile: Python 3.12.3 and NumPy 1.26.4, x86_64, IEEE-754 binary64.
The profile is recorded in every result file.

## 4. DEV choices (DEV worlds only) and frozen QUAL specifications

Commit `7cd8158` holds the laboratory source that was frozen. Its source
digest is recorded in every `frozen/*.frozen.json`. The frozen files, the
DEV evidence and the base specs were committed in `9d012be` before QUAL ran.

| experiment | DEV choice | frozen digest |
|---|---|---|
| `cubic-control` | none | `4fed32dc18c6ca1f2859c1fc69a1d4c4b8099257597f001f1aee7b9fae97f6d5` |
| `tanh-phase-sweep` | unresolved cell `(g=2.0, gamma=1)`; multistable `(3.0, 0)`; stable `(0.8, 1)` | `109dd8824feb82f00262bfd718e15b8628965cc130c63a883fa9f000b69339d9` |
| `tanh-attractors` | multistable cell `(3.0, 0)` | `acc02ceeba20fecbc647d2d7b9625d458d40b1094f2e105e6bb845621f0f14e7` |
| `tanh-delay-ladder` | cell `(2.0, 1)`; one ridge lambda per (target, representation) from grid 1e-6..1 | `ed2735d4bd811a269f4c93838a9e2fa7c2a08a4eceb69b171fa0c1ca6f8eaeda` |
| `tanh-intervention` | unresolved `(2.0, 1)`, stable `(0.8, 1)` | `283fa7b32a23f29c21c08f9b2ce1e842fc281a8c161aaf03e002d54ddb8e4d72` |
| `feedback-transition` | resolvable gains `[2.0]` (`g = 1.3` not resolvable at `d = 200` on DEV) | `a925273bb26595581532530d7a6e403c6607883722eb13fb8d5cacc5c782726f` |
| `associative-cycles` | `phi = 0.2 pi`, `beta = 6`, `alpha = 0.08` | `763cd94606b9f9ab7032d85a4703469342440a898e06b2251b1d6ca325cbb549` |
| `ecs-topology-collision` | none | `05d97d6eb869a67cbb2784cee0067ae2ddd07e7428eea11d49b172549dedd721` |

The pass rules are in `research/ecs_dynamics/experiments.py` (`*_RULE`) and
are copied into each frozen file. They were written before any QUAL run. The
smoke runs used to debug the QUAL code paths ran under a different
experiment name and seed, so they never touched QUAL worlds.

## 5. QUAL results (one run each)

| experiment | mechanics | scientific disposition | QUAL result digest |
|---|---|---|---|
| `cubic-control` | PASS | `SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME` | `638f23d7f185fe657f0835fae62e7e85bb220cb83b05088e1c3440809333f610` |
| `tanh-phase-sweep` | PASS | `DESCRIPTIVE_ONLY` | `8e96053bdf4042e1b77caada8df613aaf9d064c69c9586fd604610a25c2254ce` |
| `tanh-attractors` | PASS | `SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME` | `81c85083146cea4a9640ae284f555530f7a05ee3392e5b453ad6292c8e5dfdac` |
| `tanh-delay-ladder` | PASS | `SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME` | `a40e1b47b456d1529dc1bd5e2adeb84a2fbbe82074fb7990976c4bd76e14bd9d` |
| `tanh-intervention` | PASS | `SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME` | `86841da03729752ec27143cff95a2e5ca1c93b213f89397bc30d6118fd31c54e` |
| `feedback-transition` | PASS | `SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME` | `8a5645d65f44db594af0e9af51409fff3b1b125916f89c9e8b2902df17e31f66` |
| `associative-cycles` | PASS | `SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME` | `0da91dd5a391df3ff346f45ea431dcf27c34ee05bd3bc3bb89c795bc881ca275` |
| `ecs-topology-collision` | PASS | `SUPPORTS_UNDER_FROZEN_SYNTHETIC_REGIME` | `d110ec532d6a33a10f3bfeb195620af7b6c30c926e31b01823827d212aa4bbb1` |

"Supports" means that each hypothesis in §5.1–5.8 met its pre-registered
rule on the QUAL worlds of this synthetic regime. It means nothing beyond
that.

### 5.1 Exact cubic control: `d = 3`, `N = 12`, `eta = 0.1`, 16 QUAL worlds

**Exactness:** the microscopic sum and the moment contraction agree to a
maximum relative error of 8.1e-16.

**Fibres** (identical in every world):

| coarse map | nominal dim | rank | null dim | Gamma: instantaneous output | Gamma: one-step transition | Gamma: 3-step output trajectory |
|---|---|---|---|---|---|---|
| S1 | 3 | 3 | 33 | 16 | 19 | 33 |
| S2 | 9 | 9 | 27 | 10 | 19 | 27 |
| S3 | 19 | 19 | 17 | **0** | 16 | 17 |

For the instantaneous output, the median prediction loss is 38.9 for S1,
29.4 for S2 and 0.0 for S3. Loss and capacity are non-increasing as moments
are matched, in every world.

**Exact witnesses** (Prouhet-Tarry-Escott columns appended to each world's
rule):
* The S3 pair has identical S3, bit for bit, and outputs equal to at most
  3.6e-15. The transition target differs by 6.3e-3.
* The S2 pair has identical S2, and outputs differ by 0.011 to 0.18.

**Interventions:**
* 384 S3-fibre interventions were accepted and none rejected. The largest
  residual was 3.2e-16. Instantaneous-output change was at most 1.2e-15,
  and 100% of interventions changed the transition target by more than 1e-6.
* S2-fibre interventions changed the output in 100% of cases; the per-world
  maximum relative change was 0.012-0.087.

**Fibre design** (a prescribed change of the transition target, reached
inside the S3 fibre):
* realized to a relative error of 0.4-3.0% at L2 budget 0.025;
* coarse residual 0;
* instantaneous output unchanged (0.0).

**Findings:**
* S3 → `INSTANTANEOUS_OUTPUT`: no counterexample (algebraic identity).
* S2 → `INSTANTANEOUS_OUTPUT`: counterexample.
* S3 → `ONE_STEP_TRANSITION`: counterexample (exact witness).
* S3 → `K_STEP_TRAJECTORY`: counterexample (interventions).

This establishes instantaneous-forward sufficiency of S3 for this family
only. It establishes nothing about transition, trajectory, attractor,
intervention or production-ECS sufficiency.

### 5.2 Gain × gamma sweep: `d = 32`, 6 QUAL worlds × 6 initial conditions per cell (descriptive)

* **gamma = 0 (symmetric):** every sampled trajectory reached a fixed point
  or a 2-cycle at every gain. The finite-time growth rate is negative. The
  attractor-count proxy rises with gain, from 1.2 at `g = 0.5` to about 5
  at `g >= 1.6`. This is consistent with the classical property of
  synchronous symmetric updates (fixed points or 2-cycles); that property
  is background and was not tested here.
* **gamma = 1:**
  * `g = 0.5`: every trajectory at a fixed point (norm 0).
  * `g = 0.8`: 83% fixed and 17% 2-cycles.
  * `g = 1.0-1.6`: mixed fixed, 2-cycle and unresolved (33-67% unresolved),
    with growth between −0.03 and −0.06.
  * `g = 2.0`: 100% unresolved, growth +0.086, recovery 100% `GROW`.
  * `g = 3.0`: 83% unresolved, growth +0.126.
* **gamma >= 2:** period-4 orbits appear at `g >= 0.8` (period set `{4}`),
  with unresolved fractions of 0-83%. At `g = 0.5`, every trajectory is at a
  fixed point.
* Boundaries between these regions are changes in sampled finite-time
  classification fractions, not bifurcations.
* **Failure of one metric:** the sweep's one-step R² (ridge on `C_t`, trained
  on half the initial conditions) swings to large negative values, down to
  -163, in multistable cells, where train and test initial conditions reach
  different attractors. It is not a usable cell-level summary. It is recorded
  as it is.

### 5.3 Sampled attractors and `ATTRACTOR_CLASS`: cell `(3.0, 0)`, 8 QUAL worlds

* Per world, 24 initial conditions yielded 11-22 distinct sampled
  attractors. All were period-2 orbits, none were unresolved, and every one
  returned after a 1e-4 perturbation.
* Initial states matched exactly on `projection[0,1]` (residual 0) reached
  different attractors in 90 of 192 resolved pairs. That happened in 8 of 8
  worlds, against a rule of at least 50%.
* Finding: `C_0` → `ATTRACTOR_CLASS`: counterexample.
* Reference cells:
  * `(0.9, 0)`: 1-7 sampled attractors, all 2-cycles; one world was 92%
    unresolved.
  * `(2.5, 4)`: 4-15 period-4 orbits.
  * `(2.0, 1)`: fully unresolved in 7 of 8 worlds.

### 5.4 Delay/memory with the Lozano-Duran surrogate: cell `(2.0, 1)`, `C = projection[0,1]`, depth 4, 8 QUAL worlds

Median held-out score (R², or balanced accuracy for `EVENT`), with each
representation's dimension:

| target | weak (1) | `C_t` (2) | delay (10) | shuffled delay (10) | random projection (10) | `C_t` + seed (10) | full state (32) |
|---|---|---|---|---|---|---|---|
| one-step | 0.517 | 0.560 | 0.871 | 0.574 | 0.901 | 0.559 | 0.973 |
| 5-step | 0.345 | 0.449 | 0.745 | 0.468 | 0.793 | 0.447 | 0.893 |
| event (sign of `C_0` at t+5) | 0.736 | 0.795 | 0.893 | 0.807 | 0.912 | 0.795 | 0.940 |

**Pass rule** (median per-world gain of delay over `C_t` of at least 0.05;
delay beats shuffled delay in at least 75% of worlds; seed columns add at
most 0.01):
* one-step: gain 0.144, delay beats shuffled in 100% of worlds, seed −0.0007;
* 5-step: gain 0.165, 100%, seed −0.0010.

The event target is descriptive: gain 0.046, beats shuffled in 87.5% of
worlds.

**Surrogate of Eq. (3.24)** (medians, one-step): `eps_ref` 0.192,
`eps_Markov` 0.677, `eps_memory` 0.405; hidden penalty² 0.464, memory gain²
0.171. These are held-out errors of one ridge class. They are upper-bound
surrogates, not irreducible errors.

**Caveat:** the dimension-matched random projection of the microscopic
state beats the delay vector in 7 of 8 worlds (one-step). The delay gain is
consistent with recovering hidden-state information; it is not evidence
that temporal structure is uniquely needed. The shuffled control shows the
gain needs aligned lags. Per-world `C_t` R² ranges from 0.10 to 1.00, so the
worlds are heterogeneous.

### 5.5 Coarse-preserving state interventions: cell `(2.0, 1)`, 8 QUAL worlds

* **`projection[0,1]`:** 128 interventions accepted, 0 rejected, residual 0.
  * `K_STEP_TRAJECTORY` (10 steps): every intervention exceeded 1e-3; the
    median relative divergence was 0.019.
  * `INTERVENTION_RESPONSE` (the response to an input pulse): 100% above
    1e-3.
  * `LONG_RUN_STATISTIC` (mean `C` over a 2000-step window): median ratio to
    the within-trajectory sampling floor 0.32; per-world medians 0.00-1.47.
* **`mean+second_moment`:** 128 accepted with retraction; maximum residual
  1.2e-16. The long-run per-world ratio medians were 0.00-2.01.
* **Findings:**
  * `K_STEP_TRAJECTORY` and `INTERVENTION_RESPONSE`: counterexample.
  * `LONG_RUN_STATISTIC`: no counterexample beyond the sampling floor. This
    is the pattern of Lozano-Duran Sec. 3.8, in this probe.
* **Nuance:** a ratio of 0.00 means the perturbed trajectory reconverged,
  i.e. the world's attractor absorbs the intervention. In such worlds the
  trajectory divergence is transient.
* **Stable cell `(0.8, 1)`:** the nonlinear map accepted 0 of 128
  interventions. The fibre of (mean, second moment) through the fixed point
  0 is that single point, so every candidate was correctly rejected.

### 5.6 Vaidya: critical feedback and a static-feedback probe (`d = 200`, `gamma = 1`, 8 QUAL worlds)

**Analytic part:** Eqs. (4.5)-(4.6) at `g = 1.3` give `u_c = 0.5019` and
`yhat_c = 0.1958`. The paper states `yhat_c = 0.2` (Sec. 4, Fig. 1(f));
reproduced within ±0.01.

**Probe at `g = 2.0`** (`yhat_c = 1.279`):

| yhat / yhat_c | 0 | 0.5 | 0.75 | 1.0 | 1.33 | 2.0 |
|---|---|---|---|---|---|---|
| finite-time growth (± s.e.) | 0.141 ± .003 | 0.086 ± .014 | 0.030 ± .014 | 0.011 ± .011 | −0.065 ± .019 | −0.190 ± .020 |
| mean-field log-norm rate | 0.155 | 0.102 | 0.053 | 0.000 | −0.069 | −0.195 |
| fixed-point fraction | 0 | 0 | 0.12 | 0.12 | 0.50 | 1.00 |

* The rule held: growth is positive at 0.5 and negative at 2.0, and the
  fixed-point fraction rises by 1.0.
* The interpolated sign change of growth is at 1.05 `yhat_c`.

**`g = 1.3`** (descriptive; excluded on DEV as unresolvable):
* growth +0.002 ± 0.005 at 0.5 `yhat_c`, and −0.048 ± 0.012 at 2 `yhat_c`;
* interpolated crossing at 0.63 `yhat_c`;
* finite-size and finite-time effects dominate near the transition at this
  gain.

### 5.7 Aguilera & De Martino: Hopf line, `alpha = 0` map, small-N eigenphase comparison

* **Linearization:** for `phi` in {0, 0.1, 0.2, 0.3}·pi, the spectral radius
  of the `alpha = 0` linearization at `beta_c(phi)` is 1 to machine
  precision. Its eigen-argument equals the End-Matter `omega` to at most
  2e-17.
* **Onset at `phi = 0.2 pi`** (`beta_c = 1.2050`): at 0.9 `beta_c` the map
  decays to `m = 0` (norm 1e-91). At 1.1 `beta_c` it is unresolved and
  non-fixed, with mean `||m||` 0.34.
* **Retrieval at `phi = 0.2 pi`, `beta = 6`, `alpha = 0.08`, `N = 2000`**,
  paired patterns per world:

  | | value |
  |---|---|
  | uniform > coherent | 7 of 8 worlds |
  | median difference | 0.51 |
  | zero-load retrieval `||m||` | 0.85-0.87 |

  * Against the paper's `f = 0.5` criterion (a threshold near 0.43), uniform
    eigenphases retrieve in 6 of 8 worlds and coherent ones in 0 of 8.
  * In the exceptional world both fail: 0.09 uniform versus 0.18 coherent.
* **Control at `phi = 0`, same `beta` and `alpha`:** both retrieve (about
  0.998). No eigenphase effect is visible at this load for fixed-point
  attractors.
* No capacity `alpha_c` was estimated.

### 5.8 Public ECS: shared coarse state, different futures

**Minimal pair** (a→c then b→c, versus b→c then a→c):
* the histories are distinct and the state roots differ;
* the coarse state is equal, and so is the weak summary;
* the next processed edge differs;
* the reply-policy coarse trajectory differs at round 1 and re-converges at
  round 2;
* `C_{t-1}` distinguishes the two histories.

**QUAL** (24 random reorderings of the same enqueue multiset, 3-5 entities,
3-6 messages):
* the coarse state is equal in 24 of 24;
* the next processed edge differs in 19 of 24;
* the 2-round coarse trajectory differs in 23 of 24;
* delay depth 1 fails to separate 4 of the 19 event-divergent pairs;
* reading the analysis never changed a state root, and futures ran only on
  byte copies of closed histories.

**Findings:**
* topology-analysis `C` → `EVENT`: counterexample; → `K_STEP_TRAJECTORY`:
  counterexample.
* weak summary → `EVENT`: counterexample.
* delay-augmented depth 1 → `EVENT`: counterexample (4 unresolved pairs).
* state root → `EVENT`: no counterexample.

## 6. Paper dispositions

### arXiv:2609.19288, Vaidya

* **Used:**
  * the slow-feedback reading, in which the feedback enters as a
    quasi-static field of variance `yhat^2` (Sec. 4, Eq. 4.2);
  * the marginal condition, Eqs. (4.5)-(4.6);
  * the two-time correlation, Eq. (2.5).
* **Reproduced:** `yhat_c ≈ 0.2` at `g = 1.3`, analytically (§5.6).
* **Supported under the frozen synthetic regime:** in a discrete-time
  static-feedback probe at `g = 2`, the stability change sits near the
  paper's `yhat_c`. The derivation that transfers the condition to discrete
  time is the laboratory's own heuristic mean field.
* **Not tested:**
  * the learning rule, Eq. (2.4), and the output trajectory;
  * the critical time `t_cr`;
  * the two-time DMFT solution, Eqs. (4.7)-(4.8);
  * anything at `N = 5000`, and the continuous-time network itself.
* **Text inconsistencies, recorded rather than corrected:**
  * Sec. 2 states a coupling variance of `1/sqrt(N)` and puts `g^2 J` in
    Eq. (2.1), whereas Eqs. (3.2)/(A.2) use variance `g^2/N`. The
    laboratory follows (3.2)/(A.2).
  * Sec. 5.3 says "u = u_c ... which is 0.2 for g=1.3". Solving (4.5)-(4.6)
    gives `u_c ≈ 0.50`, so 0.2 is `yhat_c`.
  * Eq. (4.6) prints `phi^2(sqrt(u_c))` without the Gaussian argument `z`;
    the laboratory uses `int Dz phi^2(sqrt(u_c) z)`.

### arXiv:2609.07341, Aguilera & De Martino

* **Used:**
  * the model, Eqs. (1)-(3), with `J_ii = 0`, coherent vs uniform block
    eigenphases, and `Delta = 0.1` (End Matter);
  * the `alpha = 0` overlap map, `beta_c(phi)` and `omega` (End Matter);
  * the retrieval norm and the `f = 0.5` criterion (End Matter, capacity
    calculation).
* **Reproduced:** the Hopf line and frequency, derived independently as the
  modulus-one point of the linearization (§5.7).
* **Supported under the frozen synthetic regime, at `N = 2000`:** uniform
  eigenphases retrieve an oscillatory target better than coherent ones at
  the DEV-selected load.
* **Not tested:**
  * the critical capacities `alpha_c` (Figs. 1 and 3), including
    `alpha_c(T=0) ≈ 0.27` at `phi = 0`;
  * the chaotic 4×4 attractor of Eq. (30);
  * the kernels `R`, `K` of Eqs. (11)-(15) and the Monte Carlo mean field;
  * Region II/III stability, `beta_chi`;
  * any `N >= 50,000`.

### arXiv:2609.19424, Lozano-Duran

* **Used:**
  * the delay vector, Eq. (3.11);
  * the reference-penalty-gain form, Eq. (3.24);
  * nested predictors, Eqs. (3.17)-(3.18); in-sample OLS monotonicity is
    checked exactly;
  * the independent-seed statement, Eq. (3.44);
  * statistics versus trajectories, Sec. 3.8.
* **Supported under the frozen synthetic regime:**
  * delay improves one-step and 5-step prediction over `C_t`;
  * independent seed columns do not help;
  * coarse-preserving interventions change trajectories but not the
    long-run statistic beyond sampling error.
* **Surrogate only:** every error is a ridge held-out error. No irreducible
  error, mutual information or shape factor was estimated.
* **Not tested:** the KS and Lorenz examples; the density-valued and
  generative-law identities, Eqs. (3.36) and (3.48); the event identity,
  Eq. (3.41), as an identity.

### arXiv:2609.29834, Wen & Lu

* **Used:**
  * the fibre and `ker DC` (main text);
  * response capacity, Eq. (2);
  * prediction loss, Eq. (3), adapted to a Euclidean budget;
  * design within a fibre ("Designed oscillations and their limits").
* **Supported under the frozen synthetic regime (cubic control):**
  * `Gamma = 0` for the target the coarse map fixes (the instantaneous
    output under S3) and `Gamma > 0` for targets it does not fix;
  * matching more moments never increases `Gamma` or the loss;
  * a prescribed first-order change is realized inside the fibre to within
    a few percent.
* **Also used as a frame:** the ECS topology-analysis fibre (§5.8).
* **Not tested:** the active-mixture and reaction-diffusion systems, the
  L-infinity budget of Eq. (3), and positivity-constrained kernels.

## 7. Supported and unsupported claims

**Supported under this frozen synthetic regime:** the eight dispositions in
§5, each as scoped there.

**Not claimed:**
* anything about production ECS, trained models, the DSV4 fixture or Elpis
  neural components;
* sufficiency of any coarse map for any target beyond the regime tested;
* Lyapunov exponents, chaos or bifurcations. Every diagnostic is
  finite-time and every attractor is sampled;
* capacity values, critical times or DMFT solutions from the papers;
* that the discrete tanh probe is the papers' continuous-time model.

## 8. What this means for ECS

Public ECS is a deterministic, discrete-event kernel. It has no real-valued
microscopic state and no recurrent law, and its entities have no behaviour.
The laboratory therefore carries the papers' questions to ECS in one way
only: through its coarse observables.

* The topology analysis (without its binding digest) is a many-to-one map on
  histories. Its fibre contains valid histories whose next event and
  near-term coarse trajectory differ. So "topology analysis is sufficient"
  is false as a general statement; any such claim must name a target.
* Adding one prefix of history (delay depth 1) resolves most, but not all,
  sampled divergences. This matches the delay pattern in §5.4.
* The coarse trajectory can diverge and then re-converge. The target must
  name the horizon.

The sealed Structural R0 reference family (the "cubic ECS reference family")
is private and absent from this repository. The cubic control here is a
public algebraic stand-in chosen by the laboratory. It is not a
reconstruction of that family, and no private rule, dataset or reference
byte was inferred.

## 9. Failures and repairs

* The sweep's one-step R² is unstable in multistable cells (§5.2). It is
  recorded as it is and not repaired in v1.
* `g = 1.3` in the static-feedback probe was not resolvable at `d = 200`
  (§5.6). DEV excluded it from the pass rule; the QUAL result for it is
  reported descriptively.
* Pre-freeze defects, found in smoke runs under a separate name and seed
  and fixed before `7cd8158`:
  * a random projection drawn per trajectory instead of per world;
  * a single-window long-run floor;
  * an endpoint-only ECS trajectory target, which hides re-convergence;
  * a non-finite R² for constant coarse signals.
* No QUAL run was repeated. A future repair must be frozen as version 2.
