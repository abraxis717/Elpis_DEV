# CGC-LAO bounded learner: final qualification, native reference and closure

Research only, with no runtime authority. The study is synthetic (SLB2 benchmark) and nothing in it is a
production admission. The frozen v1 study in this directory is unchanged and remains authoritative for its own
claims.

**Closure: Level B.**
- Level A (algorithmic closure) and Level B (native research component) are achieved.
- Level C (production readiness) is not claimed. It depends on an actual host and an audited measurement provider,
  and the repository contains neither.

## 1. Disposition

| Contract | Readings that reach the learner | Learner | v4 disposition |
|---|---|---|---|
| **I**: audited independent re-reads | DUP2_4: two readings of the selected event at steps 1–4, one at steps 5–8 | `RMF3` | `QUALIFIED_SYNTHETIC_CONTRACT_I` (VALIDATION and HOLDOUT, every gate) |
| **II**: independence not established | first reading only | `SAFE4_P8` | `QUALIFIED_SYNTHETIC_CONTRACT_II_CONSERVATIVE` (F1–F6 on both) |

Corruption families per contract:

- **Contract I is qualified for:** clean feedback, independent symmetric flips up to q = 0.45, per-episode
  uniform q, and bursts.
- **Contract I is NOT qualified for:** FLIP48 (q = 0.48, harm +0.00049 vs NOOP); readings that share their
  corruption (EVENT40, +0.0059); reading correlation ρ ≥ 0.5.
- **Contract II** is useful under every one of these families, including EVENT40 and every ρ, except FLIP48, where
  it is safe (+0.00017, upper +0.00023). It keeps half of the clean gain.

Records:
- [`v4/evidence/DISPOSITION_V4.json`](v4/evidence/DISPOSITION_V4.json) is recomputed from the per-seed records
  by `test_v4`.
- [`v4/evidence/ood/characterization.json`](v4/evidence/ood/characterization.json) holds ρ_admit.

## 2. The final learner

`RMF3` is a reliability-mixture filter: the v1 mechanism with a smaller hypothesis set.

**Hypotheses.** Reading-level corruption with independent symmetric flips at rate q ∈ {0, 0.4, 0.48}, with a
uniform prior. Under hypothesis h the two readings of one event are independent given the clean outcome C:

$$P(y_1,y_2\mid C,q)=\prod_r q^{[y_r\neq C]}(1-q)^{[y_r=C]},\qquad P(Y_1\neq Y_2)=2q(1-q).$$

**Belief.** Under each hypothesis, a Gaussian over every expert's hidden bias b_k (mean and variance). The
probability of a positive clean outcome is σ(z_k + b_k), where z is the public router logit.

**Update.** Only the selected expert's readings update the state. The learner applies assumed-density
filtering: it moment-matches the Gaussian with a 16-node Gauss–Hermite rule. Mixture weights follow Bayes' rule
from each hypothesis's marginal likelihood of the readings.

**Decision.** At the end it proposes the expert, or STOP (payoff 0.6), with the highest posterior expected
utility. Acquisition is the reference PROBE_EARLY rule applied to the posterior predictive success
probabilities:
- choose the argmax;
- at the second and third steps, probe the best expert not yet observed;
- force coverage of unseen experts when the remaining steps require it.

**Bounds.**
- State: 3 × (1 log weight + 3 means + 3 variances) + 3 counts = **24 logical scalars** per episode.
- Fixed work: 3 hypotheses × 16 nodes = 48 likelihood evaluations per update.
- Every coefficient is frozen. No weights change at inference. No hidden simulator variable is visible to the
  learner.

**Why these three hypotheses.**
- q = 0.48 covers the near-noise band. As readings approach coin flips, the posterior moves to a hypothesis whose
  likelihood is almost flat, and the learner stops adapting instead of chasing noise. This turns the v1 learner's
  FLIP45 harm into a small gain: FLIP45 vs `RMF_V1` −0.00147 [−0.00159, −0.00134] on HOLDOUT.
- q = 0.2 was removed. With only four duplicate pairs, a run of chance agreements under q ≈ 0.45 leaves weight on
  q = 0.2, which then over-adapts. That is why `RMF_NOEVENT` ({0, 0.2, 0.4, 0.48}) failed the near-noise gate G9
  on two VALIDATION partitions. The cost of removing it is FLIP20 utility: +0.00357 against `RMF_V1`.
- The event hypothesis was removed. When readings share their corruption they always agree, so disagreement
  carries no information. On EVENT40 the posterior P(clean) is 0.69 (CLEAN 0.91, FLIP40 0.05). Hedging inside the
  learner costs about 20% of the clean gain (v1 diagnostic). Shared corruption is therefore excluded by the
  host's admission contract (Contract II), not absorbed by the learner.
- The directed five-hypothesis set {0, 0.2, 0.4, 0.48, event} (`RMF5_E25`) was a candidate in v2–v4. It failed G9
  at the v4 DEV 99.9% bound, and is not the smallest effective mechanism.

## 3. Qualification evidence (v4, the final version)

**Protocol.**
- Fresh RNG root `elpis/cgc_lao/solver_lab/v4/`. Partitions DEV 256, VALIDATION 32, HOLDOUT 64 and OOD 64 router
  seeds, disjoint from every v1–v3 partition (tested).
- [`PREREGISTRATION_V4.md`](v4/PREREGISTRATION_V4.md) was committed before any v4 run.
- One DEV selection round, then the freeze ([`FROZEN_V4.json`](v4/FROZEN_V4.json)).
- VALIDATION: 32 seeds × 4096 episodes.
- HOLDOUT: 64 × 4096, opened once after VALIDATION passed.
- OOD: 64 × 2048, opened once after the HOLDOUT. Git chronology is tested.

**Statistics.**
- Router seed is the pairing unit: paired Student-t intervals on n − 1 degrees of freedom.
- Core gates are intersection–union; G7 uses 97.5%; family claims are Bonferroni-adjusted (99.375%).
- Gate scales come from the v2 TRAIN partition: δ_clean 0.00411486, δ_all 0.00145059, online budget 0.10061226,
  τ_safe +0.0005.

**Selection (robust Occam).** Every primary gate must hold at 99.9% on DEV; among eligible arms, the fewest state
scalars wins.
- Eligible: `RMF3` (24 scalars), `RMF4_W` and `RMF4_E25` (31). Not eligible: `RMF_NOEVENT`, `RMF5_E25` and
  every equal-information control.
- Fallback: `SAFE4_P8` was eligible; `SAFE3_P4` failed F3.

Each primary gate below is the 95% interval of the deciding difference (terminal regret, ALL = mean over CLEAN,
FLIP20 and FLIP40):

| Gate | Requirement | VALIDATION | HOLDOUT |
|---|---|---|---|
| G1 FLIP40 − NOOP | upper < 0 | −0.001371 [−0.001584, −0.001158] | −0.001106 [−0.001230, −0.000983] |
| G2 CLEAN − NOOP | upper < 0 | −0.022957 [−0.025245, −0.020669] | −0.020783 [−0.021966, −0.019600] |
| G3 CLEAN − CLEAN_BAYES | upper ≤ 0.004115 | +0.000331 [+0.000261, +0.000401] | +0.000360 [+0.000319, +0.000401] |
| G4 ALL − CGC_REFERENCE | upper ≤ 0.001451 | −0.001908 [−0.002016, −0.001799] | −0.001774 [−0.001861, −0.001686] |
| G5 online regret | upper ≤ 0.100612 | 0.090968 [0.088352, 0.093583] | 0.088434 [0.087063, 0.089805] (NOOP 0.092645) |
| G7 QUNIF / BURST − NOOP | one upper < 0 (97.5%) | −0.006632 / −0.006224 | −0.006074 / −0.005548 |
| G9 FLIP45 − NOOP | upper ≤ +0.0005 | −0.000174 [−0.000310, −0.000038] | −0.000147 [−0.000218, −0.000076] |
| G9 FLIP45 − RMF_V1 | upper < 0 | −0.001567 (upper −0.001328) | −0.001466 (upper −0.001342) |
| G10 CLEAN − RMF_V1 | upper ≤ +0.0005 | −0.000499 [−0.000563, −0.000434] | −0.000552 [−0.000596, −0.000508] |

**Clean-gain retention** (NOOP − RMF3) / (NOOP − CLEAN_BAYES) on CLEAN: **98.6%** (VALIDATION) and **98.3%**
(HOLDOUT). G10 shows the clean advantage of the v1 learner is preserved, and slightly improved.

**Family claims on HOLDOUT** (Bonferroni 99.375%):
- USEFUL: CLEAN, FLIP20, FLIP40, FLIP45, QUNIF, BURST. FLIP45 is SAFE rather than USEFUL on VALIDATION.
- NOT_QUALIFIED: FLIP48 (+0.000487 [+0.000366, +0.000608]) and EVENT40 (+0.00592).

**Fallback `SAFE4_P8` on HOLDOUT.** Retention 49.1%; Δ vs NOOP:

| Regime | Δ vs NOOP |
|---|---|
| FLIP40 | −0.00114 |
| FLIP45 | −0.00031 |
| FLIP48 | +0.00017 (upper +0.00023) |
| EVENT40 | −0.00122 |
| QUNIF | −0.00432 |
| BURST | −0.00418 |

F1–F6 pass on both partitions.

**Diagnostics** (HOLDOUT means for RMF3, with NOOP in brackets where the metric applies):

| Regime | STOP rate | STOP-boundary regret | Ranking regret | P(clean) |
|---|---|---|---|---|
| CLEAN | 0.277 (0.289) | 0.0219 (0.0322) | 0.0205 (0.0309) | 0.906 |
| FLIP20 | 0.314 (0.289) | 0.0311 (0.0322) | 0.0260 (0.0309) | 0.179 |
| FLIP40 | 0.324 (0.289) | 0.0341 (0.0323) | 0.0280 (0.0309) | 0.052 |
| FLIP45 | 0.325 (0.288) | 0.0347 (0.0323) | 0.0283 (0.0309) | 0.044 |
| EVENT40 | 0.343 (0.289) | 0.0385 (0.0322) | 0.0305 (0.0308) | 0.694 |

Terminal regret is the sum of the STOP-boundary and ranking regrets. The posterior identifies the regime:
P(clean) is 0.91 on CLEAN and 0.05 on FLIP40. Under corruption the learner STOPs more often than NOOP. It then
makes more STOP-boundary errors but fewer ranking errors (FLIP40: +0.0018 and −0.0029), and its net effect is
the balance of the two.

## 4. Robustness and out-of-distribution characterisation

**Terminal regret on HOLDOUT** (Δ vs NOOP; ~ marks a 95% interval that contains 0):

| | CLEAN | FLIP20 | FLIP40 | FLIP45 | FLIP48 | QUNIF | BURST | EVENT40 |
|---|---|---|---|---|---|---|---|---|
| RMF3 (Contract I) | −0.02078 | −0.00596 | −0.00111 | −0.00015 | +0.00049 | −0.00607 | −0.00555 | +0.00592 |
| SAFE4_P8 (Contract II) | −0.01038 | −0.00509 | −0.00114 | −0.00031 | +0.00017 | −0.00432 | −0.00418 | −0.00122 |
| KNOWN_Q_DUP (privileged q, same readings) | −0.02114 | −0.01041 | −0.00168 | −0.00067 | −0.00037 | −0.00879 | n/a | −0.00130 |

FLIP48 is close to the information limit: the privileged oracle gains only 0.00037 there.

**Correlated readings** (OOD partition). Readings follow
F₁ = F₀ with probability ρ, else an independent flip, at q = 0.4. Then corr(F₀, F₁) = ρ and
P(Y₀ ≠ Y₁) = (1 − ρ)·2q(1 − q).

| ρ | 0 (FLIP40) | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 1 (EVENT40) |
|---|---|---|---|---|---|---|---|
| RMF3, Contract I | −0.00114 | −0.00095 | −0.00038 | +0.00074 | +0.00284 | +0.00468 | +0.00603 |
| 95% upper | −0.00100 | −0.00078 | −0.00024 | +0.00093 | +0.00314 | +0.00504 | +0.00643 |
| SAFE4_P8, Contract II | −0.00126 | −0.00127 | −0.00123 | −0.00128 | −0.00125 | −0.00120 | −0.00127 |

**ρ_admit = 0.25.** This is the preregistered rule: the largest grid ρ such that every ρ′ ≤ ρ has a 95% upper
bound below 0. Correlation erodes the disagreement signal, and from ρ = 0.5 the learner mistakes correlated
corruption for clean feedback. The fallback reads one reading, so it is invariant to ρ.

**Other OOD regimes.** RMF3 (Contract I) is USEFUL in every one:

| Regime | Δ vs NOOP |
|---|---|
| FLIP30 | −0.00348 |
| ASYM40 (one-sided flips) | −0.00553 |
| CTXQ40 (context-dependent corruption) | −0.00639 |
| BIASWIDE (bias scale 1.6) | −0.05717 |
| BIASWIDE40 | −0.00406 |
| ROUTERWIDE40 | −0.00061 |
| SHARED_SRC (a shared source with q 0.3 plus independent noise 0.15) | −0.00080 |
| SECOND_EVENT40 (the second reading belongs to another event) | −0.00120 |

The fallback is USEFUL in all of them too.

## 5. Measurement-independence contract (Milestone B)

Code: `v2/src/cgc_lao_v2/providers.py` and `host.py`.

**Readings.** A reading carries only an event id, a reading id, a channel id, an expert and a binary value. No
corruption label or hidden variable is exposed.

**Provider declaration.** A provider declares one relation: SINGLE, INDEPENDENT_REREAD, COPY, SHARED_SOURCE,
CORRELATED (with a bound) or SECOND_EVENT. It also declares a mechanism digest and a maximum reading count.

**Admission.** The host admits Contract I only when both hold:
- the digest resolves in its **host-owned audited registry** with a matching relation, and
- the relation is either INDEPENDENT_REREAD (audited bound 0), or CORRELATED with an audited bound ≤ ρ_admit
  (0.25; the host's default of 0.0 is stricter).

Anything else runs Contract II. That includes an unknown digest, a mismatched relation, an over-declared
reading count, COPY, SHARED_SOURCE, SECOND_EVENT, or a non-declaration. Agreement between readings, or readings
arriving from different API calls, never establishes independence.

**Fail-closed integrity checks.** The host checks reading count, value range, the selected expert, replayed
reading ids, stale events, re-reads of another event, re-reads on the same channel, COPY mismatch (honest copies
are deduplicated), unaccepted selections, lifecycle order, horizon, and context and query contracts. Any failure
zeroises the learner and refuses the episode until reset.

**Tests.**
- A privileged statistical audit of every registered SLB2 mechanism: independent re-read (corr < 0.01), correlated
  0.1 and 0.5 (± 0.01), shared source (corr 1), and second event.
- A 15-case admission decision table.
- End-to-end lifecycles through the host.
- A test that Contract II never passes a second reading to the learner.
- A test that admission of correlated providers follows the recorded ρ_admit (`test_v4_closure`).

**Scope.** This qualifies the interface mechanics and the synthetic SLB2 mechanisms only. It does not show that
any external data source supplies independent measurements. No such provider exists in the repository, and none
has been verified.

## 6. Equal-information and system-level comparison (Milestone C)

Each cell is RMF3 − X on HOLDOUT terminal regret; negative favours RMF3. ~ marks a 95% interval that contains 0.

**Algorithm level.** Every control receives the same DUP2_4 readings.

| Control (logical scalars) | CLEAN | FLIP20 | FLIP40 | FLIP45 | FLIP48 | QUNIF | BURST | EVENT40 | ALL | Gates failed |
|---|---|---|---|---|---|---|---|---|---|---|
| RMF_V1, v1 learner (31) | −0.00055 | +0.00357 | −0.00021 | −0.00147 | −0.00208 | +0.00159 | +0.00130 | +0.00199 | +0.00093 | G9 |
| PLUGIN_Q_BAYES (11) | −0.00020 | +0.00375 | −0.00109 | −0.00277 | −0.00379 | +0.00136 | +0.00080 | −0.00260 | +0.00082 | G1, G9 |
| DISAGREE_SGD (8) | −0.00086 | +0.00147 | −0.00082 | −0.00158 | −0.00201 | +0.00035 | −0.00041 | −0.00392 | −0.00007 | G9 |
| BETA_ALL (12) | −0.00238 | +0.00213 | −0.00447 | −0.00663 | −0.00783 | −0.00104 | −0.00080 | +0.00061 | −0.00157 | G1, G9, G10 |
| SGD_ALL (6) | −0.00178 | +0.00279 | −0.00401 | −0.00632 | −0.00761 | −0.00050 | −0.00012~ | +0.00116 | −0.00100 | G1, G9, G10 |
| STATIC_ATT_ALL (6) | −0.01099 | −0.00038 | +0.00005~ | −0.00016 | −0.00018 | −0.00161 | −0.00112 | +0.00688 | −0.00377 | G3, G4, G10 |

**System level.** RMF3 with its extra readings, against original-contract learners that receive one reading per
step.

| Baseline | CLEAN | FLIP20 | FLIP40 | FLIP45 | FLIP48 | QUNIF | BURST | EVENT40 | ALL |
|---|---|---|---|---|---|---|---|---|---|
| NOOP | −0.02078 | −0.00596 | −0.00111 | −0.00015 | +0.00049 | −0.00607 | −0.00555 | +0.00592 | −0.00928 |
| SGD | −0.00178 | +0.00152 | −0.00582 | −0.00814 | −0.00936 | −0.00182 | −0.00114 | +0.00116 | −0.00203 |
| BETA | −0.00238 | +0.00080 | −0.00633 | −0.00855 | −0.00974 | −0.00240 | −0.00182 | +0.00061 | −0.00264 |
| CGC_REFERENCE (trained) | −0.00495 | +0.00176 | −0.00212 | −0.00351 | −0.00422 | −0.00068 | −0.00005~ | +0.00489 | −0.00177 |
| STATIC_ATTENUATION | −0.01099 | −0.00056 | −0.00019 | −0.00039 | −0.00040 | −0.00181 | −0.00126 | +0.00688 | −0.00391 |
| CLEAN_BAYES | +0.00036 | −0.00099 | −0.01342 | −0.01711 | −0.01925 | −0.00567 | −0.00501 | −0.00655 | −0.00469 |

**Interpretation.**
- *System-level gain.* The extra readings plus RMF3 beat every original-contract learner on ALL. Against the
  trained CGC reference: −0.00177 on ALL and −0.00212 on FLIP40.
- *Algorithm-level gain is safety at equal information, not lower mean regret.*
  - The more aggressive same-information learners (`RMF_V1`, `PLUGIN_Q_BAYES`) have lower ALL regret because they
    adapt harder at FLIP20.
  - Every same-information control fails at least one gate, both on HOLDOUT (95%) and on DEV (99.9%). The other
    DEV-eligible Bayes arms, `RMF4_W` and `RMF4_E25`, are larger (31 scalars) and were not carried forward.
  - RMF3 has lower regret than every control on CLEAN, FLIP40, FLIP45 and FLIP48. The one exception is a tie with
    `STATIC_ATT_ALL` at FLIP40.
- Simpler mechanisms (6–12 scalars) exist but fail G1, G9 or G3/G4/G10. No simpler method matches RMF3, so the
  preference for a smaller method does not apply. Among the eligible Bayes arms, RMF3 is the smallest: 24 scalars
  against 31.

## 7. Native C11 reference (Milestone D)

Location: `v2/native/` (header, source, ctest program). It is wired into the root build by one
`add_subdirectory` line as test-kind research targets, without system-authority registration.

**API.** `config_init` (immutable coefficients, validated, bound by an FNV-1a digest) · `reset` · `propose` ·
`observe` (normalisation, moment updates, state correction) · `weights` · `state_correction` · `decide` (expert or
STOP) · `zeroize`.

**Fail-closed behaviour.**
- Status codes cover null pointers, configuration, state, input, horizon and numerics.
- A state is bound to its configuration's digest and is refused under any other configuration.
- A numerical failure during an update poisons the state, which is refused until reset.

**Memory** (`v4/evidence/native/resources.json`):

| Quantity | Value |
|---|---|
| Logical state scalars | 24 (RMF3) / 31 (SAFE4_P8) |
| Precision | float64 (required for bitwise parity with the reference; float32 is not qualified) |
| State struct | **480 B**, no padding: 32 B header (magic, t, counts[3], hypothesis count, config digest) plus log weights, means and variances at a fixed capacity of 8 hypotheses. RMF3 uses 200 B of it. |
| Immutable configuration | 520 B (outside the per-episode state) |
| Heap | none. Wrapping malloc, calloc, realloc and free at link time counts 0 allocations over 48,000 learner calls. |
| Stack | static frames only (gcc `-fstack-usage`): observe 432 B, decide 64 B, largest 656 B (`config_init`). The call graph is acyclic (no recursion, tested), so any call chain is bounded by the sum of all frames, 2,312 B. |
| Temporaries | fixed-size locals only; no variable-length arrays, no growth, no file or network I/O |

**Parity** (records `v4/evidence/{validation,holdout}/native_parity.json`; tolerances preregistered in v2):
- **Bitwise** equal to the explicit-loop scalar reference on every proposal, state scalar and decision: 2,048
  complete 8-step episodes per arm per phase, 0 failures.
- Against the vectorised learner: 16,384 episodes per arm per phase over all design and robustness regimes
  (including EVENT40's shared readings). 0 trajectory divergences, 0 decision mismatches, 0 near-tie exceptions.
  Max |Δ state| is 1.8e-12, against a tolerance of 1e-9.
- The mechanical C test covers layout, validation, invariants, contradictory readings, extreme logits, q → 0.5,
  binding, poisoning and zeroisation.

**Sanitizers and builds.** Every CI native job builds and runs the mechanics test:
- gcc and clang × Debug and Release, with warnings as errors;
- ASan+UBSan;
- TSan.

The allocator-wrapping test runs in the uninstrumented Linux builds, because sanitizer runtimes interpose the
allocator themselves. The source includes only `<math.h>` and `<string.h>`.

**Latency** (gcc 13.3 `-O2`, x86_64 shared container; a measurement, not a gate):

| Configuration | observe (one update) | decide (expert or STOP) |
|---|---|---|
| RMF3 | mean 1.33 µs, p50 1.18 µs, p99 4.6 µs | mean 1.81 µs, p50 1.59 µs, p99 5.4 µs |
| SAFE4_P8 | mean 1.31 µs, p99 2.3 µs | mean 1.97 µs, p99 4.7 µs |

Loop bounds depend only on the frozen configuration. The observed maxima (up to about 190 µs) are scheduler
preemption, not algorithmic variation.

## 8. Research host (Milestone E)

`ResearchHost` exposes `initialize(declaration)`, `reset_episode(router)`, `propose_expert(context)`,
`observe_selected(expert, readings)`, `propose_terminal_decision(queries)` and `close_episode()`.

- **The host owns** admission, independence verification, input and resource validation, the lifecycle and the
  final acceptance.
- **The learner owns** its bounded posterior and the proposals.

End-to-end tests run complete episodes through the host with instrumented `SimulatedProvider`s (honest,
undeclared, copying, shared-source and tampering), using the **native** backend:
- Contract I runs the frozen RMF3 and Contract II the frozen SAFE4_P8. Both match the vectorised frozen learners
  decision for decision (`v4/cgc_lao_v4_tests/test_v4_closure.py`, `v2/cgc_lao_v2_tests/test_v2_host.py`).
- The scalar and native backends agree.
- Every tampering case fails closed with the named reason and zeroises the learner.

## 9. Research path and disclosure

| Version | Primary (selection rule) | VALIDATION | HOLDOUT / OOD |
|---|---|---|---|
| v1 (SLB1) | `RMF_U3E25` {0, 0.2, 0.4, event 0.4} | passed (v1 gates) | qualified under DUP2_4; FLIP45 +0.00126 and EVENT40 +0.00404 disclosed. Frozen and unchanged. |
| v2 (SLB2) | `RMF_NOEVENT` {0, 0.2, 0.4, 0.48} (simplest arm within 0.0002 of the best ALL, gates by point estimate) | **G9 failed**: FLIP45 +0.000496, upper +0.000655 | sealed, never opened |
| v3 | `RMF_NOEVENT` (same rule, gates by their 95% bounds) | **G9 failed**: +0.000473, upper +0.000632 | sealed, never opened |
| v4 | `RMF3` (robust Occam: 99.9% eligibility, then the smallest) | all gates pass | HOLDOUT all gates pass; OOD opened once |

**Diagnosis behind v4.** Maximising ALL regret among eligible arms selected the arm on the FLIP45 boundary, a
winner's curse.

**Disclosure.** The selection rule changed twice, each time informed by an earlier version's VALIDATION. Every
version therefore used fresh, disjoint partitions, and only v4's VALIDATION, HOLDOUT and OOD support the claims
here. Every version's records are kept, including the failures.

## 10. Reproducibility

**Frozen digests.**
- Sources (v1–v4 Python): `29a88a8a34824eedd5fd71c103bd39d2a8bd25b31e3913d3fe26070ceb267dda`.
- Specifications: SLB2 `22d2654c8d628bf65aa7addf2330456cdeaa82688cdf1d87667b87862d5b4c65`, SLB1
  `ef965e1dddcf3269a1a32cd3730716569cb40293d5b6322dc5c27ea8da7ee122`.
- Coefficients:
  - `baselines.v2.json` `84309be05cb11aaad8ab9b9f9f8e3a9458678a62ed077b548a9f2ae14db3305c`
  - `cgc_reference.v2.json` `f909333eedfa5c7413be3506fc35ecf608ddaae2c00ada3799f72908f87cb8dc`
- Native source `src/cgc_lao_rmf.c`: `bbf47ca72db7f27f8366785a597ca5991bc5ade5c4b968f0f74e808d8c3a9e3f`.

`FROZEN_V4.json` holds every per-file digest, the plans and the arm specifications.

**Evidence.** The combined v4 evidence digest is `583c0f54ae8f615101057f815a92e67c1d208993281871245fee68e075269bf3`.
To recompute it:

```bash
cd research/cgc_lao/solver_lab/v4
find evidence SELECTION_V4.json FROZEN_V4.json -type f -name '*.json' | LC_ALL=C sort | xargs sha256sum | sha256sum
```

Individual records:

| Record | SHA-256 |
|---|---|
| DEV selection round | `efc5767613c25674c833930f98ab410923dba49ae77eef3fd1f56356bd4f0cb0` |
| VALIDATION | `2a594b9f667f91d6ee5174d9fa225eeec38aab015a6e17a63a326e147e4aa90f` |
| VALIDATION parity | `1d48896ecf7a72c538126c6fee5a7cd888d809c49ec4e190c927132017598dbf` |
| HOLDOUT | `0434cf0c7025af3e11cadd2df930ae708e5df6461147d7c52ab67e63d989272c` |
| HOLDOUT parity | `83d56c59347031b4df5a697792461a6a7b5f35c532ad1b0d32963c87638db112` |
| OOD | `2111bb1eb15aafe18b94a6fdec9a21b948e063e3332a9f8823803a51d570fe65` |

**RNG.** Partitions are SHA-256-keyed splitmix64 counter streams (`specs/slb2.spec.json`), bitwise identical
under numpy 1.26 and 2.x.

**Deterministic replay** (`test_v4_closure`, following `docs/CI_POLICY.md`). The test recomputes router seed 0
of every sealed v4 record.
- Under the recorded profile (Python 3.13.18, numpy 2.5.3, this CPU) it demands the recorded values bit for bit,
  to the 10 recorded significant digits. It passes.
- In any other runtime, a non-skippable regression demands agreement within 1e-7. That cannot change any gate or
  family claim, because the closest claim, FLIP45 on VALIDATION, is 2.2e-5 from its threshold.
- Under the CI pin (Python 3.11, numpy 1.26.4) the regression passes. During development the values there also
  matched all 10 recorded digits.

**Commands.** Run from `research/cgc_lao/solver_lab/v4` with `PYTHONPATH=src`. Each data-producing command ran once per phase, in this order:

```bash
python3 -m cgc_lao_v4.study4 dev
```

```bash
python3 -m cgc_lao_v4.study4 select
```

```bash
python3 -m cgc_lao_v4.study4 freeze
```

```bash
python3 -m cgc_lao_v4.study4 qualify --phase VALIDATION
```

```bash
python3 -m cgc_lao_v4.study4 parity --phase VALIDATION
```

The HOLDOUT and OOD phases use the same `qualify` and `parity` commands. Derived reports come from
`scripts/report4.py` (`disposition`, `ood`, `tables`) and `scripts/native_latency.py`.

**Tests.** From the repository root:

```bash
PYTHONPATH=research/cgc_lao/solver_lab/v4/src:research/cgc_lao/solver_lab/v4 python3 -B -m unittest discover -s research/cgc_lao/solver_lab/v4/cgc_lao_v4_tests -t research/cgc_lao/solver_lab/v4
```

The same pattern applies to `v2` and `v3`. CI runs all of these in the SCIENTIFIC lane through
`tests/research/cgc_lao_solver_lab`, and the C tests through ctest (label `research`).

## 11. Closure decision

- **Level A: achieved.** RMF3 passes every preregistered gate on fresh VALIDATION and on a holdout opened once,
  under the explicitly stated Contract I. The fallback qualifies under Contract II. Everything is reproducible
  from the repository: sources, specifications, coefficients, per-seed records, gate recomputation, chronology and
  replay tests.
- **Level B: achieved.** The C11 component is bounded:
  - 24 logical scalars in a fixed 480 B float64 state;
  - no heap and no recursion;
  - static stack frames;
  - fail-closed behaviour and zeroisation.

  It is bitwise equal to the scalar reference and runs behind a fail-closed research host with tested admission.
  Weights are frozen, and the volatile state is episode-local.
- **Level C: not claimed, host-dependent.** It needs three things:
  1. an actual host whose measurement provider has registered, audited provenance for independent reading-level
     corruption, or a certified correlation bound ≤ ρ_admit = 0.25;
  2. evidence that its readings follow the qualified corruption families (not FLIP48-like near-noise channels or
     shared corruption);
  3. a separately validated integration.

  No such provider exists in this repository. Without one, a host must run Contract II.

## Post-freeze host admission boundary (PR #70 correction)

The v4 frozen scientific source and evidence remain byte-identical. Its historical
`v2/src/cgc_lao_v2/host.py` is retained **for reproducing the frozen experiment**,
not for new provider admission: that historical host permitted a Contract I
`DUP2_4` episode to advance after only one reading at a two-reading step.

New research integrations **must** import
[`host_closure.strict_host.StrictResearchHost`](host_closure/README.md) and
`strict_admit`. These enforce exact duplicate counts, valid two-reading provider
capacity, finite correlation bounds, and fail-closed state zeroization. The
supplement does not change RMF3 coefficients, SLB2 qualification results,
source/evidence digests, or imply that an external provider has been audited.
