# Relational LAO R0: final report

**Phase:** CGC_LAO_RELATIONAL_BRIDGE_R0. Research only: nothing here is canonical, composed, released, or admitted to
a runtime.

**Result.** The primary qualified as QUALIFIED_SYNTHETIC_RELATIONAL_CONTRACT_I, and the fallback as
QUALIFIED_SYNTHETIC_RELATIONAL_CONTRACT_II_CONSERVATIVE. Both are synthetic qualifications on RLB1; the scope and
limits are in [section 7](#7-what-is-claimed-and-what-is-not).

**Read first:**
- [SPEC.md](SPEC.md): census, seam, benchmark, state, invalidation, protocol.
- [DEV_PROTOCOL.md](DEV_PROTOCOL.md): arms, shared log, selection.
- [PREREGISTRATION_R6.md](PREREGISTRATION_R6.md) and [FROZEN_R6.json](FROZEN_R6.json): the frozen qualification.
- [RESEARCH_LOG.md](RESEARCH_LOG.md): every milestone, deviation and disclosure.

## 1. Question

Take the qualified LAO reliability mechanism (RMF3, SLB2 v4) and generalise it from independent expert biases to a
bounded relational belief state over admitted relational evidence. Does that improve downstream **compositional**
decisions: two-hop and three-hop paths, conjunctions, and alternative paths?

It has to beat:
- no learning;
- exact maintenance;
- static counts;
- a geometric (HCA) projection;
- above all, independent per-edge filters with the same readout.

## 2. Architecture

The four layers run in one direction only:

```
ECS_C admitted evidence (A) → bounded relational belief update (B) → research-only projection (C) → host decision / STOP (D)
```

| Layer | Component | Authority |
|---|---|---|
| A | The ECS_C relational view (ABI v1): admission, bindings, the successor law, support and dependents | Authoritative; never written by B, C or D |
| B | Relational RMF: 8 slots × 3 reliability hypotheses, 80 logical scalars, an 8-event provenance buffer | Volatile; rebuilt from LIVE evidence only |
| C | Research HCA: a u16 salience projection and its B0 readout | SALIENCE_ONLY |
| D | Research host: strict Contract I/II admission, PROBE_R proposals, terminal answers and STOP | The only admission point |

There is no reverse write path into ECS_C. A test with an audit hook attests zero writes. The host admits each reading
into the view before the learner sees it.

**Interfaces verified in source** (SPEC.md §1):
- the ECS_C relational view (`sync`, `check`, `admit`, `support_of`, `dependents`, `relation_read`, `identity`,
  `verify`, `reset`);
- `elpis_ecs_c_entity` (152 B);
- the memory observation (448 B) and the relation key;
- HCA ABI v1 (B0 closure, SALIENCE_ONLY);
- `host_closure.strict_host` (`strict_admit`, `QUALIFIED_READ_SCHEDULE`, `MAX_QUALIFIED_RHO`);
- the provider types of `cgc_lao_v2.providers`;
- the RMF3 law of `FROZEN_V4.json`.

## 3. The qualified mechanism: RRMF3_M (frozen, `FROZEN_R6.json`)

**Slots.**
- 8 relation slots, in canonical order of the relation key.
- Capacity is refused, never grown or evicted.
- Every active slot is bound to its relation identity and to its supporting evidence and provenance.

**The update law (RMF3).** For each slot r and each hypothesis h ∈ {0, 0.4, 0.48}:
- a Gaussian bias belief (μ, v), updated by 16-node Gauss–Hermite moment matching on the admitted readings;
- a log-evidence Λ[h, r].

**The coupling.** One reliability posterior is shared by every relation of the episode:
`log w_h = log π_h + Σ_r Λ[h, r]`. The sum runs over sorted terms, so slot order does not matter. Given h, slots are
independent, which makes removing a slot exact.

**The readout** is marginal:
- per relation, the mixture predictive `Σ_h w_h E_h[p_r]`;
- a path is the product of its relations;
- an alternative is the noisy-OR of its edge-disjoint paths;
- the host answers the best alternative, or STOPs at τ.

**Invalidation** is bounded replay (strategy B). When an authority event retires a binding, its events are dropped
and the state is rebuilt from the buffered LIVE events. Suspended evidence is excluded until it is restored. A stale
posterior cannot outlive its evidence.

**Acquisition: PROBE_R.** VOI_R lowered compositional regret on-policy, but it was not adopted because its online
regret exceeded the budget.

**The fallback, SAFE3_P4** (Contract II, first reading only): the same hypotheses, prior (1, 4, 4), coupled readout.

**Measurement contract.**
- Contract I goes to an audited two-reading provider on the schedule (2, 2, 2, 2, 1, 1, 1, 1) through `strict_admit`.
- This workload's own correlation bound is **ρ_admit = 0.25**, measured on the OOD below.
- Every other provider gets Contract II.

## 4. How it was qualified

**Order of the work:**
1. SPEC.md and the R1 and R2 mechanics.
2. DEV_PROTOCOL.md, committed before any record.
3. One TRAIN round, which fixes the scales.
4. One DEV round, 16 regimes, every arm on one shared log (matched information).
5. Robust-Occam selection at 99.9%.
6. PREREGISTRATION_R6.md and FROZEN_R6.json, committed before any VALIDATION episode.
7. VALIDATION once, HOLDOUT once, OOD once.

The HOLDOUT and OOD openings are enforced by `study qualify`. The analysis code was frozen with the sources.

**Gates.** Gates are at 95% (RG5 at 97.5%), paired over world seeds, on compositional regret per episode:
- every gate part had to pass on both partitions (an intersection–union test, so no adjustment);
- the family claims are Bonferroni-adjusted (99.375%).

**Mechanical suites**, in CI:
- **R1.** Exact invalidation. A revoked source cannot reach a later proposal: the state is bitwise equal after every
  sync. The host over the real view equals the vectorised learner. Dormancy restores exactly, and reclamation with a
  reused authority retires. Integrity failures fail closed. Contracts are strict. Permutation equivariance and
  deterministic replay hold, capacity is refused, and there are zero writes.
- **R2.** The frozen RMF3 is reproduced: 0 of 98,304 decisions differ.
- **R5.** The numpy HCA readout equals native B0.

## 5. Results

Values are [mean, 95% lower, 95% upper], compositional regret differences per episode. Negative is better for C.

### 5.1 Gates

Every gate part passes on VALIDATION (32 × 4096) and HOLDOUT (64 × 4096).

| Part | VALIDATION | HOLDOUT |
|---|---|---|
| RG1: ALL, C − NOOP | −0.002199 [−0.002346, −0.002052] | −0.002008 [−0.002110, −0.001906] |
| RG2: CLEAN, C − NOOP | −0.005375 [−0.005736, −0.005013] | −0.004934 [−0.005170, −0.004699] |
| RG2: CLEAN, C − KNOWN_RELIABILITY (≤ δ_clean 0.001403) | +0.000117 [+0.000090, +0.000145] | +0.000127 [+0.000107, +0.000148] |
| RG3: ALL, C − EXACT_RELATIONAL | −0.096627 [−0.102022, −0.091231] | −0.094331 [−0.097552, −0.091109] |
| RG3: ALL, C − STATIC_COUNTS | −0.001324 [−0.001523, −0.001124] | −0.001298 [−0.001423, −0.001173] |
| RG3: ALL, C − HCA_ONLY | −0.091424 [−0.097530, −0.085317] | −0.089786 [−0.093618, −0.085955] |
| RG3: ALL, C − `:IND` (independent per-edge RMF, same readout) | −0.000291 [−0.000348, −0.000235] | −0.000255 [−0.000280, −0.000231] |
| RG3: ALL, C − `:HCA` (LAO_PLUS_HCA) | −0.013368 [−0.014878, −0.011857] | −0.013066 [−0.014103, −0.012028] |
| RG4: FLIP40, C − NOOP (≤ τ_safe 0.0005) | −0.000069 [−0.000129, −0.000009] | −0.000076 [−0.000107, −0.000045] |
| RG4: FLIP45, C − NOOP (≤ τ_safe) | +0.000112 [+0.000067, +0.000158] | +0.000095 [+0.000061, +0.000130] |
| RG5: QUNIF, C − NOOP (97.5%) | −0.001201 [−0.001309, −0.001093] | −0.001127 [−0.001215, −0.001039] |
| RG5: BURST, C − NOOP (97.5%) | −0.000915 [−0.000999, −0.000831] | −0.000933 [−0.001015, −0.000852] |
| RG6: ALL, C − `:PERM` (identity ablation) | −0.003498 [−0.003665, −0.003330] | −0.003202 [−0.003342, −0.003062] |
| RG7: REVISION, C − `:STALE` | −0.000301 [−0.000334, −0.000269] | −0.000272 [−0.000291, −0.000252] |
| RG7: REVISION, C − NOOP | −0.002217 [−0.002355, −0.002079] | −0.002071 [−0.002172, −0.001969] |
| RG8: held-out topology, C − NOOP | V_LADDER (16 seeds): −0.002058 [−0.002202, −0.001914] | H_RANDOM (32 seeds): −0.001812 [−0.001935, −0.001689] |
| RG8: held-out topology, C − `:IND` | −0.000242 [−0.000311, −0.000172] | −0.000263 [−0.000293, −0.000233] |
| RG9: ALL online regret (≤ 0.216090) | 0.194850 [0.191918, 0.197783] | 0.190948 [0.188972, 0.192924] |

### 5.2 Family claims (Bonferroni 99.375%)

| Family | VALIDATION | HOLDOUT | Qualified scope |
|---|---|---|---|
| CLEAN, FLIP20, QUNIF, BURST | USEFUL | USEFUL | useful |
| FLIP40 | SAFE | USEFUL | safe |
| FLIP45, FLIP48 | SAFE | SAFE | safe |
| EVENT40 (correlated duplicates, ρ = 1) | NOT_QUALIFIED (+0.002129) | NOT_QUALIFIED (+0.002052) | excluded by ρ_admit: Contract II |

### 5.3 What the relational state contributes (HOLDOUT)

- **Direct relations.** Direct-query regret, C − NOOP: −0.003194 [−0.003475, −0.002912].
- **Per compositional kind**, C − NOOP:
  - PATH −0.002192 [−0.002334, −0.002051];
  - CONJ −0.002535 [−0.002734, −0.002337];
  - REACH −0.001020 [−0.001174, −0.000867].
- **Identity.** Destroying the relational binding (`:PERM`, the same beliefs attached to the wrong relations) costs
  0.0032 per episode. Relational learning is therefore claimed.
- **The coupling against independent per-edge filters** (`:IND`). C is better on ALL, on every held-out topology, and
  on every OOD topology: D_DIAMOND −0.000222, D_FAN −0.000310, V_LADDER −0.000241, H_RANDOM −0.000246 [−0.000321,
  −0.000171]. The contribution depends on the regime:
  - **It helps** where reliability really is shared: CLEAN −0.001564, every CLEAN authority regime about −0.0010 to
    −0.0014, FLIP45 −0.000257, FLIP48 −0.000483.
  - **It costs** where the corruption rate is not in H3 or varies: FLIP20 +0.000788, QUNIF +0.000196, BURST +0.000318,
    EVENT40 +0.001686.
- **The geometric projection.** LAO_PLUS_HCA − C = +0.013066 [+0.012028, +0.014103]. HCA adds no decision value over
  the exact compositional readout, so the simpler rule is kept. HCA stays a salience-only view.
- **Source revision.** Invalidation is worth −0.000272 [−0.000291, −0.000252] against the stale ablation, and the
  learner keeps −0.002071 against NOOP under revisions. This rests on exact view-driven invalidation; the study does
  not try to outperform exact maintenance at bookkeeping.
- **STOP.**
  - Stop rates: C 0.5820, KNOWN_RELIABILITY 0.5805, NOOP 0.5918.
  - Regret where the STOP choice disagrees with the true optimum: −0.001938 [−0.002066, −0.001809] against NOOP. That
    is about 85% of the terminal gain of −0.002274.
  - The stopping problem is not declared solved: a CLEAN gap of +0.000127 to the privileged oracle remains.
- **Clean retention.** C keeps 97.5% of the oracle's CLEAN gain over NOOP (VALIDATION 97.9%).

### 5.4 The fallback (SAFE3_P4, Contract II), HOLDOUT

| Regime | SAFE3_P4 − NOOP |
|---|---|
| CLEAN | −0.002618 [−0.002751, −0.002486] |
| FLIP40 | −0.000137 |
| FLIP45 | +0.000035 [+0.000003, +0.000068] |
| FLIP48 | +0.000128 [+0.000096, +0.000159] |
| EVENT40 | −0.000097 [−0.000132, −0.000063] |

F1–F4 pass on both phases.

### 5.5 OOD (opened once after the HOLDOUT; `evidence/ood/characterization.json`)

**The correlation sweep, at q = 0.4** (rule: the largest ρ whose every ρ′ ≤ ρ has an upper bound ≤ τ_safe):

| ρ | Regime | C − NOOP | Safe |
|---|---|---|---|
| 0 | FLIP40 | −0.000078 [−0.000125, −0.000031] | yes |
| 0.1 | CORR10 | +0.000041 [−0.000001, +0.000083] | yes |
| 0.25 | CORR25 | +0.000133 [+0.000071, +0.000195] | yes |
| 0.5 | CORR50 | +0.000531 [+0.000462, +0.000599] | no |
| 0.75, 0.9, 1 | CORR75, CORR90, EVENT40 | +0.001072, +0.001510, +0.001910 | no |

**ρ_admit = 0.25.** It was measured on this workload independently of SLB2. It coincides with the strict cap
`MAX_QUALIFIED_RHO`, so the host can use 0.25.

**OOD corruption regimes:**
- SECOND_EVENT40 −0.000098, ASYM40 −0.000782, CTXQ40 −0.001107 and FLIP30 −0.000421 are useful.
- SHARED_SRC +0.000163 [+0.000094, +0.000231] is safe only.

**Every OOD topology** keeps C − NOOP < 0 and C − `:IND` < 0.

### 5.6 Acquisition (DEV, on-policy)

VOI_R against PROBE_R for RRMF3_M:
- ALL compositional −0.000562 [−0.000791, −0.000334] at 99.9%;
- but online regret 0.327887 against the budget 0.216090.

So VOI_R is not adopted. PROBE_R's online regret is the same for every learner, 0.191870, because it forces coverage
of the 8 relations.

## 6. The native C11 adapter and the host (built after the qualification)

`native/` holds `rlao.h` and `rlao.c`: a statement-for-statement port of the scalar reference, compiled with
`-ffp-contract=off`. It is a research test library: never linked by `src/` or RuntimeCore, and it links no ECS_C or
HCA library.

**Layout and resources.**
- `rlao_state` is 1016 B (4 hypotheses × 8 slots × {μ, v, Λ}, 8 counts, an 8-event buffer); `rlao_config` is 128 B.
- 80 logical scalars for H3.
- No heap: the allocator is wrapped at link time and the test proves it.
- Capacity is refused, and a numerical failure poisons the state until reset.

**Builds.** gcc and clang, Debug and Release, with warnings as errors. gcc ASan+UBSan and TSan pass, as in CI. The
clang sanitizer runtimes are absent from the development container.

**Parity, preregistered.** Through the host over the real ECS_C view, on VALIDATION and HOLDOUT seeds 0–7 × 256
episodes per regime, for the primary and the fallback:
- the native adapter equals the scalar reference **bitwise** on 3,571,696 (VALIDATION) and 3,448,824 (HOLDOUT)
  operations;
- the host reproduces every selection and terminal decision of the vectorised frozen arms, with 0 mismatches and 0
  near-ties (`evidence/*/parity.json`).

**A post-freeze correction, disclosed.** The same parity, run first through the **frozen** host, failed closed in 3
VALIDATION and 2 HOLDOUT sampled episodes (`evidence/*/parity_frozen_host.json`):
- **The cause.** At some step, a dormant source reported every relation, so nothing was admissible. The evaluated
  rollout abstains for such a step; the frozen host aborted the episode with `NO_ADMISSIBLE_RELATION`.
- **The correction.** `host.propose_relation` now abstains, returning −1, and the step passes without evidence.
  Integrity failures still fail closed. A regression test covers it.
- **No record is affected.** The study never imports the host. `evidence/POSTFREEZE_R6.json` records the digests of
  the two changed files, `host.py` and `hostrun.py`. The records reproduce from the frozen sources e5a17b0e… at
  commit cf05013.

## 7. What is claimed and what is not

**Claimed.** On synthetic RLB1 worlds, a bounded relational reliability state, coupled across the relations of an
episode, improves compositional decisions under the strict Contract I and qualified correlation (ρ ≤ 0.25):
- **Over** no learning, exact maintenance, static counts, the HCA projection, and independent per-edge filters with an
  unchanged readout.
- **Robustness.** It is useful or safe under the corruption regimes of the qualified families. It survives source
  revision, dormancy and reclamation by exact view-driven invalidation.
- **Generalisation.** It carries over to held-out topologies: V_LADDER and per-seed random DAGs.

The Contract II fallback is a conservative, qualified alternative for every other provider.

**Not claimed:**
- No real provider, workload or correlation regime beyond the synthetic one is certified.
- EVENT40-like correlated duplicates are not qualified under Contract I.
- The coupling is not uniformly beneficial (section 5.3).
- HCA adds no decision value.
- The stopping problem is not solved.
- Nothing is admitted to a runtime, merged, released or tagged by this study.
- Inference never changes model weights. No state persists between episodes. Coefficients are frozen.

## 8. Reproduction

Records reproduce bitwise under the recorded environment, from the frozen sources (commit cf05013), with the
working directory at this study's root:

```bash
PYTHONPATH=src python3 -m relational_lao.study qualify --phase VALIDATION
```

That command refuses to rerun a phase that already has a record. To reproduce a record, run it in a checkout that
lacks it. To recompute the analysis from the records:

```bash
PYTHONPATH=src python3 -m relational_lao.study disposition
```

```bash
PYTHONPATH=src python3 -m relational_lao.study ood
```

To rerun the native and host parity:

```bash
PYTHONPATH=src:. ELPIS_NATIVE_BUILD=<build> python3 -m relational_lao_tests.parity_r6 --phase VALIDATION
```

| Evidence | SHA-256 |
|---|---|
| `evidence/validation/qualification.json` | 4c2721dd353202261e7983da1a4ccae2291d52573d1fbf74e32d42225b3715b0 |
| `evidence/holdout/qualification.json` | 61f1f1007a78f2ecc88dd8b0a0d61154b715d38cd074f2c6933d9ba20f182efd |
| `evidence/ood/qualification.json` | 8c636bf60c9c80fd27a3f7adaa9ba7b3d19891b6e8a0212eeb919236e769f8bb |
| `evidence/DISPOSITION_R6.json` | baa98fbd13f3c50c585ac95a1047060b2992ee0e932cf2fd41489b2135d4d362 |
| `evidence/ood/characterization.json` | 6011953e3e970505f97d2ef90f8b298e8972b5a966999387910a990a68551939 |
| `FROZEN_R6.json` | 2ce160911fc2b0d2aef6e9e160982c3a3c69a53793c846bf81db466b9c5b869f |
| frozen sources (`source_sha256`) | e5a17b0ef0d4d8f906fa4e161cd09cbb25d29974a0484330166f7e113e923de1 |
| sources after the post-freeze host correction | 86edce1560197d87bf888970d78864ec0443c13eeb9dd024698589bb450051fa |
