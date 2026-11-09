# Relational LAO R0: interface census and formal specification

Classification: **NEW EXPERIMENT, NOT QUALIFIED.** Research only. Nothing here is canonical, composed or admitted.
This file was committed before any experiment of this study ran. The completed LAO v1–v4 studies
(`../solver_lab/`) are reused read-only and never modified or reopened.

## 1. Interface census (R0)

The table lists every interface that is reused. Each one was read in source, not only in its documentation.

| Layer | Interface | Reused how |
|---|---|---|
| A, authority | `native/ECS_C/relational` (`elpis/ecs_c_relational.h`, ABI v1) | Through ctypes, from research code only. Calls: `config_default`, `required_bytes`, `create`, `destroy`, `sync`, `check`, `admit`, `relation_read`, `support_of`, `dependents`, `export`, `read_status`, `identity`, `verify`, `reset`. Structures: `elpis_ecs_c_rel_authority`, `_coordinate`, `_support` (176 B), `_sync_result`, `_work`. |
| A, authority rows | `elpis_ecs_c_entity` (152 B, `elpis/ecs_c_types.h`) | Synthetic authority tables of SOURCE rows (id, version, scope, lifecycle, `source_authority`), built as the view's own tests build them (`relational_test_util.h`). Every row obeys the successor law: a scope changes only with a version, TERMINATED is final, rows vanish only by reclamation. |
| A, evidence | `elpis_ecs_mem_observation` (448 B), `elpis_ecs_mem_relation_key` (`libelpis_ecs_memory`) | One binary typed relation per slot (`relation_type` 7, roles FROM=1 and TO=2). Readings are admitted as ASSERT or NEGATE. The relation key, which is canonical over participants, names the slot. |
| B, reliability | RMF3, frozen v4 (`../solver_lab/v4/FROZEN_V4.json`) | Its update law is ported: hypotheses q ∈ {0, 0.4, 0.48}, uniform prior, bias variance 1, 16-node Gauss–Hermite moment matching, `LW_FLOOR`, `V_FLOOR` (`cgc_lao_lab.filters`, `cgc_lao_v2.scalar2`). The native `libcgc_lao_rmf` (v2) is the immutable reference for the R2 gate. |
| B, contract | `host_closure.strict_host.strict_admit`, `QUALIFIED_READ_SCHEDULE`, `MAX_QUALIFIED_RHO` | Contract I only for an audited two-reading provider that delivers exactly (2,2,2,2,1,1,1,1). Every other provider gets Contract II (first reading only). The historical `cgc_lao_v2.host` is not an admission seam. |
| C, projection | `native/ECS_C/hca` (`elpis/ecs_c_hca.h`, ABI v1) | Through ctypes, read-only: an immutable graph built from the learner's exported projection, B0 closure, output `SALIENCE_ONLY`. Nothing is written back. |
| statistics | `cgc_lao_lab.rng` (counter streams), `cgc_lao_lab.stats` (paired Student-t) | The same counter-RNG construction is used under this study's own root. |

**Boundary facts that bind this study.**
- Both native libraries are DEV or research-gated: no canonical code may use them (`tests/boundary/test_ecs_c_relational.py`, `test_ecs_c_hca.py`). This study loads them only from research Python, and adds no CMake target that links them.
- The relational view admits evidence, it does not infer it. No function takes a salience or an inferred relation, and nothing reads back from a consumer.
- HCA readouts are salience, never truth.
- Exact dependency-indexed maintenance already solves source-revision bookkeeping (`research/ecsc_dynamic_relational_r0`), so this study does not compete with it.

## 2. Architecture

```
ECS_C admitted evidence (A) → bounded relational belief update (B) → research-only projection (C) → host decision / STOP (D)
```

- **A, authoritative evidence.** The relational view alone decides what is admitted and LIVE. Its laws are untouched.
- **B, relational LAO.** The volatile learner. It reads only readings the view admitted, and it never writes to A.
- **C, projection.** A bounded, non-authoritative vector: 8 slot saliences, each a u16 in [0, 1000].
- **D, host.** It admits providers (strict contract), restricts actions to admissible relations, applies authority syncs and invalidation, decides answers or STOP, and enforces resets and resource bounds. Only the host admits a learner proposal.

There is no path from B, C or D back into ECS_C authority.

## 3. The benchmark RLB1 (stochastic model, separate from utility)

**World of one pairing unit** (a *world seed*):
- a public weight matrix `W ∈ R^{8×4}` with entries 0.65·N(0,1), rounded to float32;
- a graph template, which fixes 6–8 entities, R ≤ 8 directed typed relations, and the query structures;
- a source pattern.

**Episode.**
- Hidden biases `b_r ~ N(0,1)` for each relation, drawn per episode.
- Contexts `x_t, x'_j ~ N(0, I_4)`, rounded to float32.
- The latent relation probability is `p_r(x) = σ(W_r·x + b_r)`.
- Entity and relation identities are fresh 32-byte digests in every episode, so names are never seen twice.

**Acquisition.** There are T = 8 steps. At step t the host admits the learner's proposed relation `r_t`.
- The selected event has a clean outcome `C_t = 1[U < p_{r_t}(x_t)]`.
- The readings are `Y_{t,k} = C_t ⊕ F_{t,k}`, with k ≤ 2 under Contract I (schedule 2,2,2,2,1,1,1,1).
- Duplicate readings concern the same event and have their own corruption draws. Their independence is a property of the corruption mechanism, never assumed from agreement.
- Every stochastic quantity is a correctly indexed potential outcome `[episode, step, relation(, reading)]`, so every policy faces the same world.

**Corruption mechanisms** are those of SLB2 (`../solver_lab/v2/specs/slb2.spec.json`): symmetric q, qunif, burst, event (ρ = 1), correlated (q, ρ), shared source, second event, asymmetric, and context-dependent.

**Sources and authority.**
- Each episode has 3 SOURCE rows. Each relation has a primary source, and with probability 0.3 a secondary source.
- An event of relation r at step t is reported by its sources in turn, and is admitted under that source's current binding (source id, version, scope).
- Authority regimes:

  | Regime | Event |
  |---|---|
  | STATIC | none |
  | REV_DRIFT | At step t_ev ∈ {3,4,5}, one source's version increases, retiring its earlier evidence. The relations it owns as primary get fresh `b`: the world changed, so stale evidence misleads. |
  | REV_SAME | The same revision, with no drift. |
  | DORMANT | The source is DORMANT for 2 steps, then ACTIVE at the same version. Its evidence is suspended, then restored. Its relations are inadmissible while it is dormant. |
  | RECLAIM_DRIFT | The source is TERMINATED and reclaimed. A new source reuses its 32-bit `source_authority`, and its primary relations drift. |

**Terminal queries.** Sixteen per episode, cycling through the template's query structures, each at its own fresh context.
- Each query offers alternatives and STOP. Choosing an alternative pays its utility under the true `p(x')`; STOP pays τ.
- Query kinds:

  | Kind | Alternatives | Utility | STOP τ |
  |---|---|---|---|
  | DIRECT | single relations | `p_r` | 0.6 |
  | PATH-k | k-hop compositions | `∏ p_r` | 0.6^k |
  | CONJ-m | m relations sharing an entity | `∏ p_r` | 0.6^m |
  | REACH | one alternative: an edge-disjoint path set | `1 − ∏_π (1 − ∏_{r∈π} p_r)` | 0.6 |

- Regret is `max(max_a U(a), τ) − U(choice)`.
- Compositional regret is the mean over the PATH, CONJ and REACH queries; direct regret is the mean over DIRECT.
- Online regret is `max_r p_r(x_t) − p_{r_t}(x_t)`, the cost of the evidence each step bought.
- **The privileged oracle** sees `p`. It is used for scoring only, never by a deployed learner.

**Templates.** The partition each template enters is fixed now. Templates for VALIDATION and HOLDOUT are never evaluated earlier.

| Template | Relations | Queries | Partitions |
|---|---|---|---|
| D_DIAMOND | S→M1,S→M2,S→M3, M1→T,M2→T,M3→T, M1→U,M3→U | DIRECT(S), PATH-2 S→T (3 alternatives), PATH-2 S→U (2), CONJ-2 hub→{T,U} (2), REACH S→T | all |
| D_FAN | S→A,S→B, A→C,A→D,B→C,B→D, C→T,D→T | DIRECT(S), PATH-2 S→C, PATH-2 S→D, PATH-3 S→T (4, sharing edges), REACH S→C, CONJ-2 {A,B}→{C,D} | all |
| V_LADDER | S→A1,A1→A2,A2→T, S→B1,B1→B2,B2→T, A1→B2,B1→A2 | PATH-3 S→T (4), PATH-2 S→A2, PATH-2 S→B2, DIRECT(S), REACH S→T, CONJ-2 | VALIDATION and later |
| H_RANDOM | an 8-edge DAG drawn per world seed by a fixed generator; queries derived automatically, with at least 3 compositional | — | HOLDOUT and OOD only |

**Regimes.**
- DESIGN: CLEAN, FLIP20, FLIP40, all STATIC. ALL is the mean over DESIGN.
- ROBUSTNESS: FLIP45, FLIP48, QUNIF, BURST, EVENT40.
- AUTHORITY: CLEAN and FLIP20, each crossed with {REV_DRIFT, REV_SAME, DORMANT, RECLAIM_DRIFT}.
- OOD: the correlation sweep CORR10–CORR90, SHARED_SRC, SECOND_EVENT40, ASYM40, CTXQ40, FLIP30.

**Partitions.** The RNG root is `elpis/cgc_lao/relational_lao_r0/`.

| Partition | World seeds | Episodes per seed |
|---|---|---|
| TRAIN (reserved) | 32 | — |
| DEV | 256 | 2048 |
| VALIDATION | 32 | 4096 |
| HOLDOUT | 64 | 4096 |
| OOD | 64 | 2048 |

All partitions are disjoint from each other and from every LAO v1–v4 key (this is tested). The pairing unit is the world seed.

## 4. Learning state (Layer B)

**Candidate RRMF3 (relational RMF).**

Slots:
- 8 slots at fixed capacity.
- Slot i holds the relation with the i-th smallest canonical relation key in the episode's scope.
- Relations beyond capacity are refused, never evicted.

State, for each hypothesis h ∈ H = {0, 0.4, 0.48} and slot r:

| Quantity | Meaning |
|---|---|
| `μ_{h,r}`, `v_{h,r}` | Gaussian bias belief |
| `Λ_{h,r}` | the slot's cumulative log evidence under h |
| `n_r` | live event count |

The reliability posterior is shared, `log w_h = log π_h + Σ_r Λ_{h,r}`. The sum runs over sorted values, so it does not depend on slot order.

The core is 72 + 8 = **80 logical scalars**. This is the directive's 59-scalar design (3 shared weights, 48 moments, 8 counters) with its 3 shared weights replaced by 24 per-slot evidence terms, which is what makes slot-local invalidation exact. Under any h the slots are independent, so `Σ_r Λ_{h,r}` is exactly the marginal likelihood.

**Update.** The RMF3 law, applied to the selected slot only. The 16-node GH evidence `L_h` of the readings gives the moments; `Λ_{h,r} += log L_h`. When `L_h ≤ 1e-300`, `LW_FLOOR` is added and the moments are kept.

**Composite readout, hypothesis-coupled.** For an alternative with relation set a:

- `E[U_a] = Σ_h w_h ∏_{r∈a} m_{h,r}(x')`, where `m_{h,r}(x') = Σ_g GH_w[g] σ(W_r·x' + μ_{h,r} + √v_{h,r} GH_x[g])`.
- REACH uses `1 − ∏_π (1 − ∏_{r∈π} m_{h,r})` inside the sum over h. This is exact for edge-disjoint paths.

The host answers the alternative with the largest expected utility, or STOP when that is below τ.

**Invalidation: exact by construction.** Two strategies are compared.

| Strategy | Rule |
|---|---|
| A, binding-local reset | Every slot whose support includes a retired or suspended binding goes back to its prior, and its `Λ` is removed. Suspended slots return when the binding is LIVE again. Exact because slots are independent under every h. |
| B, bounded replay | An event buffer, 8 events wide (horizon bound; at most 12 readings), indexed by provenance. After any sync that changes the live set, the state is rebuilt by replaying only the LIVE events in order. If an event cannot be held in the buffer, the transition is refused and the affected slots are reset. |

The mechanical law, for both: after every sync, the state equals bitwise the state of a fresh learner fed exactly the retained LIVE events, and the next proposal is a function of that state and the step alone. A revoked source cannot reach a later proposal.

**Controls.** Every adaptive control receives the same admitted readings, and all are scored on the same potential outcomes.

| Control | Rule |
|---|---|
| NOOP | Prior predictive, no adaptation. |
| EXACT_RELATIONAL | The view's live net polarity per relation (+1 → p = 1, −1 → p = 0, 0 → prior), composed exactly. Admitted evidence read as truth. |
| STATIC_COUNTS | Beta-Bernoulli centred on the public prior (κ = 8, the v2 TRAIN value), plug-in composition. |
| INDEPENDENT_RMF | The same sufficient statistics, but a separate reliability posterior per slot, `w_{h,r} ∝ π_h e^{Λ_{h,r}}`. Composition is the product of per-slot marginals. No relational interaction. |
| RRMF_MARGINAL | Shared weights, marginal product readout. An ablation of the coupled readout. |
| RRMF_PERMUTED | Readings bound to a fixed derangement of the slots. Marginal feedback frequencies are unchanged, relational binding is destroyed. The identity ablation. |
| RRMF_STALE | No invalidation. An authority-regime ablation. |
| HCA_ONLY | Native HCA B0 over the admitted graph (live positive relations as seeds). |
| LAO_PLUS_HCA | HCA B0 over the learner's projection: edge salience ⌊1000·E[p_r]⌉, composite = min, alternatives = max. It answers when salience ≥ 1000·τ^{1/k}. |
| KNOWN_RELIABILITY | RRMF with the true q as a point hypothesis. Privileged, diagnostic only. |

- **Contract II fallback (RRMF_SAFE).** Hypotheses {0, 0.2, 0.4, 0.48} with prior (1, 1, 8, 8) (the v4 fallback's), on first readings.
- **Acquisition.**
  - PROBE_R: the historical PROBE_EARLY over admissible slots.
  - VOI_R: the myopic expected value of sample information of the next event, over the template's query structures, at 4 planning contexts drawn from their own stream, never the terminal contexts.
  - Acquisition rules are compared on-policy.

**Resource caps.**

| Quantity | Cap |
|---|---|
| Slots | 8 |
| Hypotheses | 4 or fewer |
| GH nodes | 16 |
| Steps | 8 |
| Readings per episode | 12 |
| Buffered events | 8 |
| Sources per episode | 4 |
| Alternatives per query | 4 |
| Relations per alternative | 3 |
| Edge-disjoint paths per REACH | 3 |

Physical bytes of the native state are measured and reported when the native adapter exists. The learner holds no unbounded structure. Nothing persists between episodes, inference has no gradients, and coefficients are frozen.

## 5. Qualification protocol (R6)

- **DEV** develops and selects. Numerical gates, multiplicity handling and resource budgets are fixed in `PREREGISTRATION_R6.md`, committed before VALIDATION.
- **Freeze.** `FROZEN_R6.json` binds the sources, specifications and plans.
- **VALIDATION** runs once.
- **HOLDOUT** is opened once, and only if every VALIDATION gate passes. It is never opened to diagnose a failure. OOD (ρ sweep, admissible correlation regime of this workload) follows.
- **Gate families** (decided now; thresholds come later):
  1. Compositional regret vs NOOP and vs every matched-information control.
  2. Clean learning kept.
  3. Corrupted-evidence robustness and near-noise safety.
  4. The identity ablation and INDEPENDENT_RMF are worse, so the relational binding and the coupling matter.
  5. Authority regimes: no stale contamination, which is mechanical, plus decision value.
  6. Held-out topologies.
  7. Online-regret budget.
- **Native.** The minimal native C11 adapter is built only after the candidate qualifies.
