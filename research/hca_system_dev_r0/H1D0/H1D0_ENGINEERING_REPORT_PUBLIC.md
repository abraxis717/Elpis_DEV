<!-- Publication adaptation: local host paths redacted; source SHA-256 in manifest. -->
# H1D0 SOURCE-EVOLUTION / RECOVERY GATE — Engineering Report

Phase: `H1D0_SOURCE_EVOLUTION_RECOVERY_GATE_R0`
Contract: `H1D0_SOURCE_EVOLUTION_R0/v1`
Classification: **PROSPECTIVE DEV CONTRACT, NOT QUAL. No scientific result, no canonical promotion, no B3 authorization.**
Workspace: `[LOCAL_HOST_PATH_REDACTED]`
Read-only authorities: the transferred mirror and the canonical `Elpis_DEV_HCA` checkout were never modified.

---

## 1. What was actually implemented

A complete, isolated, runnable H1D0 research phase, not a design document.

**Task contract (`src/contract.py`, 150 lines).** A frozen, declarative specification of typed n-ary
factor worlds: opaque entity and source tokens, an ordered event log of four causal event types
(`SOURCE`, `EMIT`, `DROP`, `REBIND`), four relation types (`R0`, `R1`, `R2`, `RX`), and the declared
live-factor rule — a factor is live iff it is present, its bound source status is `VALID`, and the
source epoch equals the factor version. `RX` is declared legal but never admitted by any witness.
Three information regimes are declared separately (A exact replay, B retained-state damage,
C incomplete observation), and five acceptance criteria (G1–G5) are declared **before** any
measurement.

**Generator (`src/generator.py`, 208 lines).** Deterministic, seed-derived, split-seeded. Produces
counterfactual pairs across six intervention kinds: source swap, event-order reversal, epoch
rebind, structural rewiring, cancellation, provenance change. Regime C withholds a bounded event
set and publishes it as a public `pending` pool.

**Independent oracle (`src/oracle.py`, 196 lines).** Written directly from the contract. Imports
nothing from the generator, checker or engines. Deliberately naive: it rebuilds the whole state
from scratch at every prefix and enumerates witnesses by brute force. It also contains an exact
enumeration of reachable final states for regime C.

**Independent checker (`src/checker.py`, 216 lines).** A second, unshared implementation of the
same fold, plus structural validation: identity uniqueness, role schema, event-type enums, step
monotonicity, budget bounds, split disjointness, denominator accounting, and pair-intervention
validity. Raises `ContractViolation` with a named reason; never swallows an exception.

**Engines (`src/engines.py`, 519 lines).** Five competitors, each seeing only the public case:
`Replay` (stateless exact rescan), `Incremental` (exact dependency-maintained, with an optional
integrity checksum), `CheckpointReplay`, and `HcaNativeSemantics` implementing B0 (exact worklist
closure), B1 (synchronous rounds) and B2 (deterministic asynchronous recurrence with a bounded
per-tick update budget and retained activation).

**Reference adapter (`src/native_ref.py`, 162 lines).** Encodes H1D0 cases into the *real* Elpis
reference substrate objects (`ContextAxiom`, `ContextRelation`, `ContextView`, `Observation`) and
calls the real `research.ecsc_hca_r0.baselines` / `b2` implementations.

**Baselines (`src/baselines.py`, 149 lines).** TRAIN-only symbolic rule induction over explicit
causal atoms, plus memorisation, majority, graph-statistic, order-only and query-identity controls.

**Learned comparator (`src/learned.py`, 159 lines; `src/learned_online.py`, 176 lines).** A 19-parameter
max-plus typed recurrence over provenance classes, and a stateful online version with retained
per-query, per-round activation vectors. Explicitly **not** B3.

**Tests.** 54 pytest tests across `test_break_suite.py` (478 lines), `test_mutations.py` (364 lines),
`test_adversarial_fixtures.py` (250 lines), `test_learned_gradient.py`, `test_learned_online.py`.

---

## 2. What was independently verified

| Gate | Result |
|---|---|
| Transfer custody SHA-256 | matches `b12c2177…25d6` |
| Oracle vs independent checker | 0 disagreements over 28,928 TRAIN and 14,464 TUNE decisions |
| Counterfactual pairs | 48 pairs, every pair flips the target query and leaves nuisance fields equal |
| Nuisance ablation | 144 cases: renaming all opaque tokens is neutral; renaming one role flips the gold |
| Exact engines vs oracle | replay, incremental, checkpoint, B0, B1: 0 mismatches over 168,960 TRAIN and 84,480 TUNE prefix decisions |
| Reference-substrate differential | 4,608 comparisons (all 96 TRAIN cases × B0/B1/B2 × 2 prefixes): **0 differences** |
| Native C differential | 31,104 comparisons against the compiled `ecs_c_hca.c`, ABI v1: **0 mismatches** |
| Implementation mutants | 14 forbidden semantic changes injected into disposable copies of the real source: **14 detected** |
| Adversarial fixtures | 14 hand-built fixtures: **all pass** |
| Leakage | public cases contain exactly the 9 declared public fields; engine source contains no reference to truth/gold/oracle/labels; no engine constructor accepts them |
| Sealed split | never generated, never read |

### The one real mechanical defect found, and its explanation

B2 disagrees with the exact oracle on 29/168,960 TRAIN and 21/84,480 TUNE decisions. This is **not**
a porting error — the port matches the reference substrate and the native C implementation exactly.
It is an intrinsic property of the bounded asynchronous recurrence: after a provenance change the
retained activation is one tick behind the exact closure.

Measured directly over 42,240 decisions: 42,191 (99.884%) are already converged at the event
boundary; 49 need exactly one additional tick; **none** need more than one. Increasing `rounds`
to 8, 16 or 32 and `updates_per_tick` to 512 or 1024 does not remove the gap, because the gap is
caused by the per-event tick budget, not by the round budget.

This is a genuine, quantifiable limitation of the unlearned B2 semantics: it is a salience
approximation, not an exact solver, and it is inexact precisely at the moments the task is about.

---

## 3. Which baselines won

**TRAIN-only symbolic induction wins outright.**

| Method | TRAIN | TUNE |
|---|---:|---:|
| Exact (replay / incremental / checkpoint / B0 / B1) | 28,928/28,928 | 14,464/14,464 |
| Symbolic inducer, full causal atoms | 1.0000 | 1.0000 |
| Symbolic inducer, local atoms only (no full-chain atom) | 0.9976 | 0.9990 |
| Case memorisation | — | 0.8584 |
| Majority | — | 0.8584 |
| Order-only | — | 0.8424 |
| Graph statistic | — | 0.6659 |

The full symbolic inducer learns a **single rule** — `full_chain` — in 0.04 seconds and is perfect
on TUNE. The deliberately handicapped local variant learns a single two-hop rule and is 99.9%
accurate. Acceptance criterion G3, declared before measurement, states that if the TRAIN-only
symbolic inducer reaches 100% on TUNE, the task is `HOLD_ASSAY_SOLVED_BY_SIMPLE_TRAIN_ONLY_SYMBOLIC_BASELINE`
and no learned-model claim may be made on it. **G3 fails.**

---

## 4. Learned local recurrence: measured, and it does not help

Three seeds, all reported, no seed selection:

| Seed | TRAIN | TUNE | training s | param bytes |
|---|---:|---:|---:|---:|
| 20261023 | 1.000 | 1.000 | 81.33 | 410 |
| 20261029 | 1.000 | 1.000 | 84.15 | 407 |
| 20261031 | 1.000 | 1.000 | 82.35 | 421 |

The learned recurrence reproduces the exact answer, but it does so by learning that only
`VALID_MATCH` factors carry signal — i.e. it re-derives the rule the symbolic inducer finds
directly. Its gradient is verified analytically against finite differences (error ~1e-11) and a
negative control rejects a perturbed gradient.

**Full online cost on TUNE (isolated process, one thread):**

| Engine | accuracy | apply s | snapshot s | total s | retained bytes | logical work |
|---|---:|---:|---:|---:|---:|---:|
| replay | 1.00000 | 0.00078 | 0.12595 | 0.12673 | 433,072 | 392,528 |
| incremental | 1.00000 | 0.00436 | 0.00393 | **0.00829** | 220,092 | **32,525** |
| incremental + integrity | 1.00000 | 0.01176 | 0.00344 | 0.01520 | 220,092 | 36,853 |
| checkpoint | 1.00000 | 0.00790 | 0.13835 | 0.14625 | 719,028 | 319,128 |
| hca_b0 | 1.00000 | 0.00114 | 0.34076 | 0.34190 | 132,500 | 690,634 |
| hca_b1 | 1.00000 | 0.00123 | 0.71813 | 0.71936 | 132,500 | 2,804,812 |
| hca_b2 | 0.99858 | 0.54653 | 0.00594 | 0.55247 | **42,704** | 1,492,707 |
| learned_online | 1.00000 | 0.02840 | 0.01114 | 0.03954 | 304,519 | 63,096 |

Exact incremental dominates on time and logical work. The only axis on which any HCA-style engine
wins is retained bytes — and B2 wins that axis while being 66× slower and inexact.

**Scaling.** Sweeping world size from 64 entities/48 factors/8 queries/37 events up to
512/2000/128/300:

| Scale | replay | incremental | hca_b0 | hca_b1 | hca_b2 | learned_online |
|---|---:|---:|---:|---:|---:|---:|
| 64e/48f/8q/37ev | 0.00023 | 0.00012 | 0.00048 | 0.00094 | 0.03190 | 0.00169 |
| 64e/200f/8q/60ev | 0.00066 | 0.00036 | 0.00181 | 0.00381 | 0.20036 | 0.01443 |
| 128e/500f/32q/120ev | 0.00401 | 0.00117 | 0.01656 | 0.04461 | 4.71121 | 0.26447 |
| 256e/1000f/64q/200ev | 0.01472 | 0.00350 | 0.07366 | 0.15560 | 30.97614 | 2.21387 |
| 512e/2000f/128q/300ev | 0.06033 | 0.00881 | 0.27803 | 0.62050 | 140.95841 | 18.40700 |

Exact incremental is the fastest engine at every scale, and its advantage grows with scale. The
learned online recurrence is 2,090× slower than incremental at the largest scale.

---

## 5. Regime B — damaged retained state

Damage is applied to each engine's own retained state after a declared prefix; the external
evidence is unchanged.

| Engine | damaged units | accuracy without repair | accuracy after self-recovery | accuracy after explicit repair | repair cost (s) |
|---|---:|---:|---:|---:|---:|
| replay | 0 (no derived state) | 0.8333 | 1.0000 | 1.0000 | 1.5e-05 |
| incremental | 384 | 0.7760 | 0.7852 | 0.7852 | 1.4e-05 |
| incremental + integrity | 384 | 0.7760 | **1.0000** | 1.0000 | 5.27e-04 |
| checkpoint | 244 | 0.8333 | 1.0000 | 1.0000 | 1.3e-05 |
| hca_b2 | 384 | 0.8203 | 1.0000 | 1.0000 | 2.4e-05 |
| learned_online | 384 | 0.7891 | 0.9336 | 1.0000 | 3.13e-02 |

The integrity-checked exact engine detects corruption through a retained-state checksum and
rebuilds, reaching 100% correctness for 0.53 ms across 96 cases. The learned online recurrence
needs an explicit full re-derivation costing 31.3 ms — **59× more** — and 25 of 96 cases never
self-recover at all.

Regime B therefore shows the opposite of the hoped-for advantage: local recurrent state is *more*
fragile and *more* expensive to restore than an exact dependency cache with a checksum.

---

## 6. Regime C — incomplete observation

Withheld events are published as a public pool. A decision is identifiable iff every subset of the
pool inserted at every position yields the same answer.

| Split | decisions | identifiable | underdetermined | identifiability cap | forced-answer accuracy | accuracy on identifiable only |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 768 | 622 | 146 | 0.8099 | 0.8112 | **1.0000** |
| TUNE | 384 | 301 | 83 | 0.7839 | 0.7891 | **1.0000** |

The exact visible-only engine achieves the information-theoretic ceiling exactly: every
identifiable decision is correct, and the underdetermined ones are not ordinary mistakes. The
identifiability computation itself was cross-checked against an exhaustive enumeration of all
interleavings of the withheld pool, and the two agree.

No learned model can exceed this ceiling, and computing the ceiling requires no learning.

---

## 7. Scientific disposition

```
mechanical:   PASS_TASK_CONTRACT_DEV
scientific:   NO_HCA_ADVANTAGE
hold reason:  HOLD_ASSAY_SOLVED_BY_SIMPLE_TRAIN_ONLY_SYMBOLIC_BASELINE
B3:           NOT AUTHORISED
```

Every acceptance criterion declared before measurement:

- **G1** oracle == exact replay == exact incremental on every decision — **PASS** (0 mismatches over 253,440 prefix decisions).
- **G2** every paired intervention flips the target and preserves nuisance fields — **PASS** (48/48).
- **G3** the TRAIN-only symbolic inducer must not reach 100% on TUNE — **FAIL** (1.000).
- **G4** mutation tests must demonstrate actual rejection — **PASS** (14/14 mutants detected, each by a named assertion).
- **G5** a learned local recurrence is promising only if it matches exact correctness at strictly lower total cost on regime B or C — **FAIL**.

**Answer to the primary question.** On this workload, no. A bounded learned local typed recurrence
does not provide a quality-versus-total-cost advantage over an equally informed exact incremental
engine. It is slower at every scale, retains more state than the smallest HCA variant, recovers
more expensively from damaged state, and cannot exceed the identifiability ceiling in regime C.
The unlearned B2 recurrence is the only HCA-style engine that wins any axis (retained bytes), and
it is inexact at exactly the event boundaries the task is designed to test.

**What is genuinely unresolved.** The task as constructed is decided by a single shallow symbolic
rule. That is a property of the task, not of HCA. Any future HCA claim must first survive a task
that a TRAIN-only symbolic inducer cannot solve. The smallest justified next experiment is therefore
not B3 training; it is a task redesign in which the witness relation is not enumerable from the
public factor set at all — for example, where the relevant state is a bounded summary of an
unbounded log, so exact maintenance is provably superlinear while a fixed-size recurrence is not.
That is a complexity argument, and it must be proven before any learned model is trained.

**Unsupported claims.** Nothing here supports: that learned recurrent hypergraph dynamics are
useful; that B3 is close to being established; that the native HCA has any admission or semantic
authority; or that the H1C1 HOLD should be overturned. The H1A, H1B and H1C results and their
classifications are preserved unchanged.

---

## 8. Evidence

- Machine-readable report: `runs/20261023T231531Z_H1D0_ADJUDICATION/H1D0_ADJUDICATION.json`
  SHA-256 `3b2ac77cca8f73323d1175f928466d4ecef36be5c99355dafcbf1b66b678eaba`
- Raw logs: `runs/pytest_final2.log`, `runs/mutant_injection_final.log`, `runs/fixtures.log`,
  `runs/native_diff.log`, `runs/reference_full_diff.log`, `runs/pareto.log`,
  `runs/scaling_grid.json`, `runs/regime_b_recovery.json`, `runs/bench_isolated.jsonl`,
  `runs/probe_b2_lag.log`
- Native library built from the mirror's C source without modifying the mirror:
  `native_build/libelpis_ecs_c_hca.so`

## 9. Next authorised phase

None beyond H1D0 closure. B3 training is not authorised. The next step is a task-design decision
by Joe, not an experiment by this phase.
