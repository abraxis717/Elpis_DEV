# Relational LAO R0: research log

Each entry is in chronological order and is written when its work is done. No entry is rewritten later.

## R0: interface census and specification

- [`SPEC.md`](SPEC.md) records the census, the four-layer seam, RLB1, the learning state, the two invalidation
  strategies, the controls and the protocol.
- Every reused C interface was read in source.
- The ECS_C relational view was probed through the new ctypes binding (`src/relational_lao/ecsc.py`):
  - admission under bindings;
  - `BINDING_MISMATCH` for a misbound `source_authority`;
  - retirement on a version increase;
  - DORMANT suspension and exact restoration;
  - LIFECYCLE refusal of a dormant source's admission;
  - release on termination and reclamation, with a new source reusing the 32-bit authority.

  Every one matches `docs/ECS_C_RELATIONAL.md`.

## R1: mechanics

**Built.**
- `world.py`: RLB1 potential outcomes.
- `learners.py`: the vectorised relational filter and controls.
- `scalar.py`: the explicit-loop reference with a bounded buffer.
- `host.py` and `hostrun.py`: the host over the real view, under the strict Contract I/II admission of `host_closure`.
- `acquisition.py` and `harness.py`.

**A defect found and fixed before any DEV record.**
- The differential between the host over the real view and the vectorised learner disagreed on 2 of 48
  RECLAIM_DRIFT episodes.
- Cause: the vectorised learner's `kept` mask did not record that a newly observed event had been applied. When that
  event's binding was retired at the very next sync, nothing appeared to change, so the stale event survived.
- Fix: every applied event is marked kept.
- After the fix there are 0 action or decision mismatches over 2 seeds × 6 regimes × both strategies.
- The real ECS_C view caught a stale-evidence defect in the simulation, which is the reason this differential exists.

**Tests.** `relational_lao_tests/test_mechanics.py` has 18 tests, all passing:
- benchmark freshness and partitions;
- causality, with hidden variables poisoned;
- invalidation exactness against the scalar reference, for both strategies;
- slot reset exact without replay (bitwise);
- the stale ablation;
- the identity ablation holding bitwise the same beliefs, attached to the wrong relations;
- contradictions;
- the host differential;
- a revoked source never reaching a later proposal (bitwise state after every sync);
- exact restoration after dormancy;
- retirement under reclamation with a reused authority;
- fail-closed integrity and the strict contract;
- permutation equivariance under renamed entities;
- deterministic replay, capacity, scope changes, and zero writes (audit hook).

## R2: independent-edge gate

`relational_lao_tests/test_r2_gate.py`. The ported law, run on the fixed-capacity slot representation (three SLB2
experts as slots, no composition), reproduces the frozen RMF3 of `FROZEN_V4.json` over 6 regimes × 2 seeds × 512
episodes:
- identical actions;
- 0 of 98,304 terminal decisions differ;
- moments and reliability weights within 1e-12.

## Disclosure: exploratory smoke runs before the DEV round

The instrument was checked on DEV world seeds 200–203 (4 seeds × 1024 episodes; CLEAN, FLIP20, FLIP40, FLIP45; PROBE_R
and VOI_R). No record was kept. These runs informed the candidate list of the DEV round. They are not evidence.

Indications, for compositional regret:
- RRMF3 beat NOOP on CLEAN (0.0362 against 0.0422).
- INDEPENDENT_RMF beat RRMF3 on FLIP20, because RRMF3 has no q = 0.2 hypothesis.
- A hypothesis set {0, 0.2, 0.4, 0.48} recovered FLIP20, with a small FLIP45 cost.
- The coupled readout was indistinguishable from the marginal readout.
- VOI_R improved terminal regret slightly, at about twice the online regret.
