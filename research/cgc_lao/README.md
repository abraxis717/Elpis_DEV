# CGC–LAO: bounded learning from selected feedback (research handoff)

**Status (2026-11-07): open research problem.** This is a **research-only**, optional addition to Elpis_DEV. It does not change `src/elpis`, the native RuntimeCore, `ELPIS_SYSTEM.json`, the packaging dependencies, or inference authority. No production CGC/LAO, robustness guarantee, adaptive HALT, RALPH integration, or model-level reasoning capability is qualified here.

## What an outside collaborator should solve

Find a **small, bounded, causal, resettable update policy** for a compact geometric processor operating with selectively observed binary feedback. The policy should preserve clean-regime decision quality while resisting corrupted rewards. It need **not** be a general meta-optimizer: the learned CGC can propose state corrections, while a minimal analytical estimator or trust controller may decide how to use them.

The obstacle is empirical, not a missing architectural buzzword: many richer optimizers, training objectives and reliability features improved FLIP40 by making smaller updates, but exceeded stringent clean-regime regression limits. Under only eight selected observations for three experts, an unknown corruption regime remains difficult to infer. The decisive scientific target is **terminal and online decision quality**, not corruption-classification AUC alone.

- [Precise learning problem, causal inputs, state and resource constraints](LEARNING_PROBLEM.md)
- [Results, negative evidence and qualification boundaries](EVIDENCE_LEDGER.md)
- [Reproduction, candidate gates, and a suggested first independent study](REPRODUCIBILITY.md)
- [Opt-in Q4E5 bounded reference (PyTorch)](reference/Q4E5_BOUNDARY.md)

## Implementation provenance

The independent [CGC repository](https://github.com/abraxis717/CGC) publishes the original E0 cubic operator, native C11 implementation, a four-output `CGC120` wrapper and [R97W meta-learning reproduction](https://github.com/abraxis717/CGC/tree/main/research/benchmarks/r97w/portable). Its R97W experiment uses a **three-output** 120-parameter CGC and **episode-local fast weight adaptation**; this is not the same mechanism as frozen-weight CGC–LAO.

This handoff additionally includes the exact, previously opt-in Q4E5 acquisition/episode source as `reference/src/cgc/lao_q4e5.py`, its previously tested unit suite, and the qualification boundary document. Its source SHA-256 is `04fc09efffd9272e64a164d512319ad5112a7c3c2433d832a0778ffd7fcb3986`. **It does not contain the trained Q4E5 weights or claim independent reconstruction of the 16-seed holdout.** The outer eight-step experiment runner and later Q4E/Q4F archives remain outside this bounded publication.

The checked source implements a float32 `[3,5,8]` frozen controller, a pure `PROBE_EARLY` selector and a strictly bounded selected-reward episode. It does **not** implement terminal STOP or arbitrary-length adaptive reasoning. Do not substitute the four-output, float64 `CGC120` wrapper from the standalone CGC repo.

## Run just the public reference unit tests

From the Elpis_DEV repository root, with an **optional** installed PyTorch and without modifying canonical Elpis modules:

```sh
PYTHONPATH=research/cgc_lao/reference/src \
  PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s research/cgc_lao/reference/tests -p test_lao_q4e5.py -v
```

These are **interface/parity tests**, not scientific reproducibility of the trained policy. A passing suite does not establish robustness, generalization, or production admission. The runtime footprint and torch dependency are limited to this opt-in research reference.

## Collaboration requests

Most useful contributions: (1) information-theoretic analysis of unknown observation corruption under selected feedback, (2) a fixed-size analytical state estimator with an admissible uncertainty model, (3) a decision-aware evaluation of whether posterior calibration transfers to regret, and (4) an acquisition design that obtains more *independent* evidence under a strictly bounded compute budget. Please preserve matched feedback, proper NOOP controls, and fresh untouched seed pools. Propose changes in a separate PR, with explicit negative results. No change to persistent runtime authority is implied by any research pass.

## Evidence provenance and honest limitations

The Q4E/Q4F results here are **operator-reported experimental summaries** and source/hash pins carried through the Elpis research sessions; not all raw qualification archives or trained weights are published in this bounded change. They should be independently reproduced before being used as definitive external evidence. Historical R97W sources and portable reproduction **are** publicly available in the separate CGC repo. Nothing in this handoff proves autonomous learning, safe self-modification, or RALPH on-policy competence.

## Independent solver lab (added 2026-11-07)

[`solver_lab/`](solver_lab/README.md) is a self-contained follow-up study on an **independent** benchmark
(SLB1). Its numbers are not comparable to the historical Q4E/Q4F measurements above, which it leaves
untouched. It finds the original eight-step, single-reading contract information-limited: no admissible
candidate passed its preregistered gates. A bounded reliability-mixture learner (31 volatile scalars)
qualifies synthetically under a changed contract that adds four independent re-readings of selected events.
See [solver_lab/RESULTS.md](solver_lab/RESULTS.md) for the scope and failure boundaries.
The final qualification on a fresh SLB2 benchmark covers:
- a smaller learner, `RMF3`;
- a measurement-independence contract (Contract I or a conservative Contract II fallback);
- a native C11 reference and a fail-closed research host.

It closes at Level B, research component only. See [solver_lab/FINAL_REPORT.md](solver_lab/FINAL_REPORT.md).

## Relational LAO R0 (added 2026-11-08)

[`relational_lao_r0/`](relational_lao_r0/README.md) generalises the qualified `RMF3` reliability law from independent
expert biases to a bounded relational belief state: 8 relation slots and 80 volatile scalars, bound to evidence that
the ECS_C relational view admitted. The flow is admitted evidence → relational belief → a salience-only HCA projection
→ host decision or STOP, with no reverse write.

On the synthetic RLB1 benchmark, the frozen primary `RRMF3_M` qualified as Contract I
(`QUALIFIED_SYNTHETIC_RELATIONAL_CONTRACT_I`), and the `SAFE3_P4` fallback as Contract II, on preregistered VALIDATION
and HOLDOUT partitions:
- compositional decisions improve over NOOP, exact maintenance, static counts, HCA, and independent per-edge filters;
- the gain carries over to held-out topologies;
- this workload's own correlation bound is ρ_admit = 0.25.

The HCA projection adds no decision value, and correlated duplicates (EVENT40) are not qualified. This is research
only: nothing is admitted to a runtime. See
[relational_lao_r0/FINAL_REPORT_R0.md](relational_lao_r0/FINAL_REPORT_R0.md).
