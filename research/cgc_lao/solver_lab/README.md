# CGC-LAO solver lab (research only)

**Final closure (2026-11-08): Level B.** See [FINAL_REPORT.md](FINAL_REPORT.md).
- `RMF3` (reliability hypotheses q ∈ {0, 0.4, 0.48}, 24 logical scalars) is `QUALIFIED_SYNTHETIC_CONTRACT_I`.
  Contract I means audited independent re-reads, or an audited reading correlation ρ ≤ 0.25. The evidence is the
  fresh SLB2 v4 VALIDATION and a holdout opened once.
- `SAFE4_P8` is the Contract II fallback (one reading per step).
- The learner has a bounded C11 reference with bitwise parity and runs behind a fail-closed research host.
- Production readiness (Level C) depends on the host and is not claimed.

The qualification versions are additive and each has its own partitions:
- [`v2/`](v2/): SLB2, the native component, the provider contract and the host. Primary not qualified (G9).
- [`v3/`](v3/): primary not qualified (G9).
- [`v4/`](v4/): final, qualified.

The only native code is the research component in `v2/native`, built as test-kind ctest targets through one
`add_subdirectory` line in the root `CMakeLists.txt`. Nothing changes in `src/elpis`, RuntimeCore, system
authority or dependencies.

The v1 study below is frozen and unchanged.

**v1 status (2026-11-07):**
- `ORIGINAL` contract (8 single readings): **NOT_QUALIFIED**, information-limited.
- `DUP2_4` contract (+4 independent readings, +0 expert queries): **QUALIFIED_SYNTHETIC** on SLB1, on
  VALIDATION and on a holdout opened once.

Research only: no change to `src/elpis`, RuntimeCore, native code, system authority or dependencies
(numpy only). Nothing here is runtime-integrated or production-admitted.

**Read first:** [RESULTS.md](RESULTS.md) has the algorithm, holdout tables, mechanism, failure
boundaries, resources, integration proposal and handoff.

## What was built

A bounded, causal **reliability-mixture filter** (RMF). It keeps a finite mixture over
feedback-reliability hypotheses (clean, channel flips at q = 0.2 and 0.4, event-level flips at
q = 0.4). Each hypothesis holds a Gaussian belief over every expert's hidden bias, updated by
assumed-density filtering (16-node Gauss–Hermite moment matching). Terminal decisions maximise posterior
expected utility, including STOP. The volatile state is 31 scalars per episode; no weights change; the
learner sees selected readings only.

Main findings:

- Under the original contract, every admissible rule (five families) sits on one Bayes frontier. Beating
  no-learning under 40% flips costs more than 20% of the clean-regime gain, because eight single readings
  identify the corruption regime with AUC 0.80.
- A second independent reading of an already selected event measures the channel independently of the
  unknown bias. With such readings at the first four steps, regime-identification AUC rises to 0.983 and
  the same learner passes every gate. Holdout: FLIP40 −0.00094 vs NOOP [−0.00109, −0.00079], 96% of the
  clean gain kept.

## Layout

| Path | Content |
|---|---|
| [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md), [`specs/slb1.spec.json`](specs/slb1.spec.json) | SLB1 benchmark, frozen before experiments |
| [PREREGISTRATION.md](PREREGISTRATION.md) | gates, partitions, selection rule (before DEV/VAL/HOLDOUT) |
| [RESEARCH_LOG.md](RESEARCH_LOG.md) | compact `CANDIDATE → MECHANISM → RESULT → DIAGNOSIS → NEXT` log |
| [RESULTS.md](RESULTS.md) | disposition, results, mechanism, integration proposal, handoff |
| [`FROZEN.json`](FROZEN.json) | frozen candidates, per-file source digests, VALIDATION/HOLDOUT commands |
| `src/cgc_lao_lab/` | `rng` counter streams · `env` simulator and potential outcomes · `selector` PROBE_EARLY port · `cgc` float32 CGC port · `filters` RMF · `agents` baselines and candidates · `harness` causal rollouts · `stats` paired t intervals · `evaluate`/`report` runs and gates · `train_cgc`/`calibrate` TRAIN-only fitting · `scalar_rmf` dependency-free reference · `resources` · `diagnostics` · `freeze` |
| `coefficients/` | TRAIN-fitted baseline coefficients and ES-trained CGC weights (JSON, digest-checked) |
| `evidence/` | sealed records: `train/`, `dev/`, `validation/`, `holdout/`, `DISPOSITION.json`, `resources.json` |
| `cgc_lao_lab_tests/` | 51 mechanical and evidence tests (unittest) |
| `scripts/` | post-holdout DEV diagnostics: fixed action log, next-experiment hypothesis grid |

## Run

From the repository root:

```sh
PYTHONPATH=research/cgc_lao/solver_lab/src:research/cgc_lao/solver_lab \
  python3 -B -m unittest discover -s research/cgc_lao/solver_lab/cgc_lao_lab_tests \
  -t research/cgc_lao/solver_lab -v
```

CI runs the same suite in the SCIENTIFIC lane through `tests/research/cgc_lao_solver_lab`. The torch parity
test (numpy CGC port vs the checked-in `cgc.lao_q4e5` module) runs only where torch is installed; it passed
with torch 2.15 CPU during development.
