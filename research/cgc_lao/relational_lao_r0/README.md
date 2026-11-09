# Relational LAO R0 (research only; synthetic qualification, no runtime admission)

**Status.** The frozen primary RRMF3_M is **QUALIFIED_SYNTHETIC_RELATIONAL_CONTRACT_I**, and the SAFE3_P4 fallback is
**QUALIFIED_SYNTHETIC_RELATIONAL_CONTRACT_II_CONSERVATIVE**. Both were qualified on the synthetic RLB1 benchmark, with
ρ_admit = 0.25. See [FINAL_REPORT_R0.md](FINAL_REPORT_R0.md) for the claims, their limits, and a disclosed post-freeze
correction of the research host.

This study extends the qualified LAO reliability mechanism (RMF3, `../solver_lab/v4`) from independent expert biases
to a bounded relational belief state. It also measures whether that state improves relational decisions under
noisy, changing evidence. The LAO v1–v4 studies are reused read-only and never modified.

```
ECS_C relational view (admitted evidence) → relational RMF (volatile, bounded) → research-only projection (HCA) → host decision / STOP
```

There is no path back into ECS_C authority.

**Read first.**
- [SPEC.md](SPEC.md): census, seam, benchmark, state, invalidation, protocol.
- [DEV_PROTOCOL.md](DEV_PROTOCOL.md): arms, the shared log, the DEV round, selection, acquisition adoption, fallback.
- [PREREGISTRATION_R6.md](PREREGISTRATION_R6.md) and [FROZEN_R6.json](FROZEN_R6.json): the frozen qualification.
- [RESEARCH_LOG.md](RESEARCH_LOG.md): the milestones, results and disclosures.

| Path | Content |
|---|---|
| `src/relational_lao/ecsc.py` | ctypes binding to the ECS_C relational view and the memory's relation key (research only) |
| `src/relational_lao/world.py` | RLB1: templates, regimes, potential outcomes, query utilities |
| `src/relational_lao/learners.py` | relational filter (RRMF), INDEPENDENT_RMF, ablations, NOOP, EXACT_RELATIONAL, STATIC_COUNTS; the decision rule |
| `src/relational_lao/scalar.py` | explicit-loop scalar reference with a bounded provenance buffer |
| `src/relational_lao/host.py`, `hostrun.py` | research host over the real view (strict Contract I/II), and its episode driver |
| `src/relational_lao/acquisition.py`, `harness.py` | PROBE_R and VOI_R; causal rollouts with matched-information shadows |
| `src/relational_lao/hca.py`, `hca_native.py` | Layer C: the salience-only projection, its HCA B0 readout, HCA_ONLY and LAO_PLUS_HCA; the native HCA binding |
| `src/relational_lao/study.py` | arms, parallel runs, sealed records, gates, selection, the frozen analysis and the qualification CLI |
| `relational_lao_tests/` | R1 mechanics, the R2 gate, the R5 HCA differential, and the R6 parity harness (`parity_r6.py`) |
| `evidence/` | compact per-seed records: `train/`, `dev/`, and one directory per qualification phase |
| `native/` | the native C11 relational adapter (`rlao.h`, `rlao.c`): bitwise equal to the scalar reference, fixed 1016 B state, no heap; a research test library |

**Run the tests** from the repository root, with the native libraries built (`cmake --build build`):

```bash
PYTHONPATH=research/cgc_lao/relational_lao_r0/src:research/cgc_lao/relational_lao_r0 python3 -B -m unittest discover -s research/cgc_lao/relational_lao_r0/relational_lao_tests -t research/cgc_lao/relational_lao_r0 -v
```

CI runs the same suite in the SCIENTIFIC lane through `tests/research/cgc_lao_relational_lao_r0`.
