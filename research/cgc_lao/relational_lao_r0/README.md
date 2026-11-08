# Relational LAO R0 (research only, not qualified)

This study extends the qualified LAO reliability mechanism (RMF3, `../solver_lab/v4`) from independent expert biases
to a bounded relational belief state. It also measures whether that state improves relational decisions under
noisy, changing evidence. The LAO v1–v4 studies are reused read-only and never modified.

```
ECS_C relational view (admitted evidence) → relational RMF (volatile, bounded) → research-only projection (HCA) → host decision / STOP
```

There is no path back into ECS_C authority.

**Read first.**
- [SPEC.md](SPEC.md): census, seam, benchmark, state, invalidation, protocol.
- [RESEARCH_LOG.md](RESEARCH_LOG.md).

| Path | Content |
|---|---|
| `src/relational_lao/ecsc.py` | ctypes binding to the ECS_C relational view and the memory's relation key (research only) |
| `src/relational_lao/world.py` | RLB1: templates, regimes, potential outcomes, query utilities |
| `src/relational_lao/learners.py` | relational filter (RRMF), INDEPENDENT_RMF, ablations, NOOP, EXACT_RELATIONAL, STATIC_COUNTS; the decision rule |
| `src/relational_lao/scalar.py` | explicit-loop scalar reference with a bounded provenance buffer |
| `src/relational_lao/host.py`, `hostrun.py` | research host over the real view (strict Contract I/II), and its episode driver |
| `src/relational_lao/acquisition.py`, `harness.py` | PROBE_R and VOI_R; causal rollouts with matched-information shadows |
| `relational_lao_tests/` | R1 mechanics and the R2 gate |

**Run the tests** from the repository root, with the native libraries built (`cmake --build build`):

```bash
PYTHONPATH=research/cgc_lao/relational_lao_r0/src:research/cgc_lao/relational_lao_r0 python3 -B -m unittest discover -s research/cgc_lao/relational_lao_r0/relational_lao_tests -t research/cgc_lao/relational_lao_r0 -v
```

CI runs the same suite in the SCIENTIFIC lane through `tests/research/cgc_lao_relational_lao_r0`.
