# CGC-LAO solver lab (research only)

Status: **in development.** A self-contained laboratory for bounded, causal learning from selectively
observed and possibly corrupted binary feedback (`research/cgc_lao/LEARNING_PROBLEM.md`). Numpy only;
no `src/elpis`, RuntimeCore, native or system-authority change; no new dependency.

- [Benchmark specification (frozen before experiments)](BENCHMARK_SPEC.md)
- [Preregistration: gates, partitions, selection rule](PREREGISTRATION.md)
- [Research log](RESEARCH_LOG.md)

## Run the mechanical suite

From the repository root (numpy only):

```sh
PYTHONPATH=research/cgc_lao/solver_lab/src:research/cgc_lao/solver_lab \
  python3 -B -m unittest discover -s research/cgc_lao/solver_lab/cgc_lao_lab_tests \
  -t research/cgc_lao/solver_lab -v
```

CI runs the same suite in the SCIENTIFIC lane through `tests/research/cgc_lao_solver_lab`.
