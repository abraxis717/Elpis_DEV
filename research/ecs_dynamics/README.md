# ECS neural-dynamics research laboratory

`RESEARCH_ONLY` · `NO_RUNTIME_AUTHORITY` · `NO_PRODUCTION_NEURAL_CLAIM`

A synthetic laboratory for questions about coarse observables, target-relative
sufficiency, delay memory, interventions, and finite-time dynamical diagnostics.
Results and paper dispositions are in
[`docs/research/ECS_DYNAMICS_RESULTS.md`](../../docs/research/ECS_DYNAMICS_RESULTS.md).

## Isolation

* The laboratory lives outside `src/`. `pyproject.toml` packages only `elpis*`,
  so the laboratory is never installed.
* `ELPIS_SYSTEM.json` lists `research` as a repository surface and as a
  forbidden Python import for production code. The boundary suite
  (`tests/boundary/test_dependency_policy.py`) therefore fails if any module
  under `src/elpis` imports it.
* The laboratory imports production code in one direction only: the public
  ECS kernel and topology analysis, in `ecs_collision.py`. It runs that
  kernel in temporary directories. It explores futures on byte copies of a
  closed history, and never changes ECS semantics, identities, replay or
  scheduling.
* NumPy is required here (the `test` extra); it is not a base dependency of
  `elpis-dev`.

## Modules

| module | contents |
|---|---|
| `spec.py` | immutable `ExperimentSpec`, canonical JSON, research-domain digests, per-world RNG (no global RNG), DEV/QUAL split, write-once `FreezeRegistry` |
| `observables.py` | coarse maps (mean, moments, norm, projections, spectral summary, combinations), the seven target classes |
| `results.py` | evidence categories, mechanics vs scientific disposition, target-relative `SufficiencyFinding`, source citations |
| `systems.py` | recurrent-system interface; tanh probe `x' = tanh(g J x + b + B u)` with `J = (S + gamma A)/sqrt(1 + gamma^2)`; linear, cyclic-shift and logistic fixtures |
| `cubic.py` | exact cubic control `phi(z) = 0.5 z + 0.5 z^2 + 0.5 z^3`, moments S1/S2/S3 with analytic Jacobians, transition fixture, exact PTE witness pairs |
| `intervention.py` | SVD null space, Gauss-Newton retraction, coarse-preserving interventions with tolerance and rejection, fibre response capacity and prediction loss |
| `diagnostics.py` | finite-time growth, recovery classes, fixed/periodic/unresolved detection, autocorrelation, two-time correlation |
| `attractors.py` | conservative sampled attractor inventory (fixed points and periodic orbits only) |
| `delay.py` | ridge predictor, delay and control representations, reference-penalty-gain surrogate |
| `sweep.py` | gain x gamma sweep |
| `feedback.py` | critical feedback of arXiv:2609.19288 Eqs. (4.5)-(4.6); static-feedback mean field |
| `associative.py` | model of arXiv:2609.07341: alpha=0 overlap map, Hopf line, small-N Glauber network |
| `ecs_collision.py` | the public-ECS experiment: equal topology-analysis coarse state, different futures |
| `export.py` | inert `DynamicsObservation` record |
| `experiments.py` | the eight experiments: base spec, DEV procedure, pass rule, QUAL procedure |
| `run.py` | `dev`, `freeze`, `qual`, `status` |

## Protocol

```
PYTHONPATH=src python -m research.ecs_dynamics.run dev
PYTHONPATH=src python -m research.ecs_dynamics.run freeze --commit <commit holding the lab source>
PYTHONPATH=src python -m research.ecs_dynamics.run qual
PYTHONPATH=src python -m research.ecs_dynamics.run status
```

1. DEV sees DEV worlds only and records operational choices, such as ridge
   strengths, regime cells and resolvable gains, in
   `evidence/dev/*.dev.json`.
2. `freeze` binds each specification with its DEV choices, the DEV evidence
   digest, the laboratory source digest and the commit, in
   `frozen/*.frozen.json`. It writes each file once.
3. `qual` refuses to run if the laboratory's Python source differs from the
   frozen digest. It runs each experiment once and writes
   `evidence/qual/*.qual.json` exclusively. A failure is recorded as it is; a
   repair needs a new version.

`tests/research/ecs_dynamics/test_run_guards_and_evidence.py` checks that the
committed frozen specifications still match the current source. Editing any
`research/ecs_dynamics/*.py` file after freezing therefore fails the test
suite until a new experiment version is frozen.

## Language

Results are stated as "supports under this frozen synthetic regime",
"descriptive", "finite-time", "sampled attractor" or "approximate coarse
preservation". No API or record says that a representation is "sufficient"
without naming a target and a regime. Nothing here is called a bifurcation,
a Lyapunov exponent or chaos.
