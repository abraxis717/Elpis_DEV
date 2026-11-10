# R122 — energy-normalized fidelity, directional safety and recurrent error localization

**Independent Elpis research. RESEARCH_ONLY. Not production-admitted.**
`R93_GATE=ENERGY_NORMALIZED_DENOMINATOR_HOLD` is binding and is **not** cleared by anything here.

R118–R121 left one scientific problem open: the frozen 88-parameter HCA diverges from its
teacher in energy-normalized terms over long horizons, and a correction can suppress or
invert a signal that the native operator keeps aligned. R122 does not try to fix this. It
asks **where the error comes from**:

1. collapse of the teacher-energy denominator,
2. an erroneous one-step transition,
3. accumulated autonomous drift of a locally acceptable operator,
4. controller-induced distortion of magnitude or direction.

Nothing is trained, no controller is fitted, the six frozen R77/R78 checkpoints and the
R119 `EXACT_GUARDED` kernel are consumed read-only, and the R120 `SIGNED_ONLY` and R121
`DELAYED8` policies are evaluated exactly as frozen.

## What is measured

For every fixture (graph, overlap k ∈ {0, 2, 6}, amplitude scale, case, seed) and every
checkpoint, the **complete relation state at every step 1…20** is recorded for:

| Path | Definition | Role |
|---|---|---|
| autonomous | engine steps from the shared initial state only | the deployable behaviour |
| teacher-forced one-step | the recorded teacher state `T_{h-1}` is the *input* of one HCA step; output compared with `T_h` | local transition defect `L_h` |
| restart | the policy is run autonomously from `T_j` (j = 4, 8, 12, 16) to step 20 | origin of the step-20 error |
| controller displacement | native step from the controlled trajectory's own incoming state | what the controller added at each step |

Teacher states never enter an autonomous trajectory; autonomous rollouts are recomputed
after all diagnostics and must be bit-identical.

Exact identities (each residual is reported and gated):

```text
E_rel = 1 + r² − 2 r c = (1 − ρ)² + |p − ρ t|²/|t|² = (1 − r)² + 2 r (1 − c)      ρ = r c
e_h   = P_h + L_h                      propagated incoming mismatch + local transition defect
e_20  = Σ_b D_b                        D_b: effect at step 20 of the defects committed in block b
Δlog E_rel = Δlog E_abs − Δlog |t|²    numerator versus denominator growth
E_rel(READOUT) = E_rel(PAIRED_STATE) · A_den
```

No denominator floor is used. Zero targets, zero predictions, subnormal-range and
ill-conditioned samples are classified and counted, never regularized. A direction class
(ALIGNED / DEVIATED / ORTHOGONAL / REVERSED) is assigned only when both norms are meaningful
(`0.01 ≤ r ≤ 100`); a zero or extinguished prediction is never called a reversal.

## Registered decisions (`PROTOCOL_R122.json`)

* **Stratum mechanism** (NATIVE, READOUT, H20, scale 1; stratum = graph × k × checkpoint
  cohort): `DENOMINATOR_COLLAPSE` if median `A_den ≥ 10`; `LOCAL_TRANSITION` if median
  `λ = |L|²/|e|² ≥ 0.5`; `ACCUMULATED_DRIFT` if `λ ≤ 0.1`; else `MIXED_LOCAL_ACCUMULATED`;
  `INVALID` below 90 % ordinary samples.
* **Disposition**: `MECHANISM_LOCALIZED_BOUNDED` iff one localizing class holds in ≥ 75 % of
  the 36 primary strata *and* in a strict majority of the strata of every graph class;
  `MECHANISM_MIXED_BOUNDED` otherwise; `MECHANISM_INDETERMINATE` if more than 10 % of the
  strata are invalid; `NONQUAL` if any integrity gate fails.
* **Directional safety** (READOUT, every step): a controller-induced reversal is a sample
  REVERSED under the policy while the NATIVE sample is ALIGNED. Any such sample at an
  in-range scale gives `HAZARD_CONTROLLER_INDUCED_REVERSAL`. Pooled NSE never overrides it.
* Growth class H8→H20 (numerator- or denominator-driven), worst graph-block harm, and four
  secondary findings are registered as fixed-criterion descriptive outputs.

## Prospective set

Six new incidence-isomorphism classes (`R122_T1_D2 … R122_T6_D3`), fresh evaluation seeds
`122101, 122103, 122107, 122109`, structure seeds `122300 + i`, four amplitude scales
(`2^-20` decayed/linear limit, `2^-2`, `1`, `2^2` above-clamp stress), all six checkpoints,
four policies. The graphs were audited against **every** historical graph definition
(54 definitions, 34 classes), not only R61/R76/R77.

Two audit findings are recorded in the protocol:

* Three of the four R121 "unseen" classes are incidence-isomorphic to earlier frozen-protocol
  graphs (`R121_HOLDOUT_1 ≅ r52_fan`, `R121_HOLDOUT_3 ≅ R72_G2`, `R121_HOLDOUT_4 ≅ R70_STAR`).
  R121's gate compared against 12 classes only. They are still outside the R77 training class.
* The ten-participant / four 4-member-relation family has only 50 valid connected classes.
  All 8 of maximum degree 4 were already used, so **no prospective degree-4 class exists**;
  17 classes were unused before R122 and 11 remain after it.

Degree-4 coverage therefore comes only from the *consumed companion*: the identical
pipeline on the four R121 graphs with R121 seeds, run before registration and published
with it (`evidence/R122_COMPANION_*`). It is discovery material and a rehearsal; it has no
decision authority.

## Layout

| File | Role |
|---|---|
| `metrics.py` | exact decomposition, explicit norm / direction classes, rational audit |
| `topology.py` | complete incidence invariants, historical inventory, registered graph scan and audit |
| `harness.py` | full-state trajectories over the R118-E composed system with the R119 engine; independent torch-free teacher reference; joint gauge transformation |
| `measure.py` | measurement pass → raw base-sum tensors and integrity counters |
| `analyze.py` | registered analysis and decision rules (pure numpy) |
| `linear.py` | decayed-regime linear operators: kernel probe versus eight-gate analytic operator, spectra |
| `controls.py` | synthetic classifier cases, consumed R121 regression fixture, gauge negative controls |
| `qualify.py` | runner (`--registered`, `--consumed-companion`), source guard, gates, SEND BACK |
| `make_protocol.py` | how `PROTOCOL_R122.json` was produced (no model execution) |
| `tests/` | identities, classifier, topology inventory, decision logic, harness on consumed material |

## Running (from the repository root)

torch is needed for the frozen R30 teacher (measured with torch 2.15.0+cpu, numpy 2.5.3,
Python 3.13.18, gcc 13.3, glibc 2.39). All scratch and all outputs are under `/dev/shm`.

```bash
python -B -m unittest discover -s research/yang_mills_hca/r122_energy_direction_forensics/tests -t .
```

```bash
python -B -m research.yang_mills_hca.r122_energy_direction_forensics.qualify --consumed-companion --out /dev/shm/r122c
```

```bash
python -B -m research.yang_mills_hca.r122_energy_direction_forensics.qualify --registered --out /dev/shm/r122
```

The registered run refuses to start unless this component is committed, the commit is
published (HEAD equals its upstream) and no registered result exists yet.

## Reproducibility boundary

R118-E cell means reproduce **bitwise** on this environment class. The published R120 and
R121 cell means were produced with another torch build and CPU (R120 records torch
2.5.1+cu121 on an i7-9800X; R121 records no environment) and reproduce here to about
1e-11 relative, not bitwise. R122 records its environment and a SHA-256 of the complete
raw measurement tensor.

## Attribution

Upstream mathematical context: [creelie/4Y-M](https://github.com/creelie/4Y-M), *Mass Gap of
Four-Dimensional Lattice Yang–Mills Theory for Every Compact Simple Group*, by Deep
Bhattacharjee, Ushashi Bhattacharya, James Montgomery Lockwood and Tristan Romanov (pinned
v4.21.0, MIT; see [../PROVENANCE.md](../PROVENANCE.md)). No upstream code is used. The
upstream authors did not develop, endorse or qualify the Elpis HCA or any measurement here.
