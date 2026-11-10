# R122 findings

**RESEARCH_ONLY. R93 `ENERGY_NORMALIZED_DENOMINATOR_HOLD` is unchanged.** A mechanism
localization is not a fidelity qualification, and no policy is promoted by this phase.

Registered run: one execution of commit `e7e1db29e0f75eb5eda9b3ad6a0615e2e8cb663a`
(protocol `fe9ec864…e31d`, source digest `4f3d407f…1f4f`) on six new topology classes,
576 fixtures, 696,960 HCA passes, 487 s. Tables: [RESULTS.md](RESULTS.md); machine-readable:
`R122_RESULT.json`, `R122_TABLES.json`, `R122_INDEX.json`.

## 1. Integrity

All twelve registered gates pass. Composed teacher equals the historical isolated teacher
bit for bit at k = 0 (3,840 step checks); an independent torch-free teacher agrees to
4.4e-16; `EXACT_GUARDED` equals `EXACT_R118` on 432 of 432 trajectories; autonomous
trajectories recomputed after every teacher-forced diagnostic are bit-identical (1,152 of
1,152); decomposition identities hold to 1.6e-15 and against exact rational arithmetic to
6.8e-16; joint gauge transformation leaves every metric invariant (2.3e-13) while a
state-only transformation changes trajectories by ≥ 0.13; six checkpoint pins and the
frozen tree are unchanged.

Non-ordinary samples are counted, not regularized: 20,736 autonomous READOUT samples at
step 1 are `BOTH_ZERO` (three of the new classes place the readout two hops from the
source, so nothing has arrived yet); no near-zero, ill-conditioned or non-finite sample
occurred.

## 2. Registered disposition: `MECHANISM_LOCALIZED_BOUNDED` — and how firmly

Primary strata (NATIVE, READOUT, step 20, scale 1): `ACCUMULATED_DRIFT` in **27 of 36**,
`LOCAL_TRANSITION` 4, `MIXED_LOCAL_ACCUMULATED` 5, `DENOMINATOR_COLLAPSE` 0, invalid 0, with
a majority in each of the six graph classes (4, 5, 4, 4, 5, 5 of 6).

Read this with three qualifications:

* **It sits exactly on the registered threshold** (27/36 = 0.75, rule ≥ 0.75). One stratum
  fewer would have given `MECHANISM_MIXED_BOUNDED`. Two UNSIGNED strata that count as
  accumulated have λ = 0.077 and 0.091 against the 0.1 limit.
* **It is carried by the SIGNED cohort**: 17 of 18 SIGNED strata are accumulated drift
  (λ ≤ 0.045 in all 17); the UNSIGNED cohort is 10 accumulated, 3 local, 5 mixed and is
  **not** localized on its own.
* **The stated expectation was wrong.** The non-binding expectation E1 (from the consumed
  companion: 13 of 24) predicted an overall mixed disposition. The new classes show more
  accumulation, mainly because composed two-hop classes over-retain strongly in both cohorts.

Outside the primary cell the label is not stable: it agrees with scale 1 in 44–61 % of
strata at the other scales, and at earlier steps the picture is different (section 3).

## 3. Where the error comes from

**Teacher-energy denominator.** At step 20 and scale 1 it is *not* the cause: `A_den`
(READOUT NSE over the NSE of the whole paired fixture state) is 0.001–1.8, never ≥ 10. It
*is* the cause in two specific places:

* early steps on two-hop classes — step 4: READOUT `E_rel` 15–611 with `A_den` 109–7,402
  while the paired-state NSE is 0.04–0.09; step 8 under composition (k ≥ 2): `E_rel`
  1.2–8.7 with `A_den` 10–50 (11 of 36 strata);
* the decayed regime — at scales 2⁻² and 2⁻²⁰, 13 and 12 of 36 strata are
  `DENOMINATOR_COLLAPSE` at step 20.

In both the teacher's signed readout is passing through or near zero. This is the R93
phenomenon in its pure form, and it is confined to identifiable steps and regimes.

**Local transition.** Teacher-forced one-step NSE is moderate everywhere: median 0.015–0.06
(UNSIGNED) and 0.018–0.14 (SIGNED) over steps 2–20, rising toward late steps. It is not
negligible: the step beats the trivial persistence baseline in only 67 % of primary strata
(`MIXED_LOCAL_SKILL`), and is worse than persistence in most k = 0 strata.

**Accumulated autonomous drift.** By step 20 the autonomous error energy is 20–1,000 times
a single local defect in the SIGNED cohort (`E_rel` 1.6–87, amplitude ratio r = 2.3–10).
The restart telescoping attributes it mainly to local defects committed in **steps 5–12**
(SIGNED shares 0.44–0.63 and 0.23–0.61); defects of steps 17–20 oppose it. Growth from
step 8 to 20 is numerator-driven in 19 strata and denominator-driven (absolute defect
falling while the normalized defect rises) in 15.

**Direction.** The READOUT error is collinear with the teacher in 94 % of primary strata
(median `E_perp/E_rel` ≤ 0.05): on this fixture the paired readout is close to a signed
scalar, so its "directional" failures are sign disagreements. At the **state level
direction is not preserved**: the paired fixture-state cosine at step 20 is 0.40–0.81 in
every stratum, and the weak-source copies reach `E_rel` 6–95 with cosines down to −0.2.
"Magnitude error with preserved direction" (R118-E) describes the readout, not the state.

**Decayed-regime operator (why it accumulates).** At vanishing amplitude one pass of either
system is exactly linear, and the HCA operator is reproduced to 1.4e-16 from its eight
limiting gate values alone. Those limits differ from the teacher's:

| | k2 A←col | k3 A←B | k4 B←B | k5 B←sal |
|---|---:|---:|---:|---:|
| teacher | 0.100 | 0.030 | 0.690 | −0.100 |
| unsigned checkpoints | 0.170–0.182 | 0.104–0.112 | 0.767–0.769 | −0.038 … −0.045 |
| signed checkpoints | 0.167–0.172 | 0.186–0.203 | 0.758–0.762 | −0.006 … +0.009 |

Consequently the HCA's spectral radius exceeds the teacher's for **every** SIGNED system
(ratio 1.077–1.100 per step, i.e. ≈ 1.19 per step in energy) and for 74 % of UNSIGNED ones
(median 1.029), and its dominant mode rotates more slowly than the teacher's. Small local
defects therefore persist and grow relative to a faster-decaying, faster-rotating target.

## 4. Controllers

| Policy | Induced reversals, in-range scales | Above-clamp scale | Verdict (in range) |
|---|---:|---:|---|
| R114 | 1,760 (scale 1: 501; 2⁻²: 1,247; 2⁻²⁰: 12) | 19,978 | `HAZARD_CONTROLLER_INDUCED_REVERSAL` |
| SIGNED_ONLY | 437 (all at scale 1, signed checkpoints, 12 of 14 cells first at step 2) | 7,675 | `HAZARD_CONTROLLER_INDUCED_REVERSAL` |
| DELAYED8 | 0 | 0 | `NO_INDUCED_REVERSAL_OBSERVED` |

An induced reversal is a sample REVERSED under the policy while the NATIVE sample of the
same fixture, checkpoint, pair and step is ALIGNED. The native operator is not reversal-free
itself: 6,368 of 69,120 scale-1 samples are REVERSED, mostly at early steps (the all-scale
census peaks at step 2 and falls steadily).

* The R104 controller is exactly `1.5 M − 0.5 I` in the decayed regime: it lowers the
  spectral radius below the teacher's in all 72 unsigned systems and raises the phase
  velocity (faster than the teacher's in 22 %). It regulates magnitude by shifting the
  timing of the readout's zero crossings, which is how it inverts signals.
* The R114 step worsens teacher-forced one-step NSE in 92 % (SIGNED) and 89 % (UNSIGNED) of
  samples: it trades local fidelity for long-horizon magnitude (the R102 pattern).
* Pooled step-20 NSE at scale 1 (descriptive, not a registered decision):

  | Overlap | NATIVE | R114 | SIGNED_ONLY | DELAYED8 |
  |---|---:|---:|---:|---:|
  | k = 0 | 0.979 | 0.391 | 0.159 | 0.445 |
  | k = 2 | 11.748 | 0.647 | 1.787 | 3.724 |
  | k = 6 | 13.896 | 0.484 | 2.083 | 3.537 |

  DELAYED8 improves on NATIVE in 6 of 6 graph blocks at every k (worst block −0.013) and
  keeps step 8 bit-identical. The R121 ordering "SIGNED_ONLY beats R114" **does not hold
  here at k = 2 and k = 6**, where unsigned checkpoints also over-retain. R114's lower
  pooled NSE coexists with its directional hazard; pooled NSE does not override it.
* Worst graph-block harm against NATIVE: R114 +0.318, SIGNED_ONLY +0.021, DELAYED8 −0.013.

## 5. Topology dependence

Three new classes are two-hop (source and readout share no participant), a configuration
present in only 7 of 54 historical graph definitions. They behave differently: denominator
collapse at steps 4–8, then the largest accumulated errors of the experiment under
composition (SIGNED `E_rel` 20–87 at step 20). One-hop classes reproduce the R121 picture
(step-8 NSE 0.007–0.04). Normalized fidelity therefore depends strongly on the
source–readout distance, which earlier phases did not vary deliberately.

## 6. Audit findings

* R121's novelty gate compared against 12 classes. Against all 54 historical definitions,
  three of its four classes are incidence-isomorphic to earlier graphs (`r52_fan`, `R72_G2`,
  `R70_STAR`); they remain outside the R77 training class.
* The ten-participant / four-relation fixture family has 50 valid connected classes; none
  of maximum degree 4 was unused, so R122 has no prospective degree-4 class. Degree-4
  behaviour appears only in the consumed companion (`COMPANION_RESULTS.md`), where the
  pattern is the same. Eleven unused classes remain (one of degree 2, ten of degree 3).
* R120's evidence records another environment (torch 2.5.1+cu121, i7-9800X) and R121 records
  none; their cell means reproduce here to 1.5e-11 relative, not bitwise. R118-E reproduces
  bitwise.

## 7. What this establishes, and what it does not

Established within the tested scope: the long-horizon normalized error of the native HCA
is predominantly accumulated drift driven by a decayed-regime operator that dissipates too
little, most clearly for the signed checkpoints; denominator collapse is a real but
separate effect at readout zero crossings; fixed radial/transverse controllers change
phase as well as magnitude and can reverse sign.

Not established: any improvement of the operator; any production or runtime claim; behaviour
on degree-4 classes prospectively; behaviour beyond twenty steps, beyond this synthetic
teacher, or on other fixture families. The overall localization is borderline by its own
rule and should be treated as such.

Indicated next step (a recommendation, not a result): correct the limiting gate values
toward the teacher's decayed-regime limits — the four entries in the table above — and
judge any such change with state-level metrics and explicit denominator classes beside the
paired readout, on topology classes that include two-hop fixtures.
