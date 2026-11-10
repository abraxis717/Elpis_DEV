# R121 — New incidence topology and temporally delayed checkpoint correction

**RESEARCH_ONLY. R93 `ENERGY_NORMALIZED_DENOMINATOR_HOLD` remains binding.**

R120 passed its frozen checkpoint-only H20 pooled criterion on four historical
OOD topology classes. Its H8 pooled NSE was worse than native in every overlap
stratum and all 12 graph-by-overlap blocks, and some H20 graph blocks regressed.
This discovery is the motivation, not part of the R121 holdout.

This phase tests exactly two independent hypotheses on four newly generated,
precommitted incidence-isomorphism classes, disjoint from the twelve archived R61, R76 and R77 training/evaluation
incidence-isomorphism classes:

1. **R120 external-topology transfer:** does `SIGNED_ONLY` beat both `NATIVE`
   and `R114` on pooled H20 NSE independently at k=0,2,6?
2. **Delayed8:** for signed checkpoints, apply NATIVE through H8, R114 from
   H9 to H20; unsigned always NATIVE. Does it preserve exact H8 native output,
   improve pooled H20 NSE over NATIVE at each k, and bound graph-block H20 harm
   to at most 0.20 NSE?

The delayed schedule is a *new prospective candidate*, not part of R120's
already-qualified fixed checkpoint policy. Both are inference-teacher-free;
all controllers are frozen, the real R119 `EXACT_GUARDED` kernel is used,
and R30 teacher enters only during offline scoring.

The experiment uses the full original frozen HCA and native sparse AWR,
constructs all trajectories independently (except exact derivation of the
static signed-only policy from its whole matched control trajectory), and
compares H8/H20 with the same composed teacher as R118-E. It does not alter
any historical source, model weights, or R118/R119/R120 evidence.

`PROTOCOL_R121.json` and source must be committed and published before
running `qualify.py`. This experiment is intentionally **not** a production
admission or an attempt to clear R93. Four graph classes are limited evidence.

Original upstream attribution remains to [creelie/4Y-M](https://github.com/creelie/4Y-M),
Deep Bhattacharjee, Ushashi Bhattacharya, James Montgomery Lockwood and
Tristan Romanov (MIT, v4.21.0); the Elpis HCA experiments are independent.
