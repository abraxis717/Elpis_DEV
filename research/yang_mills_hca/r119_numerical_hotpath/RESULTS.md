# R119 results

Protocol `PROTOCOL_R119.json` (`a8ea1bcd…`) was committed in `9526124` before the registered
measurement; it inherits the R118-D p95 targets **2 / 5 / 20 ms** unchanged. Evidence:
`evidence/` (hashes in `evidence/INDEX.json`). Hardware: 4-vCPU Intel Xeon @ 2.10 GHz
(AVX-512) KVM guest, glibc 2.39, gcc 13.3, numpy 2.5.3, torch 2.15.0+cpu (oracle only),
single-threaded measurement. Latency is re-measured for every mode in the same process with
modes interleaved per step, so modes are compared under identical conditions.

## Disposition

| Mode | Numerical class | Latency | Disposition |
|---|---|---|---|
| **EXACT_GUARDED** (R119 kernel) | **EXACT** — 0 bit mismatches vs EXACT_R118 in every verification | **qualified** in all 24 controlled cells and 4 organic trajectories | **new default exact path** |
| EXACT_R118 (frozen R118 kernel) | EXACT (reference) | **NONQUAL**: W2_PEAK A = 256 and A = 1,024, organic STRESS1M A = 256, organic SMALL A = 1,024 | reference only |
| FTZ_DAZ (opt-in) | **NONQUALIFIED** — max deviation 4.99e-285 > 2⁻¹⁰⁰⁰ | qualified | opt-in only; not numerically qualified |
| STATE_FLUSH_2P-500 | NONQUALIFIED (deviation 1.6e-150) | — | negative control, as designed |

**The R118-D numerical exception is CLOSED for the default exact path** (registered decision
rule 1): the repaired path is bit-identical to the original exact path and meets every target.
The FTZ result is a separate opt-in operating mode and is not part of that closure — and under
the registered tolerance it does not qualify numerically. R118-E remains descriptive;
R93 remains `ENERGY_NORMALIZED_DENOMINATOR_HOLD`; no capacity was expanded and the learned HCA
is unmodified. (The `R119Engine` constructor default was already `EXACT_GUARDED` in the
pre-measurement commit; the registered run passed every mode explicitly, and the decision rule
now confirms that default.)

## 1. Reproduction and forensics (development, before registration)

* **Reproduction** (`evidence/repro_r118d.json`, R118 diagnostic unchanged): SMALL, R104 on
  `54007/STATE8`, all 1,024 instances, 4,000 steps — exact path block p95 **22.2 ms** in the
  blocks ending at steps 2,500 and 2,750 (median state ≈ 1e-158); FTZ ≤ 4.4 ms.
* **Per-operation penalties** (`evidence/microbench.json`): multiply 0.47 → ≈44 ns, divide
  1.3 → ≈47 ns, sqrt 2.2 → ≈52 ns, tanh 14.7 → 47 ns with a subnormal result (or a subnormal
  operand and a nonzero result); add/sub only when normal operands cancel into the subnormal
  range (≈44 ns); add with any subnormal input, exp of tiny arguments and underflow to exact
  zero carry no penalty; FTZ/DAZ removes every penalty.
* **Census** (`evidence/forensic_census.json`): the instrumented replica is bit-identical to
  the native step at all 13 checkpoints (all 1,024 instances). Predicted penalized work tracks
  the measured excess:

| step | median state | measured excess (A = 1,024) | predicted | responsible operations |
|---|---|---|---|---|
| 1,650–1,700 | ≈ 2e-106 | +0.8 – 1.5 ms | 0.9 – 3.0 ms | controller `s·dt` products, `/den` |
| 2,350–2,400 | ≈ 2e-149 | +5.6 / +5.8 ms | 5.6 / 5.7 ms | controller `fsum` cancellations into the subnormal range |
| 2,450–2,550 | ≈ 1e-158 | +18.5 – 19.3 ms | 23 – 27 ms | gate `W·f` ≈ 46 %, invariants ≈ 23 %, link norms, clamp norms + sqrt, squash, controller |
| ≥ 2,600 | ≤ 7e-165 | ≤ +0.9 ms | ≤ 0.5 ms | products underflow to exact zero (free) |

## 2. Registered latency, STRESS1M (1,000,000 relations, 50,000 instances), R114 on `54007/STATE8_SIGNED8`

Controlled regimes (300 measured steps, fresh active set per step, state restored per step);
p95 / p99 in ms:

| regime (scale) | A | EXACT_R118 | **EXACT_GUARDED** | FTZ_DAZ |
|---|---|---|---|---|
| NORMAL (1) | 64 / 256 / 1,024 | 0.36 / 1.05 / 4.97 | **0.33 / 1.05 / 5.01** | 0.30 / 0.99 / 4.63 |
| NEARZERO (1e-20) | 64 / 256 / 1,024 | 0.31 / 1.19 / 4.67 | **0.31 / 1.30 / 4.28** | 0.31 / 1.26 / 4.31 |
| W1_RADIAL (1e-104) | 64 / 256 / 1,024 | 0.45 / 1.39 / 5.45 | **0.23 / 0.65 / 2.37** | 0.30 / 0.93 / 3.59 |
| LEAD_FSUM (1e-149) | 64 / 256 / 1,024 | 0.74 / 3.25 / 11.22 | **0.22 / 0.93 / 2.95** | 0.21 / 0.97 / 3.23 |
| W2_PEAK (1e-157) | 64 / 256 / 1,024 | 1.59 / **6.81** / **28.12** | **0.20 / 0.89 / 3.32** | 0.20 / 0.89 / 3.33 |
| TAIL (1e-300) | 64 / 256 / 1,024 | 0.25 / 0.76 / 3.69 | **0.22 / 0.66 / 3.37** | 0.23 / 0.70 / 3.33 |
| SUBNORMAL (1e-310) | 64 / 256 / 1,024 | 0.82 / 3.13 / 15.25 | **0.75 / 2.84 / 13.90** | 0.20 / 0.64 / 3.27 |
| MIXED (10^U(-320,0)) | 64 / 256 / 1,024 | 0.52 / 1.76 / 6.93 | **0.39 / 1.15 / 4.43** | 0.40 / 1.29 / 4.95 |

p99 for EXACT_GUARDED: NORMAL 0.40 / 1.10 / 5.47; W2_PEAK 0.27 / 1.00 / 3.51; worst cell
SUBNORMAL A = 1,024 p95 13.90 / p99 15.14 ms (states that are themselves subnormal: exact
products on subnormal operands are irreducible; within target, smallest margin 1.44×).

Organic decay (R104 on `54007/STATE8`, fixed active set, 4,000 steps from N(0,1); max 250-step
block p95 / overall p95 / p99, ms):

| trajectory | EXACT_R118 | **EXACT_GUARDED** | FTZ_DAZ |
|---|---|---|---|
| STRESS1M A = 64 | 1.54 / 0.75 / 1.33 | **0.40 / 0.34 / 0.44** | 0.36 / 0.30 / 0.39 |
| STRESS1M A = 256 | **5.32** / 2.43 / 5.08 | **1.19 / 0.94 / 1.19** | 1.29 / 1.20 / 1.30 |
| STRESS1M A = 1,024 | 19.62 / 10.68 / 19.15 | **4.46 / 3.26 / 3.82** | 4.49 / 3.71 / 4.44 |
| SMALL A = 1,024 (R118 failure fixture) | **23.81** / 9.16 / 22.82 | **3.46 / 3.19 / 3.46** | 4.54 / 3.73 / 4.55 |

Zero refused or failed steps in every cell. Peak RSS 1,072 MiB per latency process (registry
306 MiB plus bulk-generation transients); engine scratch 0.85 MiB; neither kernel allocates
on the heap. In the NORMAL regime the guarded kernel is within measurement noise of the
original (p95 5.01 vs 4.97 ms at A = 1,024; faster at A = 64).

Guard activity (controlled run): clamp 99.8 %, softmax 86 %, gates 86 %, radial 47 % of
evaluations taken; every guard was also declined in the differential campaign.

## 3. Numerical verification

* **Bit identity (uint64 patterns), EXACT_GUARDED vs EXACT_R118:** 0 mismatches over all
  controlled steps (24 cells × 330 steps), all organic steps (4 trajectories × 4,000 steps),
  96 historical-replay cases (NATIVE and R114, H1 and H20) and a kernel differential campaign
  of 1,296 comparisons × 32 instances (3 registry families × 6 checkpoints × 3 controllers ×
  6 state regimes incl. zeros, signed zeros, subnormals and exact threshold placements).
* **Frozen checkpoint parity:** guarded path vs torch rollout 3.6e-16 (H1), 1.1e-16 (H20);
  six checkpoint pins unchanged.
* **Gauge covariance** (overlapping windows, controller, 6 passes; abs / relative to scale):
  exact modes identical — ≤ 4.3e-16 / 1.7e-15 at scales 1 … 1e-300, 3.0e-323 / 4.3e-13 at
  1e-310 (gradual-underflow precision floor); participant-gauge invariance likewise; negative
  controls ≥ 1.37 relative. FTZ_DAZ: relative covariance error 6.2e-10 at 1e-300, and at 1e-310
  DAZ zeroes the whole state, so the covariance test is vacuous there (negative control 0).
* **FTZ_DAZ deviation:** bit-identical in NORMAL and NEARZERO; max 3.2e-308 (TAIL), 3.3e-310
  (SUBNORMAL), ≤ 3.2e-307 on organic trajectories, but **4.99e-285 in MIXED**, violating the
  registered 2⁻¹⁰⁰⁰ bound. Post-hoc diagnostic (`evidence/diag_ftz.json`, 60 MIXED steps):
  2,026 deviating coordinates, 1,489 of them normal-range, 921 above 1e-6 relative, up to 20 %
  relative — flushed subnormal intermediates are amplified by the controller's 1/den = 1e24.

## 4. Limits and open items

* Penalty structure and the latency margins are properties of this CPU; other
  microarchitectures need their own measurement.
* G_SOFTMAX relies on a libm property verified at load (glibc ≥ 2.28 tiny-argument branch),
  not proven for every libm; the guard disables itself if the check fails.
* The guarded path is exact for the R118 algebra only; any change to the kernel arithmetic
  requires re-deriving the guards.
* FTZ_DAZ is not numerically qualified; a future protocol would need a relative (not absolute)
  tolerance argument and mixed-scale covariance evidence before it could qualify.
