# R119 architecture: why the exact path was slow, and an exact repair

## 1. Mechanism (forensics)

The R118 kernel is exact IEEE-754 binary64 arithmetic with gradual underflow (no FTZ).
Without external input, repeatedly stepping the same instances is dissipative, so states
decay exponentially (≈0.88 per pass for R104 on `54007/STATE8`). On this CPU (measured by
`microbench.py`), an operation pays a microcode assist of ≈44–52 ns (≈95× a normal
multiply) exactly when:

| operation | penalized iff |
|---|---|
| multiply, divide | result subnormal, or an operand subnormal **and** the result nonzero |
| sqrt, tanh | operand subnormal |
| add, subtract | result subnormal **and** every operand normal or zero (cancellation) |
| exp | never (glibc returns `1.0 + x` for tiny x; the add is penalty-free) |
| any op whose result underflows to exactly 0 | never |

The census replica (`forensics.py`, bit-identical to the native step at every checkpoint)
attributes the failure to computations on *products of two state-scale quantities* (≈|x|²)
that land in the subnormal band:

* leading edge (median |x| ≈ 1e-106): controller `s·dt` products and `/den` divisions;
* median |x| ≈ 2e-149: controller exactly rounded sums (`fsum` error-free transforms
  `hi − x`, `y − yr` cancelling into the subnormal range) — +5.6 ms at A = 1,024;
* peak (median |x| ≈ 1e-158): gate products `W·f` (≈46 %), the nine invariant dot products
  (≈23 %), link norms, clamp norms and square roots, squash divisions, controller products
  — +18.5 to +19.3 ms at A = 1,024;
* below ≈ 1e-163 the squares underflow to exactly zero and the path is fast again.

## 2. Guarded exact fast paths (`csrc/awr_kernel_r119.c`)

Each guard is a sufficient condition, tested with integer comparisons on bit patterns
(no floating-point operation on tiny values), under which a computation can be replaced
by a cheaper one **with bit-identical results**. If the condition fails, the R118
arithmetic runs verbatim. Notation: `u = 2^-53`, `η = 2^-1075`, all finite.

**G_SOFTMAX.** If every transported A-state coordinate has |w| < 2⁻⁶⁰, then
`fl(w_k²) ≤ 2⁻¹²⁰`, `dot4 ≤ 2⁻¹¹⁸`, `sqrt ≤ 2⁻⁵⁹`, so `x_e = 8·sqrt(·) ∈ [0, 2⁻⁵⁶]` and
`x_e − max ∈ [−2⁻⁵⁶, 0]`. glibc's `exp` returns `1.0 + t` for |t| < 2⁻⁵⁴, which rounds to
1.0 here (any implementation with error < 0.875 ulp also must); `awr119_libm_selfcheck`
verifies `exp(t) == 1.0` at every binade boundary down to the smallest subnormal and the
guard is disabled if it fails. Hence `tot == ne` exactly and every weight is `1.0/ne`,
which the fast path computes directly.

**G_GATES.** For a slot, let m = max |coordinate| of A, B, sal, col. With the R118 order
`z = (((0 + W₀f₀) + …) + W₉f₉) + b`: |raw_j| ≤ 4m²(1+5u) + 5η,
|f_j| ≤ 4m²(1+6u) + 7η, and |Σ W_j f_j| ≤ 4 S m² (1 + 2⁻⁴⁷) + 2⁻¹⁰⁶⁵ for
S = Σ|W_qj| ≤ 100 (sums of subnormal-range values are exact; every other rounding is
relative). For |b| ∈ [2^E, 2^(E+1)) every value within 2^(E−54) of b rounds to b. So if
4 S m²(1+2⁻⁴⁷) + 2⁻¹⁰⁶⁵ ≤ 2^(E−53)/4 for every gate, `z == b` exactly and the gates equal
their bias values. `guards.gate_theta` computes the largest double θ with θ² below the
tightest bound using `fractions.Fraction` (no rounding); the kernel takes the guard iff
m < θ (θ ≈ 2.0–4.4e-10 for the six checkpoints). Disabled if any bias is zero or
subnormal or S > 100.

**G_CLAMP.** If max |v_k| < 1.9375 then |v| ≤ 3.875, `n + 1e-30 < 4`, `4/(n+1e-30) > 1`,
the scale is clamped to exactly 1.0 and `v·1.0 == v`; the fast path leaves v untouched.

**G_RADIAL.** Original: `r_i = fl(fl(s_i·dt)/den)`, `dt = fsum(fl(s_j d_j))`,
`den ≥ 1e-24 > 2⁻⁸⁰`, `out_i = fl(fl(p_i + fl(α r_i)) + fl(β·fl(d_i − r_i)))`, α, β ≤ 1.
With E_s, E_d the maximal exponents over the 32 coordinates:
|fl(s_j d_j)| ≤ 2^max(E_s+E_d+2, −1074), so |dt| ≤ 2^E_dt with
E_dt = max(E_s+E_d+2, −1074) + 5, and |r_i| ≤ 2^E_r with
E_r = max(E(s_i)+1+E_dt, −1074) + 80. If p_i and d_i are normal and
E_r ≤ E(p_i) − 55 and E_r ≤ E(d_i) − 55, then |fl(α r_i)| and |r_i| are below half the
smallest gap at p_i and d_i, so `fl(p_i + fl(α r_i)) == p_i`, `fl(d_i − r_i) == d_i`
and `out_i == fl(p_i + fl(β d_i))` bit for bit. s_i = 0 gives r_i = ±0 (same identities
for nonzero p_i, d_i); a coordinate with s_i, p_i, d_i all +0.0 (empty slot) gives +0 on
both paths; anything else (zero or subnormal p_i or d_i) declines the guard.

All arithmetic remains IEEE binary64 with round-to-nearest-even, no FMA contraction
(`-ffp-contract=off`) and gradual underflow. No state, weight or result is approximated.

## 3. Modes

| mode | kernel | class (by protocol) |
|---|---|---|
| EXACT_R118 | frozen R118 kernel | exact reference |
| EXACT_GUARDED | R119 kernel, all guards | EXACT iff bit-identical in every verification |
| FTZ_DAZ | frozen R118 kernel, MXCSR FTZ+DAZ for the call only (opt-in) | TOLERANCE_QUALIFIED iff deviations ≤ 2⁻¹⁰⁰⁰, bit-identical in normal and near-zero regimes, covariance ≤ 1e-9 |
| STATE_FLUSH_2P-500 | negative control (flush |x| < 2⁻⁵⁰⁰ after each step) | expected NONQUALIFIED |

## 4. Limits

* The guards remove the penalized work identified by the census in the decayed band;
  when states themselves become subnormal (|x| < 2.2e-308), transport and update products
  on subnormal states are still penalized, but the census measured them at well under a
  millisecond per A = 1,024 step.
* G_SOFTMAX depends on a libm property (checked at load, not proven for every libm).
* Penalty structure is a property of this CPU; other microarchitectures may differ.
