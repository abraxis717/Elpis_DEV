# R125 — early H1/H2 prefix + preclamp forensic replay

**Retrospective, selection-biased consumed R123 stress events. No new holdout.**

Publication uses R1 output-path erratum; original failed R0 preregistration is preserved.

Disposition: `FORENSIC_PREFIX_PASS_BOUNDED`. R93: `ENERGY_NORMALIZED_DENOMINATOR_HOLD`.

Events: 40/40. Instrumented C step parity: 84.
Guarded/R118 exact-reference run comparisons: 8.

## Prefix counterfactual totals

| Conditional history | Reversed event count |
|---|---:|
| `K1_PRECLAMP_HIT_EVENTS` | 40 |
| `K2_PRECLAMP_HIT_EVENTS` | 16 |
| `K3_PRECLAMP_HIT_EVENTS` | 0 |
| `K_BOTH_H1_H2_THEN_NATIVE_REVERSED` | 39 |
| `K_ONLY_H1_THEN_NATIVE_REVERSED` | 0 |
| `K_ONLY_H2_THEN_NATIVE_REVERSED` | 39 |
| `N1_PRECLAMP_HIT_EVENTS` | 40 |
| `N2_PRECLAMP_HIT_EVENTS` | 16 |
| `N3_PRECLAMP_HIT_EVENTS` | 0 |
| `N_BOTH_H1_H2_THEN_K_REST_REVERSED` | 1 |
| `N_H1_THEN_K_REST_REVERSED` | 40 |
| `N_H2_THEN_K_OTHER_REVERSED` | 1 |

## Actual preclamp hit totals on original paths

| Path and step | Hit proposals |
|---|---:|
| `K1_hits` | 112 |
| `K2_hits` | 16 |
| `K3_hits` | 0 |
| `N1_hits` | 112 |
| `N2_hits` | 16 |
| `N3_hits` | 0 |

## Interpretation limits

These counts describe conditioned one-step interventions and preclamp proposal saturation on previously consumed failures.
They do not prove unique trajectory causation or that the clamp caused the failures.
All causal rollouts use the original exact guarded kernel; the shadow C instrument is diagnostic only and bitwise-validated.
No weight mutation, no prospective evidence, no production admission, no R93 clearance.

Full records and hashes: `R125_RESULT.json`, `INDEX.json`.
