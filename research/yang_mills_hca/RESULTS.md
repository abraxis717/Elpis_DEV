# Frozen Yang–Mills/AWR HCA research — result ledger

Research-only condensed record. Source-pinned host evidence is retained locally; this Markdown summary is not itself a row-level numerical reproduction or a release gate.

| Phase | Reported result | Scope / gate |
|---|---|---|
| R64–R78 | 88-parameter local HCA, six frozen R77 variants reconstructed by R78, gauge-transport and signed-intervention studies | Local historical models; not integrated AWR |
| R79–R92 | Tangent amplification, Jacobians, path-dependent error attribution, amplitude reference | Diagnostic evidence only |
| R93 | Absolute defect can decline while energy-normalized defect worsens as teacher energy falls | **ENERGY_NORMALIZED_DENOMINATOR_HOLD** |
| R94–R99 | Floor-free denominator tests, error decomposition, amplitude curvature/homogeneity | Descriptive |
| R100 | Held-out amplitudes 0.75×/1.5×: interpolation beats nearest-reference | Descriptive |
| R101 | Better model: log-defect interpolated in *raw* amplitude, mean log error ≈0.00006965 vs R100 ≈0.00139897 | Original interpolation specificity nonqual |
| R102 | Shared gain +0.5: H20 autonomous error ≈4.1958→2.3771, teacher-forced one-step error worsened | Frozen shadow nonpromotable |
| R103–R104 | Matching teacher energy does not remove all directional advantage; radial chiefly magnitude, transverse chiefly direction | Retrospective mechanism diagnostics |
| R105 | Selected (0.75,0) improves raw H20 but makes matched direction ≈42.9% worse than R104 | Negative objective tradeoff |
| R106 | 0/20 fixed-gain pairs satisfy preregistered Pareto calibration | Holdout not consumed |
| R107–R108 | Frozen (0.70,0.45) improves raw H20 error ≈11.2% vs R104 and meets directional noninferiority on fresh seeds | Prospective shadow replication |
| R109–R110 | Most H20 gain associated with altered incoming autonomous state; switching-time sensitivity | Retrospective diagnostics |
| R111 | Frozen R107→R104 H16 switch: H20 raw ≈9.98% better than R104, slight matched gain | Prospective shadow, heterogeneous |
| R112–R113 | Severe checkpoint cohort heterogeneity; low-amplitude R76_H3B directional regressions; subgroup pattern replicates | Retrospective + prospective stratification |
| R114 | Gate: `SIGNED8→(0.70,0.45)` else `(0.50,0.50)`; H20 mean raw-error gain +0.31999 vs R104; 48/48 signed conditions positive | Prospective frozen shadow nonpromotable |
| R115 | Same gate transfers to amplitudes 0.25× and 3×; mean raw gain +0.318740; 32/32 signed conditions positive | OOD-amplitude shadow nonpromotable |
| R116 | Same gate transfers to four R92-excluded graphs; H20 R104 2.37668→2.05554 (~13.51%); all 48 signed conditions positive; max covariance error ≈1.75e-16 | Topology shadow nonpromotable |
| R117 | R77 documented training graphs D1_P1 and D2_P1 have same 4-relation incidence-isomorphism signature. All four R116 graphs have distinct signatures excluded from that training class | Documented training topology provenance; historical execution attestation separate |
| R118 | Clean-checkout reconstruction (R77 576/576 rows bitwise, R78 archive byte-identical); sparse AWR substrate with the authentic frozen HCA: historical rollouts replay bitwise through it, end-to-end gauge covariance with overlapping windows ≤ 2.7e-15; registered p95 0.28–0.41 / 1.0–1.1 / 3.9–4.8 ms at 64 / 256 / 1,024 active instances up to a 1,000,000-relation registry; composition changes magnitude, not direction | Engineering qualification with disclosed regime exception (input-free decay, subnormal arithmetic); descriptive science only; see [r118_scalable_awr/RESULTS.md](r118_scalable_awr/RESULTS.md) |
| R119 | Forensics of the R118-D exception: penalized subnormal multiplies, divisions, square roots and cancellations in gate, invariant, norm and controller computations; guarded exact kernel bit-identical to the R118 kernel in every verification; registered p95 at 1,000,000 relations: decayed peak 0.20 / 0.89 / 3.32 ms (original 1.59 / 6.81 / 28.12) at 64 / 256 / 1,024 active HCAs | R118-D numerical exception CLOSED for the default exact path; FTZ/DAZ opt-in is latency-qualified but numerically NONQUALIFIED (deviation 4.99e-285); see [r119_numerical_hotpath/RESULTS.md](r119_numerical_hotpath/RESULTS.md) |

**Important limitations:** R114–R116 predominantly improve magnitude regulation while preserving approximately baseline directional fidelity (noninferiority, not directional superiority). R116 matched-direction error remains ≈0.992 and predicted/teacher energy ≈1.615. The original HCA transition remains locally inaccurate under teacher forcing. R93 stays on HOLD. Historical upstream mathematics should not be represented as proof of these empirical claims.

The reproducibility boundary is in [PROVENANCE.md](PROVENANCE.md). SHA commitments for returned result bundles are in [EVIDENCE_INDEX.json](EVIDENCE_INDEX.json).
