"""Build PROTOCOL_R122.json (run once, before preregistration; no model is executed).

    python -B -m research.yang_mills_hca.r122_energy_direction_forensics.make_protocol

The protocol is the authority once committed; this script only documents how the graph
definitions, the pins and the source hashes were produced. It regenerates the registered
graphs from the registered scan and audits them against every historical graph set.
"""
from __future__ import annotations

import json
import sys

from research.yang_mills_hca.r118_scalable_awr import provenance as P

from . import metrics as M
from . import qualify as Q
from . import topology as T

GRAPH_SEED = 122400
DEGREE_SCHEDULE = [2, 2, 3, 3, 3, 3]
ATTEMPTS = 300000


def build() -> dict:
    hist = T.historical_classes()
    graphs = T.generate(GRAPH_SEED, DEGREE_SCHEDULE, set(hist), attempts=ATTEMPTS)
    named = {f"R122_T{i + 1}_D{T.max_degree(g)}": g for i, g in enumerate(graphs)}
    audit = T.audit(named)
    if not audit["all_novel"]:
        raise RuntimeError("GRAPH_NOT_NOVEL")
    ym = Q.YM
    return {
        "schema": "elpis.ym_hca.r122.protocol.v1",
        "phase": "R122",
        "title": "Energy-normalized fidelity, directional safety and recurrent error localization",
        "status": Q.REGISTERED_STATUS,
        "base_commit": "eafdb15293c6e47010c159c9adb8d7977e22e868",
        "r93": Q.R93,
        "question": ("What produces the long-horizon energy-normalized error of the frozen 88-parameter HCA against its "
                     "teacher: collapse of the teacher-energy denominator, an erroneous one-step transition, accumulated "
                     "autonomous drift, or controller-induced distortion of magnitude or direction?"),
        "scope": ("Mechanism and failure localization only. No training, no parameter or controller change, no new "
                  "operator, no production admission. A diagnostic result cannot clear R93."),
        "graphs": named,
        "graph_generation": {
            "seed": GRAPH_SEED, "degree_schedule": DEGREE_SCHEDULE, "attempts_per_graph": ATTEMPTS,
            "method": ("topology.generate: random.Random(seed) scan of four sorted 4-subsets of ten participants; four "
                       "distinct relations, all ten participants covered, connected relation overlap, registered maximum "
                       "participant degree, incidence signature absent from every historical class. Row order is slot "
                       "order: source relation 0, readout relation 3. Explicit graph definitions are the authority."),
            "family_inventory": ("Exhaustive enumeration: the ten-participant / four 4-member-relation family has 50 "
                                 "connected incidence classes with four distinct relations (11 of maximum degree 2, 31 "
                                 "of degree 3, 8 of degree 4). 33 were used before R122, including all 8 of degree 4, so "
                                 "no unused degree-4 class exists; 17 were unused (3 of degree 2, 14 of degree 3). R122 "
                                 "consumes 6 of them and leaves 11 (1 of degree 2, 10 of degree 3)."),
        },
        "topology_audit": {
            "historical_graph_definitions": audit["historical_graph_definitions"],
            "historical_incidence_classes": audit["historical_classes"],
            "audited_sources": "every FROZEN_PROTOCOL_R*.json (R61-R78), research_helpers/r92/protocol.json, PROTOCOL_R121.json",
            "registered_graphs": audit["graphs"],
            "finding_on_r121": ("R121 audited its four graphs against 12 classes (R61 r55_structural, R76, R77). Against "
                                "the complete inventory, three of them are incidence-isomorphic to earlier graphs: "
                                "R121_HOLDOUT_1 ~ R61 r52_fan, R121_HOLDOUT_3 ~ R72_G2, R121_HOLDOUT_4 ~ R70_STAR. They "
                                "remain outside the R77 training class; only R121_HOLDOUT_2 was a class unused anywhere."),
        },
        "evaluation_seeds": [122101, 122103, 122107, 122109],
        "structure_seed": 122300,
        "cases": ["weak", "anti"],
        "source_ratio": 160,
        "batch_per_seed": 4,
        "k_values": [0, 2, 6],
        "scales": {"2^-20": 2.0 ** -20, "2^-2": 0.25, "1": 1.0, "2^2": 4.0},
        "scale_definition": ("the whole historical initial state pair_state(seed, case, 160, batch=4) multiplied by the "
                             "scale. '1' is the R118-E / R120 / R121 fixture. '2^-20' is the decayed (linear-limit) "
                             "regime. '2^2' places the strong source above the operator's clamp radius 4 and is reported "
                             "as an above-clamp stress condition."),
        "horizons": [1, 4, 8, 12, 16, 20],
        "recorded_steps": "every step 1..20",
        "restart_points": [4, 8, 12, 16],
        "checkpoints": list(P.CHECKPOINTS),
        "policies": ["NATIVE", "R114", "SIGNED_ONLY", "DELAYED8"],
        "policy_definition": "frozen r121_temporal_topology.policy.controller_at; R114 gains from reference.gains",
        "numerical_mode": "R119 EXACT_GUARDED kernel (R119Engine default), MEAN reconciliation, 1 pass per step",
        "views": {
            "READOUT": "A of the readout relation, copy i minus copy i+4 (4-d): the historical paired response; primary",
            "PAIRED_STATE": "whole fixture window state, copy i minus copy i+4 (32-d)",
            "WEAK_STATE": "whole fixture window state of the weak-source copies (32-d)",
            "STRONG_STATE": "whole fixture window state of the strong-source copies (32-d)",
        },
        "paths": {
            "autonomous": "initialized once from the shared initial state; engine steps only; no teacher access",
            "teacher_forced": ("offline diagnostic: the recorded teacher state T_{h-1} is the input of one HCA step; its "
                               "output is compared with T_h and never enters an autonomous trajectory"),
            "restart": ("offline diagnostic: the policy is run autonomously from the recorded teacher state T_j to step "
                        "20; differences between consecutive restarts telescope exactly to the autonomous error"),
        },
        "metrics": {
            "per_sample": ["E_abs", "teacher energy", "predicted energy", "r", "c", "rho = r c", "E_rel", "E_gain", "E_perp",
                           "E_mag", "E_ang", "E_direction = 2(1-c)", "energy ratio r^2", "identity residuals"],
            "identities": ["E_rel = 1 + r^2 - 2 r c", "E_rel = E_gain + E_perp", "E_rel = E_mag + E_ang",
                           "e_h = P_h + L_h", "e_20 = sum_b D_b", "dlog E_rel = dlog E_abs - dlog teacher energy",
                           "E_rel(READOUT) = E_rel(PAIRED_STATE) * A_den"],
            "no_denominator_floor": True,
        },
        "classifier": {
            "r_floor": M.R_FLOOR, "cos_aligned": M.COS_ALIGNED, "cos_deviated": M.COS_DEVIATED,
            "cos_reversed": M.COS_REVERSED, "cos_antiparallel": M.COS_ANTIPARALLEL,
            "near_zero_exponent": M.NEAR_ZERO_EXPONENT, "pair_condition_limit": M.PAIR_CONDITION_LIMIT,
            "norm_classes": list(M.NORM_CLASSES), "direction_classes": list(M.DIR_CLASSES),
            "rule": ("norm class first (NONFINITE, BOTH_ZERO, ZERO_TARGET, ZERO_PREDICTION, ILL_CONDITIONED, NEAR_ZERO, "
                     "else ORDINARY); a direction class is assigned to ORDINARY samples only: EXTINGUISHED if r < "
                     "r_floor, TARGET_NEGLIGIBLE if r > 1/r_floor, else ALIGNED / DEVIATED / ORTHOGONAL / REVERSED by "
                     "the cosine thresholds. A zero or extinguished prediction is never a reversal."),
        },
        "thresholds": {
            "primary_scale": "1", "primary_horizon": 20, "linear_limit_scale": "2^-20", "above_clamp_scales": ["2^2"],
            "min_eligible_fraction": 0.9, "a_den_collapse": 10.0, "lambda_local": 0.5, "lambda_accumulated": 0.1,
            "growth_from": 8, "growth_to": 20, "dominant_fraction": 0.75, "max_invalid_strata_fraction": 0.1,
            "max_induced_reversals": 0, "collinear_perp_share": 0.05, "bootstrap_seed": 122900, "bootstrap_resamples": 2000,
        },
        "decision_rules": {
            "stratum": "graph x overlap k x scale x checkpoint cohort (SIGNED: 2 checkpoints, UNSIGNED: 4); 64 / 128 samples",
            "stratum_mechanism": ("NATIVE policy, READOUT view, one horizon; first matching rule: INVALID if the eligible "
                                  "sample fraction < min_eligible_fraction; DENOMINATOR_COLLAPSE if median A_den >= "
                                  "a_den_collapse; LOCAL_TRANSITION if median lambda >= lambda_local; ACCUMULATED_DRIFT "
                                  "if median lambda <= lambda_accumulated; else MIXED_LOCAL_ACCUMULATED. lambda = |L|^2 / "
                                  "|e|^2 with L the teacher-forced one-step defect of the same step and e the autonomous "
                                  "error; A_den = E_rel(READOUT) / E_rel(PAIRED_STATE)."),
            "dominance": ("a class is dominant iff it holds in >= dominant_fraction of the primary strata AND in a strict "
                          "majority of the strata of every graph class"),
            "primary_strata": "all 6 registered graphs x k in {0,2,6} x {SIGNED, UNSIGNED} at scale '1', horizon 20 (36 strata)",
            "disposition": {
                "NONQUAL": "any integrity gate fails",
                "MECHANISM_INDETERMINATE": "integrity holds but more than max_invalid_strata_fraction of the primary strata are INVALID",
                "MECHANISM_LOCALIZED_BOUNDED": ("integrity holds and DENOMINATOR_COLLAPSE, LOCAL_TRANSITION or ACCUMULATED_DRIFT "
                                                "is dominant over the primary strata"),
                "MECHANISM_MIXED_BOUNDED": "integrity holds and no localizing class is dominant",
            },
            "cohort_dispositions": "the same dominance rule applied to the SIGNED and UNSIGNED strata separately; reported, not the overall disposition",
            "growth_class": ("per stratum between horizons growth_from and growth_to, medians of sample-wise logs: NO_GROWTH "
                             "if median dlog E_rel <= 0; NUMERATOR_DRIVEN if median dlog E_abs > 0; else DENOMINATOR_DRIVEN "
                             "(absolute defect falls while the normalized defect rises: the R93 pattern)"),
            "directional_safety": ("READOUT view, every recorded step. A controller-induced reversal is a sample that is "
                                   "REVERSED under the policy while the NATIVE sample of the same fixture, checkpoint, pair "
                                   "and step is ALIGNED. Per policy and amplitude regime (in-range scales; above-clamp "
                                   "scale): HAZARD_CONTROLLER_INDUCED_REVERSAL if the count exceeds max_induced_reversals, "
                                   "else NO_INDUCED_REVERSAL_OBSERVED. Pooled NSE never overrides this verdict."),
            "worst_graph_harm": "per policy: max over graph blocks and k of the graph-block mean H20 E_rel minus NATIVE, scale '1'",
            "secondary_findings": {
                "S1": "readout error collinear with the teacher: median E_perp/E_rel <= collinear_perp_share in >= dominant_fraction of primary strata",
                "S2": "paired-state direction not preserved: median PAIRED_STATE cosine < cos_aligned in >= dominant_fraction of primary strata",
                "S3": "teacher-forced one-step skill against teacher persistence > 0: LOCALLY_SKILLFUL if >= dominant_fraction of primary strata, LOCALLY_UNSKILLED if <= 1 - dominant_fraction, else MIXED_LOCAL_SKILL",
                "S4": "decayed-regime operator: probe vs analytic eight-gate operator and linear-limit trajectories within tolerance (integrity gate LINEAR_OPERATOR_CONTROLS); spectra reported",
            },
            "r93": "remains ENERGY_NORMALIZED_DENOMINATOR_HOLD whatever the disposition",
        },
        "tolerances": {
            "identity_residual": 1e-9, "telescoping_residual": 1e-9, "exact_audit_relative": 1e-12,
            "teacher_k0_parity_abs": 1e-12, "reference_teacher_relative": 1e-10,
            "published_cell_relative": 1e-9, "published_cell_absolute": 1e-13,
            "gauge_covariance_abs": 1e-9, "gauge_metric_invariance": 1e-9, "gauge_negative_control_min": 1e-6,
            "linear_probe_homogeneity_abs": 1e-12, "linear_analytic_vs_probe_rel": 1e-9, "linear_limit_trajectory_rel": 1e-3,
            "bitwise": ("autonomous recheck, fresh-state replay, EXACT_GUARDED vs EXACT_R118, DELAYED8 splice and H8 "
                        "identity: zero mismatching trajectories"),
            "note": ("published R120 / R121 cell means were produced with another torch build and CPU; they reproduce here "
                     "to about 1e-11 relative, not bitwise. R118-E cells reproduce bitwise on this environment class."),
        },
        "controls": {
            "classifier_seed": 122600, "gauge_seed": 122700, "gauge_checkpoints": ["54007/STATE8", "54031/STATE8_SIGNED8"],
            "deep_check_stride": 4, "reference_teacher_steps": [1, 2, 8, 20],
            "regression_fixture": ("consumed R121_HOLDOUT_2 with R121 seeds and structure: published cell means within "
                                   "tolerance; k=0 unsigned 54007 R114 H20 all REVERSED and antiparallel with mean "
                                   "E_direction >= 3.99 while NATIVE is all ALIGNED with c >= 0.999; k=6 all EXTINGUISHED, "
                                   "none REVERSED"),
            "negative_controls": ["state-only gauge transformation (frames untouched) must change trajectories by >= 1e-6",
                                  "prediction-only half turn must be classified REVERSED, quarter turn ORTHOGONAL",
                                  "zero and extinguished predictions must not be classified as reversals"],
        },
        "integrity_gates": ["SOURCE_PINS", "FROZEN_PARITY", "CHECKPOINTS_UNCHANGED", "GRAPH_NOVELTY", "SEED_DISJOINTNESS",
                            "DETECTOR_CONTROLS", "AUTONOMOUS_GATE", "LOCAL_TRANSITION_GATE", "ENERGY_DECOMPOSITION_GATE",
                            "NUMERICAL_PARITY", "LINEAR_OPERATOR_CONTROLS", "RESOURCE_BOUND"],
        "budgets": {"max_raw_tensor_bytes": 1 << 30, "max_engine_steps": 2_000_000, "max_rss_bytes": 6 << 30,
                    "max_seconds": 7200, "max_evidence_bytes": 6_000_000,
                    "active_set": "at most 8 x (1 + k) = 56 HCA instances and 152 relation rows per system; all state volatile"},
        "consumed_companion": ("the same pipeline on the four R121 graphs with R121 seeds and structure, run before "
                               "registration and published with it. It is discovery material and a rehearsal, has no "
                               "decision authority and is the only source of maximum-degree-4 coverage."),
        "expectations_from_consumed_companion": {
            "status": "NON_BINDING: stated before the prospective run so that confirmation or refutation is visible",
            "E1": "SIGNED strata are ACCUMULATED_DRIFT with numerator-driven H8->H20 growth; UNSIGNED strata are not localized; overall disposition MECHANISM_MIXED_BOUNDED",
            "E2": "S1 and S2 are supported: the readout error is collinear with the teacher while the paired-state direction is not preserved",
            "E3": "R114 induces sign reversals at in-range scales; DELAYED8 induces none; the above-clamp scale produces many",
            "E4": "decayed-regime spectral radius of the HCA exceeds the teacher's for every SIGNED checkpoint; the R104 controller lowers it below the teacher's and raises the dominant phase velocity",
            "E5": "the teacher-forced one-step NSE is worse under the R114 step than under the NATIVE step in most samples",
        },
        "pins": {
            "frozen_manifest_sha256": P.MANIFEST_SHA256,
            "checkpoint_sha256": dict(P.CHECKPOINT_SHA256),
            "r118_component_digest": P.component_digest(),
            "r118_kernel_sha256": Q.sha(ym / "r118_scalable_awr/csrc/awr_kernel.c"),
            "r119_kernel_sha256": Q.sha(ym / "r119_numerical_hotpath/csrc/awr_kernel_r119.c"),
            "r119_engine_sha256": Q.sha(ym / "r119_numerical_hotpath/engine_r119.py"),
            "r119_guards_sha256": Q.sha(ym / "r119_numerical_hotpath/guards.py"),
            "r118_characterize_e_sha256": Q.sha(ym / "r118_scalable_awr/characterize_e.py"),
            "controller_reference_sha256": Q.sha(ym / "reference.py"),
            "r120_raw_sha256": Q.sha(ym / "r120_checkpoint_policy/evidence/R120_RAW_FRESH.json"),
            "r121_protocol_sha256": Q.sha(ym / "r121_temporal_topology/PROTOCOL_R121.json"),
            "r121_policy_sha256": Q.sha(ym / "r121_temporal_topology/policy.py"),
            "r121_qualify_sha256": Q.sha(ym / "r121_temporal_topology/qualify.py"),
            "r121_result_sha256": Q.sha(ym / "r121_temporal_topology/evidence/R121_RESULT.json"),
        },
        "source_sha256": Q.source_sha256(),
        "limitations": [
            "Synthetic teacher and synthetic fixtures; six new ten-participant / four-relation incidence classes of maximum degree 2 and 3 only.",
            "Samples within a graph are correlated; replication is counted over graph classes.",
            "The paired readout is nearly one-dimensional under the nearly flat registered transport; state-level views are reported beside it.",
            "A mechanism localization is not a fidelity qualification: R93 stays on HOLD and no policy is promoted.",
            "Bitwise claims hold for this environment class (CPU, libm, compiler, torch build) only.",
        ],
        "no_training": True, "no_teacher_in_autonomous_inference": True, "no_persistent_inference_mutation": True,
    }


def main() -> int:
    proto = build()
    Q.PROTOCOL.write_text(json.dumps(proto, indent=1, sort_keys=True) + "\n")
    print("PROTOCOL_SHA256", Q.sha(Q.PROTOCOL))
    print("COMPONENT_DIGEST", Q.component_digest())
    for name, g in proto["graphs"].items():
        print(name, g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
