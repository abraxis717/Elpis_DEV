"""v2 configurations. Hypotheses are (corruption rate or band of rates, reading correlation rho).

PRIMARY candidates run in provider mode DUPLICATE under DUP2_4 (Contract I: independent readings).
FALLBACK candidates run in mode FIRST_ONLY (Contract II: independence not established; single readings).
CONTROLS are the equal-information alternatives (Milestone C); they are eligible for selection too, so a
simpler mechanism that matches the learner is preferred by the preregistered rule.
"""
from __future__ import annotations

H_V1 = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.4, 1.0)]                       # v1 RMF_U3E25
H5 = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.48, 0.0), (0.4, 1.0)]             # + near-noise channel
H4 = [(0.0, 0.0), (0.4, 0.0), (0.48, 0.0), (0.4, 1.0)]                         # smaller: no q = 0.2
H5_BAND = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), ([0.42, 0.46, 0.49], 0.0), (0.4, 1.0)]  # interpolated band
H_NOEVENT = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.48, 0.0)]

PRIMARY = {
    "RMF5_E25": {"hypotheses": H5, "prior": [1, 1, 1, 1, 0.25]},
    "RMF5_E10": {"hypotheses": H5, "prior": [1, 1, 1, 1, 0.1]},
    "RMF5_E50": {"hypotheses": H5, "prior": [1, 1, 1, 1, 0.5]},
    "RMF4_E25": {"hypotheses": H4, "prior": [1, 1, 1, 0.25]},
    "RMF5B_E25": {"hypotheses": H5_BAND, "prior": [1, 1, 1, 1, 0.25]},
    "RMF5_E25_GREEDY": {"hypotheses": H5, "prior": [1, 1, 1, 1, 0.25], "acquisition": "greedy"},
}
FALLBACK = {
    "SAFE4_P2": {"hypotheses": H_NOEVENT, "prior": [1, 1, 2, 2]},
    "SAFE4_P4": {"hypotheses": H_NOEVENT, "prior": [1, 1, 4, 4]},
    "SAFE4_P8": {"hypotheses": H_NOEVENT, "prior": [1, 1, 8, 8]},
}
BAYES_CONTROLS = {
    "RMF_V1": {"hypotheses": H_V1, "prior": [1, 1, 1, 0.25]},
    "RMF_NOEVENT": {"hypotheses": H_NOEVENT, "prior": [1, 1, 1, 1]},
}
CONTROLS = ("NOOP", "BETA_ALL", "SGD_ALL", "DISAGREE_SGD", "PLUGIN_Q_BAYES", "STATIC_ATT_ALL") + tuple(BAYES_CONTROLS)
ORIGINAL_BASELINES = ("NOOP", "SGD", "BETA", "CGC_REFERENCE", "STATIC_ATTENUATION", "CLEAN_BAYES")
BAYES_CONFIGS = {**PRIMARY, **FALLBACK, **BAYES_CONTROLS}
