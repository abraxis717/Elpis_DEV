"""Candidate configurations of the research loop (RESEARCH_LOG.md records why each exists).

Every candidate is admissible: it uses only the public router, contexts, its own selected readings and
the terminal queries. Coefficients below are frozen design constants (hypothesis grids, prior weights,
acquisition weights, decision thresholds); none is fitted at inference.

Families: A analytical reliability estimation (bounded q-mixture filter + Bayes decision);
B selective adaptation / conservative decisions; C acquisition; E changed observation contracts
(the same filters, run under a declared contract that delivers more independent evidence).
"""
from __future__ import annotations

Q3 = [0.0, 0.2, 0.4]
Q6 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45]

CANDIDATES: dict = {
    # Family A: q-mixture assumed-density filter with Bayes terminal decision, PROBE_EARLY acquisition.
    "RMF_U3": {"family": "A", "qs": Q3, "prior": [1, 1, 1]},
    "RMF_U6": {"family": "A", "qs": Q6, "prior": [1, 1, 1, 1, 1, 1]},
    "RMF_P50": {"family": "A", "qs": Q3, "prior": [1, 1, 2]},
    "RMF_P67": {"family": "A", "qs": Q3, "prior": [1, 1, 4]},
    "RMF_P80": {"family": "A", "qs": Q3, "prior": [1, 1, 8]},
    # Family B: conservative-advantage terminal decisions; reliability-gated CGC proposals.
    "RMF_U3_CONS1": {"family": "B", "qs": Q3, "prior": [1, 1, 1], "decision": "conservative", "tau": 1.0},
    "GATED_CGC_U3": {"family": "B_GATED_CGC", "qs": Q3, "prior": [1, 1, 1]},
    # Family C: acquisition.
    "RMF_U3_GREEDY": {"family": "C", "qs": Q3, "prior": [1, 1, 1], "acquisition": "greedy"},
    "RMF_U3_INFOQ": {"family": "C", "qs": Q3, "prior": [1, 1, 1], "acquisition": "infoq", "lam": 1.0},
    # DEV round 1. Family A: two-hypothesis Bayes rules trace the CLEAN-vs-FLIP40 frontier directly
    # (a Lagrangian of CLEAN and FLIP40 risk is the Bayes risk of a prior on {clean, q=0.4} only).
    "RMF_Q2_E": {"family": "A", "qs": [0.0, 0.4], "prior": [1, 1]},
    "RMF_Q2_P60": {"family": "A", "qs": [0.0, 0.4], "prior": [2, 3]},
    "RMF_Q2_P67": {"family": "A", "qs": [0.0, 0.4], "prior": [1, 2]},
    # Family B: a less strict conservative-advantage threshold; minimax regret over plausible hypotheses.
    "RMF_U3_CONS025": {"family": "B", "qs": Q3, "prior": [1, 1, 1], "decision": "conservative", "tau": 0.25},
    "RMF_U3_MINIMAX": {"family": "B", "qs": Q3, "prior": [1, 1, 1], "decision": "minimax", "tau": 0.05},
    # Family C: reliability-information-heavy acquisition (score = MI + 0.1 * predicted success).
    "RMF_U3_INFOQ01": {"family": "C", "qs": Q3, "prior": [1, 1, 1], "acquisition": "infoq", "lam": 0.1},
    # Family E: an event-level corruption hypothesis keeps duplicate agreement from certifying cleanliness.
    "RMF_U3E": {"family": "E", "qs": [0.0, 0.2, 0.4, 0.4], "prior": [1, 1, 1, 0.5],
                "kinds": ["channel", "channel", "channel", "event"]},
    # DEV round 2 (final selection round). Design rule fixed before the round: a learner run under a
    # duplicate-reading contract keeps an event-level hypothesis, because duplicate agreement cannot test
    # the contract's own independence assumption.
    "RMF_U3E_GREEDY": {"family": "E", "qs": [0.0, 0.2, 0.4, 0.4], "prior": [1, 1, 1, 0.5],
                       "kinds": ["channel", "channel", "channel", "event"], "acquisition": "greedy"},
    "RMF_U3E25": {"family": "E", "qs": [0.0, 0.2, 0.4, 0.4], "prior": [1, 1, 1, 0.25],
                  "kinds": ["channel", "channel", "channel", "event"]},
    "RMF_P67E": {"family": "E", "qs": [0.0, 0.2, 0.4, 0.4], "prior": [1, 1, 4, 1],
                 "kinds": ["channel", "channel", "channel", "event"]},
}
