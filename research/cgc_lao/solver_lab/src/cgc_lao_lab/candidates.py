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
}
