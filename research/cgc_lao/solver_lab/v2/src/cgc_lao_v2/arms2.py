"""v2 arm registry: name -> factory(regime, contract) -> (agent, privileged spec or None).

Admissible factories ignore the regime (tested); only KNOWN_Q_ORACLE reads it. Coefficients come from the v2
TRAIN calibration (``v2/coefficients``), verified by digest; overrides are for TRAIN calibration only.
"""
from __future__ import annotations

import json
from functools import lru_cache

from cgc_lao_lab import agents as v1
from cgc_lao_lab.cgc import load_weights

from . import V2_ROOT
from .agents2 import (BayesAgent2, BetaAllAgent, CGCMeanReadingAgent, DisagreeSGDAgent, NoopAgent2,
                      PluginQBayesAgent, SGDAllAgent, known_q_oracle2)
from .candidates2 import BAYES_CONFIGS

COEFFICIENTS = V2_ROOT / "coefficients"
CGC_FILE = COEFFICIENTS / "cgc_reference.v2.json"
BASELINE_FILE = COEFFICIENTS / "baselines.v2.json"


@lru_cache(maxsize=None)
def cgc_weights():
    return load_weights(CGC_FILE)


@lru_cache(maxsize=None)
def coefficients() -> dict:
    return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))["coefficients"]


def _admissible(build):
    def factory(_regime, _contract):
        return build(), None
    factory.privileged = False
    return factory


def _oracle(regime, _contract):
    return known_q_oracle2(regime)


_oracle.privileged = True


def bayes(name: str, cfg: dict) -> BayesAgent2:
    return BayesAgent2(cfg["hypotheses"], cfg["prior"], cfg.get("bias_var", 1.0), cfg.get("acquisition", "probe"),
                       name=name)


def factory(name: str, **ov):
    c = lambda key, field: ov.get(key, None) if key in ov else coefficients()[field[0]][field[1]]  # noqa: E731
    w = lambda: ov["cgc_w"] if "cgc_w" in ov else cgc_weights()  # noqa: E731
    if name == "NOOP":
        return _admissible(NoopAgent2)
    if name == "SGD":
        eta = c("sgd_eta", ("SGD", "eta"))
        return _admissible(lambda: v1.SGDAgent(eta))
    if name == "BETA":
        kappa = c("beta_kappa", ("BETA", "kappa"))
        return _admissible(lambda: v1.BetaAgent(kappa))
    if name == "SGD_ALL":
        eta = c("sgd_all_eta", ("SGD_ALL", "eta"))
        return _admissible(lambda: SGDAllAgent(eta))
    if name == "BETA_ALL":
        kappa = c("beta_all_kappa", ("BETA_ALL", "kappa"))
        return _admissible(lambda: BetaAllAgent(kappa))
    if name == "DISAGREE_SGD":
        eta = c("disagree_eta", ("DISAGREE_SGD", "eta"))
        return _admissible(lambda: DisagreeSGDAgent(eta))
    if name == "PLUGIN_Q_BAYES":
        return _admissible(PluginQBayesAgent)
    if name == "CGC_REFERENCE":
        weights = w()
        return _admissible(lambda: v1.CGCAgent(weights))
    if name == "STATIC_ATTENUATION":
        weights = w()
        return _admissible(lambda: v1.CGCAgent(weights, 0.5, name="STATIC_ATTENUATION"))
    if name == "STATIC_ATT_ALL":
        weights = w()
        return _admissible(lambda: CGCMeanReadingAgent(weights, 0.5, name="STATIC_ATT_ALL"))
    if name == "CLEAN_BAYES":
        return _admissible(v1.clean_bayes)
    if name == "KNOWN_Q_ORACLE":
        return _oracle
    if name in BAYES_CONFIGS:
        cfg = BAYES_CONFIGS[name]
        return _admissible(lambda: bayes(name, cfg))
    raise KeyError(f"unknown v2 arm {name!r}")
