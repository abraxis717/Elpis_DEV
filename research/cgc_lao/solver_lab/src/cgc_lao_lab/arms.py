"""Arm registry: name -> factory(regime, contract) -> (agent, privileged-input spec or None).

Admissible factories never look at the regime: it is accepted only so that every factory has one
signature, and the test suite proves that swapping it changes nothing. Only ``KNOWN_Q_ORACLE`` reads it.
Frozen coefficients are loaded from ``coefficients/`` and verified by digest.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from . import agents as ag
from .cgc import load_weights
from .candidates import CANDIDATES

LAB = Path(__file__).resolve().parents[2]
COEFFICIENTS = LAB / "coefficients"
CGC_FILE = COEFFICIENTS / "cgc_reference.v1.json"
BASELINE_FILE = COEFFICIENTS / "baselines.v1.json"

BASELINES = ("NOOP", "SGD", "BETA", "CGC_REFERENCE", "STATIC_ATTENUATION", "CLEAN_BAYES")
PRIVILEGED = ("KNOWN_Q_ORACLE",)


@lru_cache(maxsize=None)
def cgc_weights():
    return load_weights(CGC_FILE)


@lru_cache(maxsize=None)
def baseline_coefficients() -> dict:
    return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))["coefficients"]


def _admissible(build):
    def factory(_regime, _contract):
        return build(), None
    factory.privileged = False
    return factory


def _oracle(regime, contract):
    return ag.known_q_oracle(regime, contract)


_oracle.privileged = True


def bayes_from_config(name: str, cfg: dict) -> ag.BayesAgent:
    return ag.BayesAgent(cfg["qs"], cfg["prior"], cfg.get("bias_var", 1.0), cfg.get("kinds"),
                         cfg.get("acquisition", "probe"), cfg.get("decision", "bayes"), cfg.get("lam", 1.0),
                         cfg.get("tau", 0.0), cfg.get("single_read", False), name=name)


def factory(name: str, coefficients: dict | None = None, sgd_eta: float | None = None,
            beta_kappa: float | None = None, cgc_w=None, attenuation: float | None = None):
    """The factory of a named arm. Optional overrides are used by TRAIN calibration only."""
    if name == "NOOP":
        return _admissible(ag.NoopAgent)
    if name == "SGD":
        eta = sgd_eta if sgd_eta is not None else baseline_coefficients()["SGD"]["eta"]
        return _admissible(lambda: ag.SGDAgent(eta))
    if name == "BETA":
        kappa = beta_kappa if beta_kappa is not None else baseline_coefficients()["BETA"]["kappa"]
        return _admissible(lambda: ag.BetaAgent(kappa))
    if name == "CGC_REFERENCE":
        w = cgc_w if cgc_w is not None else cgc_weights()
        return _admissible(lambda: ag.CGCAgent(w))
    if name == "STATIC_ATTENUATION":
        w = cgc_w if cgc_w is not None else cgc_weights()
        lam = attenuation if attenuation is not None else baseline_coefficients()["STATIC_ATTENUATION"]["attenuation"]
        return _admissible(lambda: ag.CGCAgent(w, lam, name="STATIC_ATTENUATION"))
    if name == "CLEAN_BAYES":
        return _admissible(ag.clean_bayes)
    if name == "KNOWN_Q_ORACLE":
        return _oracle
    if name in CANDIDATES:
        cfg = CANDIDATES[name]
        if cfg.get("family") == "B_GATED_CGC":
            w = cgc_w if cgc_w is not None else cgc_weights()
            return _admissible(lambda: ag.GatedCGCAgent(w, cfg["qs"], cfg["prior"], name=name))
        return _admissible(lambda: bayes_from_config(name, cfg))
    raise KeyError(f"unknown arm {name!r}")


def factories(names) -> dict:
    return {n: factory(n) for n in names}
