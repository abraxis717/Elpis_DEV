"""Delay / memory: C_t versus (C_t .. C_{t-k}) under one predictor, one split and one metric. RESEARCH_ONLY.

Every representation is scored by the same NumPy ridge predictor, trained and
evaluated on the same trajectories, against the same target and metric.
Standardization statistics come from the training rows only. The ridge
strength is chosen on DEV worlds only (:func:`choose_lambda`) and then frozen.

Representations, each named with its dimension:

* ``weak``: the first coarse coordinate only;
* ``coarse``: ``C_t``;
* ``delay``: ``[C_t, C_{t-1}, .., C_{t-k}]`` (Lozano-Duran, arXiv:2609.19424,
  Eq. 3.11);
* ``shuffled_delay``: ``C_t`` plus lag blocks taken from a random permutation
  of time indices within the same trajectory (dimension-matched to
  ``delay``; the temporal alignment is destroyed);
* ``random_projection``: ``R x_t`` for a Gaussian ``R`` with the same
  dimension as ``delay``; microscopic information in random directions;
* ``seed``: ``C_t`` plus independent Gaussian columns ``U``. Lozano-Duran
  Eq. (3.44) states that an independent seed cannot lower the irreducible
  error; for a fitted predictor, extra noise columns can only add variance;
* ``full``: the microscopic state ``x_t``.

:func:`ladder_decomposition` arranges held-out errors in the
reference/penalty/gain form of Lozano-Duran Eq. (3.24):
``eps_mem^2 = eps_ref^2 + P_hidden^2 - G_mem^2`` with
``P_hidden^2 = eps_Mar^2 - eps_ref^2`` and ``G_mem^2 = eps_Mar^2 - eps_mem^2``.
The paper's quantities are IRREDUCIBLE errors (infimum over all predictors);
here they are held-out errors of one linear ridge class. They are
upper-bound surrogates, not irreducible errors, and no information-theoretic
quantity is estimated.
"""
from __future__ import annotations

import numpy as np

REPRESENTATIONS = ("weak", "coarse", "delay", "shuffled_delay", "random_projection", "seed", "full")
TARGET_KINDS = ("ONE_STEP_TRANSITION", "K_STEP_TRAJECTORY", "EVENT")


# ---------------------------------------------------------------------------
# Ridge predictor
# ---------------------------------------------------------------------------


class Ridge:
    def __init__(self, lam: float):
        self.lam = float(lam)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "Ridge":
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        Z = (X - self.mu) / self.sd
        self.ymu = Y.mean(axis=0)
        A = Z.T @ Z + self.lam * Z.shape[0] * np.eye(Z.shape[1])
        self.coef = np.linalg.solve(A, Z.T @ (Y - self.ymu))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float64) - self.mu) / self.sd) @ self.coef + self.ymu


def r2(Y: np.ndarray, P: np.ndarray) -> float:
    """Pooled coefficient of determination over all target columns (test-set mean)."""
    Y = np.asarray(Y).reshape(len(Y), -1)
    P = np.asarray(P).reshape(len(P), -1)
    sst = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    return 1.0 - float(np.sum((Y - P) ** 2)) / sst if sst > 0 else float("nan")


def rmse(Y: np.ndarray, P: np.ndarray) -> float:
    Y = np.asarray(Y).reshape(len(Y), -1)
    return float(np.sqrt(np.mean(np.sum((Y - np.asarray(P).reshape(len(P), -1)) ** 2, axis=1))))


def balanced_accuracy(E: np.ndarray, P: np.ndarray) -> float:
    E = np.asarray(E).ravel() > 0.5
    P = np.asarray(P).ravel() > 0.5
    rates = [np.mean(P[E == v] == v) for v in (True, False) if np.any(E == v)]
    return float(np.mean(rates)) if rates else float("nan")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def make_rows(states: np.ndarray, coarse: np.ndarray, *, depth: int, horizon: int, target: str,
              extras: dict) -> dict:
    """Feature matrices for every representation and the target, from one trajectory.

    ``states`` is ``(T, d)``, ``coarse`` is ``(T, m)``. Row ``t`` runs over
    ``depth .. T - 1 - h``, where ``h = 1`` for ONE_STEP_TRANSITION and
    ``h = horizon`` otherwise. ``extras`` holds per-trajectory randomness:
    ``perm`` (a permutation for the shuffled control), ``proj`` (the random
    projection) and ``seed`` (noise columns).
    """
    h = 1 if target == "ONE_STEP_TRANSITION" else horizon
    T, m = coarse.shape
    idx = np.arange(depth, T - h)
    lags = [coarse[idx - j] for j in range(depth + 1)]
    perm = extras["perm"]
    shuffled = [coarse[idx]] + [coarse[perm[idx] - j] for j in range(1, depth + 1)]
    rows = {
        "weak": coarse[idx, :1],
        "coarse": coarse[idx],
        "delay": np.concatenate(lags, axis=1),
        "shuffled_delay": np.concatenate(shuffled, axis=1),
        "random_projection": states[idx] @ extras["proj"].T,
        "seed": np.concatenate([coarse[idx], extras["seed"][idx]], axis=1),
        "full": states[idx],
    }
    if target == "EVENT":
        y = (coarse[idx + h, 0] > extras.get("event_threshold", 0.0)).astype(np.float64)[:, None]
    else:
        y = coarse[idx + h]
    rows["_target"] = y
    return rows


def random_projection(rng: np.random.Generator, d: int, m: int, depth: int) -> np.ndarray:
    """One Gaussian projection per world, shared by every trajectory, with the delay vector's dimension."""
    return rng.standard_normal((m * (depth + 1), d)) / np.sqrt(d)


def trajectory_extras(rng: np.random.Generator, T: int, depth: int, seed_dim: int, proj: np.ndarray) -> dict:
    """Per-trajectory control randomness (permutation, seed columns) plus the world's shared projection."""
    perm = np.arange(T)
    valid = np.arange(depth, T)
    perm[valid] = rng.permutation(valid)
    return {"perm": perm, "proj": proj, "seed": rng.standard_normal((T, seed_dim))}


def stack(rows_list: list[dict]) -> dict:
    return {k: np.concatenate([r[k] for r in rows_list], axis=0) for k in rows_list[0]}


def score(train: dict, test: dict, rep: str, lam: float, target: str) -> float:
    model = Ridge(lam).fit(train[rep], train["_target"])
    pred = model.predict(test[rep])
    if target == "EVENT":
        return balanced_accuracy(test["_target"], pred)
    return r2(test["_target"], pred)


def error(train: dict, test: dict, rep: str, lam: float) -> float:
    model = Ridge(lam).fit(train[rep], train["_target"])
    return rmse(test["_target"], model.predict(test[rep]))


def choose_lambda(dev_splits: list[tuple[dict, dict]], rep: str, target: str, grid) -> tuple[float, dict]:
    """Ridge strength maximizing the mean DEV validation score (ties go to the larger lambda)."""
    table = {}
    for lam in grid:
        table[float(lam)] = float(np.mean([score(tr, va, rep, lam, target) for tr, va in dev_splits]))
    best = max(sorted(table, reverse=True), key=lambda lam: table[lam])
    return best, table


def ladder_decomposition(train: dict, test: dict, lams: dict) -> dict:
    """Held-out surrogate of Lozano-Duran Eq. (3.24) with ref = full state, Mar = C_t, mem = delay."""
    eps_ref = error(train, test, "full", lams["full"])
    eps_mar = error(train, test, "coarse", lams["coarse"])
    eps_mem = error(train, test, "delay", lams["delay"])
    return {"eps_ref": eps_ref, "eps_markov": eps_mar, "eps_memory": eps_mem,
            "hidden_penalty_sq": eps_mar ** 2 - eps_ref ** 2, "memory_gain_sq": eps_mar ** 2 - eps_mem ** 2,
            "identity_residual": abs(eps_mem ** 2 - (eps_ref ** 2 + (eps_mar ** 2 - eps_ref ** 2)
                                                    - (eps_mar ** 2 - eps_mem ** 2)))}


def insample_least_squares_monotone(X_small: np.ndarray, X_big: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    """In-sample OLS residual for nested features: adding columns can never increase it."""
    def res(X):
        A = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        return float(np.sum((Y - A @ coef) ** 2))
    return res(X_small), res(X_big)
