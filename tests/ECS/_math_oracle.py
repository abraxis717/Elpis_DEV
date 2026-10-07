"""Test-only oracle for the frozen ECS cubic/S3 mathematics."""

from __future__ import annotations

import numpy as np


def phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * z + 0.5 * z * z + 0.5 * z * z * z


def forward(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return phi(x @ w).sum(axis=1)


def project_s3(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = int(w.shape[0])

    mu = w.sum(axis=1)
    M = w @ w.T
    T3 = np.einsum("ai,bi,ci->abc", w, w, w, optimize=True)

    m_packed = np.asarray(
        [M[a, b] for a in range(dim) for b in range(a, dim)],
        dtype=np.float64,
    )
    t3_packed = np.asarray(
        [
            T3[a, b, c]
            for a in range(dim)
            for b in range(a, dim)
            for c in range(b, dim)
        ],
        dtype=np.float64,
    )
    return (
        np.ascontiguousarray(mu, dtype=np.float64),
        np.ascontiguousarray(m_packed, dtype=np.float64),
        np.ascontiguousarray(t3_packed, dtype=np.float64),
    )


def forward_s3(
    mu: np.ndarray,
    m_packed: np.ndarray,
    t3_packed: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    dim = int(mu.shape[0])
    out = np.empty(x.shape[0], dtype=np.float64)

    for r, row in enumerate(x):
        linear = float(row @ mu)

        quadratic = 0.0
        p = 0
        for a in range(dim):
            for b in range(a, dim):
                mult = 1.0 if a == b else 2.0
                quadratic += mult * m_packed[p] * row[a] * row[b]
                p += 1

        cubic = 0.0
        p = 0
        for a in range(dim):
            for b in range(a, dim):
                for c in range(b, dim):
                    if a == b == c:
                        mult = 1.0
                    elif a == b or b == c or a == c:
                        mult = 3.0
                    else:
                        mult = 6.0
                    cubic += (
                        mult
                        * t3_packed[p]
                        * row[a]
                        * row[b]
                        * row[c]
                    )
                    p += 1

        out[r] = 0.5 * linear + 0.5 * quadratic + 0.5 * cubic

    return out


def loss(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    e = forward(w, x) - y
    return float(np.mean(e * e))


def grad(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    e = forward(w, x) - y
    z = x @ w
    phip = 0.5 + z + 1.5 * z * z
    return (2.0 / len(x)) * x.T @ (e[:, None] * phip)


def gd_step(
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.002,
) -> np.ndarray:
    return w - learning_rate * grad(w, x, y)
