"""CGC -> HCA88 geometric bridge: the frozen cubic operator and its relational feature map.

The operator is the Q4E5 three-output cubic CGC of ``research/cgc_lao/reference``:
float32 weights ``[3, 5, 8]`` (120 scalars), ``out_o = 0.5 * sum_n (z + z^2 + z^3)`` with
``z = sum_d x_d w[o, d, n]``. It is evaluated here with a fixed summation order so that the
C kernel and this reference agree bit for bit.

**No trained CGC weights exist in Elpis_DEV** (the Q4E5 checkpoint is not published, and it
was trained for a different task). The weights below are SYNTHETIC and exist only to exercise
the mechanics; they carry no scientific qualification. ``zero_weights`` gives the
coefficient-free law (pure transported-evidence update), the matched lightweight alternative.

Feature map (all gauge invariant; ``v`` is the signed observation transported into the
relation frame, ``A``, ``B`` the relation state):

    x = (tanh A.v, tanh B.v, tanh A.B, tanh |v - A|^2, aux)

mirroring the Q4E5 feature style (bounded residual-type terms, a squared surprise, a clock).
The three outputs become bounded coefficients ``kappa * tanh(out)``, ``kappa`` = the Q4E5
``STEP_CAP`` = 0.25.
"""
from __future__ import annotations

import hashlib

import numpy as np

KAPPA = 0.25           # research/cgc_lao/reference/src/cgc/lao_q4e5.py STEP_CAP
RHO_B = 0.63           # frozen source fixture ratio B = 0.63 A (portable_mechanics.state_pair)
SYNTHETIC_LABEL = "SYNTHETIC_FIXED_WEIGHTS_NOT_TRAINED_NOT_QUALIFIED"


def synthetic_weights(seed: int = 120120) -> np.ndarray:
    """Deterministic SYNTHETIC float32 [3, 5, 8] weights (mechanical tests only)."""
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray((rng.normal(0.0, 0.25, size=(3, 5, 8))).astype(np.float32))


def zero_weights() -> np.ndarray:
    return np.zeros((3, 5, 8), dtype=np.float32)


def weights_sha256(w) -> str:
    return hashlib.sha256(np.ascontiguousarray(w, dtype=np.float32).tobytes()).hexdigest()


def check_weights(w) -> np.ndarray:
    w = np.ascontiguousarray(w)
    if w.shape != (3, 5, 8) or w.dtype != np.float32 or not np.all(np.isfinite(w)):
        raise ValueError("CGC_WEIGHT_CONTRACT")      # the Q4E5 contract: float32 [3, 5, 8], finite
    w.setflags(write=False)
    return w


def forward_reference(w, x) -> list:
    """The cubic operator in IEEE single precision with the kernel's summation order."""
    f = np.float32
    out = []
    for o in range(3):
        acc = f(0.0)
        for n in range(8):
            z = f(0.0)
            for d in range(5):
                z = f(z + f(f(x[d]) * w[o, d, n]))
            z2 = f(z * z)
            acc = f(acc + f(f(z + z2) + f(z2 * z)))
        out.append(f(f(0.5) * acc))
    return out
