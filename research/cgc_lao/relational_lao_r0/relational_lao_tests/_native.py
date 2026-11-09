"""Test harness only: ctypes access to the native relational adapter (research/cgc_lao/relational_lao_r0/native),
compiled into a temporary directory with ``cc -std=c11 -O2 -ffp-contract=off -fPIC -shared`` (nothing is written
to the repository). ``NativeRelational`` mirrors ``relational_lao.scalar.ScalarRelational`` call for call."""
from __future__ import annotations

import ctypes as C
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

NATIVE = Path(__file__).resolve().parents[1] / "native"
HYP, SLOTS, BUFFER, PATHS, PATH_LEN = 4, 8, 8, 3, 3
READOUT = {"coupled": 0, "marginal": 1, "independent": 2}


class NativeUnavailable(RuntimeError):
    pass


class Config(C.Structure):
    _fields_ = [("magic", C.c_uint32), ("hypotheses", C.c_uint32), ("slots", C.c_uint32), ("readout", C.c_uint32),
                ("q", C.c_double * HYP), ("rho", C.c_double * HYP), ("log_prior", C.c_double * HYP),
                ("bias_var", C.c_double), ("digest", C.c_uint64)]


class Event(C.Structure):
    _fields_ = [("step", C.c_uint32), ("slot", C.c_uint8), ("n_reads", C.c_uint8), ("reads", C.c_uint8 * 2),
                ("binding", C.c_uint32), ("reserved", C.c_uint32), ("z", C.c_double)]


class State(C.Structure):
    _fields_ = [("magic", C.c_uint32), ("slots", C.c_uint32), ("hypotheses", C.c_uint32), ("buffered", C.c_uint32),
                ("config_digest", C.c_uint64), ("counts", C.c_uint32 * SLOTS), ("mu", (C.c_double * SLOTS) * HYP),
                ("v", (C.c_double * SLOTS) * HYP), ("lam", (C.c_double * SLOTS) * HYP), ("buffer", Event * BUFFER)]


class Alternative(C.Structure):
    _fields_ = [("paths", C.c_uint32), ("length", C.c_uint32 * PATHS), ("slot", (C.c_uint32 * PATH_LEN) * PATHS)]


_LIB: list = []


def library():
    if _LIB:
        return _LIB[0]
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        raise NativeUnavailable("no C compiler")
    out = Path(tempfile.mkdtemp(prefix="rlao_"))
    so = out / "librlao.so"
    cmd = [cc, "-std=c11", "-O2", "-ffp-contract=off", "-fPIC", "-shared", "-I", str(NATIVE / "include"),
           str(NATIVE / "src" / "rlao.c"), "-lm", "-o", str(so)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        raise NativeUnavailable(res.stderr[-2000:])
    lib = C.CDLL(str(so))
    dp, sp, cp = C.POINTER(C.c_double), C.POINTER(State), C.POINTER(Config)
    sig = {"rlao_config_init": ([cp, C.c_uint32, dp, dp, dp, C.c_uint32, C.c_double, C.c_uint32], C.c_int),
           "rlao_reset": ([sp, cp], C.c_int),
           "rlao_observe": ([sp, cp, C.c_uint32, C.c_double, C.POINTER(C.c_uint8), C.c_uint32, C.c_uint32,
                             C.c_uint32], C.c_int),
           "rlao_weights": ([sp, cp, dp], C.c_int),
           "rlao_predictive": ([sp, cp, dp, dp], C.c_int),
           "rlao_expected_utility": ([sp, cp, C.POINTER(Alternative), dp, dp], C.c_int),
           "rlao_rebuild": ([sp, cp, C.c_uint32], C.c_int),
           "rlao_drop": ([sp, cp, C.c_uint32], C.c_int),
           "rlao_reset_slot": ([sp, cp, C.c_uint32], C.c_int),
           "rlao_zeroize": ([sp], None),
           "rlao_state_size": ([], C.c_size_t), "rlao_config_size": ([], C.c_size_t),
           "rlao_logical_scalars": ([cp], C.c_uint32)}
    for name, (args, res_t) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res_t
    if lib.rlao_state_size() != C.sizeof(State) or lib.rlao_config_size() != C.sizeof(Config):
        raise NativeUnavailable("ctypes layout does not match the compiled structures")
    _LIB.append(lib)
    return lib


class NativeError(RuntimeError):
    pass


def _check(code: int) -> None:
    if code != 0:
        raise NativeError(code)


class NativeRelational:
    def __init__(self, hypotheses, prior, slots=8, bias_var=1.0, readout="coupled"):
        self.lib = library()
        M = len(hypotheses)
        q, rho, pr = (C.c_double * HYP)(), (C.c_double * HYP)(), (C.c_double * HYP)()
        for m, (qq, rr) in enumerate(hypotheses):
            q[m], rho[m], pr[m] = qq, rr, prior[m]
        self.cfg, self.st, self.M, self.R = Config(), State(), M, slots
        _check(self.lib.rlao_config_init(C.byref(self.cfg), M, q, rho, pr, slots, bias_var, READOUT[readout]))
        _check(self.lib.rlao_reset(C.byref(self.st), C.byref(self.cfg)))

    def update(self, r, z, reads, step=0, binding=0):
        buf = (C.c_uint8 * 2)(*list(reads) + [0] * (2 - len(reads)))
        _check(self.lib.rlao_observe(C.byref(self.st), C.byref(self.cfg), r, z, buf, len(reads), step, binding))

    def weights(self):
        out = (C.c_double * HYP)()
        _check(self.lib.rlao_weights(C.byref(self.st), C.byref(self.cfg), out))
        return list(out[:self.M])

    def predictive(self, z_row):
        z, out = (C.c_double * SLOTS)(*z_row), (C.c_double * SLOTS)()
        _check(self.lib.rlao_predictive(C.byref(self.st), C.byref(self.cfg), z, out))
        return list(out[:self.R])

    def expected_utility(self, alternative, z_row):
        a = Alternative()
        a.paths = len(alternative)
        for i, path in enumerate(alternative):
            a.length[i] = len(path)
            for k, r in enumerate(path):
                a.slot[i][k] = r
        z, out = (C.c_double * SLOTS)(*z_row), C.c_double()
        _check(self.lib.rlao_expected_utility(C.byref(self.st), C.byref(self.cfg), C.byref(a), z, C.byref(out)))
        return out.value

    def rebuild(self, keep_mask: int):
        _check(self.lib.rlao_rebuild(C.byref(self.st), C.byref(self.cfg), keep_mask))

    def drop(self, mask: int):
        _check(self.lib.rlao_drop(C.byref(self.st), C.byref(self.cfg), mask))

    def reset_slot(self, r: int):
        _check(self.lib.rlao_reset_slot(C.byref(self.st), C.byref(self.cfg), r))

    def zeroize(self):
        self.lib.rlao_zeroize(C.byref(self.st))

    def state(self):
        s = self.st
        return (tuple(tuple(s.mu[m][r] for r in range(self.R)) for m in range(self.M)),
                tuple(tuple(s.v[m][r] for r in range(self.R)) for m in range(self.M)),
                tuple(tuple(s.lam[m][r] for r in range(self.R)) for m in range(self.M)),
                tuple(s.counts[r] for r in range(self.R)))

    def buffered(self):
        return int(self.st.buffered)

    def logical_scalars(self):
        return int(self.lib.rlao_logical_scalars(C.byref(self.cfg)))
