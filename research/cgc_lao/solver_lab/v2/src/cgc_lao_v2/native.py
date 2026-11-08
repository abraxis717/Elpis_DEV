"""ctypes binding of the native C11 learner (research harness only).

The C source is compiled with the system C compiler into a private temporary directory (removed at exit):
``cc -std=c11 -O2 -ffp-contract=off -fPIC -shared``. Nothing is written to the repository. ``NativeRMF`` has the
same API as ``scalar2.ScalarRMF2``; every C status other than OK raises ``NativeError`` with its name.
"""
from __future__ import annotations

import atexit
import ctypes as C
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import V2_ROOT

NATIVE = V2_ROOT / "native"
SOURCE = NATIVE / "src" / "cgc_lao_rmf.c"
INCLUDE = NATIVE / "include"
MAX_HYP, MAX_NODES, K, STOP = 8, 4, 3, 3
STATUS = {0: "OK", 1: "E_NULL", 2: "E_CONFIG", 3: "E_STATE", 4: "E_INPUT", 5: "E_HORIZON", 6: "E_NUMERIC"}
_lib = None


class NativeUnavailable(RuntimeError):
    pass


class NativeError(RuntimeError):
    def __init__(self, code: int):
        super().__init__(STATUS.get(code, f"E_{code}"))
        self.code = code


class Config(C.Structure):
    _fields_ = [("magic", C.c_uint32), ("hypotheses", C.c_uint32), ("horizon", C.c_uint32),
                ("acquisition", C.c_uint32), ("nodes", C.c_uint32 * MAX_HYP),
                ("q", (C.c_double * MAX_NODES) * MAX_HYP), ("node_weight", C.c_double * MAX_HYP),
                ("rho", C.c_double * MAX_HYP), ("log_prior", C.c_double * MAX_HYP), ("bias_var", C.c_double),
                ("stop_payoff", C.c_double), ("digest", C.c_uint64)]


class State(C.Structure):
    _fields_ = [("magic", C.c_uint32), ("t", C.c_uint32), ("counts", C.c_uint32 * K), ("hypotheses", C.c_uint32),
                ("config_digest", C.c_uint64), ("log_weight", C.c_double * MAX_HYP),
                ("mean", (C.c_double * K) * MAX_HYP), ("var", (C.c_double * K) * MAX_HYP)]


def compiler() -> str | None:
    return os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def build(extra_flags=(), name="libcgc_lao_rmf.so") -> Path:
    cc = compiler()
    if cc is None:
        raise NativeUnavailable("no C compiler")
    out = Path(tempfile.mkdtemp(prefix="cgc_lao_rmf_"))
    atexit.register(shutil.rmtree, out, True)
    lib = out / name
    cmd = [cc, "-std=c11", "-O2", "-ffp-contract=off", "-fPIC", "-shared", *extra_flags, "-I", str(INCLUDE),
           str(SOURCE), "-o", str(lib), "-lm"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        raise NativeUnavailable(f"compilation failed: {res.stderr.strip()[:500]}")
    return lib


def library():
    global _lib
    if _lib is None:
        lib = C.CDLL(str(build()))
        dp, up = C.POINTER(C.c_double), C.POINTER(C.c_uint32)
        sig = {
            "cgc_lao_rmf_state_size": ([], C.c_size_t), "cgc_lao_rmf_config_size": ([], C.c_size_t),
            "cgc_lao_rmf_logical_scalars": ([C.POINTER(Config)], C.c_uint32),
            "cgc_lao_rmf_config_init": ([C.POINTER(Config), C.c_uint32, up, dp, dp, dp, C.c_double, C.c_double,
                                         C.c_uint32, C.c_uint32], C.c_int),
            "cgc_lao_rmf_reset": ([C.POINTER(State), C.POINTER(Config)], C.c_int),
            "cgc_lao_rmf_predictive": ([C.POINTER(State), C.POINTER(Config), dp, dp], C.c_int),
            "cgc_lao_rmf_propose": ([C.POINTER(State), C.POINTER(Config), dp, up], C.c_int),
            "cgc_lao_rmf_observe": ([C.POINTER(State), C.POINTER(Config), C.c_uint32, C.c_double,
                                     C.POINTER(C.c_uint8), C.c_uint32, C.c_uint32], C.c_int),
            "cgc_lao_rmf_weights": ([C.POINTER(State), C.POINTER(Config), dp], C.c_int),
            "cgc_lao_rmf_state_correction": ([C.POINTER(State), C.POINTER(Config), dp], C.c_int),
            "cgc_lao_rmf_decide": ([C.POINTER(State), C.POINTER(Config), dp, up], C.c_int),
            "cgc_lao_rmf_zeroize": ([C.POINTER(State)], None),
        }
        for fname, (args, res) in sig.items():
            fn = getattr(lib, fname)
            fn.argtypes, fn.restype = args, res
        if lib.cgc_lao_rmf_state_size() != C.sizeof(State) or lib.cgc_lao_rmf_config_size() != C.sizeof(Config):
            raise NativeUnavailable("ctypes layout does not match the compiled structures")
        _lib = lib
    return _lib


def _check(code: int) -> None:
    if code != 0:
        raise NativeError(code)


class NativeRMF:
    def __init__(self, hypotheses, prior, bias_var=1.0, stop_payoff=0.6, horizon=8, acquisition=0):
        self.lib = library()
        m = len(hypotheses)
        nodes = (C.c_uint32 * MAX_HYP)()
        q = (C.c_double * (MAX_HYP * MAX_NODES))()
        rho = (C.c_double * MAX_HYP)()
        pr = (C.c_double * MAX_HYP)()
        if len(prior) != m or m > MAX_HYP:
            raise NativeError(2)
        for i, (qv, rv) in enumerate(hypotheses):
            qs = [float(qv)] if isinstance(qv, (int, float)) else [float(x) for x in qv]
            if len(qs) > MAX_NODES:
                raise NativeError(2)
            nodes[i] = len(qs)
            for j, x in enumerate(qs):
                q[i * MAX_NODES + j] = x
            rho[i] = float(rv)
            pr[i] = float(prior[i])
        self.cfg = Config()
        _check(self.lib.cgc_lao_rmf_config_init(C.byref(self.cfg), m, nodes, q, rho, pr, float(bias_var),
                                                float(stop_payoff), int(horizon), int(acquisition)))
        self.st = State()
        self.M = m
        self.reset()

    # lifecycle -----------------------------------------------------------------------------------------
    def reset(self) -> None:
        _check(self.lib.cgc_lao_rmf_reset(C.byref(self.st), C.byref(self.cfg)))

    def zeroize(self) -> None:
        self.lib.cgc_lao_rmf_zeroize(C.byref(self.st))

    def state_bytes(self) -> bytes:
        return bytes(self.st)

    def state_scalars(self) -> int:
        return int(self.lib.cgc_lao_rmf_logical_scalars(C.byref(self.cfg)))

    # algorithm -----------------------------------------------------------------------------------------
    @staticmethod
    def _z(z_row):
        return (C.c_double * K)(*[float(v) for v in z_row])

    def predictive(self, z_row):
        out = (C.c_double * K)()
        _check(self.lib.cgc_lao_rmf_predictive(C.byref(self.st), C.byref(self.cfg), self._z(z_row), out))
        return list(out)

    def select(self, z_row) -> int:
        out = C.c_uint32()
        _check(self.lib.cgc_lao_rmf_propose(C.byref(self.st), C.byref(self.cfg), self._z(z_row), C.byref(out)))
        return int(out.value)

    def update(self, a: int, z: float, reads, advance: bool = True) -> None:
        buf = (C.c_uint8 * 2)(*[int(r) for r in list(reads)[:2]])
        _check(self.lib.cgc_lao_rmf_observe(C.byref(self.st), C.byref(self.cfg), int(a), float(z), buf,
                                            len(reads), 1 if advance else 0))

    def weights(self):
        out = (C.c_double * MAX_HYP)()
        _check(self.lib.cgc_lao_rmf_weights(C.byref(self.st), C.byref(self.cfg), out))
        return list(out)[:self.M]

    def state_correction(self):
        out = (C.c_double * K)()
        _check(self.lib.cgc_lao_rmf_state_correction(C.byref(self.st), C.byref(self.cfg), out))
        return list(out)

    def decide(self, z_row) -> int:
        out = C.c_uint32()
        _check(self.lib.cgc_lao_rmf_decide(C.byref(self.st), C.byref(self.cfg), self._z(z_row), C.byref(out)))
        return int(out.value)

    # raw state views (for parity tests) ---------------------------------------------------------------
    @property
    def mu(self):
        return [[self.st.mean[m][a] for a in range(K)] for m in range(self.M)]

    @property
    def v(self):
        return [[self.st.var[m][a] for a in range(K)] for m in range(self.M)]

    @property
    def lw(self):
        return [self.st.log_weight[m] for m in range(self.M)]

    @property
    def t(self):
        return int(self.st.t)

    @property
    def counts(self):
        return [int(c) for c in self.st.counts]
