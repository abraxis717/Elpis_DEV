"""DSV4.1 transformer/MoE/Engram tower. RESEARCH_ONLY. NO_RUNTIME_AUTHORITY.

Donor/oracle/qualification code relocated out of the canonical tree by the
mission correction (docs/ELPIS_MISSION.md): DSV4 is Elpis's communication
codec, never its cognition. This package is not installed, src/ never imports
it, and no canonical runtime path composes it. No learned parameters are
provided; qualification runs only on deterministic TRAINING=NONE fixtures.
"""
from .config import TowerConfig
from .native_backend import DSV41NativeBackend
from .parameters import FileTensor, ParameterManifest, TensorBinding
from .target import DSV41Target

__all__ = ("TowerConfig", "DSV41NativeBackend", "FileTensor", "ParameterManifest", "TensorBinding", "DSV41Target")
