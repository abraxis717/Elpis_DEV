"""K1 Recovery R0: the operator surface of the bounded K1 checkpoint owner (docs/K1_RECOVERY_R0.md).

RuntimeCore owns the checkpoint protocol (``native/runtime/src/checkpoint.rs``): two fixed slot files holding the
complete ``(W, epoch, H, a)`` envelope of the authorized K1 state and of at most one newer candidate. CHECKPOINT
BYTES DO NOT AUTHORIZE THEMSELVES. CONTINUITY REMAINS THE CURRENT-AUTHORITY REGISTER.

This module holds only the explicit operator act of provisioning a store (the one way slot files come to exist)
and re-exports the recovery disposition. The runtime never provisions, resizes or adopts a store on its own:
``RuntimeConfig.k1_checkpoint_dir`` names an already provisioned directory, and a missing or malformed store
refuses the runtime's open.
"""
from __future__ import annotations

from pathlib import Path

from .core import K1Recovery, RuntimeLibrary
from .errors import CompositionError

__all__ = ("K1Recovery", "provision_k1_checkpoint")


def provision_k1_checkpoint(library: RuntimeLibrary, directory: Path, envelope_bytes: int) -> None:
    """Operator provisioning: create ``directory`` (absolute; it must not exist) with exactly two zeroed slot files
    sized for envelopes of ``envelope_bytes`` (``K1Library.envelope_bytes(dim, width)`` of the admitted shape),
    fully written and synced. Never overwrites, resizes or adopts anything (``CHECKPOINT_INVALID``)."""
    if type(library) is not RuntimeLibrary:
        raise TypeError("provision_k1_checkpoint takes an admitted RuntimeLibrary")
    if not isinstance(directory, Path) or not directory.is_absolute():
        raise CompositionError("RUNTIME_PATH", "the checkpoint directory must be an absolute Path")
    if type(envelope_bytes) is not int or envelope_bytes <= 0:
        raise CompositionError("CHECKPOINT_INVALID", "envelope_bytes: a positive int")
    raw = str(directory).encode()
    library.check(library._lib.elpis_runtime_checkpoint_provision(raw, len(raw), envelope_bytes))
