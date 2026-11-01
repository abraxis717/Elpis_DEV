"""The Python reference's process gate (docs/ELPIS_MISSION.md, Amendment 3; docs/OPERATING_ELPIS.md, "Environment").

A :class:`~elpis.runtime.composition.Runtime` opens only in a process that cannot carry its volatile state onto
persistent storage through the interpreter's or the kernel's indirect channels; otherwise it refuses
(``ENVIRONMENT_REFUSED``, fail closed). :func:`require_volatile_process` establishes what it can and reads it back:

* No bytecode cache writes: ``sys.dont_write_bytecode`` must be true (run the interpreter with ``-B`` or
  ``PYTHONDONTWRITEBYTECODE=1``). The gate cannot undo what the interpreter cached for the imports that ran before
  it; the offline CLI therefore also requires the startup flag (``sys.flags.dont_write_bytecode``).
* No core dumps: RLIMIT_CORE is lowered to 0 (soft and hard), and the process is made non-dumpable
  (PR_SET_DUMPABLE 0). A non-dumpable process is never dumped, whatever the kernel's core pattern or collector.
* The swap policy: /proc/swaps is read. With a swap area active, every page of the process is locked in RAM
  (mlockall, current and future) if the lock can hold the process: RLIMIT_MEMLOCK unlimited or the IPC_LOCK
  capability. Otherwise the runtime refuses; a small lock limit would turn later allocations into failures. The
  host's swap configuration is never read for writing or changed.

The gate is Linux-only (elsewhere it cannot verify the controls, so it refuses). It is idempotent and process-wide:
once the process is non-dumpable with a core limit of 0, both stay so and protect every later session.
The native host (docs/NATIVE_HOST.md, "Containment") enforces the same controls plus write containment (Landlock,
seccomp, no inherited descriptor); this gate adds no write containment of its own. The Python reference is an
offline interface, and write containment for it is the launcher's (``elpis_autonomous_run``) or the caller's
sandbox.
"""
from __future__ import annotations

import ctypes
import os
import resource
import sys
from dataclasses import dataclass
from pathlib import Path

from .errors import CompositionError

PR_SET_DUMPABLE, PR_GET_DUMPABLE = 4, 3
MCL_CURRENT, MCL_FUTURE = 1, 2
CAP_IPC_LOCK = 14


@dataclass(frozen=True)
class ProcessContainment:
    """What the gate established (read back)."""
    bytecode_writes: bool       # False: sys.dont_write_bytecode
    core_limit: tuple[int, int]
    dumpable: int
    swap_areas: int
    memory_locked: bool


def _refuse(detail: str) -> CompositionError:
    return CompositionError("ENVIRONMENT_REFUSED", detail)


def swap_areas(path: str | os.PathLike = "/proc/swaps") -> int:
    """The active swap areas: the lines of /proc/swaps after its header. Refuses when the policy cannot be read."""
    try:
        lines = Path(path).read_text(encoding="ascii", errors="replace").splitlines()
    except OSError as exc:
        raise _refuse(f"the swap policy is unknown ({path} unreadable: {exc.strerror})") from None
    if not lines:
        raise _refuse(f"the swap policy is unknown ({path} has no header)")
    return sum(1 for line in lines[1:] if line.strip())


def _has_ipc_lock() -> bool:
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("CapEff:"):
                return bool(int(line.split()[1], 16) >> CAP_IPC_LOCK & 1)
    except (OSError, ValueError, IndexError):
        pass
    return False


def require_volatile_process(*, proc_swaps: str | os.PathLike = "/proc/swaps") -> ProcessContainment:
    """Establish and verify the process controls, or raise ``CompositionError('ENVIRONMENT_REFUSED')``."""
    if not sys.platform.startswith("linux"):
        raise _refuse("the process controls are verified on Linux only")
    if not sys.dont_write_bytecode:
        raise _refuse("Python bytecode writing is enabled: run the interpreter with -B or PYTHONDONTWRITEBYTECODE=1")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError) as exc:
        raise _refuse(f"RLIMIT_CORE cannot be set to 0: {exc}") from None
    core = resource.getrlimit(resource.RLIMIT_CORE)
    if core != (0, 0):
        raise _refuse(f"RLIMIT_CORE reads back {core}, not (0, 0)")
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise _refuse(f"PR_SET_DUMPABLE 0 refused (errno {ctypes.get_errno()})")
    dumpable = libc.prctl(PR_GET_DUMPABLE, 0, 0, 0, 0)
    if dumpable != 0:
        raise _refuse(f"the process reads back dumpable {dumpable}, not 0")
    areas = swap_areas(proc_swaps)
    locked = False
    if areas:
        soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if soft != resource.RLIM_INFINITY and not _has_ipc_lock():
            raise _refuse(f"swap is active ({areas} area(s)) and the memory lock limit ({soft} bytes) cannot hold the "
                          "process: disable swap, raise RLIMIT_MEMLOCK, or run the native host (which locks its "
                          "memory); the host's swap configuration is never changed")
        libc.mlockall.argtypes = [ctypes.c_int]
        libc.mlockall.restype = ctypes.c_int
        if libc.mlockall(MCL_CURRENT | MCL_FUTURE) != 0:
            raise _refuse(f"swap is active and mlockall failed (errno {ctypes.get_errno()})")
        locked = True
    return ProcessContainment(bytecode_writes=False, core_limit=core, dumpable=dumpable, swap_areas=areas,
                              memory_locked=locked)
