"""ECS: the one canonical ECS / EDEN-facing cognitive substrate of Elpis.

The ECS owns the complete cognitive state ``(W, epoch, H, a)`` and its
qualified mechanics: the native cubic/S3 kernel and stateful recurrence
(``native``), the Runtime R1 executor, learning and readout, the native K1
runtime whose transactions commit the complete state (``k1``), the additive
mutable FMS residency adapter (``residency``) and the ECS-native cognitive
core (``cognition``).

Python may control the ECS; it may not execute its hot path. This package
imports only the standard library and its own native bindings: no inference,
runtime, continuity or research code. The runtime composes it.

Its native ABI (``elpis_ecsg_*`` symbols, ``elpis/ecsg_*.h`` headers and the
``libelpis_ecsg_*`` libraries in ``native/ECS``) is the qualified, byte-bound
implementation identity and is retained unchanged (docs/ARCHITECTURE.md).
"""
