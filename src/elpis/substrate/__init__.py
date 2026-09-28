"""Elpis substrate: resource authority and memory residency below every model.

The substrate manages bytes, descriptors, residency and native code. It knows
nothing about which model (if any) consumes those bytes.

* ``contracts``: typed fail-closed contract errors shared with inference.
* ``digests``: raw-byte digests (distinct from canonical structured identity).
* ``boundary``: Linux descriptor capabilities (``openat2`` beneath one trusted
  root, no symlinks) and sealed-memfd native loading with digest pins.
* ``authority``: deployment-pinned catalogs of admissible assets and native
  libraries. Inspection never creates authority.
* ``file_assets``: immutable external files admitted page by page, verified
  against the pinned catalog, and materialized on demand into bounded native
  FMS residency (HOT/WARM tiers) through leases with deterministic eviction.
* ``synthetic``: explicitly self-authorized provider for generated test
  fixtures only; the production constructor rejects its provenance.

The native side (``native/substrate``) provides SHA-256, the FMS residency
core with its POSIX platform abstraction layer, and the file-asset bridge.
"""

__all__ = ("contracts", "digests", "boundary", "authority", "file_assets", "synthetic")
