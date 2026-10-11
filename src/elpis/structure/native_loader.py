"""Only deployment-pinned, substrate-sealed native structure bridge loads.

The artifact pin is independent deployment authority, not computed from an
untrusted candidate. The substrate's one native admission mechanism
(``elpis.substrate.native_admission``: root capability, sealed memfd, exact pin)
is the single native trust mechanism. No path discovery, fallbacks or second loader.
"""
from __future__ import annotations

from pathlib import Path

from elpis.substrate.authority import PinnedAuthority
from elpis.substrate.contracts import Code, require
from elpis.substrate.native_admission import admit_library


def load_structure_bridge(root: str | Path, path: str | Path,
                          authority: PinnedAuthority, expected_library_id: str):
    """Verify the exact bridge ID in a deployment-pinned catalog, seal, load.

    Root selection and the catalog's *independent* SHA-256 pin come from the
    deployment caller; inspection of the candidate never creates authority.
    """
    require(type(authority) is PinnedAuthority, Code.IDENTITY,
            'structure bridge requires independently pinned authority')
    require(isinstance(root, (str, Path)), Code.INVALID,
            'structure bridge requires a trusted descriptor root')
    require(authority.provenance == 'deployment', Code.IDENTITY,
            'structure bridge requires deployment provenance')
    require(expected_library_id in ('elpis_retrieval_bridge', 'elpis_ingress_bridge'),
            Code.IDENTITY, 'unexpected structure bridge identifier')
    require(expected_library_id in authority.libraries, Code.IDENTITY,
            'structure bridge is not in deployment authority')
    return admit_library(root, path, authority, expected_library_id).lib
