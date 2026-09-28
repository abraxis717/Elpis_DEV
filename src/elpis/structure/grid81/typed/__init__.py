"""Typed projection of 81-cell transition rows, with D4 orbit identity.

Each source row is compiled into four typed views (transition, expansion
locus, quiescence, rationale) plus a source identity, each carrying its D4
orbit digest. :func:`compile_inventories` returns the five inventories in
memory; the structural-group stage consumes them.
"""

from elpis.structure.grid81.typed.canonical import canonicalize, domain_digest
from elpis.structure.grid81.typed.source_identity import SourceRowIdentityV1
from elpis.structure.grid81.typed.transition import T00TransitionViewV1
from elpis.structure.grid81.typed.expansion import T00ExpansionLocusViewV1
from elpis.structure.grid81.typed.quiescence import T00QuiescenceViewV1
from elpis.structure.grid81.typed.rationale import T00RationaleViewV1
from elpis.structure.grid81.typed.d4 import D4, D4_TRANSFORMS
from elpis.structure.grid81.typed.typed_orbits import (
    TransitionOrbitV1,
    ExpansionOrbitV1,
    QuiescenceOrbitV1,
    RationaleOrbitV1,
)
from elpis.structure.grid81.typed.compiler import compile_inventories, compile_row
from elpis.structure.grid81.typed.errors import (
    ElpisGridError,
    CanonicalizationError,
    SourceIdentityError,
    TransitionCompilerError,
    ExpansionCompilerError,
    QuiescenceCompilerError,
    RationaleCompilerError,
    D4Error,
    OrbitError,
)

__all__ = [
    "canonicalize",
    "domain_digest",
    "SourceRowIdentityV1",
    "T00TransitionViewV1",
    "T00ExpansionLocusViewV1",
    "T00QuiescenceViewV1",
    "T00RationaleViewV1",
    "D4",
    "D4_TRANSFORMS",
    "TransitionOrbitV1",
    "ExpansionOrbitV1",
    "QuiescenceOrbitV1",
    "RationaleOrbitV1",
    "compile_inventories",
    "compile_row",
    "ElpisGridError",
    "CanonicalizationError",
    "SourceIdentityError",
    "TransitionCompilerError",
    "ExpansionCompilerError",
    "QuiescenceCompilerError",
    "RationaleCompilerError",
    "D4Error",
    "OrbitError",
]
