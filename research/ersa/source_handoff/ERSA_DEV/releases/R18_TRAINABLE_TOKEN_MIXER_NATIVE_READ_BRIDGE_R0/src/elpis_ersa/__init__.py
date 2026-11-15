"""ELPIS ERS (ERSA) R0 — bounded relational selective attention reference.

Schema elpis.ersa.session.r0 R0, derived from (not replacing) the frozen
elpis.gdn.session.r0 R0 shared-state contract.  Reference model only:
REFERENCE_NOT_NATIVE_GDN, REFERENCE_NOT_YM_AUTHORITY.
"""
from .contract_ersa import *  # noqa: F401
from . import contract_ersa as C
from . import geometry_ersa as G
from . import bank_ersa as B

SCHEMA = C.SCHEMA
REVISION = C.REVISION
