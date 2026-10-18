"""The acknowledgment law at the Python boundary (docs/RUNTIME_CORE.md, "Recoverable and identity-only publication").

RuntimeCore resolves a recoverable request ``ACKNOWLEDGED`` (always ``recoverable``) and an identity-only request
``IDENTITY_PUBLISHED`` (never ``recoverable``); no other status is ever recoverable. The adapter re-checks that law on
every ticket it reads (``RuntimeCore._ticket``), so a library that broke it could never hand a caller an
identity-only publication, an uncertain outcome or a forgotten ticket dressed as a recoverable acknowledgment.
Every forged combination below is refused ``RUNTIME_INVALID``; every lawful one is passed through unchanged. The
lawful outcomes themselves are produced by real publications in ``test_k1_recovery.py``, ``test_query_learn.py`` and
the Rust law suite (including ``UNKNOWN`` for a ticket resolved beyond the lane's kept window).
"""
from __future__ import annotations

import pytest

from elpis.runtime import Runtime
from elpis.runtime.composition import CompositionError
from elpis.runtime.core import _Ticket

from .test_codec_ecs_turn import _config

STATUS = {"NONE": 0, "PENDING": 1, "ACKNOWLEDGED": 2, "FAILED": 3, "UNCERTAIN": 4, "SUPERSEDED": 5, "ABANDONED": 6,
          "UNKNOWN": 7, "IDENTITY_PUBLISHED": 8}
KIND = {None: 0, "RECOVERABLE": 1, "IDENTITY": 2}

LAWFUL = [("ACKNOWLEDGED", 1, "RECOVERABLE"), ("IDENTITY_PUBLISHED", 0, "IDENTITY"), ("NONE", 0, None),
          ("UNKNOWN", 0, None)] + [(s, 0, k) for s in ("PENDING", "FAILED", "UNCERTAIN", "SUPERSEDED", "ABANDONED")
                                   for k in ("RECOVERABLE", "IDENTITY")]
FORGED = [
    ("ACKNOWLEDGED", 0, "RECOVERABLE"),          # an acknowledgment that restart could not resume
    ("ACKNOWLEDGED", 1, "IDENTITY"),             # an identity-only publication dressed as a recoverable one
    ("ACKNOWLEDGED", 1, None),                   # an acknowledgment of no request
    ("IDENTITY_PUBLISHED", 1, "IDENTITY"),       # identity-only claiming the bytes are recoverable
    ("IDENTITY_PUBLISHED", 0, "RECOVERABLE"),    # a recoverable request silently downgraded to identity-only
] + [(s, 1, k) for s in ("PENDING", "FAILED", "UNCERTAIN", "SUPERSEDED", "ABANDONED", "UNKNOWN")
     for k in ("RECOVERABLE", "IDENTITY")]       # recoverable without an acknowledgment


def _forged(status, recoverable, kind):
    t = _Ticket()
    t.sequence, t.generation, t.status, t.recoverable, t.kind = 7, 3, STATUS[status], recoverable, KIND[kind]
    t.identity[:] = bytes(range(32)) if status in ("ACKNOWLEDGED", "IDENTITY_PUBLISHED") else bytes(32)
    return t


@pytest.fixture
def core(tmp_path):
    with Runtime(_config(tmp_path / "c")) as runtime:
        yield runtime._core


@pytest.mark.parametrize("status,recoverable,kind", LAWFUL)
def test_a_lawful_ticket_passes_the_boundary_unchanged(core, status, recoverable, kind):
    ticket = core._ticket(_forged(status, recoverable, kind))
    assert (ticket.sequence, ticket.generation, ticket.status, ticket.recoverable, ticket.kind) == \
        (7, 3, status, bool(recoverable), kind)
    assert ticket.acknowledged == (status == "ACKNOWLEDGED") == ticket.recoverable


@pytest.mark.parametrize("status,recoverable,kind", FORGED)
def test_a_ticket_that_breaks_the_acknowledgment_law_is_refused(core, status, recoverable, kind):
    with pytest.raises(CompositionError) as refused:
        core._ticket(_forged(status, recoverable, kind))
    assert refused.value.code == "RUNTIME_INVALID"
