"""Integration fixtures: real native libraries, a real HACF corpus and one runtime.

Every fixture is explicit: libraries are loaded from the build tree by path,
the corpus is built into a test-owned state root, and the runtime's continuity
lives in a test-owned directory. When ELPIS_REQUIRE_NATIVE=1 a missing
library is a failure, never a skip.
"""
from __future__ import annotations

import pytest

from elpis.pipeline.ingress import IngressLibrary, QueryIngress
from elpis.runtime import Runtime, RuntimeConfig
from elpis.structure.retrieval.hacf import RetrievalLibrary, build_corpus_and_index

from ..conftest import require_native_library

DOCS = [
    ("spec", "Intervals whose touching endpoints may merge keep the maximum end.", "elpis.docs", "canonical"),
    ("note", "Keep at least one merged range for every overlap.", "elpis.docs", "reference"),
]
POSITIVE = b"touching endpoints may merge; maximum end."
CONTRADICTION = b"touching endpoints do not merge; touching endpoints may merge; maximum end."


@pytest.fixture(scope="module")
def retrieval_library():
    return RetrievalLibrary(require_native_library("elpis_retrieval_bridge"))


@pytest.fixture(scope="module")
def ingress_library():
    return IngressLibrary(require_native_library("elpis_ingress_bridge"))


@pytest.fixture(scope="module")
def corpus(retrieval_library, tmp_path_factory):
    """Structural memory: one HACF corpus and vector index, owned by this module."""
    state = tmp_path_factory.mktemp("structural-memory")
    handle = build_corpus_and_index(retrieval_library, state, DOCS)
    yield handle, state / "corpus"
    handle.destroy()


@pytest.fixture
def ingress(ingress_library, corpus):
    with QueryIngress(ingress_library, corpus[1]) as handle:
        yield handle


@pytest.fixture
def runtime(tmp_path_factory, runtime_library):
    config = RuntimeConfig(tmp_path_factory.mktemp("runtime") / "continuity", runtime_library)
    with Runtime(config) as rt:
        yield rt
