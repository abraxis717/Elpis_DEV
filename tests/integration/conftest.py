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
from ..structure.native_bridge_fixture import pin_bridge

DOCS = [
    ("spec", "Intervals whose touching endpoints may merge keep the maximum end.", "elpis.docs", "canonical"),
    ("note", "Keep at least one merged range for every overlap.", "elpis.docs", "reference"),
]
POSITIVE = b"touching endpoints may merge; maximum end."
CONTRADICTION = b"touching endpoints do not merge; touching endpoints may merge; maximum end."


@pytest.fixture(scope="module")
def retrieval_library(tmp_path_factory):
    path, root, authority = pin_bridge(require_native_library("elpis_retrieval_bridge"),
                                       tmp_path_factory.mktemp("integration-retrieval-lib"),
                                       "elpis_retrieval_bridge")
    return RetrievalLibrary(path, root=root, authority=authority)


@pytest.fixture(scope="module")
def ingress_library(tmp_path_factory):
    path, root, authority = pin_bridge(require_native_library("elpis_ingress_bridge"),
                                       tmp_path_factory.mktemp("integration-ingress-lib"),
                                       "elpis_ingress_bridge")
    return IngressLibrary(path, root=root, authority=authority)


@pytest.fixture(scope="module")
def corpus(retrieval_library, tmp_path_factory):
    """Structural memory: one HACF corpus and vector index, owned by this module."""
    state = tmp_path_factory.mktemp("structural-memory")
    handle = build_corpus_and_index(retrieval_library, state, DOCS)
    assert not (state / "corpus").exists(), "HACF must not persist a corpus"
    assert not (state / "cold").exists(), "HACF must not persist cold blobs"
    yield handle, None
    handle.destroy()


@pytest.fixture
def ingress(ingress_library, corpus):
    with QueryIngress(ingress_library, corpus[0]) as handle:
        yield handle


@pytest.fixture
def runtime(tmp_path_factory, runtime_library):
    from ._turn_fixtures import TEST_CODEC_PIN
    config = RuntimeConfig(tmp_path_factory.mktemp("runtime") / "continuity", runtime_library, TEST_CODEC_PIN)
    with Runtime(config) as rt:
        yield rt
