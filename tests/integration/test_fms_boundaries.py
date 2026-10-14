"""Gate G: WARM materialization never reaches into the HOT lane's resident state (FMS boundaries).

The geometric (HOT) lane runs managed LEARN and QUERY on an FMS-resident K1 state. Concurrently the contextual (WARM)
lane resolves existing resources in place and folds them into its views, and a retrieval epoch runs searches that
never wait for residency. The HOT state's FMS context is not the retrieval epoch's: WARM work cannot promote, demote,
pin or wait on it. Afterwards the resident state holds no lease and no open transaction, stays WARM, recorded no
lease failure, and the external resources are byte-for-byte unchanged. A failed WARM materialization (an absent
resource) publishes nothing and the HOT lane never notices.

Cold-resource absence, corrupt cold replicas, exhausted residency budgets, stalled promotion (another holder's pin)
and concurrent lease use of the vector index are covered natively (ctest: structure.test_vector_fms, including
``case_nowait_never_waits``; structure.test_vector_concurrency; structure.test_vector_adversarial). Mechanics only;
the fixture map is TEST_ONLY TRAINING=NONE SEMANTICS=NONE.
"""
from __future__ import annotations

import os
import threading

from elpis.runtime import Runtime
from elpis.runtime.cognition import LearnRequest, QueryRequest
from elpis.runtime.context import ContextLane, ContextLimits, ResolveObservation
from elpis.structure.retrieval.hacf import build_corpus_and_index, hybrid_retrieve
from elpis.structure.retrieval.objects import CorpusManifest, documents_in_place

from ._turn_fixtures import LEARN, ByteTokens, FixtureMap, admitted
from .test_codec_ecs_turn import _config, _Resident, adapter, k1  # noqa: F401 (module fixtures)
from .test_hacf_in_place import BINDINGS, LIMITS, PROVENANCE, _claims, _tree, library, workstation  # noqa: F401

TRANSITIONS = 100


def test_warm_context_work_never_promotes_pins_or_waits_on_the_hot_resident_state(
        k1, adapter, library, workstation, tmp_path):   # noqa: F811
    learn = LearnRequest("hot", ByteTokens(), admitted(FixtureMap()), LEARN)
    query = QueryRequest("query", ByteTokens(), admitted(FixtureMap()))
    resources = _tree(workstation / "workspace")   # the workstation fixture shares tmp_path with the runtime
    sources = documents_in_place(workstation, BINDINGS, **LIMITS)
    errors, retrievals = [], [0]
    stop = threading.Event()

    def warm(lane, epoch, manifest):
        """WARM work: retrieval (never waiting for residency) and in-place resolution folded into views."""
        try:
            while not stop.is_set():
                claims = _claims(epoch, manifest, "pacman repositories cmake durable")
                retrievals[0] += 1
                lane.submit(ResolveObservation(workstation, manifest, claims, PROVENANCE,
                                               source_paths=sources.source_paths))
                missing = dict(sources.source_paths)
                missing[next(iter(missing))] = "workspace/absent.md"        # a failed materialization
                lane.submit(ResolveObservation(workstation, manifest, claims, "aa" * 32, source_paths=missing))
        except Exception as exc:   # pragma: no cover (asserted below)
            errors.append(exc)

    with _Resident(k1, adapter, tmp_path, max_rows=8) as state, ContextLane(ContextLimits(capacity=8)) as lane, \
            build_corpus_and_index(library, tmp_path / "epoch", list(sources.documents)) as epoch, \
            Runtime(_config(tmp_path / "c")) as runtime:
        manifest = CorpusManifest.verified(epoch.corpus_manifest_json, expected_digest=epoch.corpus_digest)
        runtime.anchor_cognition(state)
        resident = state._r.inspect(state.id)
        durable = runtime.continuity.snapshot()
        thread = threading.Thread(target=warm, args=(lane, epoch, manifest))
        thread.start()
        try:
            for n in range(TRANSITIONS):
                result = runtime.run_learn(state, learn, context=lane)
                assert result.generation_after == resident["generation"] + n + 1, n
                if n % 10 == 0:
                    runtime.run_query(state, query, context=lane)
        finally:
            stop.set()
            thread.join()
        after = state._r.inspect(state.id)
        status = lane.status()
        assert runtime.continuity.snapshot() == durable              # volatile: nothing published
    assert not errors, errors
    assert retrievals[0] > 0 and status.published > 0 and status.failed > 0 and status.failed_batches == 0
    # The resident HOT state: no lease or transaction left, still WARM, never a lease failure.
    assert (after["lease_count"], after["transaction_open"], after["lease_failures"]) == (0, 0, 0), after
    assert after["tier"] == resident["tier"] == 1                     # FMS_WARM
    assert after["generation"] == resident["generation"] + TRANSITIONS
    assert after["acquisitions"] > resident["acquisitions"]
    # External resources: not created, copied, moved or rewritten.
    assert _tree(workstation / "workspace") == resources
    assert not (tmp_path / "epoch").exists() and not os.path.exists(workstation / "workspace" / "absent.md")
