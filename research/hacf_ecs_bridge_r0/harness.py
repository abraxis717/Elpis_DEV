"""Evidence-condition harness for the HACF -> ECS research bridge. RESEARCH_ONLY. NO_CLAIM.

Runs one tiny, synthetic retrieval bundle through the bridge under five evidence conditions and records, per
condition, whether the bridge refused (and with which code) or observed (and what the K1 state answered), together
with the proof that the state did not move. It has no thresholds and makes no claim: it is the instrument a
future HACF-ECS Scientific Bridge R0 qualification would run on frozen, real evidence
(not yet run; no claim).

Conditions:

* ``VALID``: the bundle the protocol expects.
* ``WITHHELD``: the same bundle with its evidence withheld (no items): the bridge abstains (``NO_EVIDENCE``).
* ``PERMUTED``: the items reordered without re-ranking: the canonical bundle gate refuses (``RANK_ORDER``).
  ``PERMUTED_RERANKED``: reordered and consistently re-ranked: well-formed, so observed; whether the answer moves
  is recorded, not judged.
* ``STALE``: a bundle from another corpus epoch: refused (``STALE_EVIDENCE``).
* ``MISLEADING``: a bundle bound to another query: refused by the gate (``QUERY_DIGEST_MISMATCH``).
  ``MISLEADING_WELL_FORMED``: the right bindings over other documents (consistent digests): observed. Structural
  validation cannot detect semantically misleading evidence; the harness records that, it does not hide it.

The fixture map (:class:`StructuralFixtureMap`) is TEST_ONLY: TRAINING=NONE SEMANTICS=NONE.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import struct

from elpis.structure.retrieval.contracts import RetrievalBundle, RetrievalItem

from .bridge import (
    BridgeRefusal,
    ObservationContext,
    ObservationMap,
    StructuralObservationPacket,
    map_identity,
    observe,
    query,
)

CLASSIFICATION = "RESEARCH_ONLY HARNESS NO_CLAIM"
ZERO = "0" * 64
GRAPH_SOURCE = 0x04


def h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class StructuralFixtureMap(ObservationMap):
    """TEST_ONLY TRAINING=NONE SEMANTICS=NONE: one row per primary item from its structural fields only."""

    map_id = "test-structural-fixture"
    dim = 6
    max_rows = 8

    def parameter_bytes(self) -> bytes:
        return struct.pack("<3d", 0.5, 0.25, 0.125)

    def rows(self, packet):
        children = {}
        for item in packet.items:
            if item.graph_hop == 1:
                children[item.graph_parent_digest] = children.get(item.graph_parent_digest, 0) + 1
        out = []
        for item in packet.items:
            if item.graph_hop == 0 and len(out) < self.max_rows:
                out.append([1.0 / (1 + item.final_rank), 0.5 * (item.lexical_rank > 0),
                            0.25 * (item.dense_rank > 0), 0.125 * children.get(item.chunk_digest, 0),
                            int(item.doc_digest[:2], 16) / 255.0, float(len(packet.items)) / 16.0])
        return out


def _item(rank, label, *, doc, kind=1, hop=0, parent=ZERO, edge=0, source=0x03):
    text = f"fixture chunk {label} of {doc}"
    data = text.encode()
    return RetrievalItem(chunk_digest=h("chunk-" + label), doc_digest=h("doc-" + doc), namespace="fixture",
                         authority="TEST_ONLY", graph_parent_digest=parent, text_digest=hashlib.sha256(data).hexdigest(),
                         fusion_score_key=100 - rank, dense_score_key=50 - rank, lexical_rank=rank + 1,
                         dense_rank=rank + 1 if rank % 2 == 0 else 0, final_rank=rank, source_mask=source,
                         item_kind=kind, graph_hop=hop, edge_type=edge, edge_authority=1 if edge else 0, text=text,
                         text_bytes=len(data))


CONTEXT = ObservationContext(query_digest=h("query"), corpus_manifest_digest=h("corpus"),
                             graph_snapshot_digest=h("graph"), corpus_epoch=3, vector_index_epoch=3)


def bundle(context=CONTEXT, docs=("alpha", "beta", "gamma")) -> RetrievalBundle:
    """A tiny well-formed bundle: three primaries and one one-hop context item."""
    items = [_item(0, "a", doc=docs[0]), _item(1, "b", doc=docs[1]), _item(2, "c", doc=docs[2])]
    items.append(_item(3, "a-ctx", doc=docs[0], kind=2, hop=1, parent=items[0].chunk_digest, edge=1,
                       source=GRAPH_SOURCE))
    return RetrievalBundle(query_digest=context.query_digest, corpus_manifest_digest=context.corpus_manifest_digest,
                           vector_index_manifest_digest=h("vectors"), graph_snapshot_digest=context.graph_snapshot_digest,
                           fusion_policy_digest=h("fusion"), bundle_digest=h("claimed bundle"),
                           hacf_package_digest=h("hacf"), corpus_epoch=context.corpus_epoch,
                           vector_index_epoch=context.vector_index_epoch, items=tuple(items))


def _rerank(items):
    return tuple(replace(item, final_rank=rank) for rank, item in enumerate(items))


def conditions() -> dict[str, RetrievalBundle]:
    valid = bundle()
    swapped = (valid.items[1], valid.items[0], valid.items[2], valid.items[3])
    return {
        "VALID": valid,
        "WITHHELD": replace(valid, items=()),
        "PERMUTED": replace(valid, items=swapped),
        "PERMUTED_RERANKED": replace(valid, items=_rerank(swapped)),
        "STALE": replace(valid, corpus_epoch=valid.corpus_epoch - 1),
        "MISLEADING": replace(valid, query_digest=h("another query")),
        "MISLEADING_WELL_FORMED": bundle(docs=("delta", "epsilon", "zeta")),
    }


def run(state, observation_map=None, pin=None, context=CONTEXT) -> dict[str, dict]:
    """Every condition through the bridge against one K1 state. Returns a record per condition; claims nothing."""
    observation_map = observation_map if observation_map is not None else StructuralFixtureMap()
    pin = pin if pin is not None else map_identity(observation_map)
    out = {}
    for name, evidence in conditions().items():
        before = (state.state_digest(), state.epoch)
        record = {"condition": name, "classification": CLASSIFICATION}
        try:
            packet = StructuralObservationPacket.from_bundle(evidence, context)
            observation = observe(packet, observation_map, pin)
            answer = query(state, observation)
            record.update(outcome="OBSERVED", packet=packet.digest, observation=observation.digest,
                          answer=answer.digest, values=answer.values)
        except BridgeRefusal as exc:
            record.update(outcome="REFUSED", code=exc.code, detail=str(exc))
        record["state_unchanged"] = (state.state_digest(), state.epoch) == before
        out[name] = record
    return out
