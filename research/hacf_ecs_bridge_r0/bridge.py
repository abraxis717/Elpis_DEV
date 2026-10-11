"""The bridge: StructuralObservationPacket -> identified ObservationMap -> ECSObservation -> read-only QUERY.

RESEARCH_ONLY. NO_RUNTIME_AUTHORITY. UNQUALIFIED. No claim is made that any observation map carries meaning; the
only fixture map here is TEST_ONLY.

Laws (each a refusal with a stable code, or a property the tests assert):

* **Validated evidence only.** A packet is built from a :class:`~elpis.structure.retrieval.contracts.RetrievalBundle`
  only after the canonical bundle gate (``validate_bundle``) accepts it against the research protocol's frozen
  :class:`ObservationContext` (query, corpus manifest). The packet then also binds the context-graph snapshot and
  both epochs: anything else is ``STALE_EVIDENCE``. A bundle's self-declared ``bundle_digest`` is carried as a claim,
  never used as identity: the packet digest is recomputed from its structural fields.
* **Structure, not text.** The packet keeps structural fields only (identities, ranks, hops, edge types, sources);
  no chunk text crosses into ECS.
* **No evidence, no observation.** An empty packet is refused (``NO_EVIDENCE``): absence is never mapped to a default
  observation.
* **Identified maps.** An observation map is accepted only when its *measured* identity (``module:qualname``, SHA-256
  of its defining source file, SHA-256 of its ``parameter_bytes()``, dimension, row bound) equals the
  :class:`ObservationMapPin` the research protocol froze; a map never supplies the digest that identifies it
  (``MAP_UNIDENTIFIED``). Its rows must be finite and of its declared shape (``OBSERVATION_SHAPE``).
* **Read-only QUERY.** :func:`query` makes one ``query_identity`` call (the answer and the identity of the state
  that computed it) and proves the state's retained-state identity and epoch unchanged. There is no LEARN path, no
  continuity publication and no HACF writeback in this package (asserted structurally by its tests).
* **No laundering.** An :class:`ECSObservation` is a research value labelled ``RESEARCH_ONLY``; it is not a codec,
  carries no capability and is not accepted by the managed runtime (which requires an admitted codec), and the
  answer it produces is a research reading, never evidence promoted to authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math

from elpis.structure.retrieval.contracts import RetrievalBundle
from elpis.structure.retrieval.errors import BundleValidationError
from elpis.structure.retrieval.validation import validate_bundle

CLASSIFICATION = "RESEARCH_ONLY NO_RUNTIME_AUTHORITY UNQUALIFIED"
PACKET_SCHEMA = "elpis.research.hacf-ecs.structural-observation.v1"
MAP_SCHEMA = "elpis.research.hacf-ecs.observation-map.v1"
OBSERVATION_SCHEMA = "elpis.research.hacf-ecs.ecs-observation.v1"
ANSWER_SCHEMA = "elpis.research.hacf-ecs.query-answer.v1"
MAX_ROWS = 64


class BridgeRefusal(ValueError):
    """A refusal of the research bridge, with a stable code."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def _digest(domain: str, payload) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(domain.encode() + b"\0" + body).hexdigest()


@dataclass(frozen=True)
class ObservationContext:
    """What the research protocol froze about the evidence it expects (never derived from the evidence)."""

    query_digest: str
    corpus_manifest_digest: str
    graph_snapshot_digest: str
    corpus_epoch: int
    vector_index_epoch: int


@dataclass(frozen=True)
class StructuralItem:
    chunk_digest: str
    doc_digest: str
    text_digest: str
    item_kind: int
    final_rank: int
    graph_hop: int
    graph_parent_digest: str
    edge_type: int
    lexical_rank: int
    dense_rank: int
    source_mask: int


_ITEM_FIELDS = tuple(StructuralItem.__dataclass_fields__)


@dataclass(frozen=True)
class StructuralObservationPacket:
    """Validated HACF structural evidence, structure only. Built by :meth:`from_bundle` only."""

    query_digest: str
    corpus_manifest_digest: str
    graph_snapshot_digest: str
    corpus_epoch: int
    vector_index_epoch: int
    claimed_bundle_digest: str
    items: tuple[StructuralItem, ...]

    @classmethod
    def from_bundle(cls, bundle: RetrievalBundle, context: ObservationContext) -> "StructuralObservationPacket":
        if type(bundle) is not RetrievalBundle or type(context) is not ObservationContext:
            raise BridgeRefusal("INVALID", "a RetrievalBundle and the protocol's ObservationContext")
        try:
            validate_bundle(bundle, context.query_digest, context.corpus_manifest_digest)
        except BundleValidationError as exc:
            raise BridgeRefusal("EVIDENCE_REFUSED", f"{exc.code}: {exc}") from exc
        if (bundle.graph_snapshot_digest, bundle.corpus_epoch, bundle.vector_index_epoch) != (
                context.graph_snapshot_digest, context.corpus_epoch, context.vector_index_epoch):
            raise BridgeRefusal("STALE_EVIDENCE", "graph snapshot or epochs differ from the protocol's")
        items = tuple(StructuralItem(*(getattr(item, name) for name in _ITEM_FIELDS)) for item in bundle.items)
        return cls(bundle.query_digest, bundle.corpus_manifest_digest, bundle.graph_snapshot_digest,
                   bundle.corpus_epoch, bundle.vector_index_epoch, bundle.bundle_digest, items)

    @property
    def digest(self) -> str:
        return _digest(PACKET_SCHEMA, {
            "query_digest": self.query_digest, "corpus_manifest_digest": self.corpus_manifest_digest,
            "graph_snapshot_digest": self.graph_snapshot_digest, "corpus_epoch": self.corpus_epoch,
            "vector_index_epoch": self.vector_index_epoch,
            "items": [[getattr(item, name) for name in _ITEM_FIELDS] for item in self.items]})


class ObservationMap:
    """Base of a research observation map: ``rows(packet)`` -> rows of ``dim`` finite floats.

    A subclass declares ``map_id``, ``dim`` and ``max_rows`` and implements ``rows`` and ``parameter_bytes``. Its
    identity is measured by :func:`map_identity`; whatever it says about itself is not identity.
    """

    map_id: str = ""
    dim: int = 0
    max_rows: int = 0

    def parameter_bytes(self) -> bytes:   # pragma: no cover - abstract
        raise NotImplementedError

    def rows(self, packet: StructuralObservationPacket) -> list[list[float]]:   # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True)
class ObservationMapPin:
    """The identity the research protocol froze for one map."""

    map_id: str
    implementation: str
    implementation_sha256: str
    parameters_sha256: str
    dim: int
    max_rows: int

    @property
    def digest(self) -> str:
        return _digest(MAP_SCHEMA, self.__dict__)


def map_identity(observation_map) -> ObservationMapPin:
    """The measured identity of a map: its class's defining source file and its parameter bytes, hashed here."""
    cls = type(observation_map)
    if not isinstance(observation_map, ObservationMap):
        raise BridgeRefusal("MAP_UNIDENTIFIED", "an ObservationMap is required")
    try:
        path = inspect.getsourcefile(cls)
    except TypeError:
        path = None
    if not path:
        raise BridgeRefusal("MAP_UNIDENTIFIED", "the map has no source file to measure")
    with open(path, "rb") as handle:
        source = hashlib.sha256(handle.read()).hexdigest()
    parameters = cls.parameter_bytes(observation_map)
    if type(parameters) is not bytes:
        raise BridgeRefusal("MAP_UNIDENTIFIED", "parameter_bytes() must return bytes")
    dim, max_rows = observation_map.dim, observation_map.max_rows
    if type(dim) is not int or not 1 <= dim <= 64 or type(max_rows) is not int or not 1 <= max_rows <= MAX_ROWS:
        raise BridgeRefusal("MAP_UNIDENTIFIED", "dim in 1..64 and max_rows in 1..64")
    return ObservationMapPin(str(observation_map.map_id), f"{cls.__module__}:{cls.__qualname__}", source,
                             hashlib.sha256(parameters).hexdigest(), dim, max_rows)


@dataclass(frozen=True)
class ECSObservation:
    """An ECS observation of one packet through one identified map. RESEARCH_ONLY; carries no capability."""

    packet_digest: str
    map_digest: str
    dim: int
    rows: tuple[tuple[float, ...], ...]
    classification: str = CLASSIFICATION

    @property
    def digest(self) -> str:
        return _digest(OBSERVATION_SCHEMA, {
            "packet": self.packet_digest, "map": self.map_digest, "dim": self.dim,
            "rows": [[float.hex(v) for v in row] for row in self.rows], "classification": self.classification})


def observe(packet: StructuralObservationPacket, observation_map: ObservationMap,
            pin: ObservationMapPin) -> ECSObservation:
    """Map one packet through an identified map. Refuses an empty packet, an unidentified map, a bad shape."""
    if type(packet) is not StructuralObservationPacket or type(pin) is not ObservationMapPin:
        raise BridgeRefusal("INVALID", "a StructuralObservationPacket and the protocol's ObservationMapPin")
    if not packet.items:
        raise BridgeRefusal("NO_EVIDENCE", "an empty packet is never mapped to an observation")
    if map_identity(observation_map) != pin:
        raise BridgeRefusal("MAP_UNIDENTIFIED", "the map's measured identity is not the protocol's pin")
    produced = type(observation_map).rows(observation_map, packet)
    try:
        rows = tuple(tuple(float(v) for v in row) for row in produced)
    except (TypeError, ValueError) as exc:
        raise BridgeRefusal("OBSERVATION_SHAPE", "rows of numbers") from exc
    if not 1 <= len(rows) <= pin.max_rows or any(len(row) != pin.dim for row in rows):
        raise BridgeRefusal("OBSERVATION_SHAPE", f"1..{pin.max_rows} rows of {pin.dim} values")
    if not all(math.isfinite(v) for row in rows for v in row):
        raise BridgeRefusal("OBSERVATION_SHAPE", "non-finite value")
    return ECSObservation(packet.digest, pin.digest, pin.dim, rows)


@dataclass(frozen=True)
class ObservationAnswer:
    """A research reading: ``f_W(x)`` per observation row and the identity of the state that answered."""

    observation_digest: str
    state_digest: bytes
    values: tuple[float, ...]
    classification: str = CLASSIFICATION

    @property
    def digest(self) -> str:
        return _digest(ANSWER_SCHEMA, {"observation": self.observation_digest, "state": self.state_digest.hex(),
                                       "values": [float.hex(v) for v in self.values]})


def query(state, observation: ECSObservation) -> ObservationAnswer:
    """One read-only K1 QUERY with an observation; the state's identity and epoch are proven unchanged."""
    if type(observation) is not ECSObservation:
        raise BridgeRefusal("INVALID", "an ECSObservation is required")
    if state.dim != observation.dim:
        raise BridgeRefusal("OBSERVATION_SHAPE", "the observation's dimension is not the state's")
    before = (state.state_digest(), state.epoch)
    values, identity = state.query_identity([list(row) for row in observation.rows])
    if (state.state_digest(), state.epoch) != before or identity != before[0]:
        raise BridgeRefusal("STATE_MOVED", "the state is not the one that answered")   # never expected
    return ObservationAnswer(observation.digest, identity, tuple(values))
