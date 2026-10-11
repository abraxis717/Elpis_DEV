"""Independent admission authority for ECS codec maps: a codec may describe itself; it may not authorize itself.

An ECS codec map (the missing ECS<->DSV semantic codec, docs/ELPIS_MISSION.md) carries a ``classification`` string.
That string is reporting metadata written by the codec's own author. It is never the credential. A cognitive
operation runs a codec map only through an :class:`AdmittedCodec`, which binds the map to an entry of a
:class:`CodecAuthority`:

* the authority is a catalog document whose SHA-256 must equal an **independent pin** (the same pattern as the
  deployment-pinned native authority, ``elpis.substrate.authority.PinnedAuthority``). The managed runtime takes the
  pin from its deployment configuration (``RuntimeConfig.codec_authority_sha256``), never from the candidate;
* the entry names the codec's implementation (``module:qualname``) and two digests that *admission measures itself*
  from the candidate: the SHA-256 of the implementation's defining source file and the SHA-256 of the map's
  ``parameter_bytes()``. The candidate supplies bytes; it never supplies the digest that authorizes them;
* the entry grants **capabilities**, separately: ``QUERY_ENCODE`` and ``QUERY_DECODE`` (read-only QUERY),
  ``LEARN_ENCODE`` (LEARN), ``LEARN_DECODE`` (the legacy learned turn, which decodes a LEARN readout). A codec
  admitted only for QUERY never gains LEARN;
* the entry fixes the classification results report. A map that declares a different one (a forged
  ``QUALIFIED``, say) is refused.

Every operation re-verifies the admission before any ECS call: the authority's document still hashes to its pin,
the entry is still ``ADMITTED``, the map's class, its operation functions (no instance or class patching since
admission), its parameter bytes and its declared classification are the admitted ones, and the operation's
capabilities are granted. Any failure refuses before the ECS state is touched.

No ECS<->DSV semantic codec is qualified. ``QUALIFIED`` requires a byte-pinned qualification record listed in
:data:`QUALIFIED_CODEC_RECORDS`, which is empty: no catalog can admit a codec as qualified, and production text
cognition stays unavailable. Test fixtures are admitted only as ``TEST_ONLY`` with ``TRAINING=NONE
SEMANTICS=NONE`` in their classification.

Non-claims: the pins authenticate exact bytes, not a writer, and are no protection against a malicious in-process
caller (who could construct a catalog and its pin). The implementation digest covers the defining module's source
file only, not its imports, the interpreter or the tokenizer. Admission says nothing about what a codec means.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import inspect
import json
from types import MappingProxyType

from .errors import CompositionError

__all__ = ("CODEC_AUTHORITY_SCHEMA", "LEARN", "LEGACY_LEARNED_TURN_CAPABILITIES", "QUALIFIED_CODEC_RECORDS",
           "QUERY", "AdmittedCodec", "CodecAuthority", "CodecCapability", "CodecEntry", "CodecQualification",
           "admit_codec", "implementation_identity", "parameters_digest")

CODEC_AUTHORITY_SCHEMA = "elpis.ecs-codec-authority.v1"


class CodecCapability(str, Enum):
    QUERY_ENCODE = "QUERY_ENCODE"     # tokens -> query rows
    QUERY_DECODE = "QUERY_DECODE"     # f_W(x) -> tokens
    LEARN_ENCODE = "LEARN_ENCODE"     # tokens -> experience schedule
    LEARN_DECODE = "LEARN_DECODE"     # legacy learned turn only: S3 of the LEARN candidate -> tokens


class CodecQualification(str, Enum):
    TEST_ONLY = "TEST_ONLY"           # interface fixtures: TRAINING=NONE SEMANTICS=NONE
    RESEARCH_ONLY = "RESEARCH_ONLY"   # research candidates: no qualification claim
    QUALIFIED = "QUALIFIED"           # requires a byte-pinned qualification record (none exists)


QUERY = frozenset({CodecCapability.QUERY_ENCODE, CodecCapability.QUERY_DECODE})
LEARN = frozenset({CodecCapability.LEARN_ENCODE})
LEGACY_LEARNED_TURN_CAPABILITIES = frozenset({CodecCapability.LEARN_ENCODE, CodecCapability.LEARN_DECODE})

# The codec method each capability needs. Admission requires them on the class, and nowhere else.
_METHODS = {CodecCapability.QUERY_ENCODE: "encode_query", CodecCapability.QUERY_DECODE: "decode_query",
            CodecCapability.LEARN_ENCODE: "encode", CodecCapability.LEARN_DECODE: "decode"}
_BOUND = ("encode", "decode", "encode_query", "decode_query", "parameter_bytes")

# Qualification records that may admit an ECS<->DSV codec as QUALIFIED: record path -> pinned SHA-256. Empty: no
# ECS<->DSV semantic codec is qualified. Adding one is a reviewed authority change with its evidence.
QUALIFIED_CODEC_RECORDS = MappingProxyType({})


def _refuse(code, detail):
    raise CompositionError(code, detail)


def _digest(value, what):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        _refuse("CODEC_AUTHORITY", what + ": 64 lowercase hex characters")
    return value


def implementation_identity(cls) -> tuple[str, str]:
    """``(module:qualname, SHA-256 of the defining source file)`` of a codec map class, measured here."""
    try:
        path = inspect.getsourcefile(cls)
    except TypeError:
        path = None
    if not path:
        _refuse("CODEC_IDENTITY", "the codec implementation has no source file to measure")
    with open(path, "rb") as handle:
        source = handle.read()
    return f"{cls.__module__}:{cls.__qualname__}", hashlib.sha256(source).hexdigest()


def parameters_digest(codec_map) -> str:
    """SHA-256 of the codec map's ``parameter_bytes()``, measured here (the map supplies bytes, not a digest)."""
    fn = getattr(type(codec_map), "parameter_bytes", None)
    if not callable(fn):
        _refuse("CODEC_IDENTITY", "the codec map exposes no parameter_bytes()")
    data = fn(codec_map)
    if type(data) is not bytes:
        _refuse("CODEC_IDENTITY", "parameter_bytes() must return bytes")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class CodecEntry:
    codec_id: str
    implementation: str
    implementation_sha256: str
    parameters_sha256: str
    capabilities: frozenset
    classification: str
    qualification: CodecQualification
    qualification_record: str | None
    status: str


_ENTRY_FIELDS = {"codec_id", "implementation", "implementation_sha256", "parameters_sha256", "capabilities",
                 "classification", "qualification", "qualification_record", "status"}


def _entry(data, provenance) -> CodecEntry:
    if type(data) is not dict or set(data) != _ENTRY_FIELDS:
        _refuse("CODEC_AUTHORITY", "codec entry fields")
    for name in ("codec_id", "implementation", "classification"):
        if type(data[name]) is not str or not data[name].strip():
            _refuse("CODEC_AUTHORITY", name)
    if data["implementation"].count(":") != 1:
        _refuse("CODEC_AUTHORITY", "implementation: module:qualname")
    caps = data["capabilities"]
    if type(caps) is not list or not caps or len(set(caps)) != len(caps):
        _refuse("CODEC_AUTHORITY", "capabilities: a non-empty list of distinct names")
    try:
        capabilities = frozenset(CodecCapability(c) for c in caps)
        qualification = CodecQualification(data["qualification"])
    except ValueError:
        _refuse("CODEC_AUTHORITY", "unknown capability or qualification")
    if data["status"] not in ("ADMITTED", "REVOKED"):
        _refuse("CODEC_AUTHORITY", "status: ADMITTED or REVOKED")
    record, classification = data["qualification_record"], data["classification"]
    if qualification is CodecQualification.QUALIFIED:
        # No codec qualifies by declaring it: only a byte-pinned record the repository admits (none exists).
        if record not in QUALIFIED_CODEC_RECORDS:
            _refuse("CODEC_UNQUALIFIED", "no admitted qualification record: ECS codec mapping not yet qualified")
    elif record is not None:
        _refuse("CODEC_AUTHORITY", "only a QUALIFIED entry names a qualification record")
    if qualification is CodecQualification.TEST_ONLY and not all(
            mark in classification for mark in ("TEST_ONLY", "TRAINING=NONE", "SEMANTICS=NONE")):
        _refuse("CODEC_AUTHORITY", "a TEST_ONLY codec is classified TEST_ONLY TRAINING=NONE SEMANTICS=NONE")
    if qualification is CodecQualification.RESEARCH_ONLY and "RESEARCH_ONLY" not in classification:
        _refuse("CODEC_AUTHORITY", "a RESEARCH_ONLY codec is classified RESEARCH_ONLY")
    if qualification is not CodecQualification.QUALIFIED and "QUALIFIED" in classification.replace("UNQUALIFIED", ""):
        _refuse("CODEC_AUTHORITY", "an unqualified codec may not be classified QUALIFIED")
    if provenance == "test-fixture" and qualification is not CodecQualification.TEST_ONLY:
        _refuse("CODEC_AUTHORITY", "a test-fixture authority admits TEST_ONLY codecs only")
    return CodecEntry(data["codec_id"], data["implementation"],
                      _digest(data["implementation_sha256"], "implementation"),
                      _digest(data["parameters_sha256"], "parameters"), capabilities, classification, qualification,
                      record, data["status"])


class CodecAuthority:
    """An independently pinned codec admission catalog (``elpis.ecs-codec-authority.v1``).

    ``expected_sha256`` must come from trusted configuration, never from the candidate codec. The document is
    kept, and every use re-checks that it still hashes to the pin.
    """

    __slots__ = ("_document", "digest", "source", "provenance", "codecs")

    def __init__(self, document: bytes, *, expected_sha256: str):
        _digest(expected_sha256, "authority pin")
        if type(document) is not bytes or hashlib.sha256(document).hexdigest() != expected_sha256:
            _refuse("CODEC_AUTHORITY", "codec authority document does not match its independent pin")

        def unique(pairs):
            out = {}
            for key, value in pairs:
                if key in out:
                    _refuse("CODEC_AUTHORITY", "duplicate catalog key")
                out[key] = value
            return out

        try:
            data = json.loads(document, object_pairs_hook=unique)
        except (json.JSONDecodeError, UnicodeError):
            _refuse("CODEC_AUTHORITY", "malformed codec authority")
        if type(data) is not dict or set(data) != {"schema", "source", "provenance", "codecs"}:
            _refuse("CODEC_AUTHORITY", "codec authority fields")
        if data["schema"] != CODEC_AUTHORITY_SCHEMA:
            _refuse("CODEC_AUTHORITY", "codec authority schema")
        if data["provenance"] not in ("deployment", "test-fixture"):
            _refuse("CODEC_AUTHORITY", "codec authority provenance")
        if type(data["source"]) is not str or not data["source"].strip() or type(data["codecs"]) is not list:
            _refuse("CODEC_AUTHORITY", "codec authority source and codecs")
        entries = tuple(_entry(e, data["provenance"]) for e in data["codecs"])
        if len({e.codec_id for e in entries}) != len(entries):
            _refuse("CODEC_AUTHORITY", "duplicate codec identifier")
        for name, value in (("_document", document), ("digest", expected_sha256), ("source", data["source"]),
                            ("provenance", data["provenance"]),
                            ("codecs", MappingProxyType({e.codec_id: e for e in entries}))):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("CodecAuthority is immutable")

    def intact(self) -> bool:
        return hashlib.sha256(self._document).hexdigest() == self.digest


_ADMISSION = object()   # module-private: only admit_codec constructs an AdmittedCodec


class AdmittedCodec:
    """A codec map bound to one entry of an independent :class:`CodecAuthority`. Constructed by
    :func:`admit_codec` only; every operation re-verifies it with :meth:`require`."""

    __slots__ = ("codec_map", "authority", "entry", "_functions")

    def __init__(self, token, codec_map, authority, entry, functions):
        if token is not _ADMISSION:
            _refuse("CODEC_UNADMITTED", "an AdmittedCodec is issued by admit_codec only")
        for name, value in (("codec_map", codec_map), ("authority", authority), ("entry", entry),
                            ("_functions", functions)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("AdmittedCodec is immutable")

    @property
    def classification(self) -> str:
        """The classification the authority admitted (what results report), never the map's own claim."""
        return self.entry.classification

    @property
    def capabilities(self) -> frozenset:
        return self.entry.capabilities

    def require(self, needed: frozenset, deployment_pin: str | None = None):
        """Re-verify the admission for one operation needing ``needed``; returns the codec map. Raises a
        ``CODEC_*`` refusal (before any ECS call) on any discrepancy. ``deployment_pin``: the managed runtime's
        configured authority pin (``""`` when none is configured: refused); ``None`` for unmanaged use."""
        authority, entry, codec = self.authority, self.entry, self.codec_map
        if type(authority) is not CodecAuthority or not authority.intact():
            _refuse("CODEC_AUTHORITY", "the codec authority no longer matches its pin")
        if deployment_pin == "":
            _refuse("CODEC_AUTHORITY", "no deployment-pinned codec authority is configured")
        if deployment_pin is not None and authority.digest != deployment_pin:
            _refuse("CODEC_AUTHORITY", "the codec authority is not the deployment-pinned authority")
        if authority.codecs.get(entry.codec_id) is not entry:
            _refuse("CODEC_UNKNOWN", "the admitted entry is not in its authority")
        if entry.status != "ADMITTED":
            _refuse("CODEC_REVOKED", f"codec {entry.codec_id} is {entry.status}")
        cls = type(codec)
        if f"{cls.__module__}:{cls.__qualname__}" != entry.implementation:
            _refuse("CODEC_IDENTITY", "the codec map is not the admitted implementation")
        _no_instance_overrides(codec)
        if tuple(inspect.getattr_static(cls, n, None) for n in _BOUND) != self._functions:
            _refuse("CODEC_IDENTITY", "the codec implementation changed since admission")
        if parameters_digest(codec) != entry.parameters_sha256:
            _refuse("CODEC_IDENTITY", "the codec parameters are not the admitted bytes")
        if getattr(codec, "classification", None) != entry.classification:
            _refuse("CODEC_CLASSIFICATION", "the codec's declared classification is not the admitted one")
        if type(needed) is not frozenset or not needed or not needed <= entry.capabilities:
            missing = sorted(c.value for c in (needed - entry.capabilities)) if type(needed) is frozenset else needed
            _refuse("CODEC_CAPABILITY", f"codec {entry.codec_id} is not admitted for {missing}")
        return codec


def _no_instance_overrides(codec):
    try:
        own = vars(codec)
    except TypeError:
        return
    overridden = [n for n in _BOUND if n in own]
    if overridden:
        _refuse("CODEC_IDENTITY", f"instance-level override of {overridden}")


def admit_codec(codec_map, authority: CodecAuthority, codec_id: str) -> AdmittedCodec:
    """Admit ``codec_map`` as entry ``codec_id`` of ``authority`` (cold path: measures its implementation source and
    parameters). Refuses an unknown or revoked entry, a different implementation, implementation source or
    parameters, a declared classification other than the admitted one, and a capability the map cannot serve."""
    if type(authority) is not CodecAuthority or not authority.intact():
        _refuse("CODEC_AUTHORITY", "an intact, independently pinned CodecAuthority is required")
    if codec_map is None:
        _refuse("ECS_CODEC_UNQUALIFIED", "ECS codec mapping not yet qualified; text generation unavailable")
    entry = authority.codecs.get(codec_id) if type(codec_id) is str else None
    if entry is None:
        _refuse("CODEC_UNKNOWN", f"codec {codec_id!r} is not in the codec authority")
    if entry.status != "ADMITTED":
        _refuse("CODEC_REVOKED", f"codec {codec_id} is {entry.status}")
    implementation, source_sha = implementation_identity(type(codec_map))
    if implementation != entry.implementation:
        _refuse("CODEC_IDENTITY", "the codec map is not the admitted implementation")
    if source_sha != entry.implementation_sha256:
        _refuse("CODEC_IDENTITY", "the codec implementation source is not the admitted bytes")
    _no_instance_overrides(codec_map)
    for capability in entry.capabilities:
        if not callable(inspect.getattr_static(type(codec_map), _METHODS[capability], None)):
            _refuse("CODEC_IDENTITY", f"the codec map cannot serve {capability.value}")
    functions = tuple(inspect.getattr_static(type(codec_map), n, None) for n in _BOUND)
    admitted = AdmittedCodec(_ADMISSION, codec_map, authority, entry, functions)
    admitted.require(entry.capabilities)
    return admitted
