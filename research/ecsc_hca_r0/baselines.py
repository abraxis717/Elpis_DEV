"""DEV-only exact lookup and deterministic factor propagation; no admission authority."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from hashlib import sha256
import json
from types import MappingProxyType

from elpis.runtime.context import (
    BINDING, ContextAxiom, ContextLimits, ContextReference, ContextRelation, ContextView,
)
from elpis.runtime.errors import CompositionError

SOURCE_PIN = "cb530e7a0c290b9ceeb6614ab1db4ab45ada54e3"


class Refusal(Exception):
    """A mechanical refusal; used is completed logical work, never scientific NONPASS."""
    def __init__(self, code, used=0):
        self.code, self.used = code, used
        super().__init__(code)


def require(condition, code):
    if not condition:
        raise Refusal(code)


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def label(value):
    return type(value) is str and 0 < len(value) <= 64 and value.isascii()


def digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class Limits:
    axioms: int = 32
    relations: int = 32
    arity: int = 8
    degree: int = 8
    references: int = 4
    rounds: int = 8
    work: int = 100_000
    sidecar_bytes: int = 262_144
    state_bytes: int = 65_536
    updates_per_tick: int = 256

    def __post_init__(self):
        for name, low, high in (("axioms", 1, 256), ("relations", 0, 512), ("arity", 2, 16),
                                ("degree", 1, 32), ("references", 1, 16), ("rounds", 1, 32),
                                ("work", 1, 10_000_000), ("sidecar_bytes", 1, 4_194_304),
                                ("state_bytes", 1, 1_048_576), ("updates_per_tick", 1, 256)):
            require(integer(getattr(self, name), low, high), "INVALID_LIMIT")


class Meter:
    def __init__(self, cap):
        self.cap, self.used, self.counts = cap, 0, {}

    def tick(self, name, count=1):
        if self.used + count > self.cap:
            raise Refusal("WORK_LIMIT", self.used)
        self.used += count
        self.counts[name] = self.counts.get(name, 0) + count

    def result(self):
        return tuple(sorted(self.counts.items()))


def json_tokens(value):
    """Bounded scalar chunks; no asdict copy or assembled JSON string/bytes."""
    if is_dataclass(value):
        value = {f.name: getattr(value, f.name) for f in fields(value)}
    if type(value) is dict:
        yield "{"
        for i, key in enumerate(sorted(value)):
            if i:
                yield ","
            yield json.dumps(key)
            yield ":"
            yield from json_tokens(value[key])
        yield "}"
    elif type(value) is tuple:
        yield "["
        for i, item in enumerate(value):
            if i:
                yield ","
            yield from json_tokens(item)
        yield "]"
    else:
        yield json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def bounded_digest(value, limits, meter, domain):
    # Validate the complete encoding length AND reserve logical hashing work
    # before encoding a second time. Scratch is one bounded scalar, not a sidecar.
    size = 0
    for token in json_tokens(value):
        size += len(token)
        require(size <= limits.sidecar_bytes, "SIDECAR_LIMIT")
        meter.tick("sidecar_preflight_byte", len(token))
    meter.tick("sidecar_hash_byte", size)
    hasher = sha256(domain)
    for token in json_tokens(value):
        hasher.update(token.encode("ascii"))
    return hasher.hexdigest()


@dataclass(frozen=True)
class AxiomGrant:
    """Exact synthetic authority entry, including binding status and source coordinates."""
    axiom: ContextAxiom
    scope: str
    kind: str

    def __post_init__(self):
        require(type(self.axiom) is ContextAxiom and label(self.scope) and label(self.kind), "INVALID_GRANT")


@dataclass(frozen=True)
class SourceGrant:
    reference: ContextReference
    scope: str
    content_digest: str  # synthetic source bytes; NOT the HACF chunk identity algorithm

    def __post_init__(self):
        require(type(self.reference) is ContextReference and label(self.scope)
                and digest(self.content_digest), "INVALID_SOURCE")


@dataclass(frozen=True)
class SourceCheck:
    reference: ContextReference
    scope: str
    observed_digest: str | None  # None = missing; mismatch = stale/altered

    def __post_init__(self):
        require(type(self.reference) is ContextReference and label(self.scope)
                and (self.observed_digest is None or digest(self.observed_digest)), "INVALID_SOURCE")


@dataclass(frozen=True)
class Schema:
    relation_type: int
    roles: tuple[str, ...]
    kinds: tuple[str, ...]
    weight: int = 1000

    def __post_init__(self):
        require(integer(self.relation_type, 1, 2**32 - 1) and type(self.roles) is tuple
                and type(self.kinds) is tuple and 2 <= len(self.roles) <= 16
                and len(self.roles) == len(self.kinds) and all(label(x) for x in self.roles + self.kinds)
                and len(set(self.roles)) == len(self.roles) and integer(self.weight, 1, 1000), "INVALID_SCHEMA")
        # Ordered roles[:-1] are jointly necessary inputs; roles[-1] is the output.


@dataclass(frozen=True)
class Authority:
    """Caller-supplied TEST_ONLY allowlists, not a production proof verifier.

    A relation entry pins its complete members, type, provenance and status. Merely
    setting ADMITTED on an input relation cannot bypass this separate allowlist.
    """
    axioms: tuple[AxiomGrant, ...]
    sources: tuple[SourceGrant, ...]
    relations: tuple[ContextRelation, ...]
    schemas: tuple[Schema, ...]
    epoch: str = "TEST_ONLY_R1"


@dataclass(frozen=True)
class AxiomHit:
    grant: AxiomGrant
    source_valid: bool
    source_state: str = "UNKNOWN"


@dataclass(frozen=True)
class RelationHit:
    relation: ContextRelation
    scope: str
    roles: tuple[str, ...]
    source_valid: bool
    source_state: str = "UNKNOWN"


@dataclass(frozen=True)
class Result:
    coordinate: object
    evidence_digest: str
    axioms: tuple[AxiomHit, ...]
    relations: tuple[RelationHit, ...]
    operations: tuple[tuple[str, int], ...]
    scores: tuple[tuple[str, int], ...] = ()
    factors: tuple[tuple[tuple, int], ...] = ()

    @property
    def work(self):
        return sum(n for _, n in self.operations)


@dataclass(frozen=True)
class Baselines:
    """One immutable research snapshot. Construct successors off to the side.

    No queue, history, filesystem, worker, persistent index or production publication.
    Caller may replace its own reference only after build succeeds.
    """
    view: ContextView
    limits: Limits
    axioms: object = field(repr=False)
    relations: object = field(repr=False)
    schemas: object = field(repr=False)
    incidence: object = field(repr=False)
    evidence_digest: str
    build_operations: tuple
    authority: Authority = field(repr=False)
    policy_digest: str

    @classmethod
    def build(cls, view, authority, checks, limits=Limits(), *, previous=None, budget=None):
        require(type(limits) is Limits, "INVALID_LIMIT")
        require(previous is None or type(previous) is cls, "INVALID_INPUT")
        cap = limits.work if budget is None else budget
        require(integer(cap, 1, limits.work), "INVALID_BUDGET")
        meter = Meter(cap)
        try:
            return cls._build(view, authority, checks, limits, previous, meter)
        except Refusal as exc:
            exc.used = meter.used
            raise

    @classmethod
    def _build(cls, view, authority, checks, limits, previous, meter):
        require(type(view) is ContextView and type(authority) is Authority and type(limits) is Limits,
                "INVALID_INPUT")
        require(label(authority.epoch), "INVALID_AUTHORITY_EPOCH")
        arrays = ((view.axioms, limits.axioms, ContextAxiom),
                  (view.relations, limits.relations, ContextRelation),
                  (authority.axioms, limits.axioms, AxiomGrant),
                  (authority.sources, limits.axioms * limits.references, SourceGrant),
                  (authority.relations, limits.relations, ContextRelation),
                  (authority.schemas, limits.relations, Schema),
                  (checks, limits.axioms * limits.references, SourceCheck))
        for values, cap, kind in arrays:
            meter.tick("collection_check")
            require(type(values) is tuple, "INVALID_INPUT")
            require(len(values) <= cap, "CAPACITY")
            for value in values:
                meter.tick("record_check")
                require(type(value) is kind, "INVALID_INPUT")
        # Reject oversized nested values before canonical verification allocates sets/hashes.
        for a in (*view.axioms, *(g.axiom for g in authority.axioms)):
            meter.tick("axiom_check")
            require(len(a.references) <= limits.references, "REFERENCES_LIMIT")
        for r in (*view.relations, *authority.relations):
            meter.tick("relation_check")
            require(len(r.members) <= limits.arity, "ARITY_LIMIT")
        for schema in authority.schemas:
            meter.tick("schema_check")
            require(len(schema.roles) <= limits.arity, "ARITY_LIMIT")
        # Logical verification units, not CPU instructions: one record, reference or member.
        meter.tick("view_verify", 1 + len(view.axioms) + len(view.relations)
                   + sum(len(a.references) for a in view.axioms)
                   + sum(len(r.members) for r in view.relations))
        try:
            view.verify(ContextLimits(max_axioms=limits.axioms, max_relations=limits.relations,
                                      max_arity=limits.arity, max_references=limits.references))
        except (CompositionError, ValueError, OverflowError, TypeError) as exc:
            raise Refusal("VIEW_INVALID") from exc

        def index(values, key):
            out = {}
            for value in values:
                meter.tick("index_entry")
                k = key(value)
                require(k not in out, "DUPLICATE")
                out[k] = value
            return out

        grants = index(authority.axioms, lambda g: g.axiom.key)
        sources = index(authority.sources, lambda s: (s.scope, s.reference))
        observed = index(checks, lambda s: (s.scope, s.reference))
        allowed = index(authority.relations, lambda r: r.key)
        schemas = index(authority.schemas, lambda s: s.relation_type)
        for key in observed:
            meter.tick("source_authority")
            require(key in sources, "SOURCE_AUTHORITY")
        axioms, relations, incidence = {}, {}, {}
        for a in view.axioms:
            meter.tick("axiom_authority")
            g = grants.get(a.key)
            require(g is not None and g.axiom == a, "AXIOM_AUTHORITY")
            states = []
            for ref in a.references:
                meter.tick("source_check")
                k = (g.scope, ref)
                require(k in sources, "SOURCE_AUTHORITY")
                check = observed.get(k)
                states.append("UNKNOWN" if check is None or check.observed_digest is None else
                              "VALID" if check.observed_digest == sources[k].content_digest else "STALE")
            state = ("STALE" if "STALE" in states else "UNKNOWN" if "UNKNOWN" in states else
                     "VALID" if states else "AUTHORITY_ONLY")
            axioms[a.key] = AxiomHit(g, state in ("VALID", "AUTHORITY_ONLY"), state)
            incidence[a.key] = []
        for r in view.relations:
            meter.tick("relation_authority")
            require(allowed.get(r.key) == r, "RELATION_AUTHORITY")
            schema = schemas.get(r.relation_type)
            require(schema is not None and len(schema.roles) == len(r.members), "ROLE_SCHEMA")
            scope = axioms[r.members[0]].grant.scope
            valid = True
            for position, member in enumerate(r.members):
                meter.tick("incidence_build")
                hit = axioms[member]
                require(hit.grant.scope == scope, "SCOPE_MISMATCH")
                require(hit.grant.kind == schema.kinds[position], "ROLE_MISMATCH")
                require(len(incidence[member]) < limits.degree, "DEGREE_LIMIT")
                incidence[member].append((r.key, position))
                valid = valid and hit.source_valid
            states = tuple(axioms[m].source_state for m in r.members)
            state = ("STALE" if "STALE" in states else "UNKNOWN" if "UNKNOWN" in states else
                     "AUTHORITY_ONLY" if "AUTHORITY_ONLY" in states else "VALID")
            relations[r.key] = RelationHit(r, scope, schema.roles, valid, state)
        if previous is not None:
            meter.tick("predecessor_check")
            require(view.predecessor == previous.view.digest and view.version == previous.view.version + 1
                    and view.through >= previous.view.through and view.dropped >= previous.view.dropped,
                    "PREDECESSOR")
            for key, old in previous.axioms.items():
                meter.tick("binding_check")
                if old.grant.axiom.status == BINDING:
                    require(key in axioms and axioms[key].grant == old.grant, "BINDING_RETENTION")
            meter.tick("policy_continuity", len(authority.axioms) + len(authority.sources)
                       + len(authority.relations) + len(authority.schemas) + 1)
            require(authority == previous.authority and limits == previous.limits,
                    "AUTHORITY_CONTINUITY")
        # Separate sidecar coordinate: source invalidation or a role/authority change
        # must not masquerade as the same evidence just because ContextView is unchanged.
        meter.tick("sidecar_encode", 1 + len(authority.axioms) + len(authority.sources)
                   + len(authority.relations) + len(authority.schemas) + len(checks)
                   + sum(len(g.axiom.references) for g in authority.axioms)
                   + sum(len(r.members) for r in authority.relations)
                   + sum(len(s.roles) for s in authority.schemas))
        policy_digest = bounded_digest({"authority": authority, "limits": limits}, limits, meter,
                                       b"ecsc-hca-r1.TEST_ONLY.policy\0")
        evidence_digest = bounded_digest({"view": view.digest, "policy": policy_digest, "checks": checks},
                                         limits, meter, b"ecsc-hca-r1.TEST_ONLY.evidence\0")
        return cls(view, limits, MappingProxyType(axioms), MappingProxyType(relations),
                   MappingProxyType(schemas), MappingProxyType({k: tuple(v) for k, v in incidence.items()}),
                   evidence_digest, meter.result(), authority, policy_digest)

    def _meter(self, budget):
        cap = self.limits.work if budget is None else budget
        require(integer(cap, 1, self.limits.work), "INVALID_BUDGET")
        return Meter(cap)

    def _read(self, scope, keys, relation_keys, meter):
        require(label(scope) and type(keys) is tuple and type(relation_keys) is tuple, "INVALID_QUERY")
        require(len(keys) <= self.limits.axioms and len(relation_keys) <= self.limits.relations, "CAPACITY")
        selected, selected_relations = {}, {}
        for key in keys:
            meter.tick("axiom_lookup")
            require(digest(key), "INVALID_QUERY")
            hit = self.axioms.get(key)
            require(hit is not None, "NOT_FOUND")
            require(hit.grant.scope == scope, "SCOPE_MISMATCH")
            selected[key] = hit
        for key in relation_keys:
            meter.tick("relation_lookup")
            require(type(key) is tuple and len(key) == 2 and integer(key[0], 1, 2**32 - 1)
                    and type(key[1]) is tuple and 2 <= len(key[1]) <= self.limits.arity
                    and all(digest(k) for k in key[1]), "INVALID_QUERY")
            hit = self.relations.get(key)
            require(hit is not None, "NOT_FOUND")
            require(hit.scope == scope, "SCOPE_MISMATCH")
            selected_relations[key] = hit
            for member in hit.relation.members:
                meter.tick("member_read")
                selected[member] = self.axioms[member]
        # Every readout retains all binding constraints in the requested scope.
        for key, hit in self.axioms.items():
            meter.tick("binding_scan")
            if hit.grant.scope == scope and hit.grant.axiom.status == BINDING:
                selected[key] = hit
        meter.tick("readout", len(selected) + len(selected_relations))
        return tuple(selected[k] for k in sorted(selected)), tuple(selected_relations[k] for k in sorted(selected_relations))

    def b0(self, scope, keys=(), relation_keys=(), *, budget=None):
        meter = self._meter(budget)
        try:
            axioms, relations = self._read(scope, keys, relation_keys, meter)
            return Result(self.view.coordinate, self.evidence_digest, axioms, relations, meter.result())
        except Refusal as exc:
            exc.used = meter.used
            raise

    def b1(self, scope, seeds, *, rounds=1, budget=None):
        """Synchronous integer max/min propagation, a fixed non-learned B1 rule.

        Factor message = weight * min(all input-role scores) // 1000, if every
        member's source is valid. Only its output role receives that message.
        Candidate activity is salience ONLY, including when chained into another
        factor. Factor activation is NOT proof or admission/constraint truth.
        """
        meter = self._meter(budget)
        try:
            return self._propagate(scope, seeds, rounds, meter)
        except Refusal as exc:
            exc.used = meter.used
            raise

    def _propagate(self, scope, seeds, rounds, meter):
        require(label(scope) and type(seeds) is tuple, "INVALID_QUERY")
        require(len(seeds) <= self.limits.axioms, "CAPACITY")
        require(integer(rounds, 1, self.limits.rounds), "ROUND_LIMIT")
        scores, factors = {}, {}
        for key, hit in self.axioms.items():
            meter.tick("cell_init")
            if hit.grant.scope == scope:
                scores[key] = 0
        for key, hit in self.relations.items():
            meter.tick("factor_init")
            if hit.scope == scope:
                factors[key] = 0
        for key in seeds:
            meter.tick("seed_read")
            require(digest(key), "INVALID_QUERY")
            require(key in self.axioms, "NOT_FOUND")
            require(key in scores, "SCOPE_MISMATCH")
            require(self.axioms[key].source_valid, "SOURCE_INVALID")
            scores[key] = 1000
        for _ in range(rounds):
            meter.tick("cell_copy", len(scores))
            next_scores = dict(scores)
            for key in factors:
                meter.tick("factor_update")
                hit = self.relations[key]
                schema = self.schemas[hit.relation.relation_type]
                inputs = []
                for position, member in enumerate(hit.relation.members):
                    meter.tick("incidence_visit")
                    if position < len(schema.roles) - 1:
                        inputs.append(scores[member])
                message = schema.weight * min(inputs) // 1000 if hit.source_valid else 0
                factors[key] = message
                output = hit.relation.members[-1]
                meter.tick("cell_update")
                next_scores[output] = max(next_scores[output], message)
            scores = next_scores
        axioms, relations = self._read(scope, tuple(scores), tuple(factors), meter)
        meter.tick("score_readout", len(scores) + len(factors))
        return Result(self.view.coordinate, self.evidence_digest, axioms, relations, meter.result(),
                      tuple(sorted(scores.items())), tuple(sorted(factors.items())))
