"""Tiny seeded TEST_ONLY fixtures. No HACF admission or scientific authority."""
from dataclasses import dataclass, replace
from hashlib import sha256

from elpis.runtime.context import (
    ADMITTED, BINDING, CANDIDATE, EMPTY_VIEW, ContextAxiom, ContextLimits,
    ContextReference, ContextRelation, Observation, successor,
)
from .baselines import AxiomGrant, Authority, Schema, SourceCheck, SourceGrant

SEED = 1729


def h(text):
    return sha256(text.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Fixture:
    view: object
    authority: Authority
    checks: tuple
    keys: tuple  # (scope, kind, key); bounded descriptive labels only

    def key(self, scope, kind):
        return next(k for s, t, k in self.keys if (s, t) == (scope, kind))


def coding(seed=SEED, *, arity=4):
    """Two contradictory scopes and a conjunction of arity-1 input roles.

    Arch/dependency/no-system-mutation occur at separate synthetic turn coordinates.
    Source bytes are generated transiently; only digests remain. Admission entries
    are fixture allowlists, explicitly NOT native SemanticAdmission objects.
    """
    if type(seed) is not int or not 0 <= seed < 2**32 or not 4 <= arity <= 8:
        raise ValueError("bounded fixture seed/arity required")
    kinds = ("environment", "dependency", "policy") + tuple(f"extra-{n}" for n in range(arity - 4)) + ("plan",)
    axioms, sources, checks, relations, keys, observations = [], [], [], [], [], []
    for scope, environment in (("project-arch", "Arch Linux"), ("project-debian", "Debian")):
        members = []
        for turn, kind in enumerate(kinds):
            text = {"environment": environment, "dependency": "requires package-X",
                    "policy": "never mutate system-managed packages", "plan": "proposed plan"}.get(kind, kind)
            coordinate = f"TEST_ONLY:{seed}:{scope}:turn-{turn * 8}:{kind}"
            ref = ContextReference(h(coordinate), h(f"manifest:{seed}:{scope}:v1"), h(f"proposal:{coordinate}"))
            axiom = ContextAxiom(h(f"axiom:{coordinate}"), (ref,), BINDING if kind == "policy" else "PROVISIONAL")
            axioms.append(AxiomGrant(axiom, scope, kind))
            sources.append(SourceGrant(ref, scope, h(text)))
            checks.append(SourceCheck(ref, scope, h(text)))
            keys.append((scope, kind, axiom.key))
            members.append(axiom.key)
            observations.append(Observation(axiom=axiom, required=axiom.status == BINDING))
        for relation_type, status in ((71, CANDIDATE), (72, ADMITTED)):
            relation = ContextRelation(relation_type, tuple(members), h(f"TEST_ONLY:{scope}:{status}:{seed}"), status)
            relations.append(relation)
            observations.append(Observation(relation=relation))
    view = successor(EMPTY_VIEW, tuple(observations), ContextLimits(), through=len(observations), dropped=0)
    schemas = tuple(Schema(t, kinds, kinds, weight) for t, weight in ((71, 750), (72, 500)))
    return Fixture(view, Authority(tuple(axioms), tuple(sources), tuple(relations), schemas), tuple(checks), tuple(keys))


def delayed(seed=SEED, *, turns=8, drop_every=0):
    """Streaming fixed-width distractor replacement; no transcript or growing registry.

    Yields current snapshots only. Consumer must not collect them into a history.
    Optional dropped distractors are counted in the canonical view coordinate.
    """
    if turns not in (8, 32, 128, 512) or type(drop_every) is not int or not 0 <= drop_every <= 512:
        raise ValueError("bounded DEV horizons/dropout required")
    fixture = coding(seed)
    base = fixture
    for turn in range(1, turns + 1):
        key = h(f"distractor-slot:{seed}")
        ref = ContextReference(h(f"distractor:{seed}:{turn}"), h("noise-corpus"), h("noise-proposal"))
        axiom = ContextAxiom(key, (ref,))
        scope = "noise"
        dropped = bool(drop_every and turn % drop_every == 0)
        observations = () if dropped else (Observation(axiom=axiom),)
        view = successor(fixture.view, observations, ContextLimits(), through=fixture.view.through + 1,
                         dropped=fixture.view.dropped + int(dropped))
        if dropped:
            fixture = replace(fixture, view=view)
        else:
            grant = AxiomGrant(axiom, scope, "distractor")
            source = SourceGrant(ref, scope, h(f"noise:{turn}"))
            fixture = Fixture(view, replace(base.authority, axioms=base.authority.axioms + (grant,),
                                            sources=base.authority.sources + (source,)),
                              base.checks + (SourceCheck(ref, scope, source.content_digest),), base.keys)
        yield fixture
