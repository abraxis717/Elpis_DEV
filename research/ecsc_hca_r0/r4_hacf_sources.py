"""R4 DEV RESEARCH_ONLY: HACF-verified source observations for R3's exact WARM cache.

Only the canonical HACF resolver opens resources. No source bytes are retained
by this bridge. It validates document/chunk identity against a caller-pinned
manifest and requires separately granted expected normalized-text digests.
It does NOT verify semantic admission, the authenticity of an operator's
manifest pin, or provenance/claim truth. Never call it on the HOT path.
"""
from dataclasses import dataclass

from elpis.runtime.context import ContextLimits, ContextReference, ContextView
from elpis.structure.retrieval.objects import (
    ChunkClaim, CorpusManifest, ObjectResolutionError, resolve_chunks,
)

from .baselines import Authority, Limits, Refusal, SourceCheck, digest, require
from .r3_warm_exact import ExactWarmContext
from .task import Task


@dataclass(frozen=True)
class HACFBinding:
    """Caller-authorized original path and chunk claim for one exact context reference."""

    scope: str
    reference: object
    claim: ChunkClaim
    source_path: str


def _bounded(value, *, low=1, high=1 << 24):
    require(type(value) is int and low <= value <= high, "SOURCE_BUDGET")


def verified_source_checks(root, manifest: CorpusManifest, view: ContextView,
                           authority: Authority, bindings: tuple[HACFBinding, ...],
                           *, limits: Limits = Limits(), max_document_bytes: int = 1 << 20,
                           max_total_document_bytes: int = 4 << 20) -> tuple[SourceCheck, ...]:
    """Produce checks only after complete, exact, bounded HACF byte verification.

    Fail closed on stale sources, missing coverage or a mismatched independent
    SourceGrant; no fallback to synthetic checks and no partial check results.
    A SourceCheck marks byte integrity ONLY, never semantic/admission authority.
    """
    require(type(manifest) is CorpusManifest and type(view) is ContextView
            and type(authority) is Authority and type(bindings) is tuple
            and type(limits) is Limits, "INVALID_INPUT")
    _bounded(max_document_bytes)
    _bounded(max_total_document_bytes)
    require(max_document_bytes <= max_total_document_bytes, "SOURCE_BUDGET")
    require(len(view.axioms) <= limits.axioms
            and len(view.relations) <= limits.relations
            and len(authority.axioms) <= limits.axioms
            and len(authority.sources) <= limits.axioms * limits.references
            and len(authority.relations) <= limits.relations
            and len(authority.schemas) <= limits.relations
            and len(bindings) <= limits.axioms * limits.references, "CAPACITY")
    try:
        view.verify(ContextLimits(max_axioms=limits.axioms,
                                  max_relations=limits.relations,
                                  max_arity=limits.arity,
                                  max_references=limits.references))
    except Exception as exc:
        raise Refusal("VIEW_INVALID") from exc
    require(digest(manifest.digest), "SOURCE_MANIFEST")

    grants = {}
    for grant in authority.axioms:
        require(grant.axiom.key not in grants, "DUPLICATE_GRANT")
        grants[grant.axiom.key] = grant
    source_grants = {}
    for grant in authority.sources:
        key = (grant.scope, grant.reference)
        require(key not in source_grants, "DUPLICATE_GRANT")
        source_grants[key] = grant

    expected = set()
    for axiom in view.axioms:
        grant = grants.get(axiom.key)
        require(grant is not None and grant.axiom == axiom, "AXIOM_AUTHORITY")
        require(len(axiom.references) <= limits.references, "REFERENCES_LIMIT")
        for ref in axiom.references:
            key = (grant.scope, ref)
            require(ref.corpus_digest == manifest.digest, "SOURCE_COORDINATE")
            require(key in source_grants, "SOURCE_AUTHORITY")
            expected.add(key)

    declared = {}
    claims = {}
    source_paths = {}
    for binding in bindings:
        require(type(binding) is HACFBinding and type(binding.claim) is ChunkClaim
                and type(binding.reference) is ContextReference, "INVALID_BINDING")
        key = (binding.scope, binding.reference)
        require(key in expected and key not in declared, "SOURCE_BINDING")
        ref = binding.reference
        require(binding.claim.chunk_digest == ref.chunk_digest
                and ref.corpus_digest == manifest.digest, "SOURCE_COORDINATE")
        # The provenance digest belongs to the supplied ContextReference; this
        # reader does not pretend to verify the semantic proof behind it.
        old = claims.get(binding.claim.chunk_digest)
        require(old is None or old == binding.claim, "SOURCE_BINDING")
        claims[binding.claim.chunk_digest] = binding.claim
        old_path = source_paths.get(binding.claim.doc_digest)
        require(old_path is None or old_path == binding.source_path, "SOURCE_BINDING")
        source_paths[binding.claim.doc_digest] = binding.source_path
        declared[key] = binding.claim.chunk_digest
    require(set(declared) == expected, "SOURCE_BINDING")
    if not expected:
        return ()

    # Cap aggregate WHOLE DOCUMENT bytes, not merely selected chunk text.
    try:
        sizes = {d: manifest.document(d).size_bytes for d in source_paths}
        require(all(s <= max_document_bytes for s in sizes.values())
                and sum(sizes.values()) <= max_total_document_bytes, "SOURCE_BUDGET")
        resolved, omitted = resolve_chunks(
            root, manifest, tuple(claims[k] for k in sorted(claims)),
            max_objects=len(claims), max_text_bytes=max_total_document_bytes,
            max_document_bytes=max_document_bytes, source_paths=source_paths,
        )
    except ObjectResolutionError as exc:
        # Do not reinterpret an INTEGRITY refusal as merely "unknown".
        raise Refusal("SOURCE_INTEGRITY") from exc
    require(not omitted and len(resolved) == len(claims), "SOURCE_INCOMPLETE")
    observed = {c.chunk_digest: c.norm_digest for c in resolved}
    result = []
    for scope, ref in sorted(expected, key=lambda k: (k[0], k[1].chunk_digest,
                                                     k[1].corpus_digest,
                                                     k[1].provenance_digest)):
        actual = observed.get(declared[scope, ref])
        require(actual is not None and source_grants[scope, ref].content_digest == actual,
                "SOURCE_ATTESTATION")
        result.append(SourceCheck(ref, scope, actual))
    return tuple(result)


def open_verified_hacf(root, manifest, view, authority, bindings, task: Task, *,
                       limits: Limits = Limits(), max_document_bytes=1 << 20,
                       max_total_document_bytes=4 << 20):
    """WARM-only explicit cold open; identity checks precede exact-state admission."""
    checks = verified_source_checks(root, manifest, view, authority, bindings, limits=limits,
                                    max_document_bytes=max_document_bytes,
                                    max_total_document_bytes=max_total_document_bytes)
    return ExactWarmContext.open(view, authority, checks, task, limits=limits)


def advance_verified_hacf(state: ExactWarmContext, root, manifest, view, authority,
                          bindings, task: Task, *, max_document_bytes=1 << 20,
                          max_total_document_bytes=4 << 20):
    """WARM-only successor. A failed source verification leaves old state intact."""
    require(type(state) is ExactWarmContext, "INVALID_INPUT")
    limits = state.verified.graph.limits
    checks = verified_source_checks(root, manifest, view, authority, bindings, limits=limits,
                                    max_document_bytes=max_document_bytes,
                                    max_total_document_bytes=max_total_document_bytes)
    return state.advance(view, authority, checks, task, limits=limits)
