"""Edge adapters: slow-lane outputs become inference edge contracts here.

Inference accepts only :class:`~elpis.inference.structural.AddressProposal`
values; it does not know that HACF, the Regex ingress or the retrieval stage
exist. The runtime composition owns the adapters that turn their outputs into
proposals, with every check the proposal relies on:

* :func:`from_regex_hacf` consumes the zero-authority context-proposal export
  of the pipeline ingress (``elpis.regex-hacf-context-proposal.r1``) as bytes,
  pinned by its raw digest and bound to the expected source and corpus;
* :func:`from_retrieval_bundle` consumes a structure ``RetrievalBundle``,
  re-validated and pinned by its canonical export digest.

Identities keep their persisted ``elpis.inference.<kind>.r0`` domains
(``structural-rules``, ``r1-bundle-export``) and route key ``HACF_R1``; the
upstream proposal digest is kept as provenance. Adapters run before a
sequence begins, never inside one.
"""
import json

from elpis.inference.contracts import Code, ContractError, digest_value, require
from elpis.inference.structural import AddressProposal, RouteRule
from elpis.structure.retrieval.budget import RetrievalBudget
from elpis.structure.retrieval.errors import RetrievalError
from elpis.structure.retrieval.objects import ChunkClaim, ObjectResolutionError
from elpis.structure.retrieval.validation import validate_bundle
from elpis.substrate.digests import identity, raw_digest

__all__ = ("from_regex_hacf", "from_retrieval_bundle", "object_claims")


def _strict_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: '+str(key))
        result[key]=value
    return result


def _reject_constant(value):
    raise ValueError('non-finite JSON constant: '+value)


def from_regex_hacf(payload,*,expected_payload,expected_source,expected_corpus,
                    context_snapshot,query_overlay,rules):
    require(type(payload) is bytes and len(payload)<=4<<20,detail='bounded ingress export')
    require(raw_digest(payload)==expected_payload,Code.IDENTITY,'ingress transport digest')
    try:
        result=json.loads(payload,object_pairs_hook=_strict_object,parse_constant=_reject_constant)
    except (ValueError,UnicodeError) as exc:
        raise ContractError(Code.INVALID,'ingress JSON') from exc
    require(type(result) is dict and result.get('schema')=='elpis.regex-hacf-context-proposal.r1',detail='ingress schema')
    for authority in ('semantic_authority','admission_authority','execution_authority','runtime_admission'):
        require(result.get(authority) is False,Code.IDENTITY,'ingress authority widening')
    require(result.get('candidate_status')=='PROPOSED_UNADMITTED',Code.IDENTITY,'ingress status')
    require(result.get('source_sha256')==expected_source,Code.IDENTITY,'ingress source')
    proposal_digest=result.get('proposal_digest')
    digest_value(proposal_digest)
    hacf=result.get('hacf')
    require(type(hacf) is dict and hacf.get('corpus_manifest_digest')==expected_corpus,Code.STALE,'HACF corpus')
    graph_digest=hacf.get('context_graph_manifest_digest')
    digest_value(graph_digest)
    retrieval=hacf.get('retrieval')
    require(type(retrieval) is list and len(retrieval)<=1024,detail='bounded retrieval proposals')
    require(type(rules) is tuple and all(type(r) is RouteRule for r in rules))
    rule_map={r.key:r for r in rules}
    require(len(rule_map)==len(rules),detail='duplicate structural rule')
    rule_identity=identity('structural-rules',rules)
    proposals=[]
    for row in retrieval:
        require(type(row) is dict and type(row.get('pattern_id')) is str,detail='structural route')
        key=row['pattern_id']; rule=rule_map.get(key,RouteRule(key))
        hits=row.get('hacf_primary_hits')
        require(type(hits) is list and len(hits)<=1024,detail='bounded HACF candidates')
        objects=[]
        for hit in hits:
            require(type(hit) is dict,detail='HACF candidate')
            objects.append(digest_value(hit.get('chunk_digest')))
        proposals.append(AddressProposal(expected_source,proposal_digest,expected_corpus,
                         context_snapshot,query_overlay,key,rule.banks,tuple(objects),rule.expert_families,
                         None,(expected_payload,rule_identity,graph_digest)))
    return tuple(proposals)


def from_retrieval_bundle(bundle, *, expected_query, expected_corpus, expected_bundle,
                          context_snapshot, query_overlay):
    """Adapt a validated structure RetrievalBundle into one address proposal.

    The bundle is re-validated against the expected query and corpus under the
    default retrieval budget, and its canonical export must equal the pinned
    digest. Retrieval failures surface as typed contract failures.
    """
    try:
        validate_bundle(bundle, expected_query, expected_corpus, RetrievalBudget())
    except RetrievalError as exc:
        raise ContractError(Code.IDENTITY, 'retrieval bundle validation') from exc
    require(identity('r1-bundle-export', bundle.to_canonical_dict()) == expected_bundle,
            Code.IDENTITY, 'retrieval bundle export pin')
    return AddressProposal(
        expected_query, bundle.bundle_digest, expected_corpus, context_snapshot, query_overlay,
        'HACF_R1', (), tuple(item.chunk_digest for item in bundle.items), (), None,
        (expected_bundle, bundle.graph_snapshot_digest, bundle.hacf_package_digest),
    )


def object_claims(payload, *, expected_payload):
    """Where each primary HACF hit of an ingress export says its chunk lives.

    Claims come from the pinned export bytes in export order (retrieval rows,
    then hits), the same order :func:`from_regex_hacf` gives proposal objects.
    They are unverified; :func:`elpis.structure.retrieval.objects.resolve_chunks`
    verifies each against the document bytes. Graph neighbours carry no
    location and are not claimed.
    """
    require(type(payload) is bytes and len(payload) <= 4 << 20, detail='bounded ingress export')
    require(raw_digest(payload) == expected_payload, Code.IDENTITY, 'ingress transport digest')
    try:
        result = json.loads(payload, object_pairs_hook=_strict_object, parse_constant=_reject_constant)
    except (ValueError, UnicodeError) as exc:
        raise ContractError(Code.INVALID, 'ingress JSON') from exc
    hacf = result.get('hacf') if type(result) is dict else None
    require(type(hacf) is dict and type(hacf.get('retrieval')) is list, detail='ingress schema')
    claims = []
    for row in hacf['retrieval']:
        require(type(row) is dict and type(row.get('hacf_primary_hits')) is list, detail='structural route')
        for hit in row['hacf_primary_hits']:
            require(type(hit) is dict, detail='HACF candidate')
            try:
                claims.append(ChunkClaim(hit.get('chunk_digest'), hit.get('doc_digest'), hit.get('ordinal'),
                                         hit.get('byte_start'), hit.get('byte_end')))
            except ObjectResolutionError as exc:
                raise ContractError(Code.INVALID, 'HACF candidate location') from exc
    return tuple(claims)
