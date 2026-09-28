"""Structural address proposals from ingress exports and retrieval bundles.

* :func:`from_regex_hacf` consumes the zero-authority context-proposal export
  of the pipeline ingress (``elpis.regex-hacf-context-proposal.r1``) as bytes,
  pinned by its raw digest and bound to the expected source and corpus.
* :func:`from_retrieval_bundle` consumes a structure ``RetrievalBundle``,
  re-validated and pinned by its canonical export digest.

The upstream proposal digest is kept as provenance. Proposals route to banks
and expert families by explicit rules; confidence never grants authority.
"""
from dataclasses import dataclass
import json
import math

from elpis.structure.retrieval.budget import RetrievalBudget
from elpis.structure.retrieval.errors import RetrievalError
from elpis.structure.retrieval.validation import validate_bundle
from elpis.substrate.digests import raw_digest

from .contracts import Code, ContractError, ProposalOnly, digest_value, identity, require


@dataclass(frozen=True)
class RouteRule:
    key: str
    banks: tuple[str,...]=()
    expert_families: tuple[str,...]=()


@dataclass(frozen=True)
class AddressProposal(ProposalOnly):
    source: str
    regex_result: str
    corpus: str
    context_snapshot: str
    query_overlay: str
    route_key: str
    banks: tuple[str,...]
    objects: tuple[str,...]
    expert_families: tuple[str,...]
    confidence: float | None
    provenance: tuple[str,...]

    def __post_init__(self):
        for d in (self.source,self.regex_result,self.corpus,self.context_snapshot,self.query_overlay): digest_value(d)
        require(self.confidence is None or
                (type(self.confidence) in (int,float) and type(self.confidence) is not bool and
                 math.isfinite(self.confidence) and 0<=self.confidence<=1))
        require(bool(self.route_key))
        for values in (self.banks,self.objects,self.expert_families,self.provenance):
            require(type(values) is tuple and all(type(x) is str and bool(x) for x in values))

    @property
    def digest(self): return identity('structural-address',self)


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
