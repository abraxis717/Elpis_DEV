import json
import pytest
from elpis.inference.structural import RouteRule
from elpis.runtime.edges import from_regex_hacf
from elpis.substrate.file_assets import raw_digest
from elpis.inference.contracts import ContractError


def proposal_fixture():
    # Exact exported schema from build_context_proposal in the native ingress.
    return dict(schema='elpis.regex-hacf-context-proposal.r1',semantic_authority=False,
                admission_authority=False,execution_authority=False,runtime_admission=False,
                candidate_status='PROPOSED_UNADMITTED',source_sha256='1'*64,proposal_digest='2'*64,
                hacf=dict(corpus_manifest_digest='3'*64,context_graph_manifest_digest='4'*64,
                          retrieval=[dict(pattern_id='python.function',evidence_id='5'*64,
                                          hacf_primary_hits=[dict(chunk_digest='6'*64)])]))


def adapt(obj,**overrides):
    payload=json.dumps(obj,sort_keys=True).encode()
    kwargs=dict(expected_payload=raw_digest(payload),expected_source='1'*64,expected_corpus='3'*64,
                context_snapshot='7'*64,query_overlay='8'*64,
                rules=(RouteRule('python.function',('python-memory',),('code-experts',)),))
    kwargs.update(overrides)
    return from_regex_hacf(payload,**kwargs)


def test_existing_ingress_schema_consumption():
    result=adapt(proposal_fixture())
    p=result[0]
    assert p.banks==('python-memory',) and p.objects==('6'*64,)
    assert p.expert_families==('code-experts',)
    assert not p.semantic_authority and not p.execution_authority and not p.runtime_admission
    assert p.digest==adapt(proposal_fixture())[0].digest
    assert not hasattr(p,'row_ids')


def test_structural_authority_and_identity_defects():
    for field in ('semantic_authority','admission_authority','execution_authority','runtime_admission'):
        bad=proposal_fixture(); bad[field]=True
        with pytest.raises(ContractError): adapt(bad)
    for key in ('expected_payload','expected_source','expected_corpus'):
        with pytest.raises(ContractError): adapt(proposal_fixture(),**{key:'0'*64})
    with pytest.raises(TypeError):
        RouteRule('x',row_ids=(1,))


def test_structural_ingress_rejects_duplicate_keys_and_missing_digests():
    raw=(
        '{"schema":"elpis.regex-hacf-context-proposal.r1",'
        '"semantic_authority":true,"semantic_authority":false,'
        '"admission_authority":false,"execution_authority":false,'
        '"runtime_admission":false,"candidate_status":"PROPOSED_UNADMITTED",'
        '"source_sha256":"' + '1'*64 + '","proposal_digest":"' + '2'*64 + '",'
        '"hacf":{"corpus_manifest_digest":"' + '3'*64 + '",'
        '"context_graph_manifest_digest":"' + '4'*64 + '","retrieval":[]}}'
    ).encode()
    with pytest.raises(ContractError,match='INVALID:ingress JSON'):
        from_regex_hacf(
            raw,expected_payload=raw_digest(raw),expected_source='1'*64,
            expected_corpus='3'*64,context_snapshot='7'*64,query_overlay='8'*64,
            rules=()
        )
    for field in ('proposal_digest','context_graph_manifest_digest'):
        bad=proposal_fixture()
        if field=='proposal_digest':
            del bad[field]
        else:
            del bad['hacf'][field]
        with pytest.raises(ContractError):
            adapt(bad)


def test_structural_unknown_confidence_is_absent_not_fabricated():
    assert adapt(proposal_fixture())[0].confidence is None
