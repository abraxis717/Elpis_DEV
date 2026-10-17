"""R12N: Elpis-native directional structure, without an external answer model.

Research-only: LoCoMo DEV turn order admits a mechanically checkable record-order
relation. It is NOT a semantic fact, a neural language parser, or a native HACF
chunker. No DEV gold or reserved QUAL file is opened.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

from elpis.runtime.context import (
    ADMITTED, CANDIDATE, PROVISIONAL, ContextAxiom, ContextLimits,
    ContextReference, ContextRelation, EMPTY_VIEW, Observation, successor,
)
from elpis.structure.retrieval.objects import ChunkClaim, chunk_identity, normalize
from .baselines import Authority, AxiomGrant, Baselines, Limits, Refusal, Schema, SourceGrant
from .r4_hacf_sources import HACFBinding, verified_source_checks
from .r9_external_intake import digest, wire, safe_write
from .r10_blind_retrieval import indexed_turns
from .r11_candidate_graph import (source_ranges, derived_manifest,
                                   load_predictor_inputs, h, PROFILE_DIGEST, json_str)
from .task import Task, Predicate, predict as b0_predict

SCHEMA = 'elpis.ecsc.r12n.source-proven-structural-relations.v1'
NEXT, PREVIOUS = 1201, 1202
METHOD = 'ELPIS_NATIVE_B0_TYPED_STRUCTURAL'
LIMITS = Limits(axioms=8, relations=4, arity=8, degree=4, references=2,
                rounds=8, work=200_000, sidecar_bytes=65536,
                state_bytes=16384, updates_per_tick=32)
VIEW_LIMITS = ContextLimits(max_axioms=8, max_relations=4, max_arity=8, max_references=2)
SCOPE_PREFIX = 'dev-'
ANCHORS_PER_DEV_SAMPLE = 12

class NativeRefusal(Exception):
    pass

def demand(condition, reason):
    if not condition:
        raise NativeRefusal('R12N_'+reason)

def sequence_index(sample):
    """Proof authority: session-local published list order, never model outputs."""
    demand(type(sample) is dict and type(sample.get('sessions')) is list, 'SAMPLE')
    positions = {}
    for session in sample['sessions']:
        turns = session['turns']
        for ix, turn in enumerate(turns):
            did=turn['dialog_id']
            demand(did not in positions, 'DUPLICATE_ID')
            positions[did]=(session['session'], ix, tuple(t['dialog_id'] for t in turns))
    return positions

def verify_order_witness(sample, origin, target, direction):
    """Separate admission check: exact immediate document order, same session.

    This certifies ONLY structural order, not temporal truth or dialogue meaning.
    """
    demand(direction in ('NEXT','PREVIOUS'), 'INVALID_RELATION_TYPE')
    table=sequence_index(sample)
    demand(origin in table and target in table and origin!=target, 'TURN_NOT_FOUND')
    session, i, ids = table[origin]
    tgt_session, j, _ = table[target]
    demand(session==tgt_session, 'CROSS_SESSION')
    demand(j-i == (1 if direction=='NEXT' else -1), 'ORDER_PROOF_FAILURE')
    return True

def selected_centers(sample, count=ANCHORS_PER_DEV_SAMPLE):
    demand(type(count) is int and 1<=count<=32, 'COUNT')
    eligible = []
    for session in sample['sessions']:
        ids=[t['dialog_id'] for t in session['turns']]
        for ix in range(1,len(ids)-1):
            eligible.append((ids[ix-1],ids[ix],ids[ix+1]))
    demand(len(eligible)>=count, 'INSUFFICIENT_ANCHORS')
    # Hash order is frozen before evaluation and independent of question labels.
    return sorted(eligible,key=lambda row:(h('r12n.anchor.v1:'+sample['sample_id']+':'+row[1]),row[1]))[:count]

def source_graph(sample, spans, raw, manifest, row, *, status=ADMITTED):
    prev,center,nxt=row
    sid=sample['sample_id'];scope=SCOPE_PREFIX+sid
    demand(type(status) is str and status in (ADMITTED,CANDIDATE), 'RELATION_STATUS')
    # Admission proof is checked BEFORE constructing any admitted structural edge.
    verify_order_witness(sample,center,nxt,'NEXT')
    verify_order_witness(sample,center,prev,'PREVIOUS')
    turns,_=indexed_turns(sample)
    by_id={did:text for did,text in turns}
    doc=digest(raw)
    axioms={};grants=[];sources=[];bindings=[];obs=[]
    for did in sorted({prev,center,nxt}):
        demand((sid,did) in spans and did in by_id, 'SOURCE_TURN_MISSING')
        ordinal,start,end=spans[sid,did]
        rawchunk=raw[start:end]
        demand(rawchunk==json_str(by_id[did]), 'SOURCE_SPAN_DRIFT')
        norm=digest(normalize(rawchunk))
        claim=ChunkClaim(chunk_identity(doc,ordinal,start,end,norm,PROFILE_DIGEST),
                         doc,ordinal,start,end)
        ref=ContextReference(claim.chunk_digest,manifest.digest,
                             h('r12n.source-witness.v1:'+sid+':'+did))
        axiom=ContextAxiom(h('r12n.axiom.v1:'+sid+':'+did),(ref,),PROVISIONAL)
        axioms[did]=axiom
        grants.append(AxiomGrant(axiom,scope,'turn'))
        sources.append(SourceGrant(ref,scope,norm))
        bindings.append(HACFBinding(scope,ref,claim,'DEV_INPUT.json'))
        obs.append(Observation(axiom=axiom))
    relations=[]
    for typ,target,word in ((NEXT,nxt,'NEXT'),(PREVIOUS,prev,'PREVIOUS')):
        rel=ContextRelation(typ,(axioms[center].key,axioms[target].key),
            h('r12n.TEST_ONLY.structural-proof.v1:'+doc+':'+sid+':'+center+':'+target+':'+word),status)
        relations.append(rel)
        obs.append(Observation(relation=rel))
    view=successor(EMPTY_VIEW,tuple(obs),VIEW_LIMITS,through=len(obs),dropped=0)
    auth=Authority(tuple(grants),tuple(sources),tuple(relations),
        (Schema(NEXT,('origin','destination'),('turn','turn'),900),
         Schema(PREVIOUS,('origin','destination'),('turn','turn'),900)),
        epoch='TEST_ONLY_R12N_STRUCTURAL')
    return view,auth,tuple(bindings),axioms,scope

def directional_predictions(graph, scope, axioms, center, *, required_status=ADMITTED):
    """Elpis B0 computes two directional readouts on one source-verified graph."""
    result={};work={}
    lookup={v.key:k for k,v in axioms.items()}
    for direction,typ in (('NEXT',NEXT),('PREVIOUS',PREVIOUS)):
        task=Task(scope,(axioms[center].key,),
              (Predicate(typ,('origin','destination'),('turn','turn'),(required_status,)),),
              ('turn',),readout_budget=8,work_budget=150_000,steps=4,
              updates_per_tick=32)
        prediction=b0_predict(graph,task,'B0')
        demand(all(p.claim=='SALIENCE_ONLY' for p in prediction.predictions), 'EPISTEMIC_PROMOTION')
        # Root seed is expected; the only non-root activation is the typed destination.
        targets=[lookup[p.hit.grant.axiom.key] for p in prediction.predictions
                 if p.hit.grant.axiom.key!=axioms[center].key]
        demand(len(targets)<=1, 'UNBOUNDED_READOUT')
        if targets:
            selected=next(p for p in prediction.predictions if p.hit.grant.axiom.key==axioms[targets[0]].key)
            demand(selected.relations and all(r.relation.status==required_status and
                    r.relation.relation_type==typ for r in selected.relations), 'EVIDENCE_STATUS')
        result[direction]=targets[0] if targets else None
        work[direction]=prediction.work
    return result,work

def verify_one(sample,spans,raw,manifest,row,source_root):
    prev,center,nxt=row
    view,authority,bindings,axioms,scope=source_graph(sample,spans,raw,manifest,row)
    checks=verified_source_checks(source_root,manifest,view,authority,bindings,
        limits=LIMITS,max_document_bytes=1<<20,max_total_document_bytes=1<<20)
    demand(len(checks)==3, 'SOURCE_COVERAGE')
    graph=Baselines.build(view,authority,checks,LIMITS)
    out,work=directional_predictions(graph,scope,axioms,center)
    demand(out=={'NEXT':nxt,'PREVIOUS':prev}, 'NATIVE_DIRECTIONAL_FAILURE')
    demand(out['NEXT']!=out['PREVIOUS'], 'DIRECTION_COLLAPSE')
    # Controls: with candidate relations, an ADMITTED-only task must not propagate.
    cview,cauth,cbindings,caxioms,cscope=source_graph(sample,spans,raw,manifest,row,status=CANDIDATE)
    cchecks=verified_source_checks(source_root,manifest,cview,cauth,cbindings,
        limits=LIMITS,max_document_bytes=1<<20,max_total_document_bytes=1<<20)
    cgraph=Baselines.build(cview,cauth,cchecks,LIMITS)
    neg,cwork=directional_predictions(cgraph,cscope,caxioms,center)
    demand(neg=={'NEXT':None,'PREVIOUS':None}, 'CANDIDATE_UNAUTHORIZED_PROPAGATION')
    return {'center':center,'next':out['NEXT'],'previous':out['PREVIOUS'],
            'candidate_control':neg,
            'source_verified_chunks':len(checks)+len(cchecks),
            'logical_work':sum(work.values())+sum(cwork.values()),
            'graph_admission_work':sum(v for _,v in graph.build_operations)+
                sum(v for _,v in cgraph.build_operations)}

def run(work):
    root,dev,raw,dev_hash=load_predictor_inputs(work)
    demand(dev_hash=='89b8ed64d9a3cf83b7efb87611cef017dc65a2775a634851ecd6e6f711f50672',
           'DEV_PIN')
    spans,total=source_ranges(raw,dev)
    manifest=derived_manifest(raw,total)
    rows=[]
    for sample in dev['samples']:
        for triple in selected_centers(sample):
            row=verify_one(sample,spans,raw,manifest,triple,root)
            rows.append({'sample_id':sample['sample_id'],**row})
            print('R12N_STRUCTURAL=',len(rows),'CENTER=',row['center'],
                  'NEXT=',row['next'],'PREVIOUS=',row['previous'],flush=True)
    demand(len(rows)==24,'TRIAL_COUNT')
    report={'schema':SCHEMA,'status':'PASS_DEV_STRUCTURAL_ONLY_NOT_SEMANTIC_QUAL',
            'model':'ELPIS_B0_NO_EXTERNAL_LLM','r9_dev_sha256':dev_hash,
            'manifest_digest':manifest.digest,
            'authority':'TEST_ONLY_STRUCTURAL_ORDER_NOT_SEMANTIC_ENTAILMENT',
            'relationship':'IMMEDIATE_SAME_SESSION_RECORD_ORDER',
            'anchors':len(rows),'typed_tasks':len(rows)*2,
            'candidate_negative_tasks':len(rows)*2,
            'hacf_verified_chunks':sum(r['source_verified_chunks'] for r in rows),
            'logical_work_descriptive':sum(r['logical_work'] for r in rows),
            'admission_work_descriptive':sum(r['graph_admission_work'] for r in rows),
            'cases':rows,
            'scientific_qualification':False}
    path=root/'R12N_DEV_STRUCTURAL_REPORT.json'
    safe_write(path,wire(report))
    print('R12N_REPORT=',path,'SHA256=',digest(wire(report)),flush=True)
    print('R12N=PASS_DEV_STRUCTURAL_ONLY_NOT_SEMANTIC_QUAL',flush=True)
    return report

if __name__=='__main__':
    demand(len(sys.argv)==2,'USAGE_WORKTREE')
    try:run(Path(sys.argv[1]))
    except (NativeRefusal,Refusal) as exc:
        print('R12N=HOLD',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        sys.exit(3)
