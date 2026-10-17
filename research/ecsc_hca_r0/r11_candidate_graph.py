"""R11: label-blind, source-verified LoCoMo DEV turn neighborhood -> ECS_C B0.

This is NOT native HACF chunk segmentation, semantic relation admission, answer
correctness, or ECS_G integration. All turn edges are navigation CANDIDATEs.
Predictor opens DEV_INPUT + prior R10 attestation/predictions only, never gold.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import sys

from elpis.runtime.context import (CANDIDATE, PROVISIONAL, EMPTY_VIEW,
    ContextAxiom, ContextLimits, ContextReference, ContextRelation, Observation, successor)
from elpis.structure.retrieval.objects import (ChunkClaim, CorpusManifest, chunk_identity, normalize)
from .baselines import (Authority, AxiomGrant, Limits, Refusal, Schema, SourceGrant, Baselines)
from .r4_hacf_sources import HACFBinding, verified_source_checks
from .task import Task, Predicate, predict as b0_predict
from .r9_external_intake import DEV_IDS, digest, wire, safe_write
from .r10_blind_retrieval import (SCHEMA as R10_SCHEMA, SCHEMA as R10_PREFIX,
    SOURCE_SHA256, _attestation, load_exact, paths, verify_public, indexed_turns,
    lexical_ranking, K, R10Error)

SCHEMA = 'elpis.ecsc.r11.label-blind-candidate-hacf-b0.v1'
R10_PRED_SHA256 = '019aa16568d50c4f3eb668d59bd0caeee84ad0796cea1fbcf2b9cf7ea14579b8'
R10_REPORT_SHA256 = '7994a487a186b2444e7e0d32b5b64e78d50c4d2444ebea6559d40b93d377ed83'
R10_GOLD_REF_COUNT = 310
PROFILE_DIGEST = sha256(b'ecsc.r11.TEST_ONLY.json-turn-literal-range.v1').hexdigest()
RELTYPE = 1101
LIMITS = Limits(axioms=32, relations=32, arity=8, degree=8,
                references=4, rounds=8, work=500_000,
                sidecar_bytes=262144, state_bytes=65536, updates_per_tick=256)
VIEW_LIMITS = ContextLimits(max_axioms=32,max_relations=32,max_arity=8,max_references=4)


def demand(flag, code):
    if not flag: raise R10Error('R11_'+code)


def h(text):
    return sha256(text.encode('utf8')).hexdigest()


def json_str(value):
    return json.dumps(value, ensure_ascii=True).encode('ascii')


def source_ranges(raw, dev):
    """Validate exact quoted text span against the frozen R9 wire release.

    This derives an *R11 test-only chunk profile* from original released JSON,
    NOT a native HACF chunker decision. Every byte slice is later reopened and
    verified using the canonical HACF resolver.
    """
    demand(type(raw) is bytes and raw == wire(dev), 'NOT_R9_RELEASE_BYTES')
    spans={};cursor=0;ordinal=0
    for sample in dev['samples']:
        sid=sample['sample_id']
        for block in sample['sessions']:
            for turn in block['turns']:
                did=turn['dialog_id']
                marker=b'"dialog_id": '+json_str(did)
                pos=raw.find(marker,cursor)
                demand(pos >= cursor, 'TURN_ID_NOT_IN_SOURCE_ORDER')
                speaker=b'"speaker": '+json_str(turn['speaker'])
                sp=raw.find(speaker,pos+len(marker))
                textfield=b'"text": '
                textpos=raw.find(textfield,pos+len(marker))
                demand(sp>pos and textpos>sp and textpos-sp < 100_000,'TURN_FIELDS_NOT_ADJACENT')
                start=textpos+len(textfield)
                value=json_str(turn['text'])
                end=start+len(value)
                demand(raw[start:end]==value,'TURN_LITERAL_MISMATCH')
                demand((sid,did) not in spans,'DUPLICATE_SOURCE_ID')
                spans[sid,did]=(ordinal,start,end)
                cursor=end;ordinal+=1
    return spans,ordinal


def derived_manifest(raw, count):
    doc=digest(raw)
    manifest=wire({'documents':[{'digest':doc,'size_bytes':len(raw),
        'chunk_count':count,'chunk_profile_digest':PROFILE_DIGEST,
        'media_type':'application/json'}]})
    return CorpusManifest.verified(manifest,expected_digest=digest(manifest))


def scoped_graph(sample, spans, raw, manifest, question):
    """Bound 4 lexical seeds + two within-session adjacency hops; no gold used."""
    sid=sample['sample_id'];scope='dev-'+sid
    turns,adj=indexed_turns(sample)
    order=lexical_ranking(turns,question)
    seeds=list(order[:4]); nodes=set(seeds); edges=set()
    # Deterministic outward BFS per seed; cross-session links absent by construction.
    for seed in seeds:
        frontier=[seed]
        seen={seed}
        for depth in range(2):
            next_frontier=[]
            for src in frontier:
                for target in sorted(adj[src]):
                    if target not in seen:
                        seen.add(target);next_frontier.append(target)
                        nodes.add(target);edges.add((src,target))
            frontier=next_frontier
    # Fixed fairness floor: at least K source-backed active turns if adjacency
    # overlaps heavily, using additional BM25 seeds. Never use labels for this.
    for i in order:
        if len(nodes)>=K: break
        if i not in nodes:
            seeds.append(i);nodes.add(i)
    demand(8<=len(nodes)<=24 and len(edges)<=32,'GRAPH_BOUNDS')
    doc=digest(raw)
    axiom_by_idx={};grants=[];sources=[];bindings=[];obs=[]
    for i in sorted(nodes):
        did,text=turns[i]
        ordinal,start,end=spans[sid,did]
        chunk=raw[start:end]
        norm=digest(normalize(chunk))
        claim=ChunkClaim(chunk_identity(doc,ordinal,start,end,norm,PROFILE_DIGEST),
                         doc,ordinal,start,end)
        reference=ContextReference(claim.chunk_digest,manifest.digest,
                         h('r11.TEST_ONLY.provenance:'+sid+':'+did))
        key=h('r11.axiom:'+sid+':'+did)
        axiom=ContextAxiom(key,(reference,),PROVISIONAL)
        axiom_by_idx[i]=axiom
        grants.append(AxiomGrant(axiom,scope,'turn'))
        sources.append(SourceGrant(reference,scope,norm))
        bindings.append(HACFBinding(scope,reference,claim,'DEV_INPUT.json'))
        obs.append(Observation(axiom=axiom))
    relations=[]
    for src,tgt in sorted(edges):
        relation=ContextRelation(RELTYPE,(axiom_by_idx[src].key,axiom_by_idx[tgt].key),
                   h('r11.TEST_ONLY.navigation:'+sid+':'+turns[src][0]+':'+turns[tgt][0]),CANDIDATE)
        relations.append(relation)
        obs.append(Observation(relation=relation))
    view=successor(EMPTY_VIEW,tuple(obs),VIEW_LIMITS,through=len(obs),dropped=0)
    authority=Authority(tuple(grants),tuple(sources),tuple(relations),
       (Schema(RELTYPE,('source','destination'),('turn','turn'),750),),
       epoch='TEST_ONLY_R11')
    task=Task(scope,tuple(axiom_by_idx[i].key for i in seeds),
          (Predicate(RELTYPE,('source','destination'),('turn','turn'),(CANDIDATE,)),),
          ('turn',),readout_budget=32,work_budget=500_000,steps=8)
    lookup={axiom_by_idx[i].key:turns[i][0] for i in nodes}
    return view,authority,tuple(bindings),task,lookup,len(nodes),len(relations)


def run_one(sample, spans, raw, manifest, q, root):
    view,auth,bindings,task,lookup,nodes,edges=scoped_graph(
       sample,spans,raw,manifest,q['question'])
    # Actual HACF opens original frozen DEV_INPUT.json, verifies the COMPLETE
    # document digest, profile-bound byte-range claims and normalized chunks.
    checks=verified_source_checks(root,manifest,view,auth,bindings,limits=LIMITS,
                max_document_bytes=1<<20,max_total_document_bytes=1<<20)
    graph=Baselines.build(view,auth,checks,LIMITS)
    result=b0_predict(graph,task,'B0')
    selected=[lookup[p.hit.grant.axiom.key] for p in result.predictions][:K]
    demand(len(selected)==K and len(set(selected))==K,'B0_READOUT_SHORTFALL')
    demand(all(p.claim=='SALIENCE_ONLY' for p in result.predictions),'CLAIM_PROMOTION')
    demand(all(r.status==CANDIDATE for r in view.relations),'UNAUTHORIZED_ADMISSION')
    return selected,{'graph_nodes':nodes,'candidate_edges':edges,
                     'admission_work':sum(v for _,v in graph.build_operations),
                     'readout_work':result.work,
                     'source_chunks_verified':len(checks)}


def load_predictor_inputs(work):
    root,devpath,_,_=paths(work)
    dev,raw_hash=load_exact(devpath,'elpis.ecsc.external-locomo-intake.v1.public-dev',
                            ('schema','status','samples'))
    verify_public(dev)
    receipt=_attestation(root)
    demand(raw_hash==receipt['dev_input_sha256'],'DEV_ATTESTATION_MISMATCH')
    needpred,phash=load_exact(root/'R10_DEV_PREDICTIONS.json',R10_SCHEMA+'.predictions',
         ('schema','status','source_input_sha256','methods','top_k','questions','prediction_contract'))
    demand(phash==R10_PRED_SHA256 and needpred['source_input_sha256']==raw_hash,
           'R10_PREDICTIONS_DRIFT')
    return root,dev,devpath.read_bytes(),raw_hash


def predict(work):
    root,dev,raw,dev_hash=load_predictor_inputs(work)
    spans,n=source_ranges(raw,dev)
    manifest=derived_manifest(raw,n)
    rows=[];stats=defaultdict(int)
    for sample in dev['samples']:
        for q in sample['questions']:
            ids,cost=run_one(sample,spans,raw,manifest,q,root)
            rows.append({'id':q['id'],'turns':ids})
            for k,v in cost.items(): stats[k]+=v
            if len(rows)%32==0:print('R11_PREDICTED=',len(rows),flush=True)
    demand(len(rows)==304,'R11_PREDICTION_COUNT')
    result={'schema':SCHEMA+'.predictions','status':'DEV_ONLY_NO_GOLD_SEEN',
        'dev_input_sha256':dev_hash,'r10_predictions_sha256':R10_PRED_SHA256,
        'source_profile':'TEST_ONLY_JSON_TURN_RANGES_NOT_NATIVE_HACF_CHUNKER',
        'relation_authority':'CANDIDATE_NAVIGATION_ONLY_NO_SEMANTIC_ADMISSION',
        'method':'ECSC_B0_CANDIDATE_RADIUS2_TOP8','top_k':K,'questions':rows,
        'work_descriptive_total':dict(stats)}
    out=root/'R11_DEV_PREDICTIONS.json'
    safe_write(out,wire(result))
    print('R11_PREDICTIONS=',out,'SHA256=',digest(wire(result)),'DEV_CASES=',len(rows),flush=True)


def score(work):
    root,dev,_,dev_hash=load_predictor_inputs(work)
    receipt=_attestation(root)
    raw_report=(root/'R10_DEV_EVIDENCE_REPORT.json')
    demand(raw_report.is_file() and not raw_report.is_symlink() and
           digest(raw_report.read_bytes())==R10_REPORT_SHA256,'R10_REPORT_DRIFT')
    gold,ghash=load_exact(root/'SCORER_GOLD.json','elpis.ecsc.external-locomo-intake.v1.gold',
                             ('schema','status','annotations'))
    demand(ghash==receipt['scorer_gold_sha256'],'R9_GOLD_DRIFT')
    pred,pred_hash=load_exact(root/'R11_DEV_PREDICTIONS.json',SCHEMA+'.predictions',
            ('schema','status','dev_input_sha256','r10_predictions_sha256','source_profile',
             'relation_authority','method','top_k','questions','work_descriptive_total'))
    demand(pred['status']=='DEV_ONLY_NO_GOLD_SEEN' and pred['dev_input_sha256']==dev_hash
           and pred['r10_predictions_sha256']==R10_PRED_SHA256 and pred['top_k']==K,
           'R11_PREDICTION_DRIFT')
    legal={s['sample_id']:{t['dialog_id'] for b in s['sessions'] for t in b['turns']}
           for s in dev['samples']}
    expected=[q['id'] for s in dev['samples'] for q in s['questions']]
    demand([q['id'] for q in pred['questions']]==expected,'PREDICTION_IDS')
    goldby={g['id']:g for g in gold['annotations'] if g['sample_id'] in DEV_IDS}
    demand(set(goldby)==set(expected),'DEV_GOLD_IDS')
    summaries={key:{'queries':0,'gold_references':0,'matched_references':0,
         'unresolved_references':0,'any_hit':0,'all_hit':0,'empty_evidence':0}
         for key in ('answer_present','missing_answer')}
    for p in pred['questions']:
        sid=p['id'].split('/')[0];dids=p['turns'];g=goldby[p['id']]
        demand(type(dids) is list and len(dids)==K and len(set(dids))==K
                and all(type(x) is str and x in legal[sid] for x in dids),
                'PREDICTED_TURN_INVALID')
        s=summaries['answer_present' if g['answer_present'] else 'missing_answer']
        s['queries']+=1
        refs=g['evidence_original'];s['gold_references']+=len(refs)
        s['unresolved_references']+=len(g['evidence_unresolved'])
        if not refs: s['empty_evidence']+=1;continue
        n=sum(r in dids for r in refs)
        s['matched_references']+=n;s['any_hit']+=int(n>0);s['all_hit']+=int(n==len(refs))
    demand(summaries['answer_present']['queries']==235 and
            summaries['answer_present']['gold_references']==R10_GOLD_REF_COUNT,
            'GOLD_DENOMINATOR_DRIFT')
    report={'schema':SCHEMA+'.descriptive','status':'DEV_ONLY_NOT_SCIENTIFIC_QUAL',
            'source_sha256':SOURCE_SHA256,'prediction_sha256':pred_hash,
            'baseline_r10_report_sha256':R10_REPORT_SHA256,
            'methods':{'ECSC_B0_CANDIDATE_RADIUS2_TOP8':summaries},
            'work_descriptive_total':pred['work_descriptive_total'],
            'claims':['CANDIDATE_GRAPH_ONLY','SALIENCE_NOT_TRUTH',
                      'DERIVED_CHUNKS_NOT_NATIVE_HACF_SEGMENTATION',
                      'NO_GENERATED_ANSWERS','NO_QUAL_OPENED','NO_SCIENTIFIC_QUAL']}
    out=root/'R11_DEV_EVIDENCE_REPORT.json'
    safe_write(out,wire(report))
    a=summaries['answer_present']
    print('ECSC_B0_CANDIDATE_RADIUS2_TOP8','ANSWERED_QUERIES=',a['queries'],
          'GOLD_REFS=',a['gold_references'],'RETRIEVED_GOLD_REFS=',a['matched_references'],
          'ANY_HIT=',a['any_hit'],'ALL_HIT=',a['all_hit'],'UNRESOLVED=',a['unresolved_references'],flush=True)
    print('R11_REPORT=',out,'SHA256=',digest(wire(report)),flush=True)
    print('R11_DESCRIPTIVE=PASS_DEV_ONLY_NOT_SCIENTIFIC_QUAL',flush=True)

if __name__=='__main__':
    try:
        demand(len(sys.argv)==3 and sys.argv[1] in ('predict','score'),'USAGE: predict|score WORKTREE')
        (predict if sys.argv[1]=='predict' else score)(sys.argv[2])
    except Exception as exc:
        print('R11=HOLD',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        raise SystemExit(3)
