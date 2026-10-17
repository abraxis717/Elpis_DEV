"""R15: source-bound learned CANDIDATE ingress into ECS_C/B0, research only.

The R14 sparse linear projector is frozen. A candidate graph is a proposed
interpretation of text, NEVER an epistemic admission, a truth proof, or a
semantically qualified model. No hosted/local LLM is involved.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

from elpis.runtime.context import (
    CANDIDATE, ADMITTED, PROVISIONAL, ContextAxiom, ContextRelation,
    Observation, successor,
)
from .baselines import Authority, AxiomGrant, Baselines, Refusal, Schema
from .r4_hacf_sources import verified_source_checks
from .r9_external_intake import digest, wire, safe_write
from .r11_candidate_graph import source_ranges, derived_manifest, load_predictor_inputs, h
from .r12_native_structure import (
    LIMITS, VIEW_LIMITS, NativeRefusal, selected_centers, source_graph,
)
from .r13_ingress_null import PROBE_TEXT, alter_one_turn, inspected_sample
from .r14_native_projector import probability, read_exact, SCHEMA as R14_SCHEMA, MODEL_ID
from .task import Task, Predicate, predict as b0_predict

SCHEMA='elpis.ecsc.r15.source-bound-learned-candidate-ingress.v1'
AFFIRM_CANDIDATE=1501
NEGATE_CANDIDATE=1502
DEV_PIN='89b8ed64d9a3cf83b7efb87611cef017dc65a2775a634851ecd6e6f711f50672'
MODEL_PIN='9989c5e510b148b433e58fe54a6363a2fd974704c9eabcb5005b58e9704391a8'
PREDICTIONS_PIN='61dcf9c6a03199afbea9cd79ae39998b03e6e273f73be2e1b722180e5959b58c'
R14_REPORT_PIN='453dc6cd27e55d8ad74b8970780d5c25ab24bf39138ddd6f2a0daf9f6d4a9b06'


def demand(ok, code):
    if not ok:
        raise NativeRefusal('R15_'+code)


def frozen_model(root: Path):
    p=root/'R14_PROJECTOR_MODEL.json'
    model,model_hash=read_exact(p,R14_SCHEMA+'.model')
    demand(model_hash==MODEL_PIN and model['model_id']==MODEL_ID,'MODEL_PIN')
    pred,predhash=read_exact(root/'R14_PREDICTIONS.json',R14_SCHEMA+'.predictions')
    report,rephash=read_exact(root/'R14_DEV_REPORT.json',R14_SCHEMA+'.report')
    demand(predhash==PREDICTIONS_PIN and rephash==R14_REPORT_PIN,'R14_EVIDENCE_PIN')
    demand(pred['model_sha256']==model_hash and report['model_sha256']==model_hash and
           report['prediction_sha256']==predhash,'R14_CHAIN')
    # This remains a known failure, not a hidden qualification pass.
    fail=report['evaluation']['negation_scope_shift']
    demand(fail['total']==32 and fail['correct']==0,'R14_FAILURE_PRESERVATION')
    demand(report['semantic_qualification'] is False,'SEMANTIC_NONCLAIM')
    return model,model_hash


def learned_proposal(model, model_hash: str, text: str, source_chunk: str):
    """Hypothesis with source & model provenance; classifier confidence is uncalibrated."""
    demand(model_hash==MODEL_PIN and type(source_chunk) is str and len(source_chunk)==64
           and all(c in '0123456789abcdef' for c in source_chunk),'IDENTITY')
    p=probability(model,text)
    polarity=int(p>=0.5)
    typ=AFFIRM_CANDIDATE if polarity else NEGATE_CANDIDATE
    # Provenance is reproducible but is NOT an externally admitted semantic proof.
    claim_key=h('r15.hypothesis.v1:'+model_hash+':'+source_chunk+':'+str(typ))
    evidence=h('r15.MODEL_GENERATED_UNADMITTED.v1:'+model_hash+':'+source_chunk+':'+str(typ))
    return {'polarity':polarity,'type':typ,'probability_affirmative':round(p,9),
            'source_chunk':source_chunk,'model_sha256':model_hash,
            'claim_key':claim_key,'model_provenance_digest':evidence,
            'epistemic_status':CANDIDATE}


def add_candidate(view,auth,axioms,scope,center,proposal):
    """Append a source-anchored provisional hypothesis and one candidate edge.

    The same ContextReference is used by both the source turn and its hypothesis,
    avoiding fabricated proof-bearing references. The learned source relation is
    never admitted. The existing independently admitted order edges stay intact.
    """
    anchor=axioms[center]
    hyp=ContextAxiom(proposal['claim_key'],anchor.references,PROVISIONAL)
    rel=ContextRelation(proposal['type'],(anchor.key,hyp.key),
                        proposal['model_provenance_digest'],CANDIDATE)
    next_view=successor(view,(Observation(axiom=hyp),Observation(relation=rel)),
                        VIEW_LIMITS,through=view.through+2,dropped=view.dropped)
    new_auth=Authority(auth.axioms+(AxiomGrant(hyp,scope,'hypothesis'),),
                       auth.sources,auth.relations+(rel,),
                       auth.schemas+(Schema(proposal['type'],('source','hypothesis'),
                                            ('turn','hypothesis'),900),),
                       epoch=auth.epoch)
    demand(all(r.status==CANDIDATE for r in next_view.relations
               if r.relation_type in (AFFIRM_CANDIDATE,NEGATE_CANDIDATE)),
           'UNAUTHORIZED_ADMISSION')
    return next_view,new_auth,hyp,rel


def candidate_readout(graph,scope,source,hyp,typ,require_status=CANDIDATE):
    """Read out exactly one typed hypothesis; never interpret SALIENCE as truth."""
    task=Task(scope,(source.key,),
              (Predicate(typ,('source','hypothesis'),('turn','hypothesis'),
                         (require_status,)),),
              ('hypothesis',),readout_budget=8,work_budget=150_000,
              steps=4,updates_per_tick=32)
    prediction=b0_predict(graph,task,'B0')
    demand(all(x.claim=='SALIENCE_ONLY' for x in prediction.predictions),
           'EPISTEMIC_PROMOTION')
    matching=[x for x in prediction.predictions if x.hit.grant.axiom.key==hyp.key]
    demand(len(matching)<=1,'DUPLICATE_HYPOTHESIS')
    if matching:
        demand(matching[0].hit.grant.kind=='hypothesis' and
               matching[0].relations and all(x.relation.status==require_status and
               x.relation.relation_type==typ for x in matching[0].relations),
               'SOURCE_OR_STATUS_MISMATCH')
    return len(matching)==1, prediction.work


def one_pair(work:Path,dev,model,model_hash,sid,triple):
    results={}
    root=work/'research/ecsc_hca_r0/external/locomo_r0'
    center=triple[1]
    for label,text in PROBE_TEXT.items():
        variant=alter_one_turn(dev,sid,center,text)
        with TemporaryDirectory(prefix='.r15_candidate_',dir=root) as td:
            temp=Path(td)
            raw=wire(variant)
            (temp/'DEV_INPUT.json').write_bytes(raw)
            spans,total=source_ranges(raw,variant)
            manifest=derived_manifest(raw,total)
            view,auth,bindings,axioms,scope=source_graph(
                inspected_sample(variant,sid),spans,raw,manifest,triple)
            anchor=axioms[center]
            chunk=anchor.references[0].chunk_digest
            prop=learned_proposal(model,model_hash,text,chunk)
            next_view,next_auth,hyp,rel=add_candidate(
                view,auth,axioms,scope,center,prop)
            checks=verified_source_checks(temp,manifest,next_view,next_auth,bindings,
                limits=LIMITS,max_document_bytes=1<<20,max_total_document_bytes=1<<20)
            demand(len(checks)==3,'HACF_SOURCE_COVERAGE')
            graph=Baselines.build(next_view,next_auth,checks,LIMITS)
            selected,work_candidate=candidate_readout(graph,scope,anchor,hyp,prop['type'])
            admitted,work_admitted=candidate_readout(
                graph,scope,anchor,hyp,prop['type'],ADMITTED)
            opposite=NEGATE_CANDIDATE if prop['type']==AFFIRM_CANDIDATE else AFFIRM_CANDIDATE
            wrong,work_wrong=candidate_readout(graph,scope,anchor,hyp,opposite)
            demand(selected and not admitted and not wrong,'EPISTEMIC_GATE')
            demand(rel.status==CANDIDATE and hyp.status==PROVISIONAL,'STATUS_INVARIANT')
            # A source changed after verification must not be accepted as the old chunk.
            other=PROBE_TEXT['NEGATED' if label=='AFFIRMATIVE' else 'AFFIRMATIVE']
            mutated=alter_one_turn(dev,sid,center,other)
            (temp/'DEV_INPUT.json').write_bytes(wire(mutated))
            refused=False
            try:
                verified_source_checks(temp,manifest,next_view,next_auth,bindings,
                    limits=LIMITS,max_document_bytes=1<<20,max_total_document_bytes=1<<20)
            except Refusal:
                refused=True
            demand(refused,'STALE_SOURCE_ACCEPTED')
            results[label]={'document_sha256':digest(raw),'source_chunk_digest':chunk,
                            'model_sha256':model_hash,'candidate_polarity':prop['polarity'],
                            'probability_affirmative':prop['probability_affirmative'],
                            'candidate_relation_type':prop['type'],
                            'candidate_visible_to_candidate_task':selected,
                            'candidate_hidden_from_admitted_task':not admitted,
                            'opposite_relation_not_matched':not wrong,
                            'stale_source_refused':refused,'claim_status':hyp.status,
                            'relation_status':rel.status,'readout_claim':'SALIENCE_ONLY',
                            'logical_work':work_candidate+work_admitted+work_wrong,
                            'graph_admission_work':sum(n for _,n in graph.build_operations)}
    demand(results['AFFIRMATIVE']['candidate_polarity']==1 and
           results['NEGATED']['candidate_polarity']==0,'R14_PAIR_BEHAVIOR')
    demand(results['AFFIRMATIVE']['source_chunk_digest']!=
           results['NEGATED']['source_chunk_digest'],'SOURCE_IDENTITY_COLLAPSE')
    return {'sample_id':sid,'center':center,'variants':results}


def run(work:Path):
    root,dev,raw,dev_hash=load_predictor_inputs(work)
    demand(dev_hash==DEV_PIN and digest(raw)==DEV_PIN,'DEV_SOURCE_PIN')
    model,model_hash=frozen_model(root)
    rows=[]
    for sample in dev['samples']:
        triple=selected_centers(sample,1)[0]
        row=one_pair(work,dev,model,model_hash,sample['sample_id'],triple)
        rows.append(row)
        print('R15_CANDIDATE=',len(rows),'CENTER=',row['center'],
              'AFF=',row['variants']['AFFIRMATIVE']['candidate_polarity'],
              'NEG=',row['variants']['NEGATED']['candidate_polarity'],
              'ADMITTED_LEAK=',not all(x['candidate_hidden_from_admitted_task']
                                      for x in row['variants'].values()),flush=True)
    demand(len(rows)==2,'PAIR_COUNT')
    report={'schema':SCHEMA,'status':'PASS_DEV_CANDIDATE_INGRESS_ONLY_NOT_SEMANTIC_QUAL',
            'r9_dev_sha256':dev_hash,'r14_model_sha256':model_hash,
            'r14_report_sha256':R14_REPORT_PIN,'cases':rows,
            'source_variants':4,'candidate_relations':4,
            'accepted_semantic_admissions':0,'r14_scope_shift_accuracy':'0/32',
            'qual_opened':False,'external_model_used':False,
            'scientific_qualification':False,
            'interpretation': 'R14 learned hypotheses enter B0 only as source-bound CANDIDATE edges; B0 salience is not semantic truth. Scope failures unchanged.'}
    report_path=root/'R15_DEV_CANDIDATE_INGRESS_REPORT.json'
    safe_write(report_path,wire(report))
    print('R15_REPORT=',report_path,'SHA256=',digest(wire(report)),flush=True)
    print('R15=PASS_DEV_ONLY_CANDIDATE_NOT_SEMANTIC_QUAL',flush=True)
    return report


if __name__=='__main__':
    try:
        demand(len(sys.argv)==2,'USAGE_WORKTREE')
        run(Path(sys.argv[1]))
    except Exception as exc:
        print('R15=HOLD',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        sys.exit(3)
