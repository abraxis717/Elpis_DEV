"""R17 retrospective, read-only-to-prior-evidence scope sensitivity audit.

Uses the FROZEN R17 model, predictions and predictor-facing public cases.
No retraining, no new model, no gold loading, no QUAL, no scientific refreeze.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
from . import r17_raw_scope as r
from .r9_external_intake import wire, safe_write

SCHEMA='elpis.ecsc.r17.scope-ablation-audit.v1'
MODEL_SHA='222be935b8232109605966b46aedb6ad58db6bb02f6962845c4f8ef1d63d9cfb'
PRED_SHA='39ace2e70b534700303d712e6e2c95167309782e979265a2189143b779fad7b7'
REPORT_SHA='457b12146b71da29f1e5e7396fcd73c1d08bde3227e54c258139e86569c927c0'
FILE='R17_SCOPE_ABLATION_AUDIT.json'
MODIFICATIONS={
    'REMOVE_GROUPING':lambda t:t.replace('(','').replace(')',''),
    'REORDER_FALSE_THAT':lambda t:t.replace('false that (','that false ('),
    'REORDER_NOT_FALSE_THAT':lambda t:t.replace('not false that (','false not that ('),
}


def preflight(work):
    root=r.loc(Path(work))
    public,ph=r.check_json(root/r.PUBLIC_FILE,r.SCHEMA+'.eval-public',r.INPUT_PINS[r.PUBLIC_FILE])
    model,mh=r.check_json(root/r.MODEL_FILE,r.SCHEMA+'.model',MODEL_SHA)
    predictions,prh=r.check_json(root/r.PRED_FILE,r.SCHEMA+'.predictions',PRED_SHA)
    r.check_json(root/r.REPORT_FILE,r.SCHEMA+'.report',REPORT_SHA)
    r.require(model['training_sha256']==r.TRAIN_PIN and model['eval_public_sha256']==ph,'MODEL_ORIGIN')
    r.require(predictions['model_sha256']==mh and predictions['public_sha256']==ph,'PRED_ORIGIN')
    r.require(len(public['cases'])==len(predictions['rows'])==r.EVAL_COUNT,'CARDINALITY')
    return root,public['cases'],predictions['rows'],model


def analyze(cases,predictions,model):
    sums={name:{'source_cases':0,'modified':0,'same_answer':0,'changed_answer':0,'refused':0,
                'unchanged_text':0,'example_ids':[]} for name in MODIFICATIONS}
    evaluated=0;excluded=0
    for row,pred in zip(cases,predictions):
        r.require(pred['id']==row['id'],'ORDER_OR_ID')
        if pred['raw_status']=='REFUSED':
            excluded+=1
            r.require(pred['raw_answer'] is None,'REFUSAL_BINDING')
            # Refusals remain part of R17; this audit does not silently retest them.
            continue
        r.require(pred['raw_status']=='ANSWER','STATUS_UNEXPECTED')
        p=r.raw_predict(model,row)
        r.require(pred['raw_answer']==int(p>=0.5) and pred['raw_p']==round(p,9),'ORIGINAL_PRED_DRIFT')
        evaluated+=1
        for name,mutation in MODIFICATIONS.items():
            v=sums[name];v['source_cases']+=1
            text=mutation(row['text'])
            if text==row['text']:
                v['unchanged_text']+=1
                continue
            v['modified']+=1
            candidate={**row,'text':text}
            try:
                altered=r.raw_predict(model,candidate)
            except r.Hold:
                v['refused']+=1
                continue
            if int(altered>=.5)==pred['raw_answer']:
                v['same_answer']+=1
            else:
                v['changed_answer']+=1
                if len(v['example_ids'])<5:v['example_ids'].append(row['id'])
    r.require(evaluated+excluded==r.EVAL_COUNT,'COMPLETE')
    for v in sums.values():
        r.require(v['modified']==v['same_answer']+v['changed_answer']+v['refused'],'ACCOUNTING')
    return {'evaluated_answerable':evaluated,'original_refusals_excluded_from_ablation':excluded,
            'controls':sums}


def run(work):
    root,cases,preds,model=preflight(work)
    counts=analyze(cases,preds,model)
    report={'schema':SCHEMA,'disposition':'RETROSPECTIVE_STRUCTURAL_SENSITIVITY_ONLY',
            'r17_model_sha256':MODEL_SHA,'r17_predictions_sha256':PRED_SHA,
            'r17_report_sha256':REPORT_SHA,'r17_public_sha256':r.INPUT_PINS[r.PUBLIC_FILE],
            'gold_opened':False,'qual_opened':False,'model_retrained':False,
            'note':('Removing parentheses or reordering phrases outside the declared grammar '
                    'is an intervention on the input contract, not a ground-truth semantic case. '
                    'Unchanged predictions cannot establish learned syntax.'),
            **counts}
    payload=wire(report)
    safe_write(root/FILE,payload)
    print('R17_SCOPE_AUDIT_REPORT=',root/FILE,'SHA256=',r.sha(payload),flush=True)
    print('AUDIT_ORIGINAL_ANSWERABLE=',counts['evaluated_answerable'],
          'ORIGINAL_REFUSALS=',counts['original_refusals_excluded_from_ablation'],flush=True)
    for name,v in counts['controls'].items():
        print('ABLATION=',name,'MODIFIED=',v['modified'],
              'SAME=',v['same_answer'],'CHANGED=',v['changed_answer'],
              'REFUSED=',v['refused'],'NO_CHANGE=',v['unchanged_text'],flush=True)
    print('R17_SCOPE_AUDIT=PASS_DESCRIPTIVE_NO_NEW_SCIENTIFIC_QUAL',flush=True)


if __name__=='__main__':
    try:
        r.require(len(sys.argv)==2,'USAGE')
        run(sys.argv[1])
    except Exception as exc:
        print('R17_SCOPE_AUDIT=HOLD',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        sys.exit(3)
