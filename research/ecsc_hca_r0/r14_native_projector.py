"""R14 DEV-only learned content projector: source-bound candidate hypotheses, never ADMITTED.

A minimal native sparse linear learner, not an LLM, a semantic proof, or a
production ECS_C subsystem. Three explicit process commands: prepare/train/score.
The predictor never reads held-out labels or QUAL. Held-out fixtures are synthetic.
"""
from __future__ import annotations
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import re
import sys

from .r9_external_intake import wire, digest, safe_write

SCHEMA='elpis.ecsc.r14.native-content-projection.v1'
TRAIN_NAMES=('Ava','Theo','Noor','Iris','Rafi','Lea')
TRAIN_ITEMS=('copper disk','silver pin','green book','blue ribbon','glass token','iron badge')
DEV_NAMES=('Mira','Jonah','Nadia','Samir')
DEV_ITEMS=('amber key','violet folder','red ledger','marble ring')
TRAIN_PATTERNS=(
    ('The {item} belongs to {name}.','The {item} does not belong to {name}.'),
    ('{name} owns the {item}.','{name} does not own the {item}.'),
    ('{name} has the {item}.','{name} does not have the {item}.'),
    ('The {item} is registered to {name}.','The {item} is not registered to {name}.'),
)
HOLDOUT_PATTERNS={
    'unseen_entities_seen_templates':TRAIN_PATTERNS,
    'new_templates_known_negation':(
        ('{name} keeps the {item}.','{name} does not keep the {item}.'),
        ('The {item} is carried by {name}.','The {item} is not carried by {name}.'),
    ),
    'new_negation_vocabulary':(
        ('{name} owns the {item}.','{name} never owned the {item}.'),
        ('The {item} belongs to {name}.','The {item} no longer belongs to {name}.'),
    ),
    'negation_scope_shift':(
        ('It is not false that {name} owns the {item}.',
         'It is false that {name} owns the {item}.'),
    ),
}
TRAIN_EPOCHS=32
LEARNING_RATE=0.11
L2=0.0002
MAX_FEATURES=8192
MODEL_ID='NATIVE_SPARSE_POLARITY_R14_DEV_NO_LLM'
DEV_PIN='89b8ed64d9a3cf83b7efb87611cef017dc65a2775a634851ecd6e6f711f50672'
R13_PIN='a55a7066b88a242983dbce6b2f6732322a6f8603d78171690fd9c635b786e6e8'

class Hold(Exception):pass

def require(x,msg):
    if not x:raise Hold('R14_'+msg)

def read_exact(path,sch):
    require(path.is_file() and not path.is_symlink(),'MISSING:'+path.name)
    raw=path.read_bytes();data=json.loads(raw)
    require(data.get('schema')==sch,'SCHEMA:'+path.name)
    return data,digest(raw)

def rows(names,items,templates,group):
    for t,(affirm,negate) in enumerate(templates):
        for name in names:
            for item in items:
                for val,pattern in ((1,affirm),(0,negate)):
                    text=pattern.format(name=name,item=item)
                    identifier=digest(f'{group}|{t}|{name}|{item}|{val}'.encode())[:24]
                    yield {'id':identifier,'text':text,'label':val,'group':group}

def fixture_records():
    training=list(rows(TRAIN_NAMES,TRAIN_ITEMS,TRAIN_PATTERNS,'TRAIN'))
    evaluated=[record for group,patterns in HOLDOUT_PATTERNS.items()
               for record in rows(DEV_NAMES,DEV_ITEMS,patterns,group)]
    require(len(training)==288 and len(evaluated)==(128+64+64+32),'FIXTURE_CARDINALITY')
    require(len({r['id'] for r in training+evaluated})==len(training)+len(evaluated),'ID_COLLISION')
    require(not {r['text'] for r in training}&{r['text'] for r in evaluated},'TEXT_SPLIT_LEAK')
    require(all(sum(r['label'] for r in x)*2==len(x) for x in (training,evaluated)),'LABEL_BALANCE')
    return training,evaluated

def paths(w):
    root=Path(w)/'research/ecsc_hca_r0/external/locomo_r0'
    return (root,root/'R14_TRAIN.json',root/'R14_EVAL_PUBLIC.json',root/'R14_EVAL_GOLD.json',
            root/'R14_PROJECTOR_MODEL.json',root/'R14_PREDICTIONS.json',root/'R14_DEV_REPORT.json')

def prepare(w):
    root,tr,ep,eg,_,_,_=paths(w)
    source,hs=read_exact(root/'DEV_INPUT.json','elpis.ecsc.external-locomo-intake.v1.public-dev')
    require(hs==DEV_PIN and len(source['samples'])==2,'DEV_AUTHORITY')
    report,h13=read_exact(root/'R13_DEV_INGRESS_NULL_REPORT.json',
                         'elpis.ecsc.r13.native-semantic-ingress-null-diagnostic.v1')
    require(h13==R13_PIN and report['pairs']==2,'R13_AUTHORITY')
    train,evaluation=fixture_records()
    artifacts=(
        (tr,{'schema':SCHEMA+'.train','purpose':'CONTROLLED_SUPERVISION_NOT_HUMAN_ADJUDICATED','samples':train}),
        (ep,{'schema':SCHEMA+'.eval-public','purpose':'PREDICTOR_ONLY_NO_GOLD','cases':[
            {k:v for k,v in row.items() if k!='label'} for row in evaluation]}),
        (eg,{'schema':SCHEMA+'.eval-gold','purpose':'SCORER_ONLY','labels':[
            {'id':row['id'],'label':row['label'],'group':row['group']} for row in evaluation]})
    )
    for path,data in artifacts:
        b=wire(data);safe_write(path,b,private=path==eg)
        print('R14_FROZEN=',path.name,'SHA256=',digest(b),'COUNT=',len(data.get('samples',data.get('cases',data.get('labels',[])))),flush=True)

def feature_vector(text):
    require(type(text) is str and 0<len(text)<=512,'TEXT_BOUNDS')
    words=re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())
    require(1<=len(words)<=80,'TOKEN_BOUNDS')
    # Frequency-free lexical features. No keyword rule, sentiment lexicon,
    # off-the-shelf encoder or pretrained weights.
    return tuple(sorted({'w:'+w for w in words}|{'b:'+a+'_'+b for a,b in zip(words,words[1:])}))

def fit(records):
    require(type(records) is list and len(records)==288,'TRAIN_COUNT')
    v={}
    for row in records:
        require(type(row['label']) is int and row['label'] in (0,1),'TRAIN_LABEL')
        v[row['id']]=feature_vector(row['text'])
    require(len(v)==len(records),'TRAIN_IDS')
    weights=defaultdict(float); bias=0.0
    # Fixed cyclic order, L2 on active features only. No labels from evaluation.
    for epoch in range(TRAIN_EPOCHS):
        order=sorted(records,key=lambda r:digest(f'{epoch}:{r["id"]}'.encode()))
        for row in order:
            feats=v[row['id']]
            z=max(-40.,min(40.,bias+sum(weights[f] for f in feats)))
            p=1.0/(1.0+math.exp(-z))
            delta=LEARNING_RATE*(row['label']-p)
            bias+=delta
            for f in feats:weights[f]+=delta-L2*weights[f]
    require(len(weights)<=MAX_FEATURES,'FEATURE_CAPACITY')
    return {'schema':SCHEMA+'.model','model_id':MODEL_ID,
            'training':{'examples':len(records),'epochs':TRAIN_EPOCHS,'lr':LEARNING_RATE,'l2':L2},
            'bias':round(bias,12),'weights':{k:round(v,12) for k,v in sorted(weights.items())}}

def probability(model,text):
    require(model['schema']==SCHEMA+'.model' and model['model_id']==MODEL_ID,'MODEL_IDENTITY')
    z=model['bias']+sum(model['weights'].get(f,0.0) for f in feature_vector(text))
    return 1./(1.+math.exp(-max(-40.,min(40.,z))))

def inference(w):
    root,tr,ep,_,mp,pp,_=paths(w)
    training,train_sha=read_exact(tr,SCHEMA+'.train')
    evalu,eval_sha=read_exact(ep,SCHEMA+'.eval-public')
    require(len(evalu['cases'])==288,'EVAL_COUNT')
    # Predictor does not read R14_EVAL_GOLD.json, R9 gold, R10/R11 score reports, QUAL.
    model=fit(training['samples'])
    model.update({'training_sha256':train_sha,'eval_public_sha256':eval_sha})
    safe_write(mp,wire(model))
    predictions=[{'id':row['id'],'p_affirmative':round(probability(model,row['text']),9),
                  'predicted':int(probability(model,row['text'])>=0.5)} for row in evalu['cases']]
    output={'schema':SCHEMA+'.predictions','model_sha256':digest(wire(model)),
            'eval_public_sha256':eval_sha,'rows':predictions,'status':'DEV_ONLY_NO_GOLD_OPENED'}
    safe_write(pp,wire(output))
    print('R14_PREDICTED=',len(predictions),'MODEL_SHA256=',digest(wire(model)),
          'PREDICTIONS_SHA256=',digest(wire(output)),flush=True)

def score(w):
    root,tr,ep,eg,mp,pp,rp=paths(w)
    model,mhash=read_exact(mp,SCHEMA+'.model')
    preds,phash=read_exact(pp,SCHEMA+'.predictions')
    gold,ghash=read_exact(eg,SCHEMA+'.eval-gold')
    public,ehash=read_exact(ep,SCHEMA+'.eval-public')
    require(preds['model_sha256']==mhash and preds['eval_public_sha256']==ehash,'MODEL_INPUT_BINDING')
    gold_ids={r['id']:r for r in gold['labels']}
    public_ids={r['id']:r for r in public['cases']}
    require(len(gold_ids)==len(public_ids)==len(preds['rows'])==288,'CARDINALITY')
    sums={name:{'total':0,'correct':0,'false_positive':0,'false_negative':0}
          for name in HOLDOUT_PATTERNS}
    for row in preds['rows']:
        require(row['id'] in gold_ids and row['id'] in public_ids,'PREDICTION_ID')
        require(type(row['predicted']) is int and row['predicted'] in (0,1),'PREDICTION_SHAPE')
        g=gold_ids[row['id']]
        require(g['group']==public_ids[row['id']]['group'],'GROUP_MISMATCH')
        s=sums[g['group']];s['total']+=1
        s['correct']+=int(row['predicted']==g['label'])
        s['false_positive']+=int(row['predicted']==1 and g['label']==0)
        s['false_negative']+=int(row['predicted']==0 and g['label']==1)
    require(sum(x['total'] for x in sums.values())==288,'SCORING_CARDINALITY')
    native=verified_r13_probe(w,model)
    report={'schema':SCHEMA+'.report','status':'PASS_DEV_EVALUATION_EXECUTED_NO_SCIENTIFIC_QUAL',
            'model_id':MODEL_ID,'model_sha256':mhash,'prediction_sha256':phash,
            'eval_gold_sha256':ghash,'evaluation':sums,'r13_source_verified_probe':native,
            'baseline_balanced_constant_accuracy':'50%','qual_opened':False,
            'semantic_qualification':False,
            'interpretation':'Synthetic grammar polarity; learned candidate hypothesis only; no language semantics or admitted relation proven.'}
    safe_write(rp,wire(report))
    print('R14_REPORT=',rp,'SHA256=',digest(wire(report)),flush=True)
    for k,s in sums.items():print('R14_GROUP=',k,'CORRECT=',s['correct'],'TOTAL=',s['total'],
                                 'FP=',s['false_positive'],'FN=',s['false_negative'],flush=True)
    print('R14_R13_SOURCE_PROBE=',native,flush=True)
    print('R14_DEV_EVALUATION=COMPLETE_NO_SCIENTIFIC_QUAL',flush=True)
    return report

def verified_r13_probe(w,model):
    # Reuses R13's complete canonical HACF check on fresh temporary variants;
    # original DEV_INPUT.json is never modified, and no gold is consumed.
    from .r13_ingress_null import PROBE_TEXT, alter_one_turn, inspected_sample, _build
    from .r11_candidate_graph import load_predictor_inputs
    from .r12_native_structure import selected_centers
    root,dev,_,devhash=load_predictor_inputs(Path(w))
    require(devhash==DEV_PIN,'R13_DEV_DRIFT')
    cases=[]
    for sample in dev['samples']:
        row=selected_centers(sample,1)[0]
        sid=sample['sample_id'];result={}
        for label,text in PROBE_TEXT.items():
            with TemporaryDirectory(prefix='.r14_native_',dir=root) as t:
                alt=alter_one_turn(dev,sid,row[1],text)
                structural,_=_build(Path(t),alt,sid,row)
                require(structural['source_verified']==3,'R13_SOURCE_MISSING')
                p=probability(model,text)
                result[label]={'source_sha256':structural['document_digest'],
                               'candidate_polarity':int(p>=0.5),
                               'probability_affirmative':round(p,9)}
        require(result['AFFIRMATIVE']['source_sha256']!=result['NEGATED']['source_sha256'],
                'SOURCE_COLLAPSE')
        cases.append({'sample_id':sid,'center':row[1],**result})
    require(len(cases)==2,'R13_CASES')
    return cases

def main():
    require(len(sys.argv)==3 and sys.argv[1] in ('prepare','infer','score'),'USAGE')
    fn={'prepare':prepare,'infer':inference,'score':score}[sys.argv[1]]
    fn(Path(sys.argv[2]))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('R14=HOLD',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        sys.exit(3)
