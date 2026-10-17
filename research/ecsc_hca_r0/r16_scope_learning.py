"""R16: trained compositional polarity over a *declared* text grammar, DEV research only.

Syntactic parsing (parentheses, operator phrases, queried clause) is supplied.
Polarity semantics of leaves/operators are learned from TRAIN labels, not encoded
into inference. This is not open-language understanding or ECS_C admission.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from collections import defaultdict
import json
import math
import re
import sys

from .r9_external_intake import wire, digest, safe_write
from .r14_native_projector import fit as fit_lexical, probability as lexical_probability

SCHEMA='elpis.ecsc.r16.compositional-scope.v1'
R15_REPORT_PIN='4a6211f8dbb888906acc665f74d46f9afe477d2df071314a7a6ec1cc9177f8c0'
R14_MODEL_PIN='9989c5e510b148b433e58fe54a6363a2fd974704c9eabcb5005b58e9704391a8'
R14_REPORT_PIN='453dc6cd27e55d8ad74b8970780d5c25ab24bf39138ddd6f2a0daf9f6d4a9b06'
OPS=('true that','false that','not false that')
LEAVES=('owns','does not own')
TR_NAMES=('Ava','Theo','Noor','Iris','Rafi','Lea')
TR_ITEMS=('copper disk','silver pin','green book','blue ribbon','glass token','iron badge')
EV_NAMES=('Mira','Jonah','Nadia','Samir')
EV_ITEMS=('amber key','violet folder','red ledger','marble ring')
GROUPS=(('unseen_entities_depth_le2',96),('unseen_nested_depth_3_to_5',96),
        ('distractor_scope_swap',96),('double_negation_depth_3_to_5',64),
        ('unknown_operator_refusal',32))
TRAIN_SIZE=288
EVAL_SIZE=sum(n for _,n in GROUPS)
EPOCHS=110
LR=0.36
INPUT_PINS={'R16_TRAIN.json':'9bb482d62fafdc06778b1ce0d51aa0340c191e8b15f84421c8a9bb982d65663c',
            'R16_EVAL_PUBLIC.json':'c888833d279287fd0b2fe72ec7fabbbb1daa2911daf664c9a88c6d8a938289a6',
            'R16_EVAL_GOLD.json':'f5fc7f767241100ffe4415a4fa9ce8ed8f0ec964705d33c10c35e12758841bc7'}

class Hold(Exception): pass


def require(ok,reason):
    if not ok: raise Hold('R16_'+reason)


def sha(b): return sha256(b).hexdigest()


def bytes_read(path):
    require(path.is_file() and not path.is_symlink(),'SOURCE_MISSING:'+path.name)
    return path.read_bytes()


def check_json(path,schema,pin=None):
    raw=bytes_read(path)
    require(pin is None or sha(raw)==pin,'SOURCE_DRIFT:'+path.name)
    d=json.loads(raw)
    require(type(d) is dict and d.get('schema')==schema,'SCHEMA:'+path.name)
    return d,sha(raw)


def loc(work):
    return Path(work)/'research/ecsc_hca_r0/external/locomo_r0'


def leaf(name,item,truth):
    return f'{name} {"owns" if truth else "does not own"} the {item}'


def render(phrase, operators):
    for op in operators: phrase=f'{op} ({phrase})'
    return phrase


def oracle(truth, operators):
    # GOLD GENERATION ONLY. Never called by inference or learner.
    for op in operators:
        if op=='false that': truth=not truth
    return int(truth)


def stable(index,salt=0):
    return int(sha(f'r16.fixture.{index}.{salt}'.encode())[:12],16)


def fixture_one(group,pair,variant,training,nonce=0):
    names=TR_NAMES if training else EV_NAMES
    items=TR_ITEMS if training else EV_ITEMS
    targetname=names[(pair+stable(pair,1)+nonce)%len(names)]
    targetitem=items[(pair//3+stable(pair,2)+3*nonce)%len(items)]
    truth=bool(variant)
    if training:
        depth=pair%3
        distractor=(pair%3 != 0 or pair%2==0)
    elif group=='unseen_entities_depth_le2':
        depth=pair%3;distractor=(pair%2==0)
    elif group=='distractor_scope_swap':
        depth=pair%3;distractor=True
    else:
        depth=3+pair%3;distractor=(pair%2==0)
    ops=tuple(OPS[stable(pair+23*j+nonce*17,3+j)%len(OPS)] for j in range(depth))
    if group=='double_negation_depth_3_to_5':
        # Test nested operator parity without revealing gold to the predictor.
        ops=('false that','false that')+tuple('false that' if j%2 else 'not false that'
                                           for j in range(depth-2))
    if group=='unknown_operator_refusal':
        ops=('it is disputed that',)+ops
    target=render(leaf(targetname,targetitem,truth),ops)
    if distractor:
        altname=names[(names.index(targetname)+1+pair%2)%len(names)]
        altitem=items[(items.index(targetitem)+2+pair%2)%len(items)]
        # Distractor operator patterns are independent of target scope.
        alternate=render(leaf(altname,altitem,not truth),
                         tuple(OPS[stable(pair+11*j+nonce*11,8+j)%len(OPS)] for j in range((pair+1)%3)))
        sentence=(alternate+'. Meanwhile, '+target+'.') if pair%2 else (target+'. Meanwhile, '+alternate+'.')
    else:
        sentence=target+'.'
    row={'id':sha(f'{group}|{pair}|{variant}'.encode())[:24],
         'text':sentence,'query':{'name':targetname,'item':targetitem},'group':group}
    label=None if group=='unknown_operator_refusal' else oracle(truth,ops)
    return row,label


def fixtures():
    used=set()
    def distinct(group,pair,variant,training):
        # Collision resolution depends on text identity only, never on correctness.
        # Guarantees each observation and target query is unique within the study.
        for nonce in range(2048):
            row,label=fixture_one(group,pair,variant,training,nonce)
            if (row['text'],row['query']['name'],row['query']['item']) not in used:
                used.add((row['text'],row['query']['name'],row['query']['item']))
                return row,label
        raise Hold('R16_FIXTURE_UNIQUENESS_CAPACITY')
    train=[]
    for pair in range(TRAIN_SIZE//2):
        for variant in (0,1):
            row,y=distinct('TRAIN',pair,variant,True)
            train.append({**row,'label':y})
    public=[];gold=[]
    for group,n in GROUPS:
        for pair in range(n//2):
            for variant in (0,1):
                row,y=distinct(group,pair,variant,False)
                public.append(row)
                gold.append({'id':row['id'],'group':group,'label':y,
                             'expected_refusal':group=='unknown_operator_refusal'})
    require(len(train)==TRAIN_SIZE and len(public)==len(gold)==EVAL_SIZE,'CARDINALITY')
    require(len({r['id'] for r in train+public})==TRAIN_SIZE+EVAL_SIZE,'DUPLICATE_IDS')
    require(not {r['text'] for r in train}&{r['text'] for r in public},'SPLIT_TEXT_LEAK')
    require(len({(r['text'],r['query']['name'],r['query']['item']) for r in train+public})==TRAIN_SIZE+EVAL_SIZE,'FIXTURE_REPETITION')
    for group,n in GROUPS:
        vals=[x['label'] for x in gold if x['group']==group]
        require(len(vals)==n,'GROUP_SIZE')
        if group != 'unknown_operator_refusal':
            require(vals.count(0)==vals.count(1)==n//2,'GOLD_BALANCE:'+group)
    return train,public,gold


LEAF_RX=re.compile(r'(?P<name>[A-Z][a-z]+) (?P<verb>owns|does not own) the (?P<item>[a-z]+(?: [a-z]+){0,2})\Z')


def parse_expression(text):
    operators=[]
    for i in range(9):
        if LEAF_RX.fullmatch(text):
            m=LEAF_RX.fullmatch(text)
            return {'name':m.group('name'),'item':m.group('item'),
                    'leaf':m.group('verb'),'operators':tuple(operators)}
        matched=False
        for op in OPS:
            prefix=op+' ('
            if text.startswith(prefix) and text.endswith(')'):
                inner=text[len(prefix):-1]
                # Invalid parentheses are rejected, not reinterpreted as extra words.
                require(inner and inner.count('(')==inner.count(')'),'UNBALANCED_SCOPE')
                operators.append(op)
                text=inner
                matched=True
                break
        if not matched:raise Hold('R16_UNKNOWN_GRAMMAR')
    raise Hold('R16_SCOPE_DEPTH')


def parse_query(row):
    require(type(row) is dict and type(row.get('text')) is str,'INPUT_SHAPE')
    q=row.get('query')
    require(type(q) is dict and type(q.get('name')) is str and type(q.get('item')) is str,
            'QUERY_SHAPE')
    text=row['text']
    require(0<len(text)<=640 and text.endswith('.'),'TEXT_BOUNDS')
    parts=text[:-1].split('. Meanwhile, ')
    require(1<=len(parts)<=2,'SENTENCE_COUNT')
    clauses=[parse_expression(x) for x in parts]
    matched=[c for c in clauses if c['name']==q['name'] and c['item']==q['item']]
    require(len(matched)==1,'TARGET_AMBIGUOUS_OR_MISSING')
    return matched[0]


def sigmoid(z):
    return 1/(1+math.exp(-max(-35.0,min(35.0,z))))


def forward(theta,parse,with_trace=False):
    require(parse['leaf'] in LEAVES,'UNKNOWN_LEAF')
    p=sigmoid(theta['leaf'][parse['leaf']]);trace=[]
    for op in reversed(parse['operators']):
        require(op in OPS,'UNKNOWN_OPERATOR')
        a,b=map(sigmoid,theta['op'][op])
        trace.append((op,p,a,b))
        p=a*(1-p)+b*p
    return (p,trace) if with_trace else p


def fit_composition(records):
    require(type(records) is list and len(records)==TRAIN_SIZE,'TRAIN_COUNT')
    require(len({r['id'] for r in records})==TRAIN_SIZE,'TRAIN_IDS')
    pre={r['id']:parse_query(r) for r in records}
    theta={'leaf':{k:0.0 for k in LEAVES},'op':{k:[0.0,0.0] for k in OPS}}
    for epoch in range(EPOCHS):
        order=sorted(records,key=lambda r:sha(f'{epoch}|{r["id"]}'.encode()))
        for r in order:
            require(type(r['label']) is int and r['label'] in (0,1),'LABEL_INVALID')
            expr=pre[r['id']]
            p,trace=forward(theta,expr,True)
            d=(p-r['label'])/max(1e-9,p*(1-p))
            # Reverse chain rule; simultaneous update within each node.
            for op,p_before,a,b in reversed(trace):
                g0=d*(1-p_before)*a*(1-a)
                g1=d*p_before*b*(1-b)
                d=d*(b-a)
                theta['op'][op][0]-=LR*g0
                theta['op'][op][1]-=LR*g1
            p0=sigmoid(theta['leaf'][expr['leaf']])
            theta['leaf'][expr['leaf']]-=LR*d*p0*(1-p0)
    return {'schema':SCHEMA+'.model','architecture':'LEARNED_TWO_STATE_OPERATOR_COMPOSITION',
            'train_count':TRAIN_SIZE,'epochs':EPOCHS,'step':LR,
            'theta':{'leaf':{k:round(v,12) for k,v in theta['leaf'].items()},
                     'op':{k:[round(a,12),round(b,12)] for k,(a,b) in theta['op'].items()}}}


def predict(model,row):
    require(model['schema']==SCHEMA+'.model','MODEL_SCHEMA')
    return forward(model['theta'],parse_query(row))


def prepare(work):
    root=loc(work)
    _,h15=check_json(root/'R15_DEV_CANDIDATE_INGRESS_REPORT.json',
                     'elpis.ecsc.r15.source-bound-learned-candidate-ingress.v1',R15_REPORT_PIN)
    _,h14=check_json(root/'R14_DEV_REPORT.json',
                     'elpis.ecsc.r14.native-content-projection.v1.report',R14_REPORT_PIN)
    require(sha(bytes_read(root/'R14_PROJECTOR_MODEL.json'))==R14_MODEL_PIN,'R14_MODEL_DRIFT')
    training,public,gold=fixtures()
    bundle=((root/'R16_TRAIN.json',{'schema':SCHEMA+'.train','records':training}),
            (root/'R16_EVAL_PUBLIC.json',{'schema':SCHEMA+'.eval-public','cases':public}),
            (root/'R16_EVAL_GOLD.json',{'schema':SCHEMA+'.eval-gold','records':gold}))
    for path,obj in bundle:
        raw=wire(obj)
        require(sha(raw)==INPUT_PINS[path.name],'PREDECLARED_FIXTURE_DRIFT:'+path.name)
        safe_write(path,raw,private=path.name.endswith('GOLD.json'))
        print('R16_PREPARED=',path.name,'SHA256=',sha(raw),'COUNT=',len(obj.get('records',obj.get('cases',[]))),flush=True)


def inference(work):
    root=loc(work)
    training,thash=check_json(root/'R16_TRAIN.json',SCHEMA+'.train',INPUT_PINS['R16_TRAIN.json'])
    public,phash=check_json(root/'R16_EVAL_PUBLIC.json',SCHEMA+'.eval-public',INPUT_PINS['R16_EVAL_PUBLIC.json'])
    require(len(training['records'])==TRAIN_SIZE and len(public['cases'])==EVAL_SIZE,'INPUT_COUNTS')
    model=fit_composition(training['records'])
    model.update({'training_sha256':thash,'eval_public_sha256':phash})
    baseline=fit_lexical(training['records'])
    rows=[]
    for r in public['cases']:
        try:
            p=predict(model,r)
            status='ANSWER'
            inferred=int(p>=.5)
        except Hold as exc:
            # Unknown grammar has no default parse, and is never silently answered.
            status='REFUSED';inferred=None;p=None
        lexical=lexical_probability(baseline,r['text'])
        rows.append({'id':r['id'],'scope_status':status,'scope_answer':inferred,
                     'scope_probability':None if p is None else round(p,9),
                     'lexical_answer':int(lexical>=.5),
                     'lexical_probability':round(lexical,9)})
    output={'schema':SCHEMA+'.predictions','model_sha256':sha(wire(model)),
            'baseline_sha256':sha(wire(baseline)),
            'eval_public_sha256':phash,'rows':rows,
            'claim':'DEV_SYNTHETIC_ONLY_NO_ADMISSION_NO_GOLD_OPENED'}
    safe_write(root/'R16_PROJECTOR_MODEL.json',wire(model))
    safe_write(root/'R16_LEXICAL_CONTROL_MODEL.json',wire(baseline))
    safe_write(root/'R16_PREDICTIONS.json',wire(output))
    print('R16_PREDICTIONS=',len(rows),'MODEL_SHA256=',sha(wire(model)),
          'SHA256=',sha(wire(output)),flush=True)


def score(work):
    root=loc(work)
    # Check model and every prediction against predictor-facing inputs BEFORE opening gold.
    training,th=check_json(root/'R16_TRAIN.json',SCHEMA+'.train',INPUT_PINS['R16_TRAIN.json'])
    public,ph=check_json(root/'R16_EVAL_PUBLIC.json',SCHEMA+'.eval-public',INPUT_PINS['R16_EVAL_PUBLIC.json'])
    pred,prh=check_json(root/'R16_PREDICTIONS.json',SCHEMA+'.predictions')
    model,mh=check_json(root/'R16_PROJECTOR_MODEL.json',SCHEMA+'.model')
    baseline,bh=check_json(root/'R16_LEXICAL_CONTROL_MODEL.json',
                           'elpis.ecsc.r14.native-content-projection.v1.model')
    require(pred['model_sha256']==mh and pred['baseline_sha256']==bh and
            pred['eval_public_sha256']==ph and model['eval_public_sha256']==ph and
            model['training_sha256']==th,'MODEL_PREDICTION_BINDING')
    expected_model=fit_composition(training['records'])
    expected_model.update({'training_sha256':th,'eval_public_sha256':ph})
    require(wire(expected_model)==wire(model),'MODEL_REPLAY_MISMATCH')
    require(wire(fit_lexical(training['records']))==wire(baseline),'BASELINE_REPLAY_MISMATCH')
    require(len(public['cases'])==len(pred['rows'])==EVAL_SIZE,'PREDICTION_COUNT')
    for case,row in zip(public['cases'],pred['rows']):
        require(case['id']==row['id'],'PREDICTION_ORDER')
        try:
            p=predict(model,case)
            status='ANSWER';y=int(p>=.5)
        except Hold:
            p=None;status='REFUSED';y=None
        lp=lexical_probability(baseline,case['text'])
        expected={'id':case['id'],'scope_status':status,'scope_answer':y,
                  'scope_probability':None if p is None else round(p,9),
                  'lexical_answer':int(lp>=.5),'lexical_probability':round(lp,9)}
        require(row==expected,'PREDICTION_REPLAY_MISMATCH')
    gold,gh=check_json(root/'R16_EVAL_GOLD.json',SCHEMA+'.eval-gold',INPUT_PINS['R16_EVAL_GOLD.json'])
    goldmap={x['id']:x for x in gold['records']}
    publicmap={x['id']:x for x in public['cases']}
    require(len(goldmap)==len(publicmap)==len(pred['rows'])==EVAL_SIZE,'COUNT_ID')
    s={k:{'total':0,'scope_correct':0,'lexical_correct':0,'scope_refused':0,
          'lexical_wrong':0,'scope_wrong':0} for k,_ in GROUPS}
    for row in pred['rows']:
        key=row['id'];require(key in goldmap and key in publicmap,'UNKNOWN_ID')
        g=goldmap.pop(key)
        require(g['group']==publicmap[key]['group'],'GROUP_DRIFT')
        entry=s[g['group']];entry['total']+=1
        if g['expected_refusal']:
            require(g['label'] is None,'OOD_GOLD_SHAPE')
            entry['scope_refused']+=int(row['scope_status']=='REFUSED')
            continue
        require(type(g['label']) is int and g['label'] in (0,1),'GOLD_INVALID')
        entry['scope_correct']+=int(row['scope_status']=='ANSWER' and row['scope_answer']==g['label'])
        entry['lexical_correct']+=int(row['lexical_answer']==g['label'])
        entry['scope_wrong']+=int(row['scope_status']=='ANSWER' and row['scope_answer']!=g['label'])
        entry['lexical_wrong']+=int(row['lexical_answer']!=g['label'])
        entry['scope_refused']+=int(row['scope_status']!='ANSWER')
    require(not goldmap and all(s[k]['total']==n for k,n in GROUPS),'INCOMPLETE_SCORING')
    summary={'schema':SCHEMA+'.report','status':'PASS_DEV_DESCRIPTIVE_ONLY_NOT_SCIENTIFIC_QUAL',
             'train_sha256':model['training_sha256'],'eval_public_sha256':ph,
             'eval_gold_sha256':gh,'predictions_sha256':prh,'model_sha256':mh,
             'groups':s,'qual_opened':False,'semantic_qualification':False,
             'model_not_integrated_into_canonical':True,
             'scope_syntax_given_not_learned':True,
             'interpretation':'Only operator effects were learned; grammar and query-to-clause mapping are provided. Candidate-only research, no ADMITTED relations or open-language semantics.'}
    safe_write(root/'R16_DEV_SCOPE_REPORT.json',wire(summary))
    print('R16_REPORT=',root/'R16_DEV_SCOPE_REPORT.json','SHA256=',sha(wire(summary)),flush=True)
    for k,v in s.items():
        print('R16_GROUP=',k,'TOTAL=',v['total'],'SCOPE_CORRECT=',v['scope_correct'],
              'LEXICAL_CORRECT=',v['lexical_correct'],'SCOPE_REFUSED=',v['scope_refused'],flush=True)
    print('R16=PASS_DEV_DESCRIPTIVE_NOT_SCIENTIFIC_QUAL',flush=True)


def main():
    require(len(sys.argv)==3 and sys.argv[1] in ('prepare','infer','score'),'USAGE')
    command=sys.argv[1];work=Path(sys.argv[2])
    {'prepare':prepare,'infer':inference,'score':score}[command](work)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        print('R16=HOLD',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        sys.exit(3)
