"""R17: label-trained raw-token polarity sequence; no supplied operator scope tree.

DEV synthetic controlled grammar. Query entity/item identify a *sentence*, but the
learner sees the unparsed token sequence, not operator names/effects/trees.
No LLM, pretrained model, external corpus, R9 QUAL, or canonical runtime change.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from .r9_external_intake import wire, safe_write
from .r14_native_projector import fit as lexical_fit, probability as lexical_probability
from .r16_scope_learning import (SCHEMA as R16_SCHEMA, TRAIN_SIZE, INPUT_PINS as R16_PINS,
                                 check_json, loc, sha, bytes_read, fit_composition,
                                 predict as structured_predict, Hold)

SCHEMA='elpis.ecsc.r17.raw-token-scope.v1'
TRAIN_PIN=R16_PINS['R16_TRAIN.json']
R16_REPORT_PIN='95803624d72930af9d4fa4d4227ed2a7ff5243dbaf134042ca50605308624e3d'
R16_MODEL_PIN='fbe65bd4a4c018e042d4bfbfc4c2cd8b9e6577ae8613c0d91979d28873c7d003'
R16_PRED_PIN='ca644c7c3bd4fe341f67fefc1a6780463acfe0655be21cd4c991207f5e9cd6a3'
OPS=('true that','false that','not false that')  # FIXTURE ORACLE ONLY: not used by raw predictor
GROUPS=(('heldout_entities_shallow',64),('depth_3_to_7',96),
        ('distractor_scope',96),('paired_polarity_counterfactual',64),
        ('unknown_operator_refusal',32),('malformed_scope_refusal',32))
PUBLIC_FILE='R17_EVAL_PUBLIC.json'
GOLD_FILE='R17_EVAL_GOLD.json'
MODEL_FILE='R17_RAW_SEQUENCE_MODEL.json'
PRED_FILE='R17_PREDICTIONS.json'
REPORT_FILE='R17_DEV_SCOPE_REPORT.json'
INPUT_PINS={PUBLIC_FILE:'5ae65b7047c2382b912d4d9cb9898c967f621174ab0f4112f1353a12ca7dfda3',
            GOLD_FILE:'9269f6cd409870f2cb5c10ce8fc7ca57c9d3567c01145be4a20f7f60bd94de5f'}
EPOCHS=90
LR=0.15
TRAIN_COUNT=288
EVAL_COUNT=sum(n for _,n in GROUPS)
WORD_RE=re.compile(r'@[a-z]+|[a-z]+|[()]')
LEAF_RE_TEMPLATE=r'(?<![A-Za-z]){name} (?:owns|does not own) the {item}(?![A-Za-z])'
MAX_TOKENS=90


def require(ok, reason):
    if not ok: raise Hold('R17_'+reason)


def stable(*args):
    return int(sha('|'.join(map(str,args)).encode())[:14],16)


def make_item(group,i,variant):
    # Generated independently of predictions. Labels are created only at prepare.
    pair=i//2 if group=='paired_polarity_counterfactual' else i
    names=('Zara','Otto','Tessa','Luca','Ada','Micah')
    items=('gold lantern','black spool','bronze tile','teal notebook','stone brooch','white satchel')
    name=names[stable('R17name',group,pair)%len(names)]
    item=items[stable('R17item',group,pair)%len(items)]
    truth=bool(variant)
    if group=='heldout_entities_shallow': depth=i%3
    elif group=='paired_polarity_counterfactual': depth=2+pair%4
    else: depth=3+i%5
    opchain=[OPS[stable('R17op',group,pair,j)%len(OPS)] for j in range(depth)]
    if group=='paired_polarity_counterfactual':
        # Within-pair leaf-polarity intervention; operator chain is held constant.
        truth=bool(i%2)
    scope=f'{name} {"owns" if truth else "does not own"} the {item}'
    label=int(truth)
    for op in opchain:
        scope=f'{op} ({scope})'
        if op=='false that': label=1-label
    if group=='unknown_operator_refusal': scope='it is disputed that ('+scope+')'
    if group=='malformed_scope_refusal': scope='('+scope  # intentionally unmatched
    if group in ('distractor_scope','depth_3_to_7','paired_polarity_counterfactual'):
        other=names[(names.index(name)+1)%len(names)]
        other_item=items[(items.index(item)+2)%len(items)]
        distractor=f'{other} does not own the {other_item}'
        if (pair+1)%2: distractor='false that ('+distractor+')'
        full=f'{scope}. Meanwhile, {distractor}.' if pair%2 else f'{distractor}. Meanwhile, {scope}.'
    else: full=scope+'.'
    row={'id':sha(f'R17|{group}|{i}|{variant}'.encode())[:24], 'group':group,
         'text':full,'query':{'name':name,'item':item}}
    return row, (None if group.endswith('_refusal') else label)


def fixtures():
    public=[];gold=[];used=set()
    for group,count in GROUPS:
        for i in range(count):
            row,label=make_item(group,i,i%2)
            require(row['id'] not in used,'DUPLICATE_ID')
            used.add(row['id'])
            public.append(row)
            gold.append({'id':row['id'],'group':group,'label':label,
                         'expected_refusal':group.endswith('_refusal')})
    require(len(public)==len(gold)==EVAL_COUNT,'CARDINALITY')
    return public,gold


def choose_unparsed_tokens(row):
    """Query-bound sentence selection only. No extraction of operator tree/effects.

    Parenthesis balance and lexical OOD are refusal guards. Tokens are raw words
    and punctuation after masking the caller's name and object, not a parse tree.
    """
    require(type(row) is dict and type(row.get('text')) is str,'ROW_SHAPE')
    q=row.get('query')
    require(type(q) is dict and type(q.get('name')) is str and type(q.get('item')) is str,'QUERY_SHAPE')
    text=row['text']
    require(0<len(text)<=900 and text.endswith('.'),'TEXT_BOUND')
    parts=text[:-1].split('. Meanwhile, ')
    require(1<=len(parts)<=2,'SENTENCE_COUNT')
    pattern=re.compile(LEAF_RE_TEMPLATE.format(name=re.escape(q['name']),item=re.escape(q['item'])))
    matched=[(part,list(pattern.finditer(part))) for part in parts]
    require(sum(len(matches) for _,matches in matched)==1,'TARGET_AMBIGUOUS_OR_MISSING')
    clause,matches=next((part,matches) for part,matches in matched if matches)
    # Count balanced grouping; do not assign parent-child scope or operator symbols.
    bal=0
    for ch in clause:
        if ch=='(':bal+=1
        elif ch==')':
            bal-=1
            require(bal>=0,'SCOPE_UNBALANCED')
    require(bal==0,'SCOPE_UNBALANCED')
    m=matches[0]
    raw=clause[:m.start()]+' @name '+m.group(0).replace(q['name'],'',1).replace('the '+q['item'],'@item',1)+clause[m.end():]
    normalized=raw.lower()
    tokens=WORD_RE.findall(normalized)
    require(tokens and len(tokens)<=MAX_TOKENS,'TOKEN_BOUND')
    require(''.join(normalized.split())==''.join(tokens),'UNKNOWN_PUNCTUATION')
    require('@name' in tokens and '@item' in tokens,'MASK_MISSING')
    return tokens


def sigmoid(z):return 1/(1+math.exp(-max(-35.,min(35.,z))))


def token_forward(theta,tokens,trace=False):
    p=sigmoid(theta['initial'])
    steps=[]
    for tok in reversed(tokens):
        require(tok in theta['tokens'],'UNKNOWN_TOKEN:'+tok)
        a,b=map(sigmoid,theta['tokens'][tok])
        steps.append((tok,p,a,b))
        p=a*(1-p)+b*p
    return (p,steps) if trace else p


def fit_raw(train):
    require(len(train)==TRAIN_COUNT,'TRAIN_COUNT')
    require(len({x['id'] for x in train})==TRAIN_COUNT,'TRAIN_IDS')
    tokens={r['id']:choose_unparsed_tokens(r) for r in train}
    vocab=sorted({t for words in tokens.values() for t in words})
    # Identity-biased states, not operator-specific initialized meaning.
    theta={'initial':0.25,'tokens':{t:[-2.9,2.9] for t in vocab}}
    for epoch in range(EPOCHS):
        order=sorted(train,key=lambda row:sha(f'R17|{epoch}|{row["id"]}'.encode()))
        for r in order:
            y=r['label'];require(type(y) is int and y in (0,1),'LABEL_SHAPE')
            p,steps=token_forward(theta,tokens[r['id']],True)
            gradient=(p-y)/max(1e-8,p*(1-p))
            updates=[]
            for tok,previous,a,b in reversed(steps):
                da=gradient*(1-previous)*a*(1-a)
                db=gradient*previous*b*(1-b)
                updates.append((tok,da,db))
                gradient*=b-a
            for tok,da,db in updates:
                theta['tokens'][tok][0]-=LR*da
                theta['tokens'][tok][1]-=LR*db
            initial=sigmoid(theta['initial'])
            theta['initial']-=LR*gradient*initial*(1-initial)
    frozen={'initial':round(theta['initial'],10),'tokens':{t:[round(a,10),round(b,10)] for t,(a,b) in theta['tokens'].items()}}
    return {'schema':SCHEMA+'.model','architecture':'QUERY_BOUND_UNPARSED_LEXICAL_TRANSITION',
            'epochs':EPOCHS,'rate':LR,'train_count':TRAIN_COUNT,'theta':frozen}


def raw_predict(model,row):
    require(model['schema']==SCHEMA+'.model','MODEL_SHAPE')
    return token_forward(model['theta'],choose_unparsed_tokens(row))


def check_inputs(work):
    root=loc(work)
    training,train_hash=check_json(root/'R16_TRAIN.json',R16_SCHEMA+'.train',TRAIN_PIN)
    require(len(training['records'])==TRAIN_COUNT,'TRAIN_SIZE')
    require(sha(bytes_read(root/'R16_PROJECTOR_MODEL.json'))==R16_MODEL_PIN,'STRUCTURED_MODEL_DRIFT')
    require(sha(bytes_read(root/'R16_PREDICTIONS.json'))==R16_PRED_PIN,'R16_PRED_DRIFT')
    check_json(root/'R16_DEV_SCOPE_REPORT.json',R16_SCHEMA+'.report',R16_REPORT_PIN)
    return root,training['records'],train_hash


def prepare(work):
    root,training,th=check_inputs(work)
    public,gold=fixtures()
    for filename,obj in ((PUBLIC_FILE,{'schema':SCHEMA+'.eval-public','cases':public}),
                         (GOLD_FILE,{'schema':SCHEMA+'.eval-gold','rows':gold})):
        data=wire(obj)
        require(sha(data)==INPUT_PINS[filename],'FIXTURE_PIN_MISMATCH:'+filename)
        safe_write(root/filename,data,private=filename==GOLD_FILE)
        print('R17_FROZEN=',filename,'SHA256=',sha(data),'COUNT=',len(obj.get('cases',obj.get('rows',[]))),flush=True)


def infer(work):
    root,training,th=check_inputs(work)
    public,ph=check_json(root/PUBLIC_FILE,SCHEMA+'.eval-public',INPUT_PINS[PUBLIC_FILE])
    model=fit_raw(training)
    model.update({'training_sha256':th,'eval_public_sha256':ph})
    oracle_model,_=check_json(root/'R16_PROJECTOR_MODEL.json',R16_SCHEMA+'.model',R16_MODEL_PIN)
    lexical=lexical_fit(training)
    result=[]
    for row in public['cases']:
        try: p=raw_predict(model,row);status='ANSWER';y=int(p>=.5)
        except Hold: p=None;status='REFUSED';y=None
        try:sp=structured_predict(oracle_model,row);ss='ANSWER';sy=int(sp>=.5)
        except Hold:sp=None;ss='REFUSED';sy=None
        lp=lexical_probability(lexical,row['text'])
        result.append({'id':row['id'],'raw_status':status,'raw_answer':y,
                       'raw_p':None if p is None else round(p,9),
                       'structured_status':ss,'structured_answer':sy,
                       'structured_p':None if sp is None else round(sp,9),
                       'lexical_answer':int(lp>=.5),'lexical_p':round(lp,9)})
    obj={'schema':SCHEMA+'.predictions','model_sha256':sha(wire(model)),
         'structured_model_sha256':R16_MODEL_PIN,'lexical_model_sha256':sha(wire(lexical)),
         'public_sha256':ph,'rows':result,'qualification':False}
    safe_write(root/MODEL_FILE,wire(model))
    safe_write(root/PRED_FILE,wire(obj))
    print('R17_INFER=',len(result),'MODEL_SHA256=',sha(wire(model)),'PRED_SHA256=',sha(wire(obj)),flush=True)


def score(work):
    root,train,th=check_inputs(work)
    public,ph=check_json(root/PUBLIC_FILE,SCHEMA+'.eval-public',INPUT_PINS[PUBLIC_FILE])
    model,mh=check_json(root/MODEL_FILE,SCHEMA+'.model')
    predictions,predh=check_json(root/PRED_FILE,SCHEMA+'.predictions')
    require(predictions['model_sha256']==mh and predictions['public_sha256']==ph and
            model['training_sha256']==th and model['eval_public_sha256']==ph,'BINDING_MISMATCH')
    refit=fit_raw(train);refit.update({'training_sha256':th,'eval_public_sha256':ph})
    require(wire(refit)==wire(model),'MODEL_REPLAY_FAILED')
    structured,_=check_json(root/'R16_PROJECTOR_MODEL.json',R16_SCHEMA+'.model',R16_MODEL_PIN)
    lexical=lexical_fit(train)
    require(predictions['lexical_model_sha256']==sha(wire(lexical)),'LEXICAL_BINDING')
    require(len(public['cases'])==len(predictions['rows'])==EVAL_COUNT,'EVAL_LENGTH')
    for row,pred in zip(public['cases'],predictions['rows']):
        try:p=raw_predict(model,row);s='ANSWER';a=int(p>=.5)
        except Hold:p=None;s='REFUSED';a=None
        try:sp=structured_predict(structured,row);ss='ANSWER';sa=int(sp>=.5)
        except Hold:sp=None;ss='REFUSED';sa=None
        lp=lexical_probability(lexical,row['text'])
        expect={'id':row['id'],'raw_status':s,'raw_answer':a,'raw_p':None if p is None else round(p,9),
                'structured_status':ss,'structured_answer':sa,'structured_p':None if sp is None else round(sp,9),
                'lexical_answer':int(lp>=.5),'lexical_p':round(lp,9)}
        require(pred==expect,'PREDICTION_REPLAY:'+row['id'])
    # Only now is gold opened.
    gold,gh=check_json(root/GOLD_FILE,SCHEMA+'.eval-gold',INPUT_PINS[GOLD_FILE])
    require(len(gold['rows'])==EVAL_COUNT,'GOLD_LENGTH')
    summary={name:{'total':0,'raw_correct':0,'raw_wrong':0,'raw_refused':0,
                   'structured_correct':0,'structured_refused':0,'lexical_correct':0} for name,n in GROUPS}
    for src,pred,g in zip(public['cases'],predictions['rows'],gold['rows']):
        require(src['id']==pred['id']==g['id'] and src['group']==g['group'],'ID_MATCH')
        v=summary[g['group']];v['total']+=1
        if g['expected_refusal']:
            require(g['label'] is None,'REFUSAL_LABEL')
            v['raw_refused']+=int(pred['raw_status']=='REFUSED')
            v['structured_refused']+=int(pred['structured_status']=='REFUSED')
        else:
            require(type(g['label']) is int and g['label'] in (0,1),'LABEL')
            v['raw_correct']+=int(pred['raw_status']=='ANSWER' and pred['raw_answer']==g['label'])
            v['raw_wrong']+=int(pred['raw_status']=='ANSWER' and pred['raw_answer']!=g['label'])
            v['raw_refused']+=int(pred['raw_status']!='ANSWER')
            v['structured_correct']+=int(pred['structured_status']=='ANSWER' and pred['structured_answer']==g['label'])
            v['structured_refused']+=int(pred['structured_status']!='ANSWER')
            v['lexical_correct']+=int(pred['lexical_answer']==g['label'])
    require(all(summary[k]['total']==n for k,n in GROUPS),'GROUPS')
    report={'schema':SCHEMA+'.report','status':'PASS_DESCRIPTIVE_ONLY_NOT_SEMANTIC_QUAL',
            'training_sha256':th,'public_sha256':ph,'gold_sha256':gh,'predictions_sha256':predh,
            'raw_model_sha256':mh,'groups':summary,'qual_opened':False,
            'parse_tree_provided_to_raw':False,'query_clause_selector_provided':True,
            'interpretation':'Unparsed raw-token learner with query-bound sentence selection. Synthetic source grammar only.'}
    safe_write(root/REPORT_FILE,wire(report))
    print('R17_REPORT=',root/REPORT_FILE,'SHA256=',sha(wire(report)),flush=True)
    for group,v in summary.items():print('R17_GROUP=',group,'TOTAL=',v['total'],
      'RAW_CORRECT=',v['raw_correct'],'STRUCTURED_CORRECT=',v['structured_correct'],
      'LEXICAL_CORRECT=',v['lexical_correct'],'RAW_REFUSED=',v['raw_refused'],flush=True)
    print('R17=PASS_DEV_DESCRIPTIVE_NOT_SEMANTIC_QUAL',flush=True)


def main():
    require(len(sys.argv)==3 and sys.argv[1] in ('prepare','infer','score'),'USAGE')
    {'prepare':prepare,'infer':infer,'score':score}[sys.argv[1]](Path(sys.argv[2]))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('R17=HOLD',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        sys.exit(3)
