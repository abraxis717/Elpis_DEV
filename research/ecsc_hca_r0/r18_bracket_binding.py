"""R18: prospective paired bracket-binding DEV experiment.

Two surfaces per pair have the same ordered non-bracket tokens, but opposite
formal-grammar truth values. A grouping-blind predictor cannot classify both.
A trainable stack reducer receives an explicitly authored grammar/parser;
it learns leaf/operator tables, *not* the parser or natural language semantics.
No LLM, QUAL, canonical change, R17 modification, or source-derived truth claim.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
import sys
from pathlib import Path

SCHEMA='elpis.ecsc.r18.paired-binding.v1'
EVIDENCE_DIR='research/ecsc_hca_r0/external/locomo_r0'
TRAIN_FILE='R18_TRAIN.json'
PUBLIC_FILE='R18_EVAL_PUBLIC.json'
GOLD_FILE='R18_EVAL_GOLD.json'
MODEL_FILE='R18_REDUCER_MODEL.json'
PRED_FILE='R18_PREDICTIONS.json'
REPORT_FILE='R18_DEV_BINDING_REPORT.json'
FIXTURE_PINS={
 TRAIN_FILE:'df5b3c9530c79bae312dc96c9d04ab13f8a9737a30f2502c4d2ea304f96ace77',
 PUBLIC_FILE:'8fc9c852e73a8621006ff02358ad75f26dbc4c3c0136cb14608b54475a9608f8',
 GOLD_FILE:'3589d1c17638983ad45c61553fca2f9fee95b4bfbc05b8134ce3d5d439a54dd1'}
R17_AUDIT_SHA='49a6a44715dca40a78b36c0395e29d0862deed23a6ba605324b8835e5a29153c'
R17_REPORT_SHA='457b12146b71da29f1e5e7396fcd73c1d08bde3227e54c258139e86569c927c0'
R17_MODEL_SHA='222be935b8232109605966b46aedb6ad58db6bb02f6962845c4f8ef1d63d9cfb'
TRAIN_PAIRS=256
NEUTRAL_PAIRS=64
EVAL_PAIRS=128
DEEP_PAIRS=32
EVAL_NEUTRAL_PAIRS=32
REFUSALS_PER_KIND=32
EPOCHS=110
RATE=0.065
BATCH=16
MAX_TEXT=1024
TRAIN_NAMES=('Ava','Theo','Noor','Iris','Rafi','Lea')
EVAL_NAMES=('Mira','Jonah','Nadia','Samir','Tessa','Otto')
TRAIN_ITEMS=('copper disk','silver pin','green book','blue ribbon','glass token','iron badge')
EVAL_ITEMS=('amber key','violet folder','red ledger','marble ring','gold lantern','white satchel')
GROUPS=(('paired_grouping_flip',EVAL_PAIRS*2),('deeper_grouping_flip',DEEP_PAIRS*2),
        ('same_operator_control',EVAL_NEUTRAL_PAIRS*2),
        ('unknown_connective_refusal',REFUSALS_PER_KIND),
        ('malformed_parenthesis_refusal',REFUSALS_PER_KIND))

class Hold(RuntimeError): pass

def require(ok,code):
    if not ok: raise Hold('R18_'+code)

def sha(data): return hashlib.sha256(data).hexdigest()
def wire(value): return (json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'))+'\n').encode()
def digest_for(path):
    require(path.is_file() and not path.is_symlink(),'SOURCE_MISSING:'+path.name)
    return sha(path.read_bytes())
def write_once(path,data):
    if path.exists() or path.is_symlink():
        require(path.is_file() and not path.is_symlink() and path.read_bytes()==data,'CONFLICT:'+path.name)
    else:
        with path.open('xb') as f:f.write(data)

def read_json(path,schema,expected=None):
    d=digest_for(path)
    require(expected is None or d==expected,'PIN_DRIFT:'+path.name)
    try: value=json.loads(path.read_bytes())
    except Exception as exc:raise Hold('R18_BAD_JSON:'+path.name) from exc
    require(type(value) is dict and value.get('schema')==schema,'SCHEMA:'+path.name)
    return value,d

def root_for(work): return Path(work)/EVIDENCE_DIR

def stable(*x):return int(sha('|'.join(map(str,x)).encode())[:14],16)

def leaf(name,item,value): return ('leaf',name,item,'owns' if value else 'does not own')
def val(tree):
    op=tree[0]
    if op=='leaf':return int(tree[3]=='owns')
    if op=='not':return 1-val(tree[1])
    if op=='and':return val(tree[1]) & val(tree[2])
    if op=='or':return val(tree[1]) | val(tree[2])
    raise AssertionError('oracle-unknown')

def render(tree):
    op=tree[0]
    if op=='leaf':return tree[1]+' '+tree[3]+' the '+tree[2]
    if op=='not':return 'not ('+render(tree[1])+')'
    return '('+render(tree[1])+' '+op+' '+render(tree[2])+')'

def bare_tokens(text):
    # Counterfactual control: exact ordered-token identity, excluding only brackets.
    return tuple(re.findall(r'[A-Za-z]+',text.lower()))

def pair(group,index,train):
    names=TRAIN_NAMES if train else EVAL_NAMES
    items=TRAIN_ITEMS if train else EVAL_ITEMS
    offset=stable('r18',group,index)
    ns=[names[(offset+j)%len(names)] for j in range(3)]
    it=[items[(offset+2*j)%len(items)] for j in range(3)]
    distinct=[leaf(ns[j],it[j],0) for j in range(3)]
    require(len({render(x) for x in distinct})==3,'LEAF_COLLISION')
    choice=(offset//19)%2
    if group=='neutral':
        op1=op2='and' if choice else 'or'
        a,b,c=tuple(leaf(ns[j],it[j],(offset>>(j+4))&1) for j in range(3))
    elif choice:
        op1,op2='and','or'
        a=leaf(ns[0],it[0],0);b=leaf(ns[1],it[1],(offset>>8)&1);c=leaf(ns[2],it[2],1)
    else:
        op1,op2='or','and'
        a=leaf(ns[0],it[0],1);b=leaf(ns[1],it[1],(offset>>8)&1);c=leaf(ns[2],it[2],0)
    # Same leaf and connective order in both members; only parentheses move.
    left=(op2,(op1,a,b),c)
    right=(op1,a,(op2,b,c))
    wraps=(index%3 if train else (3+index%3 if group=='deep' else index%3))
    for _ in range(wraps):left=('not',left);right=('not',right)
    require(bare_tokens(render(left))==bare_tokens(render(right)),'PAIR_LEXICAL_DRIFT')
    require((val(left)!=val(right))==(group!='neutral'),'PAIR_LABEL_CONTRACT')
    def make(t,j):
        rid=sha(f'r18|{group}|{int(train)}|{index}|{j}'.encode())[:24]
        return {'id':rid,'group':group,'pair_id':f'{group}.{index}','side':j,'text':render(t)+'.'}
    return (make(left,0),val(left)),(make(right,1),val(right))

def fixtures():
    train=[];public=[];gold=[]
    for g,n in (('flip',TRAIN_PAIRS),('neutral',NEUTRAL_PAIRS)):
        for i in range(n):
            for row,y in pair(g,i,True):train.append({**row,'label':y})
    for g,n,label in (('flip',EVAL_PAIRS,'paired_grouping_flip'),('deep',DEEP_PAIRS,'deeper_grouping_flip'),('neutral',EVAL_NEUTRAL_PAIRS,'same_operator_control')):
        for i in range(n):
            for row,y in pair(g,i,False):
                row['group']=label
                public.append(row)
                gold.append({'id':row['id'],'group':label,'label':y,'refuse':False})
    for group in ('unknown_connective_refusal','malformed_parenthesis_refusal'):
        for i in range(REFUSALS_PER_KIND):
            row,y=pair('flip',i+1000,False)[0]
            t=row['text']
            if group=='unknown_connective_refusal':t=t.replace(' and ',' xor ',1) if ' and ' in t else t.replace(' or ',' xor ',1)
            else:t=t.replace('(', '',1)
            require(t!=row['text'],'REFUSAL_NOT_MUTATED')
            row={**row,'id':sha(f'r18|{group}|{i}'.encode())[:24],'group':group,'pair_id':group+'.'+str(i),'side':0,'text':t}
            public.append(row);gold.append({'id':row['id'],'group':group,'label':None,'refuse':True})
    require(len(train)==(TRAIN_PAIRS+NEUTRAL_PAIRS)*2,'TRAIN_COUNT')
    require(len(public)==len(gold)==sum(c for _,c in GROUPS),'EVAL_COUNT')
    require(len({r['id'] for r in train+public})==len(train)+len(public),'DUPLICATE_ID')
    require(not {r['text'] for r in train}&{r['text'] for r in public},'SPLIT_DUPLICATE_TEXT')
    return train,public,gold

# Explicitly SUPPLIED grammar. Not learned. All reductions below use this parse.
LEXER=re.compile(r'[A-Z][a-z]+|[a-z]+|[()]')
LEAF_RE=re.compile(r'(?P<name>[A-Z][a-z]+) (?P<verb>owns|does not own) the (?P<item>[a-z]+(?: [a-z]+){0,2})\Z')

def parse(text):
    require(type(text) is str and 0<len(text)<=MAX_TEXT and text.endswith('.'),'INPUT_BOUND')
    source=text[:-1]
    tokens=LEXER.findall(source)
    require(' '.join(source.split()).replace(' ','')==''.join(tokens),'LEXICAL_INVALID')
    require(len(tokens)<150,'TOKEN_LIMIT')
    def sub(i,depth):
        require(depth<=16 and i<len(tokens),'DEPTH_OR_EOF')
        t=tokens[i]
        if t=='not':
            require(i+1<len(tokens) and tokens[i+1]=='(','NOT_SYNTAX')
            inner,j=sub(i+2,depth+1)
            require(j<len(tokens) and tokens[j]==')','NOT_CLOSE')
            return ('not',inner),j+1
        if t=='(':
            left,j=sub(i+1,depth+1)
            require(j<len(tokens) and tokens[j] in ('and','or'),'BINARY_OPERATOR')
            op=tokens[j]
            right,k=sub(j+1,depth+1)
            require(k<len(tokens) and tokens[k]==')','BINARY_CLOSE')
            return (op,left,right),k+1
        # A leaf consumes until next connective/close; it never treats conjunction as a leaf.
        j=i
        while j<len(tokens) and tokens[j] not in ('and','or',')','(','xor'):
            j+=1
        clause=' '.join(tokens[i:j]);m=LEAF_RE.fullmatch(clause)
        require(m is not None,'LEAF_SYNTAX')
        return ('leaf',m.group('verb')),j
    tree,end=sub(0,0)
    require(end==len(tokens),'TRAILING_TOKENS')
    return tree

PARAMS=('leaf.owns','leaf.does not own','not.0','not.1')+tuple(f'{op}.{i}' for op in ('and','or') for i in range(4))

def sigmoid(v):return 1/(1+math.exp(-max(-30.,min(30.,v))))

def forward(theta,tree):
    """Probability plus per-node tape for exact reverse-mode differentiation."""
    op=tree[0]
    if op=='leaf':
        k='leaf.'+tree[1];p=sigmoid(theta[k]);return p,('leaf',k,p)
    x,tx=forward(theta,tree[1])
    if op=='not':
        t0=sigmoid(theta['not.0']);t1=sigmoid(theta['not.1'])
        p=(1-x)*t0+x*t1
        return p,('unary',x,tx,t0,t1)
    y,ty=forward(theta,tree[2])
    t=[sigmoid(theta[f'{op}.{i}']) for i in range(4)]
    weights=((1-x)*(1-y),(1-x)*y,x*(1-y),x*y)
    p=sum(w*v for w,v in zip(weights,t))
    return p,('binary',op,x,y,tx,ty,t,weights)

def backward(tape,delta,grad):
    kind=tape[0]
    if kind=='leaf':
        _,k,p=tape;grad[k]+=delta*p*(1-p)
    elif kind=='unary':
        _,x,tx,t0,t1=tape
        grad['not.0']+=delta*(1-x)*t0*(1-t0)
        grad['not.1']+=delta*x*t1*(1-t1)
        backward(tx,delta*(t1-t0),grad)
    else:
        _,op,x,y,tx,ty,t,w=tape
        for i in range(4):grad[f'{op}.{i}']+=delta*w[i]*t[i]*(1-t[i])
        dx=(1-y)*(t[2]-t[0])+y*(t[3]-t[1])
        dy=(1-x)*(t[1]-t[0])+x*(t[3]-t[2])
        backward(tx,delta*dx,grad);backward(ty,delta*dy,grad)

def fit(train):
    require(type(train) is list and len(train)==(TRAIN_PAIRS+NEUTRAL_PAIRS)*2,'TRAIN_SHAPE')
    require(len({r['id'] for r in train})==len(train),'TRAIN_DUPLICATE')
    parsed=[(r['id'],parse(r['text']),r['label']) for r in train]
    require(all(type(y) is int and y in (0,1) for _,_,y in parsed),'TRAIN_LABEL')
    theta={k:((stable('init',k)%1001)/1000-.5)*.24 for k in PARAMS}
    m={k:0. for k in PARAMS};v={k:0. for k in PARAMS};step=0
    for epoch in range(EPOCHS):
        order=sorted(parsed,key=lambda x:sha(f'{epoch}|{x[0]}'.encode()))
        for start in range(0,len(order),BATCH):
            chunk=order[start:start+BATCH]
            grads={k:0. for k in PARAMS}
            for _,tree,y in chunk:
                p,tape=forward(theta,tree)
                delta=(p-y)/max(1e-7,p*(1-p))
                backward(tape,delta,grads)
            step+=1
            for k in PARAMS:
                g=grads[k]/len(chunk)
                m[k]=.9*m[k]+.1*g;v[k]=.999*v[k]+.001*g*g
                mh=m[k]/(1-.9**step);vh=v[k]/(1-.999**step)
                theta[k]-=RATE*mh/(math.sqrt(vh)+1e-8)
    return {'schema':SCHEMA+'.model','architecture':'SUPPLIED_PARENTHESIS_PARSER_LEARNED_BOOLEAN_GATES',
            'training_examples':len(train),'epochs':EPOCHS,'batch':BATCH,'rate':RATE,
            'theta':{k:round(theta[k],10) for k in PARAMS}}

def predict(model,row):
    require(model['schema']==SCHEMA+'.model' and set(model['theta'])==set(PARAMS),'MODEL_INTEGRITY')
    p,_=forward(model['theta'],parse(row['text']))
    return p

def provenance(root):
    required={
      'R17_SCOPE_ABLATION_AUDIT.json':R17_AUDIT_SHA,
      'R17_DEV_SCOPE_REPORT.json':R17_REPORT_SHA,
      'R17_RAW_SEQUENCE_MODEL.json':R17_MODEL_SHA,
    }
    for name,expect in required.items():require(digest_for(root/name)==expect,'PRIOR_DRIFT:'+name)

def prepare(work):
    root=root_for(work);provenance(root)
    train,public,gold=fixtures()
    for name,suffix,records in ((TRAIN_FILE,'.train',train),(PUBLIC_FILE,'.public',public),(GOLD_FILE,'.gold',gold)):
        payload=wire({'schema':SCHEMA+suffix,'records':records})
        require(sha(payload)==FIXTURE_PINS[name],'FIXTURE_DRIFT:'+name)
        write_once(root/name,payload)
        print('R18_FROZEN=',name,'SHA256=',sha(payload),'COUNT=',len(records),flush=True)

def infer(work):
    root=root_for(work);provenance(root)
    train,th=read_json(root/TRAIN_FILE,SCHEMA+'.train',FIXTURE_PINS[TRAIN_FILE])
    public,ph=read_json(root/PUBLIC_FILE,SCHEMA+'.public',FIXTURE_PINS[PUBLIC_FILE])
    # No gold access anywhere in the predictor path.
    model=fit(train['records']);model.update({'train_sha256':th,'public_sha256':ph})
    predictions=[]
    for row in public['records']:
        try:
            p=predict(model,row);st='ANSWER';answer=int(p>=.5)
        except Hold:
            p=None;st='REFUSED';answer=None
        predictions.append({'id':row['id'],'status':st,'answer':answer,'p':None if p is None else round(p,9)})
    obj={'schema':SCHEMA+'.predictions','model_sha256':sha(wire(model)),
         'train_sha256':th,'public_sha256':ph,'rows':predictions,'qual_opened':False}
    write_once(root/MODEL_FILE,wire(model))
    write_once(root/PRED_FILE,wire(obj))
    print('R18_PREDICTED=',len(predictions),'MODEL_SHA256=',sha(wire(model)),'PRED_SHA256=',sha(wire(obj)),flush=True)

def score(work):
    root=root_for(work);provenance(root)
    train,th=read_json(root/TRAIN_FILE,SCHEMA+'.train',FIXTURE_PINS[TRAIN_FILE])
    public,ph=read_json(root/PUBLIC_FILE,SCHEMA+'.public',FIXTURE_PINS[PUBLIC_FILE])
    model,mh=read_json(root/MODEL_FILE,SCHEMA+'.model')
    pred,predh=read_json(root/PRED_FILE,SCHEMA+'.predictions')
    require(model['train_sha256']==th and model['public_sha256']==ph,'MODEL_BINDING')
    require(pred['model_sha256']==mh and pred['train_sha256']==th and pred['public_sha256']==ph,'PRED_BINDING')
    replay=fit(train['records']);replay.update({'train_sha256':th,'public_sha256':ph})
    require(wire(replay)==wire(model),'FIT_REPLAY')
    require(len(pred['rows'])==len(public['records'])==sum(v for _,v in GROUPS),'COUNT')
    for row,out in zip(public['records'],pred['rows']):
        try:p=predict(model,row);st='ANSWER';y=int(p>=.5)
        except Hold:p=None;st='REFUSED';y=None
        require(out=={'id':row['id'],'status':st,'answer':y,'p':None if p is None else round(p,9)},'PRED_REPLAY')
    # Gold is opened only after all prediction replays match.
    gold,gh=read_json(root/GOLD_FILE,SCHEMA+'.gold',FIXTURE_PINS[GOLD_FILE])
    require(len(gold['records'])==len(pred['rows']),'GOLD_COUNT')
    summary={g:{'total':0,'correct':0,'wrong':0,'refused':0} for g,_ in GROUPS}
    pairs={}
    for row,out,target in zip(public['records'],pred['rows'],gold['records']):
        require(row['id']==out['id']==target['id'] and row['group']==target['group'],'ALIGNMENT')
        s=summary[row['group']];s['total']+=1
        if target['refuse']:
            require(target['label'] is None,'REFUSAL_GOLD')
            s['refused']+=int(out['status']=='REFUSED')
            s['wrong']+=int(out['status']!='REFUSED')
        else:
            require(type(target['label']) is int and target['label'] in (0,1),'GOLD')
            s['correct']+=int(out['status']=='ANSWER' and out['answer']==target['label'])
            s['wrong']+=int(out['status']=='ANSWER' and out['answer']!=target['label'])
            s['refused']+=int(out['status']!='ANSWER')
            pairs.setdefault(row['pair_id'],[]).append((out['answer'],target['label'],out['status']))
    require(all(summary[g]['total']==n for g,n in GROUPS),'GROUP_CARDINALITY')
    both={g:{'pairs':0,'both_correct':0,'one_correct':0,'neither_correct':0} for g in ('paired_grouping_flip','deeper_grouping_flip','same_operator_control')}
    lookup={r['pair_id']:r['group'] for r in public['records']}
    for pair_id,p in pairs.items():
        require(len(p)==2,'PAIR_COUNT')
        g=lookup[pair_id];correct=sum(int(status=='ANSWER' and got==gold) for got,gold,status in p)
        both[g]['pairs']+=1
        both[g][('both_correct' if correct==2 else 'one_correct' if correct==1 else 'neither_correct')]+=1
    report={'schema':SCHEMA+'.report','disposition':'DEV_DESCRIPTIVE_NOT_SEMANTIC_QUAL',
            'r17_scope_audit_sha256':R17_AUDIT_SHA,'r17_model_sha256':R17_MODEL_SHA,
            'train_sha256':th,'public_sha256':ph,'gold_sha256':gh,
            'model_sha256':mh,'prediction_sha256':predh,
            'groups':summary,'pair_analysis':both,
            'grammar_supplied_to_predictor':True,'relation_admission':False,
            'llm_used':False,'qual_opened':False,
            'interpretation':'Formal bracket sensitivity with a supplied parser and trainable operation tables; not learned syntax.'}
    payload=wire(report);write_once(root/REPORT_FILE,payload)
    print('R18_REPORT=',root/REPORT_FILE,'SHA256=',sha(payload),flush=True)
    for k,s in summary.items():print('R18_GROUP=',k,'TOTAL=',s['total'],'CORRECT=',s['correct'],'WRONG=',s['wrong'],'REFUSED=',s['refused'],flush=True)
    for k,v in both.items():print('R18_PAIRS=',k,'PAIRS=',v['pairs'],'BOTH_CORRECT=',v['both_correct'],
                                   'ONE_CORRECT=',v['one_correct'],'NEITHER_CORRECT=',v['neither_correct'],flush=True)
    print('R18=PASS_DEV_DESCRIPTIVE_NO_SEMANTIC_QUAL',flush=True)

def main():
    require(len(sys.argv)==3 and sys.argv[1] in ('prepare','infer','score'),'USAGE')
    {'prepare':prepare,'infer':infer,'score':score}[sys.argv[1]](Path(sys.argv[2]))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('R18=HOLD',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        sys.exit(3)
