"""R19 prospective DEV: fixed-root discriminative composition, externally isolated scoring.

R18 is frozen. Formal truth and parser are authored and supplied, NOT learned. A
separate raw-token n-gram baseline sees exactly the text and is never parsed.
No LLM, native code edits, QUAL access, or semantic authority promotion.
"""
from __future__ import annotations
import hashlib, json, math, re, sys
from pathlib import Path
from . import r18_bracket_binding as r

SCHEMA='elpis.ecsc.r19.fixed-root.v1'
TRAIN='R19_TRAIN.json'
PUBLIC='R19_EVAL_PUBLIC.json'
GOLD='R19_EVAL_GOLD.json'
MODEL='R19_MODELS.json'
PRED='R19_PREDICTIONS.json'
REPORT='R19_DEV_FIXED_ROOT_REPORT.json'
PINS={TRAIN:'98a694a4e7f07bc4e763d35b67afeaed293619df21be34268c0fcffb0bb2615b',
      PUBLIC:'d8f3cbd00a1e0828bc2c9661413afcc182849e7a6117f135bee3df2cec01bcb6',
      GOLD:'3d19640300c6fd961b9d4c419f17617a50583714b2d6d0c456d64835060fc6ac'}
R18_SOURCE_SHA='14173dd48bd9596da67f3f83620d4ef6a336564d1bc2a9547da9eb6945a72a4d'
R18_AUDIT_SOURCE_SHA='2182fe1f8c8ec1b163a1b54c459c586a1cf33346b18e68377a0eabdd3f125859'
R18_AUDIT_REPORT_SHA='9fb6a313cfa093feead709ba128a54b84b57c7c6b724a98041948ed3f007686d'
R18_MODEL_SHA='6eb81f038f0ef9e89367a73767db6cae5798ea7710e3e93f74d7123308ee8d63'
R18_REPORT_SHA='38b1b68c429f875688509c4ffaba6c2d427f2366eb927f07b5dc2717758dea4a'
GROUPS=(('fixed_root_pairs',256),('deep_fixed_root_pairs',128),('primitive_generalization',96),('unknown_connective_refusal',32),('malformed_parenthesis_refusal',32))
TRAIN_NAMES=('Ariana','Basil','Clara','Dorian','Elias','Freya','Galen','Hector','Irene','Jasper','Keira','Loren','Mason','Nora','Olive','Pavel','Quentin','Rosa','Selma','Tomas','Una','Vera','Wendy','Xavier','Yara','Zelda','Bruno','Celia','Daphne','Edgar','Fiona','Greta')
EVAL_NAMES=('Mira','Jonah','Nadia','Samir','Tessa','Otto','Linnea','Mabel','Rohan','Paula','Dario','Ilona','Naomi','Sofia','Ingrid','Talia','Zubin','Karla','Piero','Marlon','Raquel','Hugo','Adela','Carmen','Beatriz','Lucia','Helena','Pedro','Victor','Felix','Silvia','Berta')
TRAIN_ITEMS=('copper disk','silver pin','green book','blue ribbon','glass token','iron badge','black card','white stone','cobalt tile','amber cup','crimson scarf','bronze ring','flax pouch','cedar box','opal lens','jade coin','olive tag','ivory page','tin buckle','purple cord','gray tablet','stone token','aqua clip','wooden peg','pink letter','velvet sheet','plum gem','coral key','cotton bag','linen seal','navy pouch','rubber stamp')
EVAL_ITEMS=('red ledger','violet folder','gold lantern','marble ring','amber key','white satchel','saffron book','platinum pin','copper stamp','lapis token','purple lens','sandstone tile','indigo page','orchid card','silver coin','azure feather','bronze candle','yellow badge','scarlet ribbon','teal coaster','cream disk','orange folder','jade satchel','green tablet','pink pendant','wooden dial','gray pouch','crimson box','blue button','opal ribbon','amber leaf','ivory tablet')
RAW_DIM=1024
RAW_EPOCHS=16
RAW_RATE=.065

class Hold(RuntimeError):pass

def require(cond,code):
    if not cond:raise Hold('R19_'+code)

def wire(obj):return r.wire(obj)
def sha(x):return hashlib.sha256(x).hexdigest()
def stable(*items):return int(sha('|'.join(map(str,items)).encode())[:15],16)
def root_for(work):return Path(work)/r.EVIDENCE_DIR

def fixture_row(tree,split,group,index,side=0):
    return {'id':sha(f'r19|{split}|{group}|{index}|{side}'.encode())[:24],
            'group':group,'pair_id':f'{split}.{group}.{index}','side':side,'text':r.render(tree)+'.'}

def leaves(index,train,how_many=5):
    names=TRAIN_NAMES if train else EVAL_NAMES
    items=TRAIN_ITEMS if train else EVAL_ITEMS
    off=stable('r19',int(train),index)
    base=(off+index*7)%len(names)
    vals=[]
    for j in range(how_many):
        n=names[(base+j*3)%len(names)]
        item=items[(base+index+j*5)%len(items)]
        vals.append((n,item))
    require(len({n+'|'+s for n,s in vals})==how_many,'LEAF_DISTINCT')
    return vals

def discriminative_pair(index,train,deep=False):
    # BOTH sides have the exact same ordered non-bracket tokens and the SAME
    # outer connective. Only the internal grouping changes the formal result.
    outer=('and','or')[(index//4)%2]
    core=('or_and','and_or')[(index//2)%2]
    middle=index%2
    names=leaves(index+ (10000 if deep else 0),train)
    (an,ai),(bn,bi),(cn,ci),(dn,di),(en,ei)=names
    if core=='or_and':
        a=r.leaf(an,ai,1);b=r.leaf(bn,bi,middle);c=r.leaf(cn,ci,0)
        x=('and',('or',a,b),c)
        y=('or',a,('and',b,c))
    else:
        a=r.leaf(an,ai,0);b=r.leaf(bn,bi,middle);c=r.leaf(cn,ci,1)
        x=('or',('and',a,b),c)
        y=('and',a,('or',b,c))
    if deep:
        e=r.leaf(en,ei,1)
        for _ in range(1+index%3):
            x=('and',e,('not',('not',x)))
            y=('and',e,('not',('not',y)))
    d=r.leaf(dn,di,int(outer=='and'))
    x=(outer,d,x);y=(outer,d,y)
    if (index//8)%2:
        x=('not',x);y=('not',y)
    group='deep_fixed_root_pairs' if deep else 'fixed_root_pairs'
    split='train' if train else 'eval'
    arow=fixture_row(x,split,group,index,0)
    brow=fixture_row(y,split,group,index,1)
    require(arow['text']!=brow['text'] and r.bare_tokens(arow['text'])==r.bare_tokens(brow['text']),'COUNTERFACTUAL_WORDS')
    require(r.val(x)!=r.val(y),'COUNTERFACTUAL_LABEL')
    parsed_a=r.parse(arow['text']);parsed_b=r.parse(brow['text'])
    for t in (parsed_a,parsed_b):
        while t[0]=='not':t=t[1]
        require(t[0]==outer,'ROOT_CHANGED')
    return ((arow,r.val(x)),(brow,r.val(y)))

def primitive(index,train):
    name1,item1,name2,item2=(leaves(index+23000,train,2)[0][0],leaves(index+23000,train,2)[0][1],leaves(index+23000,train,2)[1][0],leaves(index+23000,train,2)[1][1])
    code=index%12
    if code<2: t=r.leaf(name1,item1,code)
    elif code<4:t=('not',r.leaf(name1,item1,code-2))
    else:
        op='and' if code<8 else 'or'
        j=(code-4)%4
        t=(op,r.leaf(name1,item1,j//2),r.leaf(name2,item2,j%2))
    row=fixture_row(t,'train' if train else 'eval','primitive_generalization',index)
    return row,r.val(t)

def composite(index):
    # Diversify full-operator training while balancing labels within root type.
    ns=leaves(index+67000,True,4)
    bits=stable('bits',index)
    a,b,c,d=[r.leaf(n,item,(bits>>j)&1) for j,(n,item) in enumerate(ns)]
    inner=('and' if (index//2)%2 else 'or',a,b)
    other=('or' if (index//4)%2 else 'and',c,d)
    root=('and' if index%2 else 'or',inner,other)
    if index%7==0:root=('not',root)
    return fixture_row(root,'train','general_training',index),r.val(root)

def fixtures():
    train=[];public=[];gold=[]
    for i in range(128):
        for row,y in discriminative_pair(i,True):train.append({**row,'label':y})
    for i in range(192):
        row,y=primitive(i,True);train.append({**row,'label':y})
    for i in range(192):
        row,y=composite(i);train.append({**row,'label':y})
    for group,count,deep in (('fixed_root_pairs',128,False),('deep_fixed_root_pairs',64,True)):
        for i in range(count):
            for row,y in discriminative_pair(i,False,deep):
                public.append(row);gold.append({'id':row['id'],'group':group,'label':y,'refuse':False})
    for i in range(96):
        row,y=primitive(i+1000,False)
        public.append(row);gold.append({'id':row['id'],'group':row['group'],'label':y,'refuse':False})
    for group in ('unknown_connective_refusal','malformed_parenthesis_refusal'):
        for i in range(32):
            row,_=discriminative_pair(i+2000,False)[0]
            text=row['text']
            if group=='unknown_connective_refusal':
                text=text.replace(' and ',' xor ',1) if ' and ' in text else text.replace(' or ',' xor ',1)
            else:text=text.replace('(', '',1)
            require(text!=row['text'],'INVALID_UNCHANGED')
            out={'id':sha(f'r19|eval|{group}|{i}'.encode())[:24], 'group':group,'pair_id':group+'.'+str(i),'side':0,'text':text}
            public.append(out);gold.append({'id':out['id'],'group':group,'label':None,'refuse':True})
    require(len(train)==640,'TRAIN_CARDINALITY')
    require(len(public)==len(gold)==sum(n for _,n in GROUPS),'EVAL_CARDINALITY')
    require(len({row['id'] for row in train+public})==len(train)+len(public),'DUPLICATE_ID')
    require(not {row['text'] for row in train}&{row['text'] for row in public},'CROSS_SPLIT_DUPLICATE_TEXT')
    require(all('label' not in row and 'refuse' not in row for row in public),'LABEL_LEAK')
    for rows in (train,):
        pos=sum(row['label'] for row in rows)
        require(240<=pos<=400,'TRAIN_CLASS_SKEW')
    return train,public,gold

TOKEN_RE=re.compile(r'\(|\)|[a-z]+',re.I)
def raw_features(text):
    # No supplied parser / tree, no lexical truth knowledge, no oracle.
    toks=TOKEN_RE.findall(text.lower())
    feats={0:1.0}
    for n in (1,2,3):
        for j in range(len(toks)-n+1):
            key=str(n)+'|'+ '|'.join(toks[j:j+n]);bucket=1+int(sha(key.encode())[:8],16)%(RAW_DIM-1)
            feats[bucket]=feats.get(bucket,0.0)+1.0
    size=math.sqrt(sum(v*v for v in feats.values()))
    return tuple(sorted((j,v/size) for j,v in feats.items()))

def train_raw(train):
    dataset=[(row['id'],raw_features(row['text']),row['label']) for row in train]
    weights=[0.0]*RAW_DIM
    for epoch in range(RAW_EPOCHS):
        for ident,feats,y in sorted(dataset,key=lambda z:sha(f'{epoch}|{z[0]}'.encode())):
            score=sum(weights[j]*v for j,v in feats)
            p=r.sigmoid(score)
            err=(p-y)*RAW_RATE
            for j,v in feats:weights[j]-=err*v
    return {'architecture':'RAW_TOKEN_1_3GRAM_HASH_LOGISTIC_NO_PARSER','dimensions':RAW_DIM,'epochs':RAW_EPOCHS,'rate':RAW_RATE,
            'weights':[round(w,10) for w in weights]}

def predict_raw(model,text):
    feats=raw_features(text)
    return r.sigmoid(sum(model['weights'][j]*v for j,v in feats))

def source_integrity(work):
    parent=Path(work)/'research/ecsc_hca_r0';e=root_for(work)
    for path,expected in ((parent/'r18_bracket_binding.py',R18_SOURCE_SHA),(parent/'r18_shortcut_audit.py',R18_AUDIT_SOURCE_SHA),
                          (e/'R18_RETROSPECTIVE_SHORTCUT_AUDIT.json',R18_AUDIT_REPORT_SHA),
                          (e/'R18_REDUCER_MODEL.json',R18_MODEL_SHA),(e/'R18_DEV_BINDING_REPORT.json',R18_REPORT_SHA)):
        require(r.digest_for(path)==expected,'PRIOR_DRIFT:'+path.name)
    r.provenance(e)

def read(root,name,kind,expected=None):
    return r.read_json(root/name,SCHEMA+'.'+kind,expected)

def prepare(work):
    source_integrity(work)
    root=root_for(work);train,public,gold=fixtures()
    for name,kind,records in ((TRAIN,'train',train),(PUBLIC,'public',public),(GOLD,'gold',gold)):
        payload=wire({'schema':SCHEMA+'.'+kind,'records':records})
        require(sha(payload)==PINS[name],'FIXTURE_DRIFT:'+name)
        r.write_once(root/name,payload)
        print('R19_FROZEN=',name,'SHA256=',sha(payload),'COUNT=',len(records),flush=True)

def infer(work):
    source_integrity(work)
    root=root_for(work)
    data,th=read(root,TRAIN,'train',PINS[TRAIN]);public,ph=read(root,PUBLIC,'public',PINS[PUBLIC])
    # No GOLD file access in this process. Handwritten grammar is available
    # to supplied-parser branch ONLY, never to raw-token branch.
    structured=r.fit(data['records'])
    raw=train_raw(data['records'])
    model={'schema':SCHEMA+'.models','train_sha256':th,'public_sha256':ph,'structured':structured,'raw':raw,'gold_read':False}
    rows=[]
    for row in public['records']:
        raw_p=predict_raw(raw,row['text'])
        try:
            struct_p=r.predict(structured,row)
            t=r.parse(row['text']);while_not=0
            while t[0]=='not':t=t[1];while_not+=1
            root_only=(int(t[0]=='or') ^ (while_not%2)) if t[0] in ('and','or') else None
            structured_status='ANSWER';str_ans=int(struct_p>=.5)
        except r.Hold:
            struct_p=None;structured_status='REFUSED';str_ans=None;root_only=None
        rows.append({'id':row['id'],'structured_status':structured_status,'structured_answer':str_ans,
                     'structured_probability':None if struct_p is None else round(struct_p,9),
                     'root_only_answer':root_only,
                     'raw_status':'ANSWER','raw_answer':int(raw_p>=.5),'raw_probability':round(raw_p,9)})
    pred={'schema':SCHEMA+'.predictions','model_sha256':sha(wire(model)),
          'train_sha256':th,'public_sha256':ph,'records':rows,'qual_opened':False}
    r.write_once(root/MODEL,wire(model));r.write_once(root/PRED,wire(pred))
    print('R19_PREDICTIONS=',len(rows),'MODEL_SHA256=',sha(wire(model)),'PRED_SHA256=',sha(wire(pred)),flush=True)

def score(work):
    source_integrity(work)
    root=root_for(work)
    train,th=read(root,TRAIN,'train',PINS[TRAIN]);public,ph=read(root,PUBLIC,'public',PINS[PUBLIC])
    model,mh=read(root,MODEL,'models');pred,predh=read(root,PRED,'predictions')
    require(model['train_sha256']==th and model['public_sha256']==ph,'MODEL_BOUND')
    require(pred['model_sha256']==mh and pred['train_sha256']==th and pred['public_sha256']==ph,'PRED_BOUND')
    replay_struct=r.fit(train['records']);replay_raw=train_raw(train['records'])
    require(replay_struct==model['structured'] and replay_raw==model['raw'],'TRAIN_REPLAY')
    require(len(public['records'])==len(pred['records'])==sum(n for _,n in GROUPS),'PRED_COUNT')
    # Replay every prediction without consulting gold; fail on drift.
    for row,out in zip(public['records'],pred['records']):
        require(row['id']==out['id'],'ORDER')
        raw_p=predict_raw(replay_raw,row['text'])
        require(out['raw_status']=='ANSWER' and out['raw_answer']==int(raw_p>=.5) and out['raw_probability']==round(raw_p,9),'RAW_REPLAY')
        try:
            p=r.predict(replay_struct,row);t=r.parse(row['text']);wrap=0
            while t[0]=='not':t=t[1];wrap+=1
            root_ans=(int(t[0]=='or') ^ (wrap%2)) if t[0] in ('and','or') else None
            require(out['structured_status']=='ANSWER' and out['structured_answer']==int(p>=.5) and out['structured_probability']==round(p,9) and out['root_only_answer']==root_ans,'STRUCT_REPLAY')
        except r.Hold:
            require(out['structured_status']=='REFUSED' and out['structured_answer'] is None and out['structured_probability'] is None and out['root_only_answer'] is None,'REFUSAL_REPLAY')
    gold,gh=read(root,GOLD,'gold',PINS[GOLD])
    require(len(gold['records'])==len(public['records']),'GOLD_COUNT')
    summary={g:{'total':0, 'structured_correct':0,'raw_correct':0,'root_correct':0,'structured_refused':0,'raw_refused':0} for g,_ in GROUPS}
    pairs={}
    for src,predrow,target in zip(public['records'],pred['records'],gold['records']):
        require(src['id']==predrow['id']==target['id'] and src['group']==target['group'],'GOLD_ALIGN')
        g=summary[src['group']];g['total']+=1
        if target['refuse']:
            require(target['label'] is None,'GOLD_REFUSAL_LABEL')
            g['structured_correct']+=int(predrow['structured_status']=='REFUSED')
            g['raw_correct']+=int(predrow['raw_status']=='REFUSED')
            g['root_correct']+=int(predrow['root_only_answer'] is None)
        else:
            y=target['label'];require(type(y)==int and y in (0,1),'GOLD_LABEL')
            g['structured_correct']+=int(predrow['structured_status']=='ANSWER' and predrow['structured_answer']==y)
            g['raw_correct']+=int(predrow['raw_status']=='ANSWER' and predrow['raw_answer']==y)
            g['root_correct']+=int(predrow['root_only_answer']==y)
            if src['group'] in ('fixed_root_pairs','deep_fixed_root_pairs'):
                pairs.setdefault(src['pair_id'],[]).append((src['group'],y,predrow))
        g['structured_refused']+=int(predrow['structured_status']=='REFUSED')
        g['raw_refused']+=int(predrow['raw_status']=='REFUSED')
    require(all(summary[g]['total']==n for g,n in GROUPS),'GROUP_COUNTS')
    pairgroups={g:{method:{'both':0,'one':0,'neither':0} for method in ('structured','raw','root')} for g in ('fixed_root_pairs','deep_fixed_root_pairs')}
    for pid,arr in pairs.items():
        require(len(arr)==2 and arr[0][0]==arr[1][0] and arr[0][1]!=arr[1][1],'PAIR_INVALID')
        g=arr[0][0]
        for method in ('structured','raw','root'):
            matched=0
            for _,y,pr in arr:
                if method=='root':got=pr['root_only_answer']
                else:got=pr[f'{method}_answer']
                matched+=int(got==y and (method=='root' or pr[f'{method}_status']=='ANSWER'))
            pairgroups[g][method][('both' if matched==2 else 'one' if matched==1 else 'neither')]+=1
    require(len(pairs)==192,'PAIR_COUNT')
    for g in pairgroups:
        require(pairgroups[g]['root']['both']==0,'ROOT_BASELINE_CHEATED')
        require(pairgroups[g]['raw']['both']<=pairgroups[g]['raw']['one']+pairgroups[g]['raw']['both'],'RAW_ACCOUNTING')
    report={'schema':SCHEMA+'.report','status':'DEV_DESCRIPTIVE_NOT_SCIENTIFIC_QUAL',
            'historical_r18_audit_sha256':R18_AUDIT_REPORT_SHA,
            'train_sha256':th,'public_sha256':ph,'gold_sha256':gh,'model_sha256':mh,'prediction_sha256':predh,
            'groups':summary,'pairs':pairgroups,'supplied_parser':True,'raw_baseline_parser_used':False,
            'candidate_relation_admission':False,'llm_used':False,'qual_opened':False,
            'nonclaims':['Synthetic formal Boolean task only','Parser is externally authored','Leaf grammar is supplied',
                         'Raw-token baseline is a fixed ngram classifier, not an optimal parser-free learner']}
    output=wire(report);r.write_once(root/REPORT,output)
    print('R19_REPORT=',root/REPORT,'SHA256=',sha(output),flush=True)
    for name,s in summary.items():
        print('R19_GROUP=',name,'TOTAL=',s['total'],'STRUCTURED=',s['structured_correct'],
              'RAW=',s['raw_correct'],'ROOT=',s['root_correct'],'STRUCT_REFUSED=',s['structured_refused'],'RAW_REFUSED=',s['raw_refused'],flush=True)
    for name,s in pairgroups.items():
        for method,v in s.items():
            print('R19_PAIRS=',name,'METHOD=',method,'BOTH=',v['both'],'ONE=',v['one'],'NEITHER=',v['neither'],flush=True)
    print('R19=PASS_DEV_DESCRIPTIVE_NO_SEMANTIC_QUAL',flush=True)

def main():
    require(len(sys.argv)==3 and sys.argv[1] in ('prepare','infer','score'),'USAGE')
    {'prepare':prepare,'infer':infer,'score':score}[sys.argv[1]](Path(sys.argv[2]))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('R19=HOLD',type(exc).__name__,str(exc),file=sys.stderr,flush=True);sys.exit(3)
