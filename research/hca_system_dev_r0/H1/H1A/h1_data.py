#!/usr/bin/env python3
"""H1 DEV counterfactual task generator; pure stdlib; no QUAL and no canonical mutation.

RUN H1 only through RUN_H1.sh. Gold separated from inference-visible DEV records.
"""
import hashlib, json, pathlib, random, sys

DOMAIN = 'elpis.hca.system.h1.dev.v1'
TRAIN_PAIRS = 128
TRAIN_ABLATIONS = 16
DEV_PAIRS = 64
DEV_ABLATIONS = 16
NOISE_NODES = 12
NOISE_FACTORS = 16
SCHEMA = 'elpis.hca.system.h1.structural.v1'


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def digest(x):
    return hashlib.sha256(x if isinstance(x, bytes) else x.encode('utf-8')).hexdigest()


def rand_seed(*parts):
    return int(digest('|'.join(map(str, (DOMAIN,)+parts)))[:16],16)


def names(split, family, idx):
    roles=['s','x','g','h','a','b','t','u'] + ['n%02d'%j for j in range(NOISE_NODES)]
    # Salts differ across splits and pairs; graph IDs have no stable lexical role label.
    mapping={role:digest('|'.join(map(str,(DOMAIN,'NODE',split,family,idx,role)))) for role in roles}
    return mapping


def graph(split,family,idx,side):
    assert split in ('train','dev') and family in ('structural','source_ablation') and side in (0,1)
    rng=random.Random(rand_seed('GRAPH',split,family,idx))
    m=names(split,family,idx)
    def node(role):
        typ=1 if role in ('s','x') else 2 if role in ('g','h') else 3 if role in ('a','b') else 4 if role in ('t','u') else 5
        state='STALE' if family=='source_ablation' and role=='a' else 'VALID'
        return {'key':m[role],'kind':typ,'source':state,'scope':1}
    nodes=sorted([node(role) for role in m],key=lambda n:n['key'])
    # Core: one reachable relay a, unreachable relay b. Swap only outputs of R2 in paired graphs.
    factors=[
        {'rel':1,'members':[m['s'],m['g'],m['a']],'status':'ADMITTED'},
        {'rel':1,'members':[m['x'],m['h'],m['b']],'status':'ADMITTED'},
        {'rel':2,'members':[m['a'],m['g'],m['t' if side==0 else 'u']],'status':'ADMITTED'},
        {'rel':2,'members':[m['b'],m['h'],m['u' if side==0 else 't']],'status':'ADMITTED'},
    ]
    noise=[m[f'n{j:02d}'] for j in range(NOISE_NODES)]
    for i in range(NOISE_FACTORS):
        chosen=rng.sample(noise,3)
        factors.append({'rel':3+i%2,'members':chosen,'status':'ADMITTED' if i%3 else 'CANDIDATE'})
    # The nodes and noise factors are identical within the pair; labels change solely with wiring.
    factors.sort(key=lambda f:(f['rel'],tuple(f['members']),f['status']))
    row={'schema':SCHEMA,'id':digest('|'.join(map(str,(DOMAIN,'ROW',split,family,idx,side))))[:28],
         'graph':{'nodes':nodes,'factors':factors},'query':{'seeds':sorted([m[x] for x in ('s','g','h')]),'target':m['t'],'scope':1}}
    return row


def gold(row):
    nodes={n['key']:n for n in row['graph']['nodes']}
    active=set(row['query']['seeds'])
    for k in active:
        if nodes[k]['source'] not in ('VALID','AUTHORITY_ONLY'):
            raise ValueError('invalid query seed')
    changed=True
    while changed:
        changed=False
        for f in row['graph']['factors']:
            if f['rel'] not in (1,2) or f['status']!='ADMITTED':continue
            a,b,out=f['members']
            if a in active and b in active and all(nodes[z]['source'] in ('VALID','AUTHORITY_ONLY') for z in (a,b,out)) and out not in active:
                active.add(out);changed=True
    return int(row['query']['target'] in active)


def write_jsonl(path, rows):
    with open(path,'x',encoding='utf-8',newline='\n') as f:
        for row in rows:f.write(canon(row)+'\n')


def make(out):
    out=pathlib.Path(out)
    if not out.is_dir():raise ValueError('preexisting unique run directory required')
    train=[]; pub=[]; dev_gold=[]; meta=[]
    for split,families in (('train', [('structural',TRAIN_PAIRS),('source_ablation',TRAIN_ABLATIONS)]),
                           ('dev',[('structural',DEV_PAIRS),('source_ablation',DEV_ABLATIONS)])):
        for fam,n in families:
            for idx in range(n):
                pair=[]
                for side in (0,1):
                    row=graph(split,fam,idx,side)
                    label=gold(row)
                    pair.append((row,label))
                    if split=='train': train.append({**row,'label':label})
                    else:
                        pub.append(row)
                        dev_gold.append({'id':row['id'],'label':label})
                expect=(1,0) if fam=='structural' else (0,0)
                if tuple(label for _,label in pair)!=expect:
                    raise ValueError(f'non-discriminating pair {split} {fam} {idx}')
                meta.append({'split':split,'family':fam,'row_ids':[pair[0][0]['id'],pair[1][0]['id']]})
    # Explicit shuffle: side ordering and pair adjacency cannot be assumed by downstream learners.
    random.Random(rand_seed('ORDER','train')).shuffle(train)
    rr=random.Random(rand_seed('ORDER','dev')); rr.shuffle(pub)
    # DEV gold kept in a different file and shuffled independently; only the scorer may read it.
    random.Random(rand_seed('ORDER','dev','gold')).shuffle(dev_gold)
    write_jsonl(out/'TRAIN.jsonl',train)
    write_jsonl(out/'DEV_PUBLIC.jsonl',pub)
    write_jsonl(out/'DEV_GOLD.jsonl',dev_gold)
    write_jsonl(out/'PAIR_AUDIT_ONLY.jsonl',meta)
    counts={'train':len(train),'dev_public':len(pub),'dev_gold':len(dev_gold),
            'train_structural_pairs':TRAIN_PAIRS,'train_ablation_pairs':TRAIN_ABLATIONS,
            'dev_structural_pairs':DEV_PAIRS,'dev_ablation_pairs':DEV_ABLATIONS}
    (out/'COUNTS.json').write_text(json.dumps(counts,sort_keys=True,indent=2)+'\n')
    print('H1_DATA_GENERATED='+json.dumps(counts,sort_keys=True),flush=True)


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('usage: h1_data.py OUTPUT_RUN_DIRECTORY')
    make(sys.argv[1])
