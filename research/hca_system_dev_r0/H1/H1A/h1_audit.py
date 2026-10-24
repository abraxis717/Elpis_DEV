#!/usr/bin/env python3
"""Independent H1 DEV contract audit. Does NOT import generator, Elpis, or prior validators."""
import collections, hashlib, json, pathlib, sys


def load(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def need(condition,code):
    if not condition:raise ValueError('H1_AUDIT_HOLD_'+code)


def closure(row,rels=(1,2)):
    nodes={n['key']:n for n in row['graph']['nodes']}
    need(len(nodes)==len(row['graph']['nodes']),'NODE_DUP')
    known=set(row['query']['seeds'])
    need(all(k in nodes and nodes[k]['source'] in ('VALID','AUTHORITY_ONLY') for k in known),'SEED_INVALID')
    while True:
        next_set=set(known)
        for e in row['graph']['factors']:
            if e['rel'] not in rels or e['status']!='ADMITTED':continue
            members=e['members']
            need(len(members)==3,'FACTOR_ARITY')
            need(all(m in nodes for m in members),'FACTOR_MISSING_NODE')
            if all(x in known for x in members[:2]) and all(nodes[x]['source'] in ('VALID','AUTHORITY_ONLY') for x in members):
                next_set.add(members[2])
        if next_set==known:break
        known=next_set
    return int(row['query']['target'] in known)


def signature(row):
    # View with no graph adjacency: only raw relation/status distributions and per-node degree profiles.
    nodes={n['key']:n for n in row['graph']['nodes']}
    indeg=collections.Counter();outdeg=collections.Counter()
    for e in row['graph']['factors']:
        for u in e['members'][:2]:indeg[u]+=1
        outdeg[e['members'][2]]+=1
    degree=sorted((nodes[k]['kind'],nodes[k]['source'],indeg[k],outdeg[k]) for k in nodes)
    return (tuple(sorted((e['rel'],e['status']) for e in row['graph']['factors'])), tuple(degree),
            tuple(sorted((n['kind'],n['source']) for n in row['graph']['nodes'])), tuple(row['query']['seeds']),row['query']['target'])


def audit(run):
    run=pathlib.Path(run)
    train=load(run/'TRAIN.jsonl'); pub=load(run/'DEV_PUBLIC.jsonl'); gold=load(run/'DEV_GOLD.jsonl'); pairs=load(run/'PAIR_AUDIT_ONLY.jsonl')
    need(len(train)==288 and len(pub)==160 and len(gold)==160 and len(pairs)==224,'COUNTS')
    labels={x['id']:x['label'] for x in gold}
    need(len(labels)==len(gold),'GOLD_DUP')
    t_ids={x['id'] for x in train};d_ids={x['id'] for x in pub}
    need(len(t_ids)==len(train) and len(d_ids)==len(pub) and not (t_ids&d_ids),'IDS')
    need(all('label' not in x and 'family' not in x and 'side' not in x and 'pair_id' not in x for x in pub),'DEV_LABEL_LEAK')
    need(all(x['label'] in (0,1) for x in train) and all(x in (0,1) for x in labels.values()),'LABEL_RANGE')
    rows={x['id']:x for x in train+pub}
    need(len(rows)==len(train)+len(pub),'ALL_IDS')
    need(set(labels)==d_ids,'PUBLIC_GOLD_MISMATCH')
    hashes={'train':set(),'dev':set()}
    for x in train+pub:
        split='train' if x['id'] in t_ids else 'dev'
        nodes=x['graph']['nodes'];edges=x['graph']['factors']
        need(len(nodes)==20 and len(edges)==20,'GRAPH_SIZE')
        need(len(set(n['key'] for n in nodes))==20,'NODE_KEYS')
        need(nodes==sorted(nodes,key=lambda n:n['key']),'NODE_ORDER')
        need(edges==sorted(edges,key=lambda e:(e['rel'],tuple(e['members']),e['status'])),'EDGE_ORDER')
        need(x['query']['target'] in {n['key'] for n in nodes},'TARGET')
        hashes[split].update(n['key'] for n in nodes)
        y=closure(x)
        expected=x['label'] if split=='train' else labels[x['id']]
        need(y==expected,'ORACLE_DISAGREE')
    need(not(hashes['train']&hashes['dev']),'NODE_ID_SPLIT_LEAK')
    by_split=collections.defaultdict(lambda:collections.Counter())
    for p in pairs:
        split=p['split'];fam=p['family'];ids=p['row_ids']
        need(len(ids)==2 and ids[0]!=ids[1] and all(i in rows for i in ids),'PAIR_IDENTITY')
        need(all((i in t_ids)==(split=='train') for i in ids),'PAIR_SPLIT')
        y=[rows[i]['label'] if i in t_ids else labels[i] for i in ids]
        expected=[1,0] if fam=='structural' else [0,0]
        need(y==expected,'PAIR_GOLD')
        first,second=rows[ids[0]],rows[ids[1]]
        need(first['graph']['nodes']==second['graph']['nodes'],'PAIR_NODE_DRIFT')
        need(first['query']==second['query'],'PAIR_QUERY_DRIFT')
        need(signature(first)==signature(second),'MARGINAL_SHORTCUT')
        e0=first['graph']['factors'];e1=second['graph']['factors']
        need(e0!=e1,'NO_STRUCTURAL_INTERVENTION')
        # Difference in R2 output bindings only; all other facts must be byte-identical.
        need([e for e in e0 if e['rel']!=2]==[e for e in e1 if e['rel']!=2],'EXTRA_PAIR_DRIFT')
        by_split[split][fam]+=1
    need(by_split['train']=={'structural':128,'source_ablation':16},'TRAIN_FAMILIES')
    need(by_split['dev']=={'structural':64,'source_ablation':16},'DEV_FAMILIES')
    # Non-graph structural/marginal-only baseline is forced to tie on every opposing-label pair.
    report={'schema':'elpis.hca.system.h1.audit.v1','status':'PASS_TASK_CONTRACT_DEV_NOT_QUAL',
            'rows':{'train':len(train),'dev_public':len(pub)},'pairs':{s:dict(x) for s,x in by_split.items()},
            'oracle_exact':len(train)+len(pub),'marginal_only_both_correct_structural_pairs':0,
            'leakage_checks':'PASS','independent_oracle':'PASS','formal_qual':'NOT_RUN'}
    (run/'AUDIT.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print('PASS_HCA_H1_DEV_TASK_CONTRACT_NO_SCIENCE',flush=True)
    print('H1_AUDIT='+json.dumps(report,sort_keys=True),flush=True)

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('usage: h1_audit.py OUTPUT_RUN_DIRECTORY')
    audit(sys.argv[1])
