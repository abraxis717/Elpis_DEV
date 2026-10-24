#!/usr/bin/env python3
"""H1 DEV-only controls. A simple symbolic learner is fit strictly on TRAIN labels.

It is intentionally a strong baseline and a falsifier of HCA-specific benefit.
"""
import collections, json, pathlib, sys


def load(p):return [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]


def closure(row, rels):
    nodes={n['key']:n for n in row['graph']['nodes']}
    a=set(row['query']['seeds'])
    while True:
        b=set(a)
        for e in row['graph']['factors']:
            if e['rel'] not in rels or e['status']!='ADMITTED':continue
            u,v,w=e['members']
            if u in a and v in a and all(nodes[z]['source']=='VALID' for z in (u,v,w)):
                b.add(w)
        if b==a:break
        a=b
    return int(row['query']['target'] in a)


def run(root):
    root=pathlib.Path(root)
    train=load(root/'TRAIN.jsonl');dev=load(root/'DEV_PUBLIC.jsonl')
    g={row['id']:row['label'] for row in load(root/'DEV_GOLD.jsonl')}
    # This train-only model selects a set of one or two relation types by minimizing
    # TRAIN classification mistakes. Candidate class is declared in the protocol.
    candidates=[(i,) for i in (1,2,3,4)] + [(i,j) for i in (1,2,3,4) for j in (i+1,i+2,i+3) if j<=4]
    ranked=[]
    for rels in candidates:
        wrong=sum(closure(x,rels)!=x['label'] for x in train)
        ranked.append((wrong,len(rels),rels))
    wrong,_,choice=min(ranked)
    dev_correct=sum(closure(x,choice)==g[x['id']] for x in dev)
    truth_correct=sum(closure(x,(1,2))==g[x['id']] for x in dev)
    true_counts=collections.Counter(x['label'] for x in train)
    majority=min((-n,label) for label,n in true_counts.items())[1]
    majority_correct=sum(g[x['id']]==majority for x in dev)
    report={'schema':'elpis.hca.system.h1.controls.dev.v1',
            'train_only_symbolic_model':{'candidate_relation_sets':len(candidates),'chosen_relation_types':choice,
                                          'train_wrong':wrong,'dev_correct':dev_correct,'dev_total':len(dev)},
            'known_rule_exact_upper_bound':{'dev_correct':truth_correct,'dev_total':len(dev),'rule_available_to_B3':False},
            'majority_no_graph_control':{'majority_label':majority,'dev_correct':majority_correct,'dev_total':len(dev)},
            'primary_assay_scientifically_discriminating':dev_correct<len(dev),
            'interpretation':'If simple TRAIN-only symbolic rule induction is perfect, this assay is a necessary compositional test, not evidence of HCA-specific advantage.',
            'classification':'DEV_DESCRIPTIVE_NOT_SCIENTIFIC_QUAL'}
    (root/'CONTROLS.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('H1_CONTROLS='+json.dumps(report,sort_keys=True),flush=True)
    if dev_correct==len(dev):
        print('H1_SCIENCE_PRIMARY_HOLD=SYMBOLIC_RULE_BASELINE_SOLVES_ASSAY; BUILD_H1B_DYNAMIC_WORKLOAD_BEFORE_B3_CLAIMS',flush=True)
    else:
        print('H1_SCIENCE_PRIMARY_STATUS=INVESTIGATE_CONTROLS',flush=True)


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('usage: h1_controls.py OUTPUT_DIR')
    run(sys.argv[1])
