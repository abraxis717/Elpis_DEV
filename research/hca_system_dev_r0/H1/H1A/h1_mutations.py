#!/usr/bin/env python3
"""Negative controls for independent H1 auditor; mutate isolated DEV copies only."""
import copy, json, pathlib, shutil, subprocess, sys, tempfile

FILES=['TRAIN.jsonl','DEV_PUBLIC.jsonl','DEV_GOLD.jsonl','PAIR_AUDIT_ONLY.jsonl']

def load(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def write(p,rows):p.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in rows))

def mutate(kind,wd):
    def rows(f):return load(wd/f)
    if kind=='gold_label_flip':
        a=rows('DEV_GOLD.jsonl');a[0]['label']^=1;write(wd/'DEV_GOLD.jsonl',a)
    elif kind=='dev_label_leak':
        a=rows('DEV_PUBLIC.jsonl');a[0]['label']=1;write(wd/'DEV_PUBLIC.jsonl',a)
    elif kind=='pair_count_dropped':
        a=rows('PAIR_AUDIT_ONLY.jsonl');write(wd/'PAIR_AUDIT_ONLY.jsonl',a[1:])
    elif kind=='node_status_pair_drift':
        p=rows('PAIR_AUDIT_ONLY.jsonl');ident=next(x['row_ids'][0] for x in p if x['split']=='dev' and x['family']=='structural');
        a=rows('DEV_PUBLIC.jsonl');r=next(x for x in a if x['id']==ident);r['graph']['nodes'][0]['source']='STALE';write(wd/'DEV_PUBLIC.jsonl',a)
    elif kind=='pair_query_drift':
        p=rows('PAIR_AUDIT_ONLY.jsonl');ident=next(x['row_ids'][0] for x in p if x['split']=='dev');
        a=rows('DEV_PUBLIC.jsonl');r=next(x for x in a if x['id']==ident);r['query']['target']=r['query']['seeds'][0];write(wd/'DEV_PUBLIC.jsonl',a)
    elif kind=='graph_edge_deleted':
        a=rows('TRAIN.jsonl');a[0]['graph']['factors'].pop();write(wd/'TRAIN.jsonl',a)
    elif kind=='train_label_flip':
        a=rows('TRAIN.jsonl');a[0]['label']^=1;write(wd/'TRAIN.jsonl',a)
    elif kind=='gold_duplicate':
        a=rows('DEV_GOLD.jsonl');a[1]=copy.deepcopy(a[0]);write(wd/'DEV_GOLD.jsonl',a)
    elif kind=='noncanonical_edge_order':
        a=rows('TRAIN.jsonl');a[0]['graph']['factors'][0],a[0]['graph']['factors'][-1]=a[0]['graph']['factors'][-1],a[0]['graph']['factors'][0];write(wd/'TRAIN.jsonl',a)
    else:raise ValueError(kind)

def run(root):
    root=pathlib.Path(root)
    auditor=pathlib.Path(__file__).with_name('h1_audit.py')
    cases=['gold_label_flip','dev_label_leak','pair_count_dropped','node_status_pair_drift','pair_query_drift','graph_edge_deleted','train_label_flip','gold_duplicate','noncanonical_edge_order']
    outcome=[]
    for kind in cases:
        with tempfile.TemporaryDirectory(prefix='h1-mut-',dir=root) as tmp:
            d=pathlib.Path(tmp)
            for name in FILES:shutil.copyfile(root/name,d/name)
            mutate(kind,d)
            p=subprocess.run([sys.executable,'-I','-B',str(auditor),str(d)],capture_output=True,text=True,timeout=60)
            outcome.append({'mutant':kind,'rejected':p.returncode!=0,'returncode':p.returncode,
                            'reason':(p.stderr or p.stdout)[-500:]})
            print('H1_MUTANT='+kind+' '+('REJECTED' if p.returncode else 'SURVIVED'),flush=True)
    out={'schema':'elpis.hca.h1.mutation.v1','classification':'DEV_INSTRUMENT_MECHANICS',
         'mutants':len(cases),'rejected':sum(x['rejected'] for x in outcome),'results':outcome}
    (root/'MUTATIONS.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    if out['rejected']!=len(cases):raise ValueError('H1_AUDITOR_NOT_MUTATION_SENSITIVE')
    print('PASS_HCA_H1_AUDITOR_MUTANTS='+str(out['rejected'])+'/'+str(len(cases)),flush=True)
if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('usage: h1_mutations.py OUTPUT_DIR')
    run(sys.argv[1])
