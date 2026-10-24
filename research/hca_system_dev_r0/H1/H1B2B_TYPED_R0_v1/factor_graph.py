"""Independent typed factor-node learner. Never imports a generator or oracle.

Roles are directed FROM, TO, SCOPE, SOURCE on a bipartite entity/factor graph.
Only admitted public source events are interpreted into per-source features.
All training labels originate in the TRAIN_GOLD file and visible support labels.
"""
from __future__ import annotations
import copy
import math
import time
from collections import defaultdict

ROLE_NAMES=('from','to','scope','source')
RELATIONS=('A','B','C','D')
STATUSES=('VALID','STALE','UNKNOWN')


def assert_public(scene):
    forbidden={'label','gold','rule','is_positive','target_label','answer','hidden_rule'}
    def visit(obj):
        if isinstance(obj,dict):
            if forbidden.intersection(obj):
                raise ValueError('forbidden target/oracle key in public probe')
            for v in obj.values():visit(v)
        elif isinstance(obj,list):
            for v in obj:visit(v)
    visit(scene)


def source_snapshot(events, *, reverse=False):
    state={}
    ev=list(reversed(events)) if reverse else events
    prev=None
    for event in ev:
        if (event.get('kind')!='SOURCE' or event.get('status') not in STATUSES or
            not isinstance(event.get('epoch'),int) or event['epoch']<0):
            raise ValueError('bad SOURCE event')
        stamp=(event['tick'],event['ordinal'])
        if not reverse and prev is not None and stamp<=prev:
            raise ValueError('uncanonical/duplicate source event time')
        prev=stamp
        state[event['source']]=(event['status'],event['epoch'])
    return state


def vectorize_case(scene, support_labels, *, blind_source=False, reverse_events=False,
                   swap_roles=False, erase_factors=False):
    """Return role-aware graph tensor source material, no labels or rule induction.

    The two supervised support labels are visible for EVERY comparator and never
    transformed into an explicit rule identifier or exact solver output here.
    """
    assert_public(scene)
    if len(support_labels)!=2 or any(v not in (0,1) for v in support_labels):
        raise ValueError('exactly two visible binary support labels required')
    snap=source_snapshot(scene['events'],reverse=reverse_events)
    query=scene['query']; seen={}; features=[]
    def add(name, category):
        key=(category,name)
        if key in seen:return seen[key]
        idx=len(features);seen[key]=idx
        if category=='source':
            status,epoch=snap.get(name,('UNKNOWN',-1))
            row=[0.,0.,0.,1.,float(status=='VALID'),float(status=='STALE'),float(status=='UNKNOWN')]
            if blind_source:row=[0.,0.,0.,1.,0.,0.,0.]
        else:
            row=[float(name==query['from']),float(name==query['to']),
                 float(name==query['scope']),0.,0.,0.,0.]
        features.append(row)
        return idx
    qidx=[add(query[k],'entity') for k in ('from','to','scope')]
    factors=[]; links=[]; seen_fids=set()
    for f in ([] if erase_factors else scene['factors']):
        if f['fid'] in seen_fids:raise ValueError('duplicate factor id')
        seen_fids.add(f['fid'])
        if f['rel'] not in RELATIONS:raise ValueError('unexpected relation token')
        st,epoch=snap.get(f['source'],('UNKNOWN',-1))
        ft=[float(f['rel']==r) for r in RELATIONS]
        ft.extend([float(st==s) if not blind_source else 0. for s in STATUSES])
        ft.extend([float(epoch==f['source_version']) if not blind_source else 0.,
                   float(f['roles']['scope']==query['scope'])])
        fi=len(factors);factors.append(ft)
        for ri,role in enumerate(ROLE_NAMES):
            if role=='source':name=f['source'];kind='source'
            else:name=f['roles'][role];kind='entity'
            # Swap only FROM/TO edge roles; retain the same typed n-ary factor count.
            edge_role=(1-ri) if (swap_roles and ri<2) else ri
            links.append((add(name,kind),fi,edge_role))
    return {'node_x':features,'factor_x':factors,'links':links,
            'query_idx':qidx,'support':list(map(float,support_labels))}


def prepare_batch(worlds,labels=None,*,blind_source=False,reverse_events=False,
                  swap_roles=False,erase_factors=False):
    import torch
    nodes=[];factors=[];edge_n=[];edge_f=[];edge_r=[];q=[];supports=[];truth=[]
    factor_batch=[];examples=[]
    for w in worlds:
        support=[int(s['label']) for s in w['support']]
        for p in w['probes']:
            graph=vectorize_case(p,support,blind_source=blind_source,
                  reverse_events=reverse_events,swap_roles=swap_roles,erase_factors=erase_factors)
            offset_n=len(nodes);offset_f=len(factors);case_num=len(examples)
            nodes.extend(graph['node_x']);factors.extend(graph['factor_x'])
            for ni,fi,role in graph['links']:
                edge_n.append(ni+offset_n);edge_f.append(fi+offset_f);edge_r.append(role)
            q.append([offset_n+i for i in graph['query_idx']]);supports.append(graph['support'])
            factor_batch.extend([case_num]*len(graph['factor_x']))
            examples.append((p['case_id'],p['pair_id'],p['pair_type']))
            if labels is not None:truth.append(int(labels[p['case_id']]))
    if not examples:raise ValueError('empty batch')
    def longs(xs):return torch.tensor(xs,dtype=torch.long)
    data={'node_x':torch.tensor(nodes,dtype=torch.float32),
          'factor_x':torch.tensor(factors,dtype=torch.float32).reshape(-1,9),
          'edge_n':longs(edge_n),'edge_f':longs(edge_f),'edge_r':longs(edge_r),
          'q':longs(q),'support':torch.tensor(supports,dtype=torch.float32),
          'f_batch':longs(factor_batch),'ids':examples}
    if labels is not None:data['y']=torch.tensor(truth,dtype=torch.float32)
    return data


def create_model(hidden=24, rounds=3):
    """No arithmetic closure, symbolic law interpreter, memorization table, or oracle."""
    import torch
    from torch import nn
    class FactorNodeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden=hidden;self.rounds=rounds
            self.node_init=nn.Linear(7,hidden)
            self.factor_init=nn.Linear(9,hidden)
            self.to_factor=nn.Parameter(torch.empty(4,hidden,hidden))
            self.to_node=nn.Parameter(torch.empty(4,hidden,hidden))
            nn.init.xavier_uniform_(self.to_factor)
            nn.init.xavier_uniform_(self.to_node)
            self.f_update=nn.Linear(hidden*2,hidden)
            self.n_update=nn.Linear(hidden*2,hidden)
            self.readout=nn.Sequential(nn.Linear(hidden*4+2,hidden),nn.ReLU(),
                                       nn.Linear(hidden,1))
        def forward(self,b):
            n=torch.relu(self.node_init(b['node_x']))
            f=torch.relu(self.factor_init(b['factor_x']))
            nidx=b['edge_n'];fidx=b['edge_f'];rid=b['edge_r']
            for _ in range(self.rounds):
                # Entity -> typed factor, then factor -> entity. Each is separately
                # role-conditioned; a factor connects all four n-ary participants.
                msg=torch.bmm(n[nidx].unsqueeze(1),self.to_factor[rid]).squeeze(1)
                agg_f=torch.zeros_like(f).index_add(0,fidx,msg)
                f=torch.relu(self.f_update(torch.cat((f,agg_f),dim=-1)))
                msg_back=torch.bmm(f[fidx].unsqueeze(1),self.to_node[rid]).squeeze(1)
                agg_n=torch.zeros_like(n).index_add(0,nidx,msg_back)
                n=torch.relu(self.n_update(torch.cat((n,agg_n),dim=-1)))
            fb=torch.zeros((len(b['ids']),self.hidden),device=n.device)
            fb.index_add_(0,b['f_batch'],f)
            counts=torch.bincount(b['f_batch'],minlength=len(b['ids'])).clamp(min=1).unsqueeze(1)
            fb=fb/counts
            query_states=n[b['q']].reshape(len(b['ids']),3*self.hidden)
            return self.readout(torch.cat((query_states,fb,b['support']),dim=-1)).flatten()
    return FactorNodeNet()


def train_fixed(train,*,seed=11021,epochs=120,lr=0.008,hidden=24,rounds=3):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    net=create_model(hidden,rounds)
    optimizer=torch.optim.Adam(net.parameters(),lr=lr)
    y=train['y'];history=[]
    t0=time.perf_counter_ns()
    for ep in range(epochs):
        net.train();optimizer.zero_grad(set_to_none=True)
        logits=net(train)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
        if not bool(torch.isfinite(loss)):raise RuntimeError('non-finite TRAIN loss')
        loss.backward()
        if ep==0:
            if net.to_factor.grad is None or not torch.isfinite(net.to_factor.grad).all():
                raise RuntimeError('missing/non-finite factor-role gradient')
            if net.to_node.grad is None or not torch.isfinite(net.to_node.grad).all():
                raise RuntimeError('missing/non-finite entity-role gradient')
        optimizer.step()
        if ep in (0,19,59,epochs-1):
            history.append({'epoch':ep+1,'train_loss':round(float(loss.item()),7)})
    elapsed=time.perf_counter_ns()-t0
    net.eval()
    return net,history,elapsed


def predict(model,batch):
    import torch
    with torch.no_grad():
        scores=torch.sigmoid(model(batch)).tolist()
    return {case_id:{'score':float(score),'label':int(score>=0.5)}
            for (case_id,_,_),score in zip(batch['ids'],scores)}


def metrics(batch,predictions):
    gold={key:int(y) for (key,_,_),y in zip(batch['ids'],batch['y'].tolist())}
    pair={};wrong=[];fp=0;fn=0
    for case_id,pair_id,kind in batch['ids']:
        pr=predictions[case_id]['label'];y=gold[case_id]
        if pr!=y:wrong.append(case_id)
        if pr and not y:fp+=1
        if not pr and y:fn+=1
        pair.setdefault(pair_id,[]).append((kind,int(pr==y)))
    bytype={}
    for kind in ('source_swap','event_order','epoch_rebind','structural'):
        rows=[v for v in pair.values() if v[0][0]==kind]
        if any(len(v)!=2 or v[0][0]!=v[1][0] for v in rows):raise ValueError('pair contract')
        bytype[kind]={'pairs':len(rows),'both':sum(sum(x[1] for x in v)==2 for v in rows),
                      'one':sum(sum(x[1] for x in v)==1 for v in rows),
                      'neither':sum(sum(x[1] for x in v)==0 for v in rows)}
    return {'correct':len(gold)-len(wrong),'total':len(gold),
            'false_positives':fp,'false_negatives':fn,'pairs':bytype,'errors':wrong}
