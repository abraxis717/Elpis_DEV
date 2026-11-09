#!/usr/bin/env python3
"""Elpis YM/AWR R63-M: source-bound, archive-independent H1 intervention seam.

Uses exact unmodified R30 invariant/teacher, R62 transport and graph_step_split,
R63 inputs and hybrid_step from verified captured sources. This is a MECHANICAL
fixture qualification with deterministic control HCA weights, NOT the frozen
R52 learned models, R63 scientific evaluation, or physical Spark GDN execution.
"""
from __future__ import annotations
import copy, hashlib, io, json, math, os, pathlib, statistics, sys, traceback, zipfile
import torch as t
import r30_source as r30
import r62_source as r62
import r63_source as r63

ROOT = pathlib.Path(__file__).resolve().parent
EXPECTED_SOURCE = {
    'r30_source.py': '51e05a242b9ec6411d8a2bb093f779b4d7ac8bab4e1ac9def28d4d89f6b3d8b9',
    'r62_source.py': 'fe272da88e3833ad5deaad305dc5ab4d07b666af1fe357a07da7df10e1a799ee',
    'r63_source.py': '6f2fb05aabbc06e8b1892f35949bdcc168941831a368ede9b1f895a6b4877eea',
    'FROZEN_PROTOCOL_R61.json': '632222d5ecfdb8fd0acb51a6de732a819251bfe2cdc3351e900be78445a9dda4',
    'FROZEN_PROTOCOL_R62.json': '39ae807dd4a40039912240933c2d5004b002acd3f8458c96636cedb42ef1a886',
    'FROZEN_PROTOCOL_R63.json': 'c900364137f00309c34a7ca49c89f7cb8bfffb63b10257b6e2d28cb5a4ccd81b',
}
MODES = ('BASE','INJECT_ONLY','GATE_ONLY','FULL')
GRAPHS=('D1_P1','D1_P2','D2_P1','D2_P2')
CASES=('weak','reverse','anti')
ARMS=('CONTROL_A','CONTROL_B')
SEEDS=(9103,9109,9119,9127,9133,9151,9161,9173)
class Hold(RuntimeError):pass
def require(p,msg):
    if not p:raise Hold(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def read_json(p):return json.loads((ROOT/p).read_bytes())
def norm(v):return t.linalg.vector_norm(v,dim=-1)
def dot(u,v):return (u*v).sum(-1)
def fmt(x):return format(float(x),'.9g')

def source_admission():
    for name,digest in EXPECTED_SOURCE.items():
        require((ROOT/name).is_file() and sha((ROOT/name).read_bytes())==digest, 'PIN_MISMATCH:'+name)
    p=read_json('FROZEN_PROTOCOL_R61.json')
    p62=read_json('FROZEN_PROTOCOL_R62.json')
    p63=read_json('FROZEN_PROTOCOL_R63.json')
    require(list(p['r55_structural']['graphs'])==list(GRAPHS),'R55_GRAPH_NAMES')
    require(all(len(g)==4 and all(len(members)==4 for members in g) for g in p['r55_structural']['graphs'].values()),'R55_GRAPH_SHAPES')
    require(p['r55_structural']['cases']['weak']==[.0625,4.,'independent'],'R55_WEAK_CASE')
    require(p['r55_structural']['cases']['reverse']==[4.,.0625,'independent'],'R55_REVERSE_CASE')
    require(p['r55_structural']['cases']['anti']==[.0625,4.,'antialigned'],'R55_ANTI_CASE')
    require(p63['gate_injection_bits']=={'BASE':[0,0],'INJECT_ONLY':[0,1],'GATE_ONLY':[1,0],'FULL':[1,1]},'R63_FACTORIAL')
    require(p62['parent_r61_sha256']=='0d09605632e30850902d9f558f753aa6051e991a23d401ac016e5b9fd215bda3','R62_IDENTITY')
    require(p['r56_transport']['nonflat']['r_scale']>0,'R56_BAD_SCALE')
    return p,p62,p63

def rotation(th1,th2,torch=t):
    c1,s1=math.cos(th1),math.sin(th1)
    c2,s2=math.cos(th2),math.sin(th2)
    return torch.tensor([[c1,-s1,0,0],[s1,c1,0,0],[0,0,c2,-s2],[0,0,s2,c2]],dtype=torch.float64)

class R3Adapter:
    """Only the rotation interface used by the exact source-pinned R62 transport law."""
    @staticmethod
    def so2x2_torch(theta1,theta2,torch):return rotation(theta1,theta2,torch)

def link_builder(graph,matrices):
    """Construct U_{dst<-p} U_{src<-p}^{-1} for each authenticated shared p.
    Finite synthetic incidence only; NOT a real HACF authority/admission ticket.
    """
    out={}
    for dst in range(4):
        neighbors=[];ops=[]
        for src in range(4):
            if src==dst:continue
            for p in sorted(set(graph[dst]).intersection(graph[src])):
                neighbors.append(src)
                ops.append(matrices[(dst,p)]@matrices[(src,p)].T)
        if neighbors:
            out[dst]=(t.tensor(neighbors,dtype=t.long),t.stack(ops))
    return out

def gauged_links(adjs,gauges):
    return {dst:(src.clone(),t.stack([gauges[dst]@op@gauges[int(s)].T for s,op in zip(src,ops)]))
            for dst,(src,ops) in adjs.items()}

def gauge_state(state,g):
    return tuple(t.stack([t.einsum('ij,bj->bi',g[i],channel[:,i,:]) for i in range(4)],dim=1) for channel in state)

class HCAControl:
    def __init__(self,arm):
        self.arm=arm
        self.gates=t.nn.Linear(10,8).to(dtype=t.float64)
        with t.no_grad():
            g=t.Generator(device='cpu').manual_seed(811 if arm=='CONTROL_A' else 1237)
            if arm=='CONTROL_A':
                self.gates.weight.copy_(t.randn((8,10),generator=g,dtype=t.float64)*.13)
                self.gates.bias.copy_(t.tensor([.6,.2,.3,.05,.55,-.1,.25,.09],dtype=t.float64))
            else:
                self.gates.weight.copy_(t.randn((8,10),generator=g,dtype=t.float64)*.05)
                self.gates.bias.copy_(t.tensor([.9,.25,.15,-.04,.8,-.17,.3,.07],dtype=t.float64))
        self.gates.requires_grad_(False)
    def __call__(self,state,sal,col):return r63.hybrid_step(self,state,sal,col,col,r30,t)[0]

def state_pair(seed,case,batch=8):
    """Synthetic source fixture: both groups share non-source state, source differs.
    Intentional independent mechanical data generator; not a substitute for r3.generate.
    """
    ci=CASES.index(case)
    rng=t.Generator(device='cpu').manual_seed(seed*139+ci*83)
    vector=t.randn((batch,4),generator=rng,dtype=t.float64)
    vector=vector/norm(vector).unsqueeze(-1)
    vector2=t.randn((batch,4),generator=rng,dtype=t.float64)
    vector2=vector2/norm(vector2).unsqueeze(-1)
    amp_a,amp_b,relation=read_json('FROZEN_PROTOCOL_R61.json')['r55_structural']['cases'][case]
    if relation=='antialigned':vector2=-vector
    a=t.zeros((2*batch,4,4),dtype=t.float64)
    b=t.zeros_like(a)
    a[:batch,0,:]=amp_a*vector
    a[batch:,0,:]=amp_b*vector2
    b[:batch,0,:]=amp_a*.63*vector
    b[batch:,0,:]=amp_b*.63*vector2
    return (a,b)

def graph_adjs(graph,cfg):
    base={}
    for rid,members in enumerate(graph):
        for p in members:
            base[(rid,p)]=rotation(.19+.17*rid-.067*p,-.41+.103*rid+.049*p)
    original=r62.transport_matrices(None,base,'BASE_SEPARABLE',cfg,R3Adapter(),t)
    nonflat=r62.transport_matrices(None,base,'PARTICIPANT_NONFLAT',cfg,R3Adapter(),t)
    return link_builder(graph,original),link_builder(graph,nonflat)

def run_one(seed,graph_name,case,arm,cfg,preflight=False):
    graph=cfg['r55_structural']['graphs'][graph_name]
    eb,en=graph_adjs(graph,cfg)
    state=state_pair(seed,case)
    model=HCAControl(arm)
    weight_before=sha(b''.join(x.detach().cpu().numpy().tobytes() for x in model.gates.parameters()))
    teacher=r30.NonlinearTeacher(t)
    ref=state
    for _ in range(20):ref=r62.graph_step_split(ref,eb,eb,teacher,t)
    target=r63.paired(ref[0],8)
    den=dot(target,target)
    require(bool((den>1e-25).all()),'SMALL_FIXED_REFERENCE:'+graph_name+':'+case)
    sa,cb0=r63.inputs(state,eb,eb,t)
    sa1,cb1=r63.inputs(state,eb,en,t)
    require(float((sa-sa1).abs().max())<=1e-13,'A_TRANSPORT_DRIFT')
    k0=r63.coeff(model,state,sa,cb0,r30,t)
    k1=r63.coeff(model,state,sa,cb1,r30,t)
    # Batch all four independent interventions into one recurrent unroll.
    h1={};pred={};h20={}
    for mode,(gb,inj) in {'BASE':(0,0),'INJECT_ONLY':(0,1),'GATE_ONLY':(1,0),'FULL':(1,1)}.items():
        out,_,_=r63.hybrid_step(model,state,sa,cb1 if gb else cb0,cb1 if inj else cb0,r30,t)
        h1[mode]=out
    packed=tuple(t.cat([h1[m][k] for m in MODES],dim=0) for k in (0,1))
    for _ in range(19):packed=r62.graph_step_split(packed,eb,eb,model,t)
    step_batch=state[0].shape[0]
    for ix,mode in enumerate(MODES):
        sl=slice(ix*step_batch,(ix+1)*step_batch)
        next_state=(packed[0][sl],packed[1][sl])
        v=r63.paired(next_state[0],8)
        e=dot(v-target,v-target)
        pred[mode]={'error':float((e/den).mean()),'energy_error':float(e.sum()/den.sum()),
                    'abs_error':float(e.mean()),'gain':float((dot(v,target)/den).mean()),
                    'cosine':float((dot(v,target)/(norm(v)*norm(target)+1e-30)).mean()),
                    'worst_error':float((e/den).max()),'vector':v.tolist()}
        h20[mode]=next_state
    baseline_native=r62.graph_step_split(state,eb,eb,model,t)
    full_native=r62.graph_step_split(state,eb,en,model,t)
    local_parity=max(float((h1[mo][z]-native[z]).abs().max()) for mo,native in [('BASE',baseline_native),('FULL',full_native)] for z in (0,1))
    require(local_parity<=1e-12,'H1_NATIVE_PARITY')
    q={m:pred[m]['error'] for m in MODES}
    gate=.5*((q['GATE_ONLY']-q['BASE'])+(q['FULL']-q['INJECT_ONLY']))
    inject=.5*((q['INJECT_ONLY']-q['BASE'])+(q['FULL']-q['GATE_ONLY']))
    interaction=q['FULL']-q['GATE_ONLY']-q['INJECT_ONLY']+q['BASE']
    closure=abs(gate+inject-(q['FULL']-q['BASE']))
    require(closure <= 1e-12*max(1.,*(abs(x) for x in q.values())),'SHAPLEY_TWO_FACTOR_IDENTITY')
    maxcov=0.
    if preflight:
        gauges={r:rotation(.17+.09*r,-.13-.11*r) for r in range(4)}
        gs=gauge_state(state,gauges)
        gb= gauged_links(eb,gauges);gn=gauged_links(en,gauges)
        gsa,gcol0=r63.inputs(gs,gb,gb,t)
        _,gcol1=r63.inputs(gs,gb,gn,t)
        gh1={}
        for mode,(gatebit,injectbit) in {'BASE':(0,0),'INJECT_ONLY':(0,1),'GATE_ONLY':(1,0),'FULL':(1,1)}.items():
            gx,_,_=r63.hybrid_step(model,gs,gsa,gcol1 if gatebit else gcol0,gcol1 if injectbit else gcol0,r30,t)
            gh1[mode]=gx
            expected=gauge_state(h1[mode],gauges)
            maxcov=max(maxcov,*(float((gx[z]-expected[z]).abs().max()) for z in (0,1)))
        gpacked=tuple(t.cat([gh1[m][z] for m in MODES],dim=0) for z in (0,1))
        for _ in range(19):gpacked=r62.graph_step_split(gpacked,gb,gb,model,t)
        for ix,mode in enumerate(MODES):
            sl=slice(ix*step_batch,(ix+1)*step_batch)
            expected=gauge_state(h20[mode],gauges)
            maxcov=max(maxcov,*(float((gpacked[z][sl]-expected[z]).abs().max()) for z in (0,1)))
        require(maxcov<1e-9,'GAUGE_COVARIANCE')
    weight_after=sha(b''.join(x.detach().cpu().numpy().tobytes() for x in model.gates.parameters()))
    require(weight_before==weight_after,'INFERENCE_WEIGHT_MUTATION')
    return {'seed':seed,'graph':graph_name,'case':case,'arm':arm,'modes':pred,
            'h1':{'gate_effect':gate,'inject_effect':inject,'interaction':interaction,'closure':closure,
                  'aggregate_shift':float((cb1-cb0).square().mean().sqrt()),
                  'gate_shift':float((k1-k0).square().mean().sqrt()),
                  'native_parity':local_parity,'gauge_covariance':maxcov if preflight else None,
                  'fixed_reference_min':float(norm(target).min())},'REFERENCE4':target.tolist()}

def run_seed_batch(graph_name,case,arm,cfg):
    """Eight independent paired fixtures through one mathematically identical unroll.
    The batch axis has no inter-seed edges, gates, normalization, or reductions.
    """
    b=8
    n=len(SEEDS)
    B=b*n
    states=[state_pair(seed,case,b) for seed in SEEDS]
    pack=lambda ch:t.cat((t.cat([s[ch][:b] for s in states],dim=0),
                           t.cat([s[ch][b:] for s in states],dim=0)),dim=0)
    state=(pack(0),pack(1))
    graph=cfg['r55_structural']['graphs'][graph_name]
    eb,en=graph_adjs(graph,cfg)
    model=HCAControl(arm)
    param_pin=sha(b''.join(x.detach().numpy().tobytes() for x in model.gates.parameters()))
    teacher=r30.NonlinearTeacher(t)
    ref=state
    for _ in range(20):ref=r62.graph_step_split(ref,eb,eb,teacher,t)
    targets=r63.paired(ref[0],B)
    require(bool((dot(targets,targets)>1e-25).all()),'BATCH_FIXED_REF:'+graph_name+':'+case)
    sal,c0=r63.inputs(state,eb,eb,t)
    sal2,c1=r63.inputs(state,eb,en,t)
    require(float((sal-sal2).abs().max())<1e-12,'BATCH_A_TRANSPORT_DRIFT')
    k0=r63.coeff(model,state,sal,c0,r30,t)
    k1=r63.coeff(model,state,sal,c1,r30,t)
    h1={}
    for mode,(gb,inj) in {'BASE':(0,0),'INJECT_ONLY':(0,1),'GATE_ONLY':(1,0),'FULL':(1,1)}.items():
        h1[mode]=r63.hybrid_step(model,state,sal,c1 if gb else c0,c1 if inj else c0,r30,t)[0]
    native={
        'BASE':r62.graph_step_split(state,eb,eb,model,t),
        'FULL':r62.graph_step_split(state,eb,en,model,t),
    }
    parity=max(float((h1[m][z]-native[m][z]).abs().max()) for m in native for z in (0,1))
    require(parity<=1e-12,'BATCH_H1_SOURCE_STEP_PARITY')
    packed=tuple(t.cat([h1[m][z] for m in MODES],dim=0) for z in (0,1))
    for _ in range(19):packed=r62.graph_step_split(packed,eb,eb,model,t)
    predictions={}
    for mi,mode in enumerate(MODES):
        sl=slice(mi*2*B,(mi+1)*2*B)
        predictions[mode]=r63.paired(packed[0][sl],B)
    def slice_paired(x,si):
        return t.cat((x[si*b:(si+1)*b],x[B+si*b:B+(si+1)*b]),dim=0)
    rows=[]
    for si,seed in enumerate(SEEDS):
        sl=slice(si*b,(si+1)*b)
        ref=targets[sl]
        den=dot(ref,ref)
        pred={}
        for mode in MODES:
            vec=predictions[mode][sl]
            e=dot(vec-ref,vec-ref)
            pred[mode]={'error':float((e/den).mean()),'energy_error':float(e.sum()/den.sum()),
                        'abs_error':float(e.mean()),'gain':float((dot(vec,ref)/den).mean()),
                        'cosine':float((dot(vec,ref)/(norm(vec)*norm(ref)+1e-30)).mean()),
                        'worst_error':float((e/den).max()),'vector':vec.tolist()}
        values={m:pred[m]['error'] for m in MODES}
        gate=.5*(values['GATE_ONLY']-values['BASE']+values['FULL']-values['INJECT_ONLY'])
        inject=.5*(values['INJECT_ONLY']-values['BASE']+values['FULL']-values['GATE_ONLY'])
        mix=values['FULL']-values['GATE_ONLY']-values['INJECT_ONLY']+values['BASE']
        closure=abs(gate+inject-(values['FULL']-values['BASE']))
        require(closure<1e-12*max(1.,*(abs(x) for x in values.values())),'BATCH_TWO_FACTOR_IDENTITY')
        rows.append({'seed':seed,'graph':graph_name,'case':case,'arm':arm,'modes':pred,
                     'h1':{'gate_effect':gate,'inject_effect':inject,'interaction':mix,'closure':closure,
                           'aggregate_shift':float((slice_paired(c1,si)-slice_paired(c0,si)).square().mean().sqrt()),
                           'gate_shift':float((slice_paired(k1,si)-slice_paired(k0,si)).square().mean().sqrt()),
                           'native_parity':parity,'gauge_covariance':None,
                           'fixed_reference_min':float(norm(ref).min())},'REFERENCE4':ref.tolist()})
    require(param_pin==sha(b''.join(x.detach().numpy().tobytes() for x in model.gates.parameters())),
            'BATCH_WEIGHT_MUTATION')
    return rows

def independent_tests(cfg):
    # Real negatives: a forbidden input must throw, no broad handler swallows the test assertion.
    def rejected(fn):
        try:fn()
        except Hold:return
        raise AssertionError('NEGATIVE_TEST_FAILED_TO_REJECT')
    rejected(lambda:require(False,'BAD'))
    assert len(SEEDS)==len(set(SEEDS))
    assert all(set(g[0])=={0,1,2,7} for g in cfg['r55_structural']['graphs'].values())
    for name,graph in cfg['r55_structural']['graphs'].items():
        eb,en=graph_adjs(graph,cfg)
        assert set(eb)==set(en)
        assert all(t.equal(eb[d][0],en[d][0]) for d in eb)
        assert all(len(eb[d][0])==eb[d][1].shape[0] for d in eb)
        assert all(bool(t.isfinite(eb[d][1]).all()) and bool(t.isfinite(en[d][1]).all()) for d in eb)
        for mat in (eb,en):
            for _,op in mat.values():
                identity=t.eye(4,dtype=t.float64).expand(len(op),4,4)
                assert float((op@op.transpose(-1,-2)-identity).abs().max())<1e-12
    model=HCAControl('CONTROL_A')
    state=state_pair(9103,'weak')
    a=state[0]
    assert (a[:,1:,:]==0).all()
    # No relation bridges: an isolated destination must remain exactly zero under recurrence.
    isolated={}
    for _ in range(20):state=r62.graph_step_split(state,isolated,isolated,model,t)
    assert float(state[0][:,3,:].abs().max())==0
    # Fault injection: gate versus injection arm distinction must not collapse.
    graph=cfg['r55_structural']['graphs']['D2_P1'];eb,en=graph_adjs(graph,cfg)
    state=state_pair(9103,'weak');sal,c0=r63.inputs(state,eb,eb,t);_,c1=r63.inputs(state,eb,en,t)
    assert float((c1-c0).abs().max())>1e-10
    gate_only=r63.hybrid_step(model,state,sal,c1,c0,r30,t)[0]
    inject_only=r63.hybrid_step(model,state,sal,c0,c1,r30,t)[0]
    assert max(float((gate_only[k]-inject_only[k]).abs().max()) for k in (0,1))>1e-12
    return 'PASS'

def evaluate(full):
    t.set_num_threads(1);t.use_deterministic_algorithms(True);t.set_default_dtype(t.float64)
    cfg,p62,p63=source_admission()
    independent_tests(cfg)
    if not full:
        print('LOCAL_PREFLIGHT graph=D2_P1 case=weak arm=CONTROL_A',flush=True)
        row=run_one(SEEDS[0],'D2_P1','weak','CONTROL_A',cfg,preflight=True)
        return [row]
    rows=[]
    with t.no_grad():
        for gi,graph in enumerate(GRAPHS,1):
            print('BATCHED_MECHANICS_PROGRESS graph='+graph+' '+str(gi)+'/4',flush=True)
            for case in CASES:
                for arm in ARMS:
                    rows.extend(run_seed_batch(graph,case,arm,cfg))
        # Independent serial seed path must agree with the batched execution.
        direct=run_one(SEEDS[0],'D2_P1','weak','CONTROL_A',cfg,preflight=True)
        match=next(r for r in rows if (r['seed'],r['graph'],r['case'],r['arm']) ==
                   (SEEDS[0],'D2_P1','weak','CONTROL_A'))
        for mode in MODES:
            for field in ('error','gain','cosine','energy_error','worst_error'):
                av=direct['modes'][mode][field]
                bv=match['modes'][mode][field]
                require(abs(av-bv)<1e-10*max(1.,abs(av),abs(bv)),
                        'BATCH_SERIAL_PARITY:'+mode+':'+field)
        match['h1']['gauge_covariance']=direct['h1']['gauge_covariance']
    require(len(rows)==8*4*3*2,'CELL_COVERAGE')
    return rows

def report(rows):
    out={}
    for arm in ARMS:
        for case in CASES:
            subset=[r for r in rows if r['arm']==arm and r['case']==case]
            if not subset:continue
            def avg(k):return statistics.mean(r['h1'][k] for r in subset)
            out[arm+'_'+case]={'cells':len(subset),'gate_effect_mean':avg('gate_effect'),'inject_effect_mean':avg('inject_effect'),
                'interaction_mean':avg('interaction'),'full_delta_mean':statistics.mean(r['modes']['FULL']['error']-r['modes']['BASE']['error'] for r in subset),
                'favorable_cells':sum(r['modes']['FULL']['error']<r['modes']['BASE']['error'] for r in subset),
                'graph_full_delta':{g:statistics.mean(r['modes']['FULL']['error']-r['modes']['BASE']['error'] for r in subset if r['graph']==g) for g in GRAPHS if any(r['graph']==g for r in subset)}}
    return out

def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'preflight'
    result='HOLD';blocker='NONE';report_hash='NONE';report_path='NONE';coverage=0;cov=0.;par=0.;out={};evidence_gate='NOT_RUN'
    try:
        require(mode in ('selftest','preflight','run'),'MODE')
        cfg,_,_=source_admission();independent_tests(cfg)
        if mode=='selftest':
            result='SELFTEST_PASS';return 0
        rows=evaluate(full=mode=='run');coverage=len(rows)
        cov=max(r['h1']['gauge_covariance'] for r in rows if r['h1']['gauge_covariance'] is not None)
        par=max(r['h1']['native_parity'] for r in rows)
        require(cov<=1e-9 and par<=1e-12,'MECHANICAL_PARITY')
        out=report(rows)
        if mode=='run':
            evidence={'phase':'YM_AWR_R63_PORTABLE_MECHANICAL_SEAM_R0','scope':'MECHANICAL_FIXTURE_ONLY',
                      'source_pins':EXPECTED_SOURCE,'model_weights':'DETERMINISTIC_CONTROL_WEIGHTS_NOT_R52',
                      'transitive_archive_admission':'NOT_EXECUTED_NOT_CLAIMED',
                      'native_gdn':'NOT_EXECUTED','frozen_learned_utility':'R52_NONQUAL_UNCHANGED',
                      'rows':rows,'summary':out}
            import verify_evidence
            check_rows,check_cov=verify_evidence.validate(evidence)
            verify_evidence.adversarial(evidence)
            require(check_rows==192 and check_cov>=1,'EVIDENCE_RECOMPUTATION')
            evidence_gate='PASS_192_ROWS_5_ADVERSARIAL_REJECTIONS'
            blob=json.dumps(evidence,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            require(len(blob)<3_000_000,'EVIDENCE_BOUNDS')
            z=io.BytesIO()
            with zipfile.ZipFile(z,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
                zi=zipfile.ZipInfo('R63_PORTABLE_MECHANICAL_EVIDENCE.json',(2026,11,6,0,0,0))
                zi.compress_type=zipfile.ZIP_DEFLATED
                archive.writestr(zi,blob)
            buf=z.getvalue();report_hash=sha(buf)
            target=pathlib.Path(os.environ.get('YM_HANDOFF_DIR','/mnt/primesauce/Downloads'))
            require(target.is_dir() and os.access(target,os.W_OK),'MAILBOX_NOT_WRITABLE')
            path=target/('YM_AWR_R63_PORTABLE_MECHANICAL_RESULT_'+report_hash[:16]+'.zip')
            if path.exists():require(sha(path.read_bytes())==report_hash,'EVIDENCE_COLLISION')
            else:with_file_write(path,buf)
            require(sha(path.read_bytes())==report_hash,'EVIDENCE_POSTWRITE')
            report_path=str(path)
        result='MECHANICAL_PASS' if mode=='run' else 'PREFLIGHT_PASS'
        return 0
    except Exception as ex:
        blocker=type(ex).__name__+':'+str(ex)
        if not isinstance(ex,Hold):traceback.print_exc(limit=5)
        return 2
    finally:
        print('\n=== SEND BACK ===')
        print('PHASE=YM_AWR_R63_PORTABLE_MECHANICAL_SEAM_R0')
        print('MODE='+mode+' RESULT='+result)
        print('SOURCE_PINS_R30_SCRIPT_R61_PROTOCOL_R62_R63='+('PASS' if blocker=='NONE' else 'CHECK_LOG'))
        print('SCOPE=MECHANICAL_FIXTURE_ONLY;NO_FROZEN_R52_WEIGHTS;NO_HISTORICAL_ARCHIVE_DEPENDENCY')
        print('EVAL_SEEDS_COMPLETED='+('8/8' if result=='MECHANICAL_PASS' else '0/8'))
        print('CELLS='+str(coverage)+'/192' if mode=='run' else 'CELLS='+str(coverage)+'/1')
        print('R62_GRAPH_STEP_PARITY='+fmt(par)+' GAUGE_COVARIANCE_MEASURED='+fmt(cov)+' GAUGE_CELLS='+str(sum(r['h1']['gauge_covariance'] is not None for r in rows)) if coverage else 'R62_GRAPH_STEP_PARITY=NOT_MEASURED GAUGE_COVARIANCE=NOT_MEASURED')
        for key,value in sorted(out.items()):
            print('ATTR_'+key+'=GATE:'+fmt(value['gate_effect_mean'])+' INJECT:'+fmt(value['inject_effect_mean'])+' MIX:'+fmt(value['interaction_mean'])+' FULL:'+fmt(value['full_delta_mean'])+' FAV:'+str(value['favorable_cells'])+'/'+str(value['cells']))
        print('EVIDENCE_RECOMPUTATION='+evidence_gate)
        print('TRANSITIVE_SOURCE_ADMISSION=NOT_EXECUTED')
        print('SCIENTIFIC_DISPOSITION=R63_MECHANICAL_ONLY_NO_NEW_LEARNED_UTILITY')
        print('RESULT_REPORT='+report_path+' REPORT_SHA256='+report_hash)
        print('BLOCKER='+blocker+' CHILD_EXIT='+('0' if result in ('MECHANICAL_PASS','PREFLIGHT_PASS','SELFTEST_PASS') else '2'))
        print('=== END SEND BACK ===')

def with_file_write(path,payload):
    with path.open('xb') as f:f.write(payload)

if __name__=='__main__':sys.exit(main())
