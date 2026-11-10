"""R121 prospective new-topology whole-trajectory evaluation.

New topology/structure/evaluation seeds are committed before execution.
Frozen R118 and R119 cores are consumed read-only. All solver trajectories
are produced autonomously; R30 teacher is evaluation-only. No fitted policy.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import pathlib
import time
import math

import numpy as np

from .policy import controller_at

HERE = pathlib.Path(__file__).resolve().parent
PROTOCOL = HERE / "PROTOCOL_R121.json"


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def signature(graph):
    """Four-arity relation/ten-participant incidence isomorphism diagnostic."""
    if len(graph) != 4 or any(len(set(x)) != 4 for x in graph) or any(
            not all(type(p) is int and 0 <= p < 10 for p in x) for x in graph):
        raise ValueError("INCIDENCE_SHAPE")
    return min(tuple(sum(sum(1 << i for i,r in enumerate(perm) if p in graph[r]) == mask
                         for p in range(10)) for mask in range(16))
               for perm in itertools.permutations(range(4)))


def source_guard(proto):
    from research.yang_mills_hca.r118_scalable_awr import characterize_e as e
    from research.yang_mills_hca.r118_scalable_awr import provenance as P
    from research.yang_mills_hca.r119_numerical_hotpath import engine_r119 as er
    if digest(__file__) != proto["qualify_sha256"] or digest(HERE / "policy.py") != proto["policy_sha256"]:
        raise RuntimeError("R121_REGISTERED_SOURCE_DRIFT")
    if digest(e.__file__) != proto["r118_characterize_e_sha256"]:
        raise RuntimeError("R118E_DRIFT")
    if P.component_digest() != proto["r118_component_digest"]:
        raise RuntimeError("R118_COMPONENT_DRIFT")
    if digest(HERE.parent / 'r119_numerical_hotpath/csrc/awr_kernel_r119.c') != proto["r119_kernel_sha256"]:
        raise RuntimeError("R119_KERNEL_DRIFT")
    if digest(HERE.parent / 'r120_checkpoint_policy/evidence/R120_RAW_FRESH.json') != proto["r120_raw_sha256"]:
        raise RuntimeError("R120_EVIDENCE_DRIFT")
    # Source-pinned frozen oracle is authenticated before any new test states.
    from research.yang_mills_hca.r118_scalable_awr.frozen_oracle import get_oracle
    o = get_oracle()
    if o.parameter_pins() != P.CHECKPOINT_SHA256:
        raise RuntimeError("FROZEN_CHECKPOINT_DRIFT")
    source_graphs = (dict(o.r61['r55_structural']['graphs']) |
                     dict(o.pm.read_json('FROZEN_PROTOCOL_R76.json')['graphs']) |
                     dict(o.r77.CFG['graphs']))
    original = {signature(g) for g in source_graphs.values()}
    candidate = [signature(g) for g in proto['graphs'].values()]
    if len(candidate) != len(set(candidate)) or original.intersection(candidate):
        raise RuntimeError("INCIDENCE_ISOMORPHISM_LEAK")
    for g in proto['graphs'].values():
        if len({x for rel in g for x in rel}) != 10:
            raise RuntimeError("GRAPH_PARTICIPANT_COVERAGE")
    old_seeds = e.historical_seeds() | set(e.EVAL_SEEDS) | {e.STRUCTURE_SEED} | {120101,120103,120107,120109,120300}
    new_seeds = set(proto['evaluation_seeds']) | {proto['structure_seed']+i for i in range(4)}
    if len(new_seeds) != len(proto['evaluation_seeds'])+4 or old_seeds.intersection(new_seeds):
        raise RuntimeError("SEED_LEAK")
    return e, P, er, o, {n: list(x) for n,x in zip(proto['graphs'],candidate)}


def simulate(e, er, st, handles, fid, checkpoint, policy, horizons, budgets):
    """Run a full trajectory, switching at a predetermined step boundary only."""
    from research.yang_mills_hca.r119_numerical_hotpath.engine_r119 import R119Engine
    engines = {}
    ret = {}
    for step in range(1, max(horizons)+1):
        choice = controller_at(checkpoint, policy, step)
        if choice not in engines:
            engines[choice] = R119Engine(st, checkpoint, controller=choice, mode='EXACT_GUARDED', budgets=budgets)
            # R119 disables only the softmax guard when this optional libm
            # precondition fails; the exact fallback remains valid.
        result = engines[choice].step(handles)
        if not result.exact or result.disposition != 'EXECUTED':
            raise RuntimeError('ENGINE_REFUSAL:' + str(result.disposition))
        if step in horizons:
            ret[step] = e.readout(st, fid).copy()
    return ret


def evaluate():
    started = time.monotonic()
    proto = json.loads(PROTOCOL.read_text())
    if proto['status'] != 'SOURCE_PREREGISTERED_BEFORE_NEW_GRAPH_RUN':
        raise RuntimeError('UNREGISTERED_SOURCE')
    e,P,er,o,signatures = source_guard(proto)
    if proto['horizons'] != [8,20] or proto['k_values'] != [0,2,6]:
        raise RuntimeError('PROTOCOL_DRIFT')
    from research.yang_mills_hca.r118_scalable_awr import engine as E
    horizons = tuple(proto['horizons']); kvals=tuple(proto['k_values'])
    budgets = E.Budgets(max_passes=20)
    records = {}
    parity_max = 0.0
    early_mismatches = 0
    modes = ('NATIVE','R114','SIGNED_ONLY','DELAYED8')
    for gi,(name,graph) in enumerate(sorted(proto['graphs'].items())):
        print(f'R121_GRAPH_START={name} ({gi+1}/{len(proto["graphs"])})', flush=True)
        spec=e.structure(np.random.default_rng(proto['structure_seed']+gi))
        reg, windows, fid = e.composed_registry(o, graph, spec)
        for case in proto['cases']:
            for seed in proto['evaluation_seeds']:
                st0=o.r64.pair_state(seed,case,160,batch=4)
                targets={}
                for k in kvals:
                    st,h=e.build_state(reg,windows(k),fid,st0)
                    targets[k]=e.teacher_rollout(o,reg,st,h,fid)
                # Independent isolated-source parity for each new graph/seed.
                _,adj=o.graph_adjs(graph)
                with o.torch.no_grad():
                    true=o.r64.rollout(st0,adj,o.r30.NonlinearTeacher(o.torch),20,keep=horizons)
                for H in horizons:
                    observed=o.r63.paired(true[H][0],4).numpy()
                    parity_max=max(parity_max,float(np.abs(targets[0][H]-observed).max()))
                for ck in P.CHECKPOINTS:
                    for k in kvals:
                        out={}
                        for mode in ('NATIVE','R114'):
                            st,h=e.build_state(reg,windows(k),fid,st0)
                            out[mode]=simulate(e,er,st,h,fid,ck,mode,horizons,budgets)
                        out['SIGNED_ONLY']=out[controller_at(ck,'SIGNED_ONLY',20)]
                        if ck.endswith('/STATE8_SIGNED8'):
                            st,h=e.build_state(reg,windows(k),fid,st0)
                            out['DELAYED8']=simulate(e,er,st,h,fid,ck,'DELAYED8',horizons,budgets)
                            if not np.array_equal(out['DELAYED8'][8].view(np.uint64),out['NATIVE'][8].view(np.uint64)):
                                early_mismatches+=1
                        else:
                            out['DELAYED8']=out['NATIVE']
                        for mode in modes:
                            for H in horizons:
                                m=e.metrics(out[mode][H],targets[k][H])
                                tag=f'{name}/k{k}/{mode}/{ck}/H{H}'
                                rec=records.setdefault(tag,{'nse':[],'matched_nse':[],'energy_ratio':[],
                                                            'teacher_shift_nse':[]})
                                shift=e.metrics(targets[k][H],targets[0][H])['nse']
                                for key,val in [('nse',m['nse']),('matched_nse',m['matched_nse']),
                                                ('energy_ratio',m['energy_ratio']),('teacher_shift_nse',shift)]:
                                    rec[key].extend(float(x) for x in val)
        print('R121_GRAPH_COMPLETE='+name,flush=True)
    if parity_max>1e-12 or early_mismatches:
        raise RuntimeError('NUMERICAL_PARITY_OR_EARLY_IDENTITY:'+str((parity_max,early_mismatches)))
    if any(len(v['nse'])!=32 or not all(math.isfinite(x) for vv in v.values() for x in vv)
           for v in records.values()):
        raise RuntimeError('CELL_COMPLETENESS_OR_FINITE')
    by_cell={key:{n:float(np.mean(v)) for n,v in rec.items()} | {'n':len(rec['nse'])}
             for key,rec in records.items()}
    aggregate={}
    for k in kvals:
        for H in horizons:
            for mode in modes:
                vals=[v for key,v in by_cell.items() if f'/k{k}/{mode}/' in key and key.endswith(f'/H{H}')]
                if len(vals)!=24:
                    raise RuntimeError('BAD_CELL_COUNT:'+str((k,H,mode,len(vals))))
                aggregate[f'k{k}/{mode}/H{H}']={m:float(np.mean([v[m] for v in vals]))
                                              for m in ('nse','matched_nse','energy_ratio')}
    # Registered and non-overlapping decisions: topology transfer of prior fixed policy;
    # distinct temporal proposal to preserve native H8 and repair H20.
    transfer={};temporal={};blocks={}
    for k in kvals:
        n=aggregate[f'k{k}/NATIVE/H20']['nse']
        a=aggregate[f'k{k}/R114/H20']['nse']
        s=aggregate[f'k{k}/SIGNED_ONLY/H20']['nse']
        d=aggregate[f'k{k}/DELAYED8/H20']['nse']
        transfer[str(k)]={'delta_native':n-s,'delta_r114':a-s,'pass':s<n and s<a}
        graph_harms=[]
        for g in proto['graphs']:
            def mn(mode):
                return np.mean([v['nse'] for key,v in by_cell.items()
                    if key.startswith(f'{g}/k{k}/{mode}/') and key.endswith('/H20')])
            graph_harms.append(float(mn('DELAYED8')-mn('NATIVE')))
        temporal[str(k)]={'delta_native':n-d,'delta_signed_only':s-d,
                          'max_graph_harm_vs_native':max(graph_harms),
                          'pass':d<n and max(graph_harms)<=proto['maximum_graph_harm_vs_native']}
        blocks[str(k)]={g:float(np.mean([v['nse'] for key,v in by_cell.items()
                    if key.startswith(f'{g}/k{k}/DELAYED8/') and key.endswith('/H20')]))
                      for g in proto['graphs']}
    early_exact=early_mismatches==0 and all(
        aggregate[f'k{k}/DELAYED8/H8']['nse']==aggregate[f'k{k}/NATIVE/H8']['nse'] for k in kvals)
    transfer_pass=all(v['pass'] for v in transfer.values())
    temporal_pass=early_exact and all(v['pass'] for v in temporal.values())
    return {'phase':'R121','schema':'elpis.ym_hca.r121.v1','source_commit_parent':proto['source_commit_parent'],
            'protocol_sha256':digest(PROTOCOL),'qualify_source_sha256':digest(__file__),
            'policy_sha256':digest(HERE/'policy.py'),'signatures':signatures,
            'eval_seeds':proto['evaluation_seeds'],'structure_seed':proto['structure_seed'],
            'r93':'ENERGY_NORMALIZED_DENOMINATOR_HOLD', 'frozen_checkpoint_pins_unchanged':o.parameter_pins()==P.CHECKPOINT_SHA256,
            'teacher_k0_parity_max_abs':parity_max,'delayed_vs_native_h8_bit_mismatches':early_mismatches,
            'early_identity':early_exact,'decisions':{'r120_transfer_on_unseen_topology':
                         'PASS_BOUNDED' if transfer_pass else 'NONQUAL',
                         'delayed8_temporal_policy':'PASS_BOUNDED' if temporal_pass else 'NONQUAL'},
            'transfer':transfer,'temporal':temporal,'by_graph_delayed_h20':blocks,'aggregate':aggregate,
            'by_cell':by_cell,'elapsed_seconds':round(time.monotonic()-started,2)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    args=p.parse_args()
    path=pathlib.Path(args.out).resolve()
    if not path.is_relative_to('/dev/shm'):
        raise SystemExit('VOLATILE_OUTPUT_ONLY')
    obj=evaluate()
    path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print('=== SEND BACK ===',flush=True)
    print('PHASE=R121_NEW_TOPOLOGY_TEMPORAL_POLICY',flush=True)
    for k,v in obj['decisions'].items():print('GATE_'+k.upper()+'='+v,flush=True)
    for k,v in obj['transfer'].items():print('TRANSFER_K'+k+'='+json.dumps(v,sort_keys=True),flush=True)
    for k,v in obj['temporal'].items():print('DELAYED_K'+k+'='+json.dumps(v,sort_keys=True),flush=True)
    print('H8_BIT_IDENTITY='+str(obj['early_identity'])+' TEACHER_K0_PARITY='+str(obj['teacher_k0_parity_max_abs']),flush=True)
    print('R93='+obj['r93']+' BLOCKER=NONE CHILD_EXIT=0',flush=True)

if __name__=='__main__': main()
