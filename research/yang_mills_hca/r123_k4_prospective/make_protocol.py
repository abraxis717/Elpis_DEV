"""Construct R123 prospective protocol without running a model or touching fresh seeds."""
from __future__ import annotations
import hashlib
import json
import math
import pathlib
import random

from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r122_energy_direction_forensics import topology as T
from research.yang_mills_hca.r122_energy_direction_forensics import qualify as Q122

HERE = pathlib.Path(__file__).resolve().parent
YM = HERE.parent
BASE = '501be02d33201ce9c95e211d6ba9a1b531656f43'


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def candidate_graphs():
    """Deterministic, no-model scan. Require 2 one-hop and 2 two-hop unseen classes."""
    prior = set(T.historical_classes())
    r122 = json.loads((YM / 'r122_energy_direction_forensics/PROTOCOL_R122.json').read_text())
    prior.update(T.signature(g) for g in r122['graphs'].values())
    rng = random.Random(123400)
    found = {}
    for hop, label in ((1,'R123_H1_A'), (2,'R123_H2_A'), (1,'R123_H1_B'), (2,'R123_H2_B')):
        for _ in range(500_000):
            g = [sorted(rng.sample(range(10), 4)) for _ in range(4)]
            if len(set(map(tuple,g))) != 4 or T.participants(g) != 10 or not T.connected(g):
                continue
            if T.max_degree(g) > 3 or T.hop_distance(g) != hop:
                continue
            sig = T.signature(g)
            if sig in prior:
                continue
            found[label] = g
            prior.add(sig)
            break
        else:
            raise RuntimeError('NO_UNUSED_TOPOLOGY_WITH_REQUIRED_HOPS:'+str(hop))
    audit = T.audit(found)
    if not audit['all_novel'] or len(set(T.signature(g) for g in found.values())) != 4:
        raise RuntimeError('GRAPH_NOVELTY_FAILED')
    return found


def build():
    graphs = candidate_graphs()
    old = Q122.consumed_seeds() | {122101,122103,122107,122109,122300,122301,122302,122303,122304,122305,122400}
    ev = [123101,123103,123107,123109]
    structure = 123300
    new = set(ev) | {structure + i for i in range(4)} | {123400,123900}
    derived = Q122.pair_state_generator_seeds(ev) & Q122.pair_state_generator_seeds(old)
    if old & new or len(new)!=10 or derived:
        raise RuntimeError('SEED_LEAK')
    import research.yang_mills_hca.r122_energy_direction_forensics.linear as L
    teacher_k4 = float(L.teacher_limits()[4])
    if abs(teacher_k4 - (1.0/(1.0+math.exp(-0.8)))) > 1e-14:
        raise RuntimeError('TEACHER_GATE_LIMIT_DRIFT')
    src = {
        'qualify.py': sha(HERE/'qualify.py'),
        'make_protocol.py': sha(__file__),
        'r118_component_digest':P.component_digest(),
        'r119_kernel':sha(YM/'r119_numerical_hotpath/csrc/awr_kernel_r119.c'),
        'r119_engine':sha(YM/'r119_numerical_hotpath/engine_r119.py'),
        'r119_guard':sha(YM/'r119_numerical_hotpath/guards.py'),
        'r122_harness':sha(YM/'r122_energy_direction_forensics/harness.py'),
        'r122_linear':sha(YM/'r122_energy_direction_forensics/linear.py'),
        'r122_metrics':sha(YM/'r122_energy_direction_forensics/metrics.py'),
        'r122_topology':sha(YM/'r122_energy_direction_forensics/topology.py'),
        'r122_result':sha(YM/'r122_energy_direction_forensics/evidence/R122_RESULT.json'),
        'r121_result':sha(YM/'r121_temporal_topology/evidence/R121_RESULT.json'),
    }
    return {
        'schema':'elpis.ym_hca.r123.prospective.v1','phase':'R123',
        'mode':'PREREGISTERED_FRESH_DATA_CAUSAL_SHADOW',
        'status':'FROZEN_BEFORE_EVALUATION',
        'base_commit':BASE,
        'r93':'ENERGY_NORMALIZED_DENOMINATOR_HOLD',
        'graphs':graphs,'graph_seed':123400,'required_hops':[1,2,1,2],
        'evaluation_seeds':ev,'structure_seed':structure,
        'cases':['weak','anti'],'k_values':[0,2,6],
        'scales':{'2^-20':2.0**-20,'2^-2':0.25,'1':1.0,'2^2':4.0},
        'source_ratio':160,'batch':4,
        'checkpoints':list(P.CHECKPOINTS),
        'arms':[0.0,0.25,0.5],
        'primary_fraction':0.25,'secondary_fraction':0.50,
        'target_gate':'k4 B<-B; interpolate post-sigmoid limit, modify only shadow bias[4]',
        'teacher_k4_limit':teacher_k4,
        'mode_native':'R119 EXACT_GUARDED; MEAN reconciliation; H1-H20',
        'views':{'primary':'all eight fixture-copy states, each four relations x eight coordinates; no distractor-node averaging',
                 'secondary':['paired fixture state','paired readout A at relation slot 3','teacher forced H8/H20 one-step from T[h-1]','readout reversal each H1..H20']},
        'decision_rules':{
            'induced_reversal':'READOUT REVERSED in shadow while matched NATIVE READOUT ALIGNED, from R122 exact classifier; every step',
            'in_range_scales':['2^-20','2^-2','1'],
            'above_clamp':['2^2'],
            'max_in_range_induced_reversals_primary':0,
            'primary_h20_scale':'1',
            'primary_cohort_k_min_relative_improvement':0.20,
            'primary_graphs_improving_min_per_cohort_k':3,
            'primary_max_graph_cohort_k_abs_h20_harm':0.10,
            'primary_h8_pooled_noninferiority_relative':0.05,
            'primary_teacher_forced_h20_pooled_noninferiority_relative':0.10,
            'secondary_has_no_rescue_authority':True,
            'disposition':'PASS_BOUNDED only if all source/numerical/finite/weight/budget gates and all primary conditions pass; else NONQUAL or SCIENTIFIC_FAIL_BOUNDED',
        },
        'source_pins':src,
        'frozen_manifest':P.MANIFEST_SHA256,
        'checkpoint_pins':dict(P.CHECKPOINT_SHA256),
        'budget':{'max_engine_steps':250_000,'max_seconds':2400,'max_rss_mib':2450,'max_result_bytes':6_000_000},
        'limitations':['Oracle-derived gate target is offline causal calibration, not learned or teacher-free calibration.',
                       'Four new graph classes in the 10-participant/4-relation family, degree <=3 only.',
                       'Four independent graph classes give limited statistical power; repeated cases/seeds are correlated.',
                       'Zero observed induced reversals does not prove universal directional safety.',
                       'Even PASS_BOUNDED cannot clear R93 or authorize runtime promotion.'],
        'no_training':True,'no_teacher_in_autonomous_inference':True,
        'no_persistent_inference_mutation':True,
    }


def main():
    data = build()
    target = HERE/'PROTOCOL_R123.json'
    target.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('PROTO_GENERATED='+str(target))
    print('PROTO_SHA256='+sha(target))
    for name,g in data['graphs'].items():
        print('GRAPH='+name+' HOPS='+str(T.hop_distance(g)))

if __name__ == '__main__':main()
