"""R124 retrospective, source-pinned R123 above-clamp reversal forensics.

No new seed, no graph novelty claim, no training, no coefficient fit.
Autonomous trajectories use real R119 EXACT_GUARDED private-shadow biases.
The four one-step counterfactuals are diagnostic, not inference policies.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import numpy as np

from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r118_scalable_awr.frozen_oracle import get_oracle
from research.yang_mills_hca.r122_energy_direction_forensics import harness as H
from research.yang_mills_hca.r122_energy_direction_forensics import metrics as M
from research.yang_mills_hca.r122_energy_direction_forensics import topology as TOP
from research.yang_mills_hca.r123_k4_prospective import qualify as R123

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
YM = HERE.parent
PROTO = HERE / 'PROTOCOL_R124.json'
R123_PROTO = YM / 'r123_k4_prospective/PROTOCOL_R123.json'
R123_RESULT = YM / 'r123_k4_prospective/evidence/R123_RESULT.json'
R93 = 'ENERGY_NORMALIZED_DENOMINATOR_HOLD'
BASE = '53057654cd8f11a4e8d95d0dcad469d294ca9498'
BRANCH = 'research/ym-r124-stress-reversal-forensics-r0'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(cond, why):
    if not cond:
        raise RuntimeError(why)


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], stderr=subprocess.PIPE, text=True).strip()


def verify_source(proto):
    need(proto['phase'] == 'R124' and proto['mode'] == 'CONSUMED_R123_FORENSICS', 'PROTOCOL_MODE_DRIFT')
    need(proto['r93'] == R93 and proto['base_commit'] == BASE, 'BASE_OR_R93_DRIFT')
    need(proto['source_sha256'] == sha(__file__), 'QUALIFY_SOURCE_DRIFT')
    for rel,expected in proto['source_pins'].items():
        need(sha(YM / rel) == expected, 'UPSTREAM_SOURCE_DRIFT:' + rel)
    need(sha(R123_PROTO) == proto['r123_protocol_sha256'], 'R123_PROTOCOL_DRIFT')
    need(sha(R123_RESULT) == proto['r123_result_sha256'], 'R123_RESULT_DRIFT')
    need(P.component_digest() == proto['r118_component_digest'], 'R118_DIGEST_DRIFT')
    need(P.verify_manifest()['files_verified'] == 39, 'FROZEN_MANIFEST_DRIFT')
    need(P.CHECKPOINT_SHA256 == proto['checkpoint_pins'], 'CHECKPOINT_PINS_DRIFT')
    need(tuple(proto['arms']) == (0.0, 0.25), 'ARM_DRIFT')
    need(proto['expected_induced'] == 40 and proto['scale'] == '2^2', 'EVENT_TARGET_DRIFT')
    need(proto['reversal_definition'] == 'R123_NATIVE_ALIGNED_SHADOW_REVERSED', 'CLASSIFIER_DRIFT')
    need(git('rev-parse', '--abbrev-ref', 'HEAD') == BRANCH, 'WRONG_RESEARCH_BRANCH')
    need(git('status', '--porcelain') == '', 'WORKTREE_UNCLEAN_BEFORE_EVALUATION')
    head = git('rev-parse', 'HEAD')
    need(git('rev-parse', 'HEAD^') == BASE, 'PREREG_PARENT_DRIFT')
    need(not (HERE / 'evidence/R124_RESULT.json').exists(), 'ALREADY_PUBLISHED_RESULT')
    remote = os.environ.get('ELPIS_GITHUB_REMOTE', '')
    need('github.com' in remote and 'abraxis717/Elpis_DEV' in remote, 'REMOTE_NOT_GITHUB')
    expected = git('ls-remote', remote, 'refs/heads/' + BRANCH).split()
    need(expected and expected[0] == head, 'PREREG_NOT_PUBLISHED_OR_REMOTE_DRIFT')
    return head


def paths_from(X, S):
    """Postreconciliation state norms: clamp proxies, NOT actual preclamp activations."""
    a = np.linalg.norm(X[:, :4], axis=-1)
    b = np.linalg.norm(X[:, 4:], axis=-1)
    allv = np.concatenate([a, b])
    return {'max_post_relation_norm': float(allv.max()),
            'post_relation_vectors_near_4': int(np.sum((allv >= 3.99) & (allv <= 4.000000001)))}


def viewpoint(S, X, T, step, pair):
    d = M.decompose(S.view(X, 'READOUT'), S.view(T[step], 'READOUT'))
    p = S.view(X, 'READOUT')[pair]
    t = S.view(T[step], 'READOUT')[pair]
    return {'direction_class': M.DIR_CLASSES[int(d['dir_class'][pair])],
            'cosine': float(d['c'][pair]) if np.isfinite(d['c'][pair]) else None,
            'norm_ratio': float(d['r'][pair]) if np.isfinite(d['r'][pair]) else None,
            'relative_error': float(d['erel'][pair]) if np.isfinite(d['erel'][pair]) else None,
            'prediction_norm': float(np.linalg.norm(p)),
            'teacher_norm': float(np.linalg.norm(t))}


def cross_step(S, state, engine):
    S.load(state)
    result = engine.step(S.h)
    need(result.exact and result.disposition == 'EXECUTED', 'ONE_STEP_REFUSAL')
    return S.dump()


def pathway_label(direct, inherited):
    """One-step diagnostic sufficiency, NOT an identified global causal model."""
    if direct and inherited:
        return 'BOTH_ONE_STEP_PATHS_REVERSED'
    if direct:
        return 'DIRECT_PATH_REVERSED'
    if inherited:
        return 'INHERITED_PATH_REVERSED'
    return 'JOINT_PATH_ONLY_REVERSED'


def evaluate(out_path):
    t0 = time.monotonic()
    proto = json.loads(PROTO.read_bytes())
    pre = verify_source(proto)  # completed before any replay
    prior = json.loads(R123_RESULT.read_bytes())
    registered = json.loads(R123_PROTO.read_bytes())
    need(prior['disposition'] == 'PASS_BOUNDED', 'R123_DISPOSITION_DRIFT')
    need(prior['prereg_commit'] == '0c3da9f2da5d3aa9840132133e00eb8cadafd407', 'R123_PREREG_DRIFT')
    need(prior['directional_hazards']['2^2/0.25'] == 40, 'R123_STRESS_SUMMARY_DRIFT')
    need(registered['checkpoints'] == list(P.CHECKPOINTS), 'R123_CHECKPOINT_SET_DRIFT')
    need(M.D_REVERSED == 3 and M.D_ALIGNED == 0, 'CLASSIFIER_INDEX_DRIFT')
    # Build a complete inventory from published prospective measurements only.
    raw = prior['raw_records']
    positive = [r for r in raw if r['frac'] == 0.25 and r['scale'] == '2^2' and r['induced'] > 0]
    need(bool(positive) and sum(r['induced'] for r in positive) == 40, 'R123_EVENT_INVENTORY_INCOMPLETE')
    keys = [(r['g'], r['k'], r['ck'], r['seed'], r['case']) for r in positive]
    need(len(set(keys)) == len(keys), 'DUPLICATE_EVENT_RUN_KEYS')
    need(all(r['seed'] in registered['evaluation_seeds'] and r['case'] in registered['cases'] and
             r['g'] in registered['graphs'] and r['k'] in registered['k_values'] and
             r['ck'] in registered['checkpoints'] for r in positive), 'EVENT_FIXTURE_OUT_OF_PROTOCOL')
    need(len(positive) <= proto['budgets']['max_hazard_runs'], 'RUN_BUDGET_REFUSAL')

    oracle = get_oracle()
    need(oracle.parameter_pins() == P.CHECKPOINT_SHA256, 'FROZEN_CHECKPOINT_AUTH')
    need(M is not None, 'METRICS_MISSING')
    events, run_checks, parity = [], [], 0
    total_steps = 0
    target = float(registered['teacher_k4_limit'])
    graph_names = list(registered['graphs'])
    group = defaultdict(list)
    for r in positive:
        group[r['g']].append(r)
    for name in graph_names:
        selected = group.get(name, [])
        if not selected:
            continue
        gi = graph_names.index(name)
        graph = registered['graphs'][name]
        print(f'R124_GRAPH_START={name} HOPS={TOP.hop_distance(graph)} RUNS={len(selected)}', flush=True)
        systems, _ = H.build_systems(oracle, graph, registered['structure_seed'] + gi, ks=tuple(registered['k_values']))
        for r in selected:
            S = systems[r['k']]
            a, b = oracle.r64.pair_state(r['seed'], r['case'], registered['source_ratio'], batch=registered['batch'])
            X0 = S.x0(a.numpy(), b.numpy(), scale=registered['scales']['2^2'])
            T = S.teacher(oracle, X0)
            en = R123.engine(S, r['ck'], 0.0, target)
            ek = R123.engine(S, r['ck'], 0.25, target)
            N = R123.simulate(S, X0, en)
            K = R123.simulate(S, X0, ek)
            total_steps += 40
            need(np.isfinite(N).all() and np.isfinite(K).all() and np.isfinite(T).all(), 'NONFINITE_STATE')
            # Replay R123 published aggregate observations (not newly measured scores).
            for x, traj in ((0.0, N), (0.25, K)):
                relevant = [z for z in raw if z['g'] == name and z['k'] == r['k'] and z['ck'] == r['ck'] and
                            z['seed'] == r['seed'] and z['case'] == r['case'] and z['scale'] == '2^2' and z['frac'] == x]
                need(len(relevant) == 1, 'PUBLISHED_RUN_RECORD_NOT_UNIQUE')
                for h,field in ((8,'h8'),(20,'h20')):
                    actual = R123.state_nse(S,traj[h],T[h])
                    want = relevant[0][field]
                    need(abs(actual-want) <= 1e-10*max(1.0, abs(want)), 'PUBLISHED_NSE_REPLAY_DRIFT')
            n_dirs = np.stack([M.decompose(S.view(N[h], 'READOUT'), S.view(T[h], 'READOUT'))['dir_class'] for h in range(1, 21)])
            k_dirs = np.stack([M.decompose(S.view(K[h], 'READOUT'), S.view(T[h], 'READOUT'))['dir_class'] for h in range(1, 21)])
            loc = np.argwhere((n_dirs == M.D_ALIGNED) & (k_dirs == M.D_REVERSED))
            need(len(loc) == r['induced'], 'PUBLISHED_REVERSALS_REPLAY_DRIFT')
            # In-range matched fixture must have no induced reversals (R123 registered claim).
            x1 = S.x0(a.numpy(), b.numpy(), scale=1.0)
            t1 = S.teacher(oracle, x1)
            n1 = R123.simulate(S, x1, R123.engine(S,r['ck'],0.0,target))
            k1 = R123.simulate(S, x1, R123.engine(S,r['ck'],0.25,target))
            total_steps += 40
            nd1 = np.stack([M.decompose(S.view(n1[h],'READOUT'), S.view(t1[h],'READOUT'))['dir_class'] for h in range(1,21)])
            kd1 = np.stack([M.decompose(S.view(k1[h],'READOUT'), S.view(t1[h],'READOUT'))['dir_class'] for h in range(1,21)])
            need(not np.any((nd1 == M.D_ALIGNED) & (kd1 == M.D_REVERSED)), 'R123_IN_RANGE_REPLAY_DRIFT')
            # Fixed small sample: exact guarded vs R118 reference, no learned updates.
            if parity < proto['budgets']['max_reference_parity_runs']:
                n_ref = R123.simulate(S, X0, R123.engine(S,r['ck'],0.0,target,mode='EXACT_R118'))
                k_ref = R123.simulate(S, X0, R123.engine(S,r['ck'],0.25,target,mode='EXACT_R118'))
                total_steps += 40
                need(np.array_equal(n_ref.view(np.uint64), N.view(np.uint64)) and
                     np.array_equal(k_ref.view(np.uint64), K.view(np.uint64)), 'GUARDED_EXACT_REFERENCE_MISMATCH')
                parity += 1
            first_by_pair = {}
            for h0, pair0 in loc:
                h, pair = int(h0)+1, int(pair0)
                first_by_pair.setdefault(pair, h)
            for h0, pair0 in loc:
                h, pair = int(h0)+1, int(pair0)
                # Four one-step possibilities with the same target T[h].
                from_native_k = cross_step(S, N[h-1], ek)
                from_k_native = cross_step(S, K[h-1], en)
                from_native_native = cross_step(S, N[h-1], en)
                from_k_k = cross_step(S, K[h-1], ek)
                total_steps += 4
                need(np.array_equal(from_native_native.view(np.uint64), N[h].view(np.uint64)), 'N_TO_N_ONE_STEP_NOT_IDENTICAL')
                need(np.array_equal(from_k_k.view(np.uint64), K[h].view(np.uint64)), 'K_TO_K_ONE_STEP_NOT_IDENTICAL')
                view = { 'NN': viewpoint(S, N[h], T, h, pair),
                         'NK': viewpoint(S, from_native_k, T, h, pair),
                         'KN': viewpoint(S, from_k_native, T, h, pair),
                         'KK': viewpoint(S, K[h], T, h, pair)}
                need(view['NN']['direction_class'] == 'ALIGNED' and view['KK']['direction_class'] == 'REVERSED', 'EVENT_DETECTOR_DRIFT')
                direct = view['NK']['direction_class'] == 'REVERSED'
                inherited = view['KN']['direction_class'] == 'REVERSED'
                # Report explicit near-zero and clamp proxies; NEVER infer true preclamp hits from pooled states.
                event = {'graph':name,'k':r['k'],'hops':TOP.hop_distance(graph),'seed':r['seed'],'case':r['case'],
                         'checkpoint':r['ck'],'cohort':r['cohort'],'pair':pair,'step':h,
                         'first_induced_for_pair':h == first_by_pair[pair],
                         'paths':view,'pathway_class':pathway_label(direct,inherited),
                         'before':{'native':viewpoint(S,N[h-1],T,h-1,pair),
                                   'shadow':viewpoint(S,K[h-1],T,h-1,pair)} if h>1 else None,
                         'post_norm_proxy':{'native':paths_from(N[h],S), 'shadow':paths_from(K[h],S)},
                         'incoming_state_cosine':float(np.dot(N[h-1].ravel(),K[h-1].ravel()) /
                            (np.linalg.norm(N[h-1])*np.linalg.norm(K[h-1]))) if np.linalg.norm(N[h-1]) and np.linalg.norm(K[h-1]) else None}
                events.append(event)
            run_checks.append({'graph':name,'k':r['k'],'checkpoint':r['ck'],'seed':r['seed'],'case':r['case'],
                               'published_count':r['induced'],'replayed_count':int(len(loc)),
                               'in_range_reversals':0})
        print(f'R124_GRAPH_DONE={name} EVENTS={sum(e["graph"]==name for e in events)}', flush=True)
        need(time.monotonic()-t0 <= proto['budgets']['max_seconds'], 'TIME_BUDGET_EXCEEDED')
    need(len(events) == proto['expected_induced'], 'TOTAL_EVENT_COUNT_MISMATCH')
    need(total_steps <= proto['budgets']['max_engine_steps'], 'STEP_BUDGET_EXCEEDED')
    need(parity >= 1, 'NO_NUMERICAL_PARITY_CHECK')
    need(oracle.parameter_pins() == P.CHECKPOINT_SHA256 and P.verify_manifest()['files_verified'] == 39, 'FROZEN_CHANGED')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    seconds = time.monotonic()-t0
    need(rss <= proto['budgets']['max_rss_mib'] and seconds <= proto['budgets']['max_seconds'], 'RSS_OR_TIME_BUDGET')
    result = {'schema':'elpis.ym_hca.r124.forensics.v1','phase':'R124','mode':'PUBLISHED_R123_CONSUMED_REPLAY',
              'disposition':'FORENSIC_REPLAY_PASS_BOUNDED','r93':R93,'base_commit':BASE,'prereg_commit':pre,
              'source_sha256':sha(__file__),'protocol_sha256':sha(PROTO),
              'r123_result_sha256':sha(R123_RESULT),'original_event_count':40,
              'replayed_events':len(events),'by_pathway_class':dict(Counter(e['pathway_class'] for e in events)),
              'by_graph':dict(Counter(e['graph'] for e in events)),
              'by_cohort':dict(Counter(e['cohort'] for e in events)),
              'by_step':{str(k):v for k,v in sorted(Counter(e['step'] for e in events).items())},
              'first_events':sum(e['first_induced_for_pair'] for e in events),
              'events':events,'run_checks':run_checks,'integrity':{'all_original_hazards_replayed':True,
              'source_guard':True,'in_range_replay_zero':True,'counterfactual_endpoint_bit_identity':True,
              'guarded_vs_exact_r118_parity_runs':parity,'checkpoint_pins_unchanged':True},
              'resources':{'seconds':round(seconds,2),'peak_rss_mib':round(rss,1),'engine_steps':total_steps},
              'limits':['Retrospective on published R123 samples; no new generalization claim.',
                        'One-step paths are conditional counterfactuals, not full dynamical causal identification.',
                        'Postreconciliation norms are NOT evidence of preclamp activation.',
                        'No trained calibrator, no teacher available at inference, no R93 clearance.']}
    blob = (json.dumps(result, sort_keys=True, indent=1, allow_nan=False)+'\n').encode()
    need(len(blob) <= proto['budgets']['max_result_bytes'], 'RESULT_BUDGET')
    need(not out_path.exists(), 'RESULT_ALREADY_WRITTEN')
    out_path.parent.mkdir(parents=True,exist_ok=True)
    out_path.write_bytes(blob)
    print('=== SEND BACK ===',flush=True)
    for line in [
        'PHASE=R124_STRESS_REVERSAL_FORENSICS',
        'DISPOSITION=FORENSIC_REPLAY_PASS_BOUNDED',
        'MODE=CONSUMED_R123_NO_NEW_HOLDOUT',
        'BASE_COMMIT='+BASE,'PREREG_COMMIT='+pre,
        'PROTOCOL_SHA256='+sha(PROTO),'RESULT_SHA256='+hashlib.sha256(blob).hexdigest(),
        'EVENTS_REPLAYED='+str(len(events))+'/40',
        'ONE_STEP_PATHWAYS='+json.dumps(result['by_pathway_class'], sort_keys=True),
        'BY_GRAPH='+json.dumps(result['by_graph'],sort_keys=True),
        'BY_COHORT='+json.dumps(result['by_cohort'],sort_keys=True),
        'BY_STEP='+json.dumps(result['by_step'],sort_keys=True),
        'FIRST_PAIR_EVENTS='+str(result['first_events']),
        'GUARDED_EXACT_PARITY_RUNS='+str(parity),
        'IN_RANGE_REPLAY_INDUCED=0',
        'FROZEN=UNCHANGED','R93='+R93,
        'BLOCKER=NONE CHILD_EXIT=0']:
        print(line,flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    if not out.is_relative_to('/dev/shm'):
        raise SystemExit('VOLATILE_OUTPUT_ONLY')
    try:
        evaluate(out)
    except Exception as exc:
        traceback.print_exc()
        print('=== SEND BACK ===\nPHASE=R124_STRESS_REVERSAL_FORENSICS\nDISPOSITION=NONQUAL_NOT_COMPLETED'
              '\nBLOCKER='+type(exc).__name__+':'+str(exc)[:300]+'\nR93='+R93+'\nCHILD_EXIT=2',flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
