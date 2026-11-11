"""R125 preregistered consumed-data protocol constructor. NO model calls, NO new seeds."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from research.yang_mills_hca.r118_scalable_awr import provenance as P
from research.yang_mills_hca.r122_energy_direction_forensics import metrics as M

HERE=Path(__file__).resolve().parent
YM=HERE.parent
BASE='7e8536ab2deafae6aabb423f3639ba4d218ce357'
BRANCH='research/ym-r125-early-prefix-clamp-forensics-r0'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build():
    upstream={
       'r119_numerical_hotpath/csrc/awr_kernel_r119.c',
       'r119_numerical_hotpath/engine_r119.py',
       'r119_numerical_hotpath/native_r119.py',
       'r119_numerical_hotpath/guards.py',
       'r122_energy_direction_forensics/harness.py',
       'r122_energy_direction_forensics/metrics.py',
       'r123_k4_prospective/qualify.py',
       'r124_stress_reversal_forensics/qualify.py',
       'r124_stress_reversal_forensics/PROTOCOL_R124.json',
    }
    return {
        'schema':'elpis.ym_hca.r125.early_prefix.forensics.protocol.v1',
        'phase':'R125','mode':'CONSUMED_R123_R124_RETROSPECTIVE',
        'status':'PREPUBLISHED_CONSUMED_FORENSICS',
        'base_commit':BASE,'branch':BRANCH,'r93':'ENERGY_NORMALIZED_DENOMINATOR_HOLD',
        'question':('For each of 40 published R123 scale-4 K4-quarter induced readout reversals, which H1/H2 operator '
                    'prefixes are conditionally sufficient to reproduce the event under fixed suffix contexts, and does '
                    'actual preclamp proposal saturation occur in the first three recurrent updates?'),
        'evidence_scope':'Published R123/R124 events and seeds only; retrospective, no fresh held-out examples.',
        'arms':[0.0,0.25], 'expected_events':40,'target_scale':'2^2','prefix_steps':[1,2],
        'prefix_interventions':('Complete 2^2 choices at steps H1,H2; N or K uniform intervening steps H3..H-1; '
                                'either N or K at target horizon. Enumerate 8 paths for H3 and 16 for H9/H11; '
                                'all original N^H and K^H endpoints must match native and K4 trajectories.'),
        'preclamp_instrument':('Diagnostic-only instrumented R119 C source is compiled exclusively in /dev/shm; '
                               'count per live slot A/B un-clamped proposal norms whose exact clamp scale < 1, '
                               'from native/K4 autonomous incoming states at H1,H2,H3; bitwise parity with '
                               'unmodified R119 is mandatory. R119 original code and weights are never modified.'),
        'decisions':{
            'integrity':'All 40 R124 events and H8/H20 R123 scores replay; exactly 38 H3, one H9 and one H11; '
                       'source/protocol/frozen pins and budget gates pass; instrumented C bitwise equals original '
                       'R119 on every probe; bounded R118 parity checks pass.',
            'disposition':'FORENSIC_PREFIX_PASS_BOUNDED if integrity holds, otherwise NONQUAL_NOT_COMPLETED.',
            'first_step_sufficiency':'A prefix word is conditionally sufficient at the measured event when the matched '
                                     'readout is classified REVERSED by the frozen R122 classifier. Full trajectory '
                                     'causation cannot be deduced uniquely.',
            'preclamp':'Count actual raw per-instance, pre-reconciliation A/B proposal clamp activations. '
                       'A positive clamp count indicates clamp use, but not causal responsibility for a sign reversal.',
            'no_selection_tuning':True,
        },
        'budgets':{'max_hazard_events':40,'max_model_steps':15000,'max_reference_parity_runs':8,
                   'max_seconds':1800,'max_rss_mib':2450,'max_result_bytes':750000},
        'source_pins':{'qualify.py':sha(HERE/'qualify.py'),'instrument.py':sha(HERE/'instrument.py'),
                       'make_protocol.py':sha(__file__)},
        'upstream_pins':{x:sha(YM/x) for x in sorted(upstream)},
        'r124_result_sha256':sha(YM/'r124_stress_reversal_forensics/evidence/R124_RESULT.json'),
        'r123_result_sha256':sha(YM/'r123_k4_prospective/evidence/R123_RESULT.json'),
        'r123_protocol_sha256':sha(YM/'r123_k4_prospective/PROTOCOL_R123.json'),
        'r118_component_digest':P.component_digest(),
        'checkpoint_pins':dict(P.CHECKPOINT_SHA256),
        'classifier':{'aligned':M.D_ALIGNED,'reversed':M.D_REVERSED},
        'constraints':{'no_new_seeds':True,'no_new_graphs':True,'no_training':True,
                       'no_gain_tuning':True,'no_persistent_inference_mutation':True,'no_unbounded_growth':True},
        'limitations':['Retrospective, selection-biased sample of 40 failures, not prospective validation.',
                       'Context-dependent finite H1/H2 interventions do not uniquely identify the whole-trajectory cause.',
                       'Preclamp hit attribution only localizes saturation; causal effect of clamp requires its own intervention.',
                       'No production admission, no new learned policy, R93 hold remains.']
    }


def main():
    p=build();dest=HERE/'PROTOCOL_R125.json'
    dest.write_text(json.dumps(p,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('R125_PROTO_SHA256='+sha(dest),flush=True)
    print('R125_MODEL_DATA_CONSUMED=NONE',flush=True)

if __name__=='__main__':main()
