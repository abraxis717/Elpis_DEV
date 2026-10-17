"""Run frozen R8, post-hoc refusal-inclusive accounting. No experiment rerun."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import os
import sys

WORKTREE = Path('/mnt/primesauce/Elpis_DEV_WORK/ecsc_hca_r0')
RESEARCH = WORKTREE / 'research/ecsc_hca_r0'
RESULT = RESEARCH / 'R8_DEV_FALSIFICATION.json'
SOURCE = RESEARCH / 'r8_falsification.py'
OUTPUT = RESEARCH / 'R8_DESCRIPTIVE_ACCOUNTING.json'
EXPECTED_RESULT_SHA256 = '4d405f8bfcd1e563e1f71e1cc19c7167927781a69b595b562deb181f270fa4aa'
EXPECTED_SOURCE_SHA256 = '0a1a28ecc1b88b19200fdb9ea787aae5d104fbd83955d39ed1cfead75c0851f8'
METHODS = ('INDEX_ONLY', 'ADMITTED_ONLY', 'B0', 'B0I', 'B2S')
SCHEMA = 'elpis.ecsc.r8.descriptive-accounting.v1'


def demand(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(label)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def accounting(report: dict) -> dict:
    demand(report.get('schema') == 'elpis.ecsc.r8.dev-epistemic-falsification.v1', 'SCHEMA_MISMATCH')
    profiles = report.get('profiles')
    demand(type(profiles) is list and len(profiles) == 4, 'PROFILE_COUNT')
    expected_profiles = {(4, False), (4, True), (8, False), (8, True)}
    demand({(p['arity'], p['reverse']) for p in profiles} == expected_profiles, 'PROFILE_IDENTITIES')
    records = {method: [] for method in METHODS}
    targets = 0
    for p in profiles:
        demand(type(p.get('events')) is list and len(p['events']) == 20, 'EVENT_COUNT')
        names = []
        for event in p['events']:
            name = event['event']
            names.append(name)
            target_count = event.get('withheld_count')
            demand(type(target_count) is int and 0 <= target_count <= 256, 'WITHHELD_COUNT')
            targets += target_count
            entries = event.get('methods')
            demand(type(entries) is dict and set(entries) == set(METHODS), 'METHOD_SET')
            for method in METHODS:
                entry = entries[method]
                refusal, s = entry.get('refusal'), entry.get('score')
                demand((refusal is None) != (s is None), 'EXACTLY_ONE_DISPOSITION:' + method + ':' + name)
                if refusal is not None:
                    demand(type(refusal) is str and refusal, 'INVALID_REFUSAL')
                    records[method].append({'refusal': refusal, 'tp': 0, 'fp': 0,
                                            'fn_answered': 0, 'fn_refused': target_count,
                                            'invalid': 0, 'work': None, 'keys': None})
                else:
                    demand(type(s) is dict, 'INVALID_SCORE')
                    for key in ('tp', 'fp', 'fn', 'invalid', 'logical_work'):
                        demand(type(s.get(key)) is int and s[key] >= 0, 'INVALID_METRIC:'+key)
                    demand(s['tp'] + s['fn'] == target_count, 'PER_EVENT_TARGET_ACCOUNTING:'+method+':'+name)
                    demand(type(s.get('predicted_keys')) is list, 'MISSING_PREDICTIONS')
                    demand(s['tp'] + s['fp'] <= len(s['predicted_keys']), 'PREDICTION_CONSISTENCY')
                    records[method].append({'refusal': None, 'tp': s['tp'], 'fp': s['fp'],
                                            'fn_answered': s['fn'], 'fn_refused': 0,
                                            'invalid': s['invalid'], 'work': s['logical_work'],
                                            'keys': s['predicted_keys']})
        demand(len(set(names)) == 20, 'DUPLICATE_EVENTS')
    demand(targets == 300, 'FROZEN_EXPECTED_TARGET_COUNT')
    demand(len(records['B0']) == 80, 'TOTAL_EVENT_COUNT')

    summary = {}
    for method, rows in records.items():
        success = [x for x in rows if x['refusal'] is None]
        refusals = Counter(x['refusal'] for x in rows if x['refusal'] is not None)
        tp, fp = sum(x['tp'] for x in rows), sum(x['fp'] for x in rows)
        fn_answered = sum(x['fn_answered'] for x in rows)
        fn_refused = sum(x['fn_refused'] for x in rows)
        fn_all = fn_answered + fn_refused
        demand(tp + fn_all == targets, 'DENOMINATOR_DRIFT:' + method)
        demand(sum(x['invalid'] for x in rows) == 0, 'INVALID_SOURCE_PREDICTION:' + method)
        partial_work = sum(x['work'] for x in success)
        summary[method] = {
            'answered_events': len(success),
            'refused_events': len(rows) - len(success),
            'refusals': dict(sorted(refusals.items())),
            'tp': tp,
            'fp': fp,
            'fn_answered_only': fn_answered,
            'fn_from_refused_events': fn_refused,
            'fn_all_events': fn_all,
            'expected_targets_all_events': targets,
            'recall_fraction_all_events': f'{tp}/{targets}',
            'reported_work_answered_events_only': partial_work,
            'work_at_refusal': 'UNMEASURED',
        }
        original = report['summary'][method]
        for key, expected in (('tp',tp), ('fp',fp), ('fn',fn_answered), ('logical_work',partial_work), ('answered',len(success))):
            demand(original[key] == expected, 'FROZEN_SUMMARY_MISMATCH:'+method+':'+key)
        demand(original['refusals'] == summary[method]['refusals'], 'REFUSAL_COUNTS_DRIFT:'+method)
    for a,b in zip(records['B0'], records['B0I']):
        demand(a['refusal'] == b['refusal'], 'B0I_REFUSAL_DRIFT')
        demand((a['tp'],a['fp'],a['fn_answered'],a['fn_refused'],a['keys']) ==
               (b['tp'],b['fp'],b['fn_answered'],b['fn_refused'],b['keys']), 'B0I_PREDICTION_DRIFT')
    demand(summary['B0']['tp'] == 260 and summary['B0I']['tp'] == 260, 'BASELINE_TARGET_DRIFT')
    demand(summary['B2S']['tp'] == 256, 'RECURRENT_TARGET_DRIFT')
    return {
        'schema': SCHEMA,
        'classification': 'RETROSPECTIVE_DEV_DESCRIPTIVE_AUDIT_NOT_PRE_REGISTERED_NOT_SCIENTIFIC_QUAL',
        'input_report_sha256': EXPECTED_RESULT_SHA256,
        'input_source_sha256': EXPECTED_SOURCE_SHA256,
        'events': 80, 'expected_targets': targets,
        'operation_cost_caveat': 'Recorded logical work includes successful readouts only; refused-turn costs and graph admission are excluded. Values are not end-to-end hardware performance.',
        'independence_caveat': 'Four deterministic profiles share the same authored fixture and labels; reversed ordering is a sensitivity test, not independent evidence.',
        'oracle_caveat': 'All withheld labels are synthetic and graph-derived, not independently adjudicated semantic truth.',
        'posthoc_accounting': summary,
        'decision': {
            'research_candidate': 'B0_FRESH_EXACT',
            'retain_comparator': 'B0I_INCREMENTAL_EXACT',
            'do_not_promote': 'B2S_RECURRENCE',
            'next_qualification': 'PROSPECTIVE_INDEPENDENTLY_ADJUDICATED_SOURCE_BACKED_TASKS',
            'canonical_integration': 'NOT_AUTHORIZED',
        },
    }


def main():
    demand(SOURCE.is_file() and RESULT.is_file(), 'FROZEN_R8_INPUTS_MISSING')
    demand(digest(SOURCE) == EXPECTED_SOURCE_SHA256, 'R8_SOURCE_HASH_MISMATCH')
    demand(digest(RESULT) == EXPECTED_RESULT_SHA256, 'R8_RESULTS_HASH_MISMATCH')
    report = accounting(json.loads(RESULT.read_text()))
    wire = (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True)+'\n').encode()
    if OUTPUT.exists():
        demand(OUTPUT.read_bytes() == wire, 'EXISTING_AUDIT_CONFLICT_NO_OVERWRITE')
    else:
        temp = OUTPUT.with_name(OUTPUT.name+'.tmp.'+str(os.getpid()))
        try:
            with open(temp, 'xb') as out:
                out.write(wire)
                out.flush()
                os.fsync(out.fileno())
            # Refuse unintentional overwrite if competing run created output during write.
            demand(not OUTPUT.exists(), 'EXISTING_AUDIT_CONFLICT_NO_OVERWRITE')
            os.link(temp, OUTPUT)
        finally:
            temp.unlink(missing_ok=True)
    print('R8_DESCRIPTIVE_AUDIT_REPORT=',OUTPUT)
    print('R8_DESCRIPTIVE_AUDIT_SHA256=',sha256(wire).hexdigest())
    for method, s in report['posthoc_accounting'].items():
        print(method, 'TP=',s['tp'], 'FN_ANSWERED=',s['fn_answered_only'],
              'FN_REFUSED=',s['fn_from_refused_events'], 'ALL_TARGETS=', s['expected_targets_all_events'],
              'ANSWERED=',s['answered_events'], 'REFUSALS=',s['refused_events'],
              'RECORDED_WORK_PARTIAL=',s['reported_work_answered_events_only'])
    print('R8_DESCRIPTIVE_ACCOUNTING=PASS_POSTHOC_NOT_SCIENTIFIC_QUAL')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('R8_DESCRIPTIVE_ACCOUNTING=HOLD', type(exc).__name__, str(exc),file=sys.stderr)
        sys.exit(3)
